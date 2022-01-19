import glob
import logging
import math
import multiprocessing
import os
import pickle
import time
from typing import List, Union, Dict, Any

import numpy as np


# Calculates the frequencies of each activity in a sequence and stores them in a dict
# Ignored features are not included in the result
def calc_frequencies(seq: Union[list, np.ndarray], normalize=True, ignore_features=None) -> dict:
    if ignore_features is None:
        ignore_features = []
    if type(seq) == np.ndarray:
        seq = seq.tolist()
    # Don't calculate frequencies for ignored features
    freq = {x: seq.count(x) for x in seq if x not in ignore_features}
    if normalize:
        freq = normalize_freq(freq)
    return freq


# def normalize_freq_list(freqs: List[dict]):
#     freqs2 = []
#     for freq in freqs:
#         freqs2.append(normalize_freq(freq))
#     return freqs


# Normalizes a frequency histogram (= dict) to length 1
def normalize_freq(freq: dict) -> dict:
    total = get_vector_length(freq)
    return {k: v / total for k, v in freq.items()}


# Calculates the angular similarity between two sequences
# Note, this does not use n-grams or anything, just frequencies
# The frequencies can be n-gram frequencies though!
# Frequencies MUST be normalized!
def get_angular_sim(freq1: dict, freq2: dict, assert_norm: bool = True):
    if assert_norm:
        _assert_normalized_frequencies([freq1, freq2])

    angular_sim = 1 - get_angular_dist(freq1, freq2, assert_norm=False)
    return angular_sim


# Calculates the angular distance between two frequency dicts
# Frequencies must be normalized
def get_angular_dist(freq1: dict, freq2: dict, assert_norm: bool = True):
    if assert_norm:
        _assert_normalized_frequencies([freq1, freq2])

    cos_sim = get_cosine_sim(freq1, freq2, assert_norm=False)
    angular_dist = 2 / math.pi * math.acos(cos_sim)
    return angular_dist


# Asserts that all frequencies are normalized (i.e. euclidean length of vector = 1)
def _assert_normalized_frequencies(freqs: List[dict], eps=1e-9):
    for freq in freqs:
        assert 1 - eps < get_vector_length(freq) < 1 + eps


def get_vector_length(vector: dict):
    return math.sqrt(sum([val * val for val in vector.values()]))


# Calculates the cosine similarity between two frequency histograms
# frequencies MUST be normalized
def get_cosine_sim(freq1: dict, freq2: dict, assert_norm: bool = True):
    if assert_norm:
        _assert_normalized_frequencies([freq1, freq2])

    # Collect all unique activities
    all_grams = list(set(freq1.keys()).union(set(freq2.keys())))

    # Calculate frequencies of all features
    full_freq1 = [freq1[a] if a in freq1 else 0 for a in all_grams]
    full_freq2 = [freq2[a] if a in freq2 else 0 for a in all_grams]

    # Calculate cosine similarity (recall: length is already normalized)
    cos_sim = sum([full_freq1[i] * full_freq2[i] for i in range(len(all_grams))])

    # Round to correct domain
    eps = 1e-15
    if cos_sim + eps > 1:
        cos_sim = 1
    elif cos_sim - eps < -1:
        cos_sim = -1
    return cos_sim


# Calculates a list of distances from a list of sequences
def calc_distance_matrix(freqs: List[dict], max_sequences=None, function=get_angular_dist, ignore_features=None):
    # freqs = calc_multiple_frequencies(sequences, ignore_features=ignore_features, max_sequences=max_sequences)
    dist = [[function(freq1, freq2) for freq1 in freqs.values()] for freq2 in freqs.values()]
    return dist


# Calculates activity frequencies from a list of activity sequences (i.e. multiple clickstreams at once)
def calc_multiple_frequencies(sequences, ignore_features=None, max_sequences=None, normalize=False):
    if max_sequences is None:
        max_sequences = len(sequences)
    return {index: calc_frequencies(sequence, ignore_features=ignore_features, normalize=normalize)
            for index, sequence in enumerate(sequences[:max_sequences])}


# Calculates a distance matrix from a histogram (=dict) of normalized activity frequencies and returns the distance
# matrix. The matrix is also stored as a pickle (and temp matrices too) if the save_directory is not None
# TODO make this work for partial matrices
def calc_distance_matrix_multithreaded(freqs: List[dict], function=get_angular_dist,
                                       num_threads=4, save_directory=None):
    if len(freqs) < 200:
        print("[INFO] Distance matrix < 200x200, defaulting to single threaded calculation.")
        return calc_distance_matrix(freqs)

    if num_threads > len(freqs) / 200:
        print("[INFO] Adjusted number of threads for distance calculation from %d to %d"
              % (num_threads, int(len(freqs) / 200) + 1))
        num_threads = int(len(freqs) / 200) + 1

    # shared queue to store received distance rows
    results = multiprocessing.Queue(len(freqs))
    threads = []
    # variables for dividing tasks
    start = 0
    step = int(len(freqs) / num_threads + 1)
    # start threads and delegate distance matrix calculation
    start_time = time.time()
    for i in range(num_threads):
        indices = range(start, min(start + step, len(freqs)))
        print("Creating process with indices " + str(indices))
        start += step
        thread = multiprocessing.Process(target=calc_partial_distance_matrix, daemon=True,
                                         kwargs={"freqs": freqs, "indices_from": indices, "function": function,
                                                 "output": results, "save_directory": save_directory, "thread_id": i})
        threads.append(thread)
        thread.start()

    # retrieve calculated distances from other processes and store in own dict
    results_dict = {}
    for i in range(len(freqs)):
        val = results.get(block=True)
        # print("Retrieved row %d" % val[0])
        results_dict[val[0]] = val[1]

    # wait for threads to finish (should not be necessary, since previous process is blocking)
    print("Waiting for processes to shutdown...")
    for thread in threads:
        thread.join()
    print("Processes all finished")
    print("Finished in %.2f seconds" % (time.time() - start_time))

    # Convert dict to distance matrix
    dist = [results_dict[key] for key in sorted(results_dict.keys())]
    # Save results and remove temp files
    if save_directory is not None:
        print("Saving full matrix")
        with open(save_directory + "\\full.pkl", "wb") as file:
            pickle.dump(dist, file)
        print("Removing temp files")
        files = glob.glob(save_directory + "\\tmp*.pkl")
        for f in files:
            os.remove(f)
    return dist


def prepare_data(freqs: Dict[int, Dict[Any, int]], exclude_features=None, exclude_users=None):
    """
    Excludes features and normalizes frequencies
    :param users: list of users to include
    :param freqs: a id -> frequency (=activity -> count mapping) mapping
    :param exclude_features: features to be excluded
    :return: the frequencies, normalized and without the exclusions
    """
    if exclude_features is not None:
        logging.info(f"Excluding {len(exclude_features)} features")
        freqs = exclude_features_from_freqs(freqs, exclude_features)

    # normalize
    logging.info(f"Normalizing frequencies")
    if exclude_users is None:
        return {user: normalize_freq(freq) for user, freq in freqs.items()}
    else:
        return {user: normalize_freq(freq) for user, freq in freqs.items() if user not in exclude_users}


def exclude_features_from_freqs(freqs, exclude_features=None):
    if exclude_features is None:
        exclude_features = []
    modified_freqs = {}
    for user, freq in freqs.items():
        modified_freqs[user] = {feature: count for feature, count in freq.items() if feature not in exclude_features}
    return modified_freqs


# Calculates the rows of the distance table with indices given in index_from. Returns a dictionary containing the
# indices and the related distance row calculated. Result must be a multiprocessing queue for multithreaded
# implementation Note, if excluded features should be removed from the frequencies beforehand
# if filepath is not set, the partial matrices are not stored.
# save percent determines at what percentage the intermediate results are store (e.g. 0.1 -> save at 10%, 20%, etc)
def calc_partial_distance_matrix(freqs: dict, indices_from: Union[range, list], function=get_angular_dist, output=None,
                                 save_directory=None, thread_id=None, save_percent=0.1, exclude_features=None, full_width=True):
    # Handle save file/directory
    save_file = None
    if save_directory is not None:
        if not os.path.exists(save_directory):
            print("Creating directory for saved matrix %s " % save_directory)
            os.mkdir(save_directory)
        save_file = "%s\\tmp_%d.pkl" % (save_directory, indices_from[0])

    # Handle feature exclusions
    if exclude_features is not None:
        freqs = exclude_features_from_freqs(freqs, exclude_features)
        freqs = {user: normalize_freq(freq) for user, freq in freqs.items()}

    # Setup basic information
    partial_dist = {}
    total = len(indices_from)
    # full width -> so we can concat rows later
    # if false, just do a square matric of only indices selected
    if full_width:
        row = range(len(freqs))
    else:
        row = indices_from
    # For logging/saving purposes: determines at what counts we store temp results
    save_moment = [int(total * save_percent * i) for i in range(1, int(1 / save_percent) + 1)]
    for count, index in enumerate(indices_from):
        # Calculate one row of the distance matrix
        dist_row = [function(freqs[index], freqs[j]) for j in row]
        partial_dist[index] = dist_row

        # If actually using multiple threads, send result to main process
        if output is not None:
            output.put((index, dist_row))

        # Store intermediate results in temp directory
        if count in save_moment:
            print("%sFinished %d/%d = %.1f%%. %s"
                  % ("[thread-%03d] " % thread_id if thread_id is not None else "",
                     count, total, count / total * 100,
                     "Saving to " + save_file if save_file is not None else "")
                  )
            if save_file is not None:
                with open(save_file, 'wb') as file:
                    pickle.dump(partial_dist, file)

    # if it is full width, we will merge later so we can return as dict
    if full_width:
        return partial_dist
    else:
        # else we need to convert to list
        return [row for row in partial_dist.values()]
