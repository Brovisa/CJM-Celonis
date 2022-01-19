import multiprocessing
import os
import pickle
import time
from typing import Dict, Iterable, Sized, Any

from numpy import array_split

import src.features.custom_clustering.DistanceMatrixUtils as DMU

MIN_CALC_PER_THREAD = 500000


def calc_distance_matrix(freqs: Dict[int, Dict[Any, int]], indices_from: [Iterable, Sized] = None,
                         indices_to: [Iterable, Sized] = None,
                         function=DMU.get_angular_dist, save_directory=None, verbose=False,
                         save_percent=0.1, exclude_features=None, num_threads: int = 1):
    if verbose:
        print("[INFO] Creating distance matrix")
    # Handle save directory creation
    if save_directory is not None:
        if not os.path.exists(save_directory):
            if verbose:
                print(f"[INFO] Creating directory for saved matrix {save_directory}")
            os.mkdir(save_directory)

    if indices_to is None or indices_from is None:
        all_indices = [key for key in freqs.keys()]
        if indices_to is None:
            indices_to = list(all_indices)
        if indices_from is None:
            indices_from = list(all_indices)

    # prepare data (normalize + exclude features and ids)
    exclude_users = set(freqs.keys() - set(indices_from + indices_to))  # exclude whatever we do not need
    # this will probably crash if we get a user without features, which should be checked somewhere
    freqs = DMU.prepare_data(freqs, exclude_features=exclude_features, exclude_users=exclude_users)

    result_dict = {}
    start_time = time.time()

    # hand off calculation for single threaded
    if num_threads == 1 or len(freqs) ** 2 < MIN_CALC_PER_THREAD:
        save_file = None
        if save_directory is not None:
            save_file = "%s\\tmp_%d.pkl" % (save_directory, indices_from[0])
        print(f"[INFO] Using single threaded calculation (len(freqs) = {len(freqs)})")
        result_dict = _calc_partial_distance_matrix(freqs, indices_from=indices_from, indices_to=indices_to,
                                                    function=function, save_file=save_file, save_percent=save_percent)
    # setup multithreaded approach
    else:
        calc = len(freqs) ** 2
        num_threads = min(1 + calc // MIN_CALC_PER_THREAD, num_threads)
        print(f"[INFO] Using {num_threads} threads for distance matrix calculation (Number of rows = {len(freqs)}).")
        action_time = time.time()
        results_queue = multiprocessing.Queue()
        threads = []
        ranges = array_split(indices_from, num_threads)
        for range_index, partial_indices_from in enumerate(ranges):
            print(
                f"[INFO] Setting up thread {range_index} for indices {partial_indices_from[0]}-{partial_indices_from[-1]} (# of rows: {len(partial_indices_from)})")
            save_file = None
            if save_directory is not None:
                save_file = "%s\\tmp_%d.pkl" % (save_directory, partial_indices_from[0])
            thread = multiprocessing.Process(target=_calc_partial_distance_matrix,
                                             kwargs={"freqs": freqs, "indices_from": partial_indices_from,
                                                     "indices_to": indices_to, "function": function,
                                                     "output": results_queue, "save_file": save_file,
                                                     "thread_id": range_index, "save_percent": save_percent,
                                                     "verbose": verbose},
                                             daemon=True)
            threads.append(thread)
            thread.start()

        print(f"[INFO] Threads started. Time spent setting up threads: {time.time() - action_time:.2f}s")
        # retrieve results from queue

        for i in range(len(freqs)):
            row_index, row = results_queue.get()
            result_dict[row_index] = row
            # update statement
            divisor = max(len(freqs) // 1000, 5)
            if len(freqs) > 100 and i % divisor == 1:
                percent = i / len(freqs) * 100
                duration = time.time() - start_time
                expected = 100 / percent * duration
                print(f"[INFO] Calculated {i}/{len(freqs)} rows ({percent:.1f}%) of dist matrix. "
                      f"Time spent: {duration:.2f}s (expected: {expected:.0f}s)", end="\r")

        print("")  # empty line

        for thread in threads:
            thread.join()

    print(f"[INFO] Finished calcuting {len(freqs)} rows of the the distance matrix. "
          f"Total time spent: {time.time() - start_time:.2f} seconds")
    print(f"[INFO] Merging partial distance matrices...")
    result = [result_dict[index] for index in sorted(result_dict.keys())]
    return result


def _calc_partial_distance_matrix(freqs: Dict[int, Dict[Any, int]], indices_from: [Iterable, Sized],
                                  indices_to: [Iterable, Sized],
                                  function=DMU.get_angular_dist, save_file=None, thread_id=-1, verbose=False,
                                  save_percent=0.1, output: multiprocessing.Queue = None):
    # prepare loop and logging
    partial_dist = {}
    save_moments = [int(len(indices_from) * save_percent * i) for i in range(1, int(1 / save_percent) + 1)]
    for count, index in enumerate(indices_from):
        # Calculate one row of the distance matrix
        dist_row = [function(freqs[index], freqs[j]) for j in indices_to]
        partial_dist[index] = dist_row

        # push result to main thread/process, if multithreaded
        if output is not None:
            output.put((index, dist_row))

        # Store intermediate results in temp directory (and display progress)
        if count in save_moments:
            if verbose:
                print(
                    f"{'' if thread_id < 0 else f'[thread-{thread_id:>03d}]'}[INFO] Finished {count}/{len(indices_from)} = {count / len(indices_from):>5.2%}")
            if save_file is not None:
                with open(save_file, 'wb') as file:
                    pickle.dump(partial_dist, file)

    return partial_dist
