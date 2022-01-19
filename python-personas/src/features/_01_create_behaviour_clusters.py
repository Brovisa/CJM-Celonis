import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

import config
from config import timed
from src.features.custom_clustering import DistanceMatrixUtils as dmu
from src.features.custom_clustering import recursiveDivisiveClustering as rdc

logger = logging.getLogger(__name__)

FULL_FILE = config.DATA_FOLDER_PROCESSED / "sept_all_sequences.parquet"
DICT_FILE = config.DATA_FOLDER_PROCESSED / "sept_sequences_dict.parquet"
DEFAULT_TREE_FILE = config.DATA_FOLDER_PROCESSED / "clusters.json"
DEFAULT_EXTRACT_CLUSTERS_FILE = config.DATA_FOLDER_PROCESSED / "clusters_df.parquet"


@timed
def filter_sequences(sequences: pd.DataFrame, on: str = "all"):
    """Filters sequences so they must contain and/or must not contain certain events. 4 options:
    'purchase' -> must contain checkout:thankyou:pageview
    'near_purchase' -> must contain cart:pageview, must not contain checkout:thankyou:pageview
    'no_purchase' -> must not contain 'checkout:thankyou:pageview
    'all' -> no filtering"""
    valid_on = ["purchase", "near_purchase", "no_purchase", "all"]
    if on not in valid_on:
        raise ValueError("on must be in " + str(valid_on) + ". Your value is " + on)

    must_contain = []
    must_not_contain = []
    if on == "all":
        return sequences
    if on == "purchase":
        must_contain = ['checkout:thankyou:pageview']
    elif on == "near_purchase":
        must_contain = ['checkout:payment:pageview']
        must_not_contain = ['checkout:thankyou:pageview']
    elif on == "no_purchase":
        must_not_contain = ['checkout:thankyou:pageview']

    activity_dict = get_activity_dict()
    activity_dict = {v: k for k, v in activity_dict.items()}

    must_contain = [activity_dict[act] for act in must_contain]
    must_not_contain = [activity_dict[act] for act in must_not_contain]

    def check_sequence(seq):
        for act in must_contain:
            if act not in seq:
                return False
        for act in must_not_contain:
            if act in seq:
                return False
        return True

    logger.info(f"Filtering users. Criteria: '{on}'")
    selection = sequences['sequence'].apply(check_sequence)
    selected_sequences = sequences.loc[selection].copy()
    logger.info(f"Finished filtering users. Total rows selected: {selection.sum()}")
    return selected_sequences


@timed
def cluster(sample=2000, override_random=None, sequences=None, min_cluster_size=0, exclude: Optional[List] = None):
    """Clusters the user behaviour using Wang. et al's methodology (recursive, divisive).
    Can pass sample to select the number of users to cluster.
    Can override the random state by passing a random_state like object (e.g. an int).
    Can also override the input by passing a df in sequences. In that case, the sample should be None.
    Can also override the minimum cluster size by passing min_cluster_size > 0"""
    if sequences is None:
        sequences = get_sequences()

    if sample:
        logger.info(f"Sampling {sample} users out of {sequences.shape[0]} (~{sample/sequences.shape[0]:.2%})")
        sequences = sequences.sample(sample, random_state=override_random)

    sequences = sequences.sort_index()
    freqs = dmu.calc_multiple_frequencies(sequences["sequence"], normalize=False)


    result = rdc.run_divisive_clustering(freqs,
                                         max_threads=os.cpu_count(),
                                         min_cluster_size=min_cluster_size,
                                         exclusions=exclude)
    return sequences, result


def get_sequences(file: Path = None) -> pd.DataFrame:
    """Reads a set of sequences from the processed folder. Can be a sample or the full dataset.
    If no file is given, the default file is used (FULL_FILE) (no n-grams)"""
    if file is None:
        file = FULL_FILE
    sequences = pd.read_parquet(file)
    return sequences


def get_activity_dict() -> dict:
    """Reads the id -> activity name dictionary from processed folder."""
    return pd.read_parquet(DICT_FILE).to_dict(orient="dict")['act_name']


def _get_tree(path=None):
    """Reads a stored clustering tree"""
    if path is None:
        path = DEFAULT_TREE_FILE
    with open(path) as file:
        tree = json.load(file)
    return tree


def _store_tree(tree, path=None):
    """Stores a (correctly formatted) tree as a json object."""
    if path is None:
        path = DEFAULT_TREE_FILE
    logger.info(f"Storing parsed clustering results as json tree at {path}")
    with open(path, "w") as file:
        json.dump(tree, file, indent=4)


def _extract_clusters(result: dict = None, depth=4):
    """Annotates each user with their cluster (up to a depth of 'depth').
    If no result is passed, the default location is used to retrieve the result.
    Pass None or -1 to use the maximum depth."""
    logger.info("Extracting clusters...")
    if result is None:
        try:
            result = _get_tree()
        except FileNotFoundError as e:
            logger.error("Clustering not yet ran! Cannot extract clusters")
            raise e

    users = _traverse_tree(result, {})

    # Calc max depth
    if depth is None or depth < 0:
        depth = max([len(val.split(".")) for val in users.values()]) - 1  # -1 to ignore the root cluster (that has all)

    header = ["cluster" + str(i) for i in range(depth)]
    results = {}

    logger.info("Iterating over clustering tree...")
    # Iterate over each users
    for user, path in users.items():
        path = path.split(".")[1:]
        paths = []
        # expand paths
        for index in range(depth):
            if index >= len(path):
                paths.append(None)
            else:
                paths.append(".".join([str(item) for item in path[:index + 1]])) # start at 1 to ignore root cluster
        results[user] = paths

    logger.info("Creating dataframe...")
    results = pd.DataFrame.from_dict(results, orient="index", columns=header).sort_index()
    return results


def _traverse_tree(tree: dict, users: dict):
    """Recursively traverses a (correctly formatted) tree to find each user and their path in the tree. """
    if tree["type"] == "l":
        # leaf
        for user in tree["nodes"]:
            users[user] = tree["path"]
    else:
        # tree
        for subtree in tree['children']:
            users = _traverse_tree(subtree, users)
    return users


def create_and_extract_clusters(sample: Optional[int] = 2000,
                                override_random: Optional[int] = None,
                                max_cluster_depth: Optional[int] = None,
                                min_cluster_size: int = 0,
                                exclude: Optional[List] = None,
                                sequences: pd.DataFrame = None,
                                output_file: Optional[Path] = None,
                                output_tree: Optional[Path] = None) -> Tuple[pd.DataFrame, dict]:
    """Samples 'sample' users (or all if False/None) with random_state 'override_random', stores the result to disk
    and extract the clusters from them. Returns the set of samples user_ids with their clusters. Clusters up to
    'max_cluster_depth' are extracted (or all depths if negative). You can also pass your own selection of users
    to apply the clustering too. You can select where the json tree is stored with output tree and where the merged
    user table is stored with output_file. Ignore_account is True ignores the account activity."""

    sequences, result = cluster(sample=sample,
                                override_random=override_random,
                                sequences=sequences,
                                min_cluster_size=min_cluster_size,
                                exclude=exclude)

    activity_dict = get_activity_dict()  # maps activivity id -> activity name
    node_to_user_id = sequences.index.to_list()  # maps node id -> user_id

    parsed_result = rdc.result_to_tree(result,
                                       activity_id_to_name_map=activity_dict,
                                       node_to_user_id=node_to_user_id)[1]

    _store_tree(parsed_result, output_tree)
    clusters = _extract_clusters(parsed_result, depth=max_cluster_depth)
    merged = sequences.merge(clusters, left_index=True, right_index=True)

    if output_file is None:
        output_file = DEFAULT_EXTRACT_CLUSTERS_FILE

    logger.info(f"Storing case table with clustering results at {output_file}")
    merged.to_parquet(output_file)
    return merged, parsed_result


def get_clusters(path=None):
    """Reads the extracted clusters and user id dataframe from disk; if it exists!"""
    if path is None:
        path = DEFAULT_EXTRACT_CLUSTERS_FILE
    return pd.read_parquet(path)


if __name__ == '__main__':
    print(create_and_extract_clusters(sample=5000, override_random=42))
