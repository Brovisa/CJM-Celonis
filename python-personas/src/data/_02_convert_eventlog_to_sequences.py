import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

import config
from config import timed

logger = logging.getLogger(__name__)

CASE_TABLE_FILE = config.DATA_FOLDER_INTERIM / 'sept_case_table.parquet'
EVENTLOG_FILE = config.DATA_FOLDER_INTERIM / 'sept_eventlog.parquet'


@timed
def get_data_from_eventlog(input=None, add_category_codes=True):
    """Retrieves the eventlog and (optional) appends the category codes."""
    if input is None:
        input = EVENTLOG_FILE
    logger.info(f"Reading eventlog: {input}...")
    df = pd.read_parquet(input)
    activity_dict = None
    # Append codes
    if add_category_codes:
        logger.info("Encoding activities as numeric...")
        df["activity_code"] = df['activity'].cat.codes
        activity_dict = dict(enumerate(df['activity'].cat.categories))
    return df, activity_dict


def create_ngrams(items, n: int = 3, fill: int = -1) -> list:
    """Creates n grams out of a list of items. It pads the first and last items with the `fill` parameter."""
    if n <= 1:
        return list(items)

    result = []
    # pad with meaningless activities
    if len(items) < n:
        num = n - len(items)
        items = np.append(items, [fill] * num)

    # create ngrams
    for i in range(len(items) - n + 1):
        result.append(list(items[i:i + n]))

    return result


def create_sequences(df, activity_col="activity_code", ngrams=0):
    """Groups the data by users and creates sequences of ngrams out of them."""
    start_time = time.time()

    logger.info("Sorting dataframe (precaution, should be unnecessary)...")
    action_start = time.time()
    df = df.sort_values(["user_id", "timestamp", "sorting"])
    logger.info(f"Dataframe sorted. Time spent: {time.time() - action_start:.1f}s")

    logger.info("Extracting users and activities...")
    action_start = time.time()
    keys, values = df[["user_id", activity_col]].values.T
    logger.info(f"Users and activities extracted. Time spent: {time.time() - action_start:.1f}s")

    logger.info(f"Calculating unique keys...")
    action_start = time.time()
    ukeys, index = np.unique(keys, True)  # preserves the first index of the unique key
    logger.info(f"Unique keys calculated. Time spent: {time.time() - action_start:.1f}s")

    logger.info("Creating new dataframe...")
    action_start = time.time()
    arrays = np.split(values, index[1:])  # splits the list of activity at indices where a new user begins
    result = pd.DataFrame({"user_id": ukeys, 'sequence': [create_ngrams(item, n=ngrams) for item in arrays]}). \
        set_index(["user_id"])

    logger.info(f"Dataframe created. "
                f"Time spent: {time.time() - action_start:.1f}s. "
                f"Total time spent: {time.time() - start_time:.1f}s")
    return result


@timed
def filter_sequence_length(sequences, min_length=4):
    """Removes sequences of length < min_length"""
    mask = (sequences["sequence"].str.len() >= min_length)
    return sequences[mask]


@timed
def process(input=None, sample_size: Optional[int] = 2000, min_length: int = 1, ngrams: int = 0):
    """Process an eventlog into a list of sequences"""
    assert min_length > 0
    assert ngrams >= 0
    # Read data
    df, act_dict = get_data_from_eventlog(input)

    # Create sequences
    seq = create_sequences(df, ngrams=ngrams)

    # filter by length
    filtered = filter_sequence_length(seq, min_length=min_length)

    # Sample
    if sample_size is not None:
        sample = filtered.sample(n=sample_size)
    else:
        sample = filtered

    # Return results
    return sample, act_dict


def convert_eventlog(input=None, output_prefix='sept', store_sample=False, ngrams=0):
    """Converts the eventlog to sequences of activities and stores it in the processed folder.
    Select the output prefix with 'output_prefix'. If 'store_sample' is True, a smaller sample is also stored.
    The number of ngrams will be """
    logger.info("Created sequences and activity dictionary...")
    all_sequences, act_dict = process(input=input, sample_size=None, ngrams=ngrams)
    if ngrams > 1:
        output_prefix += f"_n-{ngrams}"

    all_file = config.DATA_FOLDER_PROCESSED / f"{output_prefix}_all_sequences.parquet"
    logger.info(f"Storing sequences in {all_file}")
    all_sequences.to_parquet(all_file, )

    dict_file = config.DATA_FOLDER_PROCESSED / f"{output_prefix}_sequences_dict.parquet"
    logger.info(f"Storing number to activity dict in {dict_file}")
    pd.DataFrame.from_dict(data=act_dict, orient="index", columns=["act_name"]).to_parquet(dict_file)

    logger.info("Sequences and dictionary are created!")

    if store_sample:
        sample_size = 500
        logger.info(f"Sampling {sample_size} users...")
        sample_sequences = all_sequences.sample(n=sample_size, random_state=42)
        sample_file = config.DATA_FOLDER_PROCESSED / f"{output_prefix}_{sample_size}_sequences.parquet"
        logger.info(f"Storing sample of sequences in {sample_file}")
        sample_sequences.to_parquet(sample_file)


if __name__ == '__main__':
    convert_eventlog()
