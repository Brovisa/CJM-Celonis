import collections
import datetime
import logging
import pickle
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn import preprocessing

import config
from config import timed
from src.data._02_convert_eventlog_to_sequences import EVENTLOG_FILE
from src.features._02_merge_behaviour_cluster_with_case_table import get_case_table

ENRICHED_CASE_TABLE_FILE = config.DATA_FOLDER_PROCESSED / 'enriched_case_table.parquet'
SCALER_FILE = config.DATA_FOLDER_PROCESSED / 'enriched_case_table_scaler.pickle'
FEATURE_NAMES_FILE = config.DATA_FOLDER_PROCESSED / 'enriched_case_table_feature_names.pickle'

DEFAULT_SEQUENCE_FILE = config.DATA_FOLDER_PROCESSED / "sept_all_sequences.parquet"
DEFAULT_DICT_FILE = config.DATA_FOLDER_PROCESSED / "sept_sequences_dict.parquet"

logger = logging.getLogger(__name__)


def _create_indicator(df: pd.DataFrame):
    """Transforms numeric values to indicators."""
    indicators = df.transform(lambda x: x > 0)
    indicators = indicators.rename(columns=lambda name: name + "_ind") * 1
    return df.join(indicators)


@timed
def create_time_of_day_features(end_of_session_eventlog: pd.DataFrame, indicator=False, as_proportion=True):
    """Transforms timestamps to moment of day"""

    def time_to_group(dt: datetime.datetime):
        if 0 <= dt.hour < 5 or 21 <= dt.hour < 24:
            return "night"
        elif 5 <= dt.hour < 12:
            return "morning"
        elif 12 <= dt.hour < 17:
            return "afternoon"
        elif 17 <= dt.hour < 21:
            return "evening"

    # filtered = eventlog.loc[eventlog["activity"] == "end of session"].copy()
    filtered = end_of_session_eventlog.copy()
    filtered["moment"] = filtered["timestamp"].transform(time_to_group)
    uniques = filtered.groupby(["user_id", "moment"]).size()
    result = uniques.unstack().fillna(0)

    if as_proportion:
        logger.info(f"Calculating proportions")
        result = result.div(end_of_session_eventlog.groupby("user_id").size(), axis=0)

    if indicator:
        result = _create_indicator(result)
    return result


@timed
def create_day_feature(end_of_session_eventlog: pd.DataFrame, indicator=False, as_proportion=True):
    def to_name(dt):
        day = dt.weekday()
        if day == 0:
            return "mon"
        elif day == 1:
            return "tue"
        elif day == 2:
            return "wed"
        elif day == 3:
            return "thu"
        elif day == 4:
            return "fri"
        elif day == 5:
            return "sat"
        elif day == 6:
            return "sun"

    # filtered = eventlog.loc[eventlog["activity"] == "end of session"].copy()
    filtered = end_of_session_eventlog.copy()
    filtered["weekday"] = filtered["timestamp"].transform(to_name)
    uniques = filtered.groupby(["user_id", "weekday"]).size()
    result = uniques.unstack().fillna(0)

    if as_proportion:
        logger.info(f"Calculating proportions...")
        result = result.div(end_of_session_eventlog.groupby("user_id").size(), axis=0)

    if indicator:
        result = _create_indicator(result)

    result['week'] = result[["mon", "tue", "wed", "thu", "fri"]].sum(axis=1)
    result['weekend'] = result[["sat", "sun"]].sum(axis=1)

    return result


@timed
def get_num_sessions(eventlog: pd.DataFrame, use_timestamps=False):
    """Calculates the number of sessions per user. Can use the timestamps or end of session events."""
    if use_timestamps:
        groups = eventlog.groupby("user_id", sort=False)
        num_sessions = groups['timestamp'].nunique()
    else:
        # filtered = eventlog.copy()
        filtered = eventlog[eventlog["activity"] == "end of session"]
        grouped = filtered.groupby("user_id")
        num_sessions = grouped.size()

    return num_sessions


@timed
def create_event_features(end_of_session_eventlog: pd.DataFrame,
                          start_date=datetime.datetime(2020, 9, 1),
                          end_date=datetime.datetime(2020, 10, 1),
                          as_proportion=True):
    """Creates events that indicate the number of sessions before, during and after major events in the US."""
    days_before = [7, 3, 0]
    days_after = [3, 7]

    # local time is bad, so we use a custom date
    logger.info("Filtering fashion data for usable events...")
    start_date = max(end_of_session_eventlog['timestamp'].min(), start_date) - datetime.timedelta(days=max(days_before))
    end_date = min(end_of_session_eventlog['timestamp'].max(), end_date) + datetime.timedelta(days=max(days_after))

    fashion_data = pd.read_excel(config.DATA_FOLDER_EXTERNAL / "US_fashion.xlsx", sheet_name="Summarized events",
                                 header=0, parse_dates=["Start date", "End date"])
    missing_end = fashion_data["End date"].isna()
    fashion_data.loc[missing_end, "End date"] = fashion_data.loc[missing_end, "Start date"]
    fashion_data = fashion_data[(fashion_data["Start date"] >= start_date)
                                & (fashion_data["End date"] <= end_date)]

    logger.info(
        f"Fashion data filtered from {start_date} until {end_date}. Time: {end_date - start_date}. Have {fashion_data.shape[0]} events.")

    logger.info(f"Calculating sessions before and after each event...")
    filtered = end_of_session_eventlog.copy()
    all_colnames = []
    for i, event in fashion_data.iterrows():
        logger.debug(f"Calculating sessions for event {event}")
        event_name = event['Event'].strip().replace(" ", "_").replace("_spring/summer_2021", " ").lower()
        for day in days_before:
            colname = str(day) + "d_before_" + event_name
            if day == 0:
                colname = "on_" + event_name
            start = event["Start date"] - datetime.timedelta(days=day)
            if start < start_date:
                continue
            end = event['End date']
            filtered[colname] = ((start <= filtered['timestamp']) & (filtered['timestamp'] <= end)) * 1
            all_colnames.append(colname)

        for day in days_after:
            colname = str(day) + "d_after_" + event_name
            start = event["Start date"]
            end = event['End date'] + datetime.timedelta(days=day)
            if end > end_date:
                continue
            filtered[colname] = ((start <= filtered['timestamp']) & (filtered['timestamp'] <= end)) * 1
            all_colnames.append(colname)

    logger.info(f"Aggregating sessions on per-user basis.")
    group = filtered[["user_id"] + all_colnames].groupby("user_id").aggregate("sum")

    if as_proportion:
        logger.info(f"Calculation proportions...")
        group = group.div(end_of_session_eventlog.groupby("user_id").size(), axis=0)

    logger.info(f"Dropping {(group != 0).any(axis=0).sum()} zero-only entries...")
    logger.debug(f"{(group != 0).any(axis=0)}")
    group = group.loc[:, (group != 0).any(axis=0)].copy()

    return group


def get_tables(case_table_file=None, eventlog_file=None):
    """Reads the case table and eventlog files"""
    case_table = get_case_table(case_table_file)
    if eventlog_file is None:
        eventlog_file = EVENTLOG_FILE
    eventlog = pd.read_parquet(eventlog_file)
    return case_table, eventlog


def standardize_columns(df: Union[pd.DataFrame, pd.Series], suffix='_standardized'):
    """Transforms a feature column to have 0 mean and 1 sd, preserving indices and column names."""
    col_names = df.add_suffix(suffix).columns
    index = df.index
    scaler = preprocessing.StandardScaler()
    result = scaler.fit_transform(df)
    result = pd.DataFrame(result, index=index, columns=col_names)
    return result, scaler


@timed
def run(store: Union[bool, Path] = False, standardize=True, sequence_file=None, sequence_dict_file=None,
        filter_seq_length=1, case_table_file=None, eventlog_file=None):
    """Enriches the case table and (optionally) stores it.
    store can be a bool (True defaults to default location) or a pathlike (stores at location)."""
    # Sequence/ngrams counts
    sequences, act_dict = _get_sequences(sequence_file=sequence_file, dict_file=sequence_dict_file)
    sequences = sequences[sequences.str.len() >= filter_seq_length]
    seq_counts = get_sequence_count(sequences, act_dict)

    # purchase indicator
    purchase_indicator = (seq_counts['checkout:thankyou:pageview'] > 0) * 1  # * 1 -> cast to int (not bool)

    # sequence length
    sequence_length = pd.Series([len(this_list) for this_list in sequences.tolist()], index=sequences.index)

    case_table, eventlog = get_tables(case_table_file, eventlog_file)
    case_table = case_table.set_index("user_id")
    end_of_session_eventlog = eventlog.loc[eventlog["activity"] == "end of session"].copy()

    # num sessions
    num_sessions = get_num_sessions(eventlog)

    # average length / session
    act_per_session = sequence_length / num_sessions

    # event features
    # event_features = create_event_features(end_of_session_eventlog)

    # time of day
    time_of_day = create_time_of_day_features(end_of_session_eventlog)

    # weekday
    weekday = create_day_feature(end_of_session_eventlog)

    # merge all
    merged = case_table.join([
        # event_features,
        time_of_day, weekday, seq_counts])
    merged['num_sessions'] = num_sessions
    merged['sequence_length'] = sequence_length
    merged['act_per_session'] = act_per_session
    merged['purchase_indicator'] = purchase_indicator

    # standardize numeric (except indicators)
    scaler, feature_names = None, None
    if standardize:
        logger.info("Standardizing features...")
        drop = ["7d_before_labor_day",
                "3d_before_labor_day",
                "on_labor_day",
                "3d_after_labor_day",
                "7d_after_labor_day",
                "7d_before_new_york_fashion_week",
                "3d_before_new_york_fashion_week",
                "on_new_york_fashion_week",
                "3d_after_new_york_fashion_week",
                "7d_after_new_york_fashion_week",
                "7d_before_london_fashion_week",
                "3d_before_london_fashion_week",
                "on_london_fashion_week",
                "3d_after_london_fashion_week",
                "7d_after_london_fashion_week",
                "7d_before_milan_fashion_week",
                "3d_before_milan_fashion_week",
                "on_milan_fashion_week",
                "3d_after_milan_fashion_week",
                "7d_after_milan_fashion_week",
                "7d_before_paris_fashion_week_spring/summer_2021",
                "3d_before_paris_fashion_week_spring/summer_2021",
                "on_paris_fashion_week_spring/summer_2021",
                "m20help:pageview",
                "end of session",
                "checkout:delivery:pageview",
                "checkout:gateway:pageview",
                "checkout:payment:pageview",
                "checkout:thankyou:pageview",
                "purchase_indicator"]
        to_scale = merged.select_dtypes(include="number").drop(columns=drop, errors="ignore")
        merged_std, scaler = standardize_columns(to_scale, suffix="_standardized")
        merged = merged.join([merged_std])

    if store:
        if type(store) is bool:
            store = ENRICHED_CASE_TABLE_FILE
        merged.to_parquet(str(store) + ".parquet")
        merged.to_csv(str(store) + ".csv")

        if standardize:
            with open(SCALER_FILE, "wb") as f:
                pickle.dump(scaler, f)

    return merged, scaler, purchase_indicator


def get_sequence_count(sequences, act_dict):
    """Calculates how often each object in the sequence/ngrams list occurs as features
    (and transforms their id to names)"""
    sequence_count = [collections.Counter(this_list) for this_list in sequences.tolist()]
    sequences_normalized = [{k: v / (sum(counts.values())) for k, v in counts.items()} for counts in sequence_count]
    df_seq = pd.DataFrame(sequences_normalized, index=sequences.index)
    df_seq = df_seq.rename(columns=act_dict)
    df_seq = df_seq.fillna(0)
    df_seq = df_seq.sort_index(axis="columns")
    return df_seq


def _get_sequences(sequence_file=None, sequence_col_name='sequence', dict_file=None, dict_col_name='act_name'):
    """Retrieves sequence/ngrams data from disk"""
    if sequence_file is None:
        sequence_file = DEFAULT_SEQUENCE_FILE
    if dict_file is None:
        dict_file = DEFAULT_DICT_FILE
    sequences = pd.read_parquet(sequence_file)[sequence_col_name]
    act_dict = pd.read_parquet(dict_file).to_dict(orient="dict")[dict_col_name]
    return sequences, act_dict


@timed
def retrieve(location: Optional[Path] = None, with_scaler: bool = False):
    """Retries an already enriched case table.
    @:param all retrieve scalar and feature names too
    """
    logger.info("Retrieving stored case_table and associated data...")
    if location is None:
        location = ENRICHED_CASE_TABLE_FILE

    case_table = pd.read_parquet(location)

    if with_scaler:
        with open(SCALER_FILE, "rb") as f:
            with_scaler = pickle.load(f)
        return case_table, with_scaler
    return case_table


if __name__ == '__main__':
    run()
