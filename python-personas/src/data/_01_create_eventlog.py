import csv
import datetime
import io
import logging
import time
import zipfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import config
import src.data.UserParser
from src.data.ActivityParser import ActivityParser


logger = logging.getLogger(__name__)

ZIP_FILE = 'SGH_w42_export_v4.zip' # 'SGH_sept.zip'
CSV_NAME = "Luxor_SGH_w42_export_v4.csv" # "Luxor_SGH_US_-_Sept2020.csv"
PARQUET_FILE = 'SGH_w42_sept.parquet'
PARQUET_FILE_FILTERED = 'SGH_w42_filtered.parquet'

NAME_MAPPER = {
    'Visitor_ID': "visitor_id",
    'Country (Pages c2) (prop2)': "country",
    'Cities': "location",
    'Local time (Session v7) (evar7)': "local_time",
    'Mobile Device Type': "device",
    'OS Type': "os",
    'Last Touch Channel': "ltt",
    'URL (Pages c8) (prop8)': "url",
    'Pages name (Pages c58) DEPRECATED (prop58)': "page_name",
    'Adobe Events - Tags (v73) (evar73)': "events",
    'Combo (Action c38) (prop38)': "combo",
    'Actions (x44) (evar44)': "action",
    'Keyword (Internal Search v12) (evar12)': "search_keyword",
    'Action - serial (e45) (event45)': "action_serial",
}

TYPE_MAPPER = {
    'visitor_id': "category",
    'country': "category",
    'location': "category",
    'local_time': str,
    'device': "category",
    'os': "category",
    'ltt': "category",
    'url': str,
    'page_name': str,
    'events': str,
    'combo': str,
    'action': "category",
    'search_keyword': str,
    'action_serial': np.float32,
}

ap = ActivityParser(parse_actions=True)
up = src.data.UserParser.UserParser()


def _parse_time(time_string: str) -> Optional[datetime.datetime]:
    """Parses a SGH weblog local date time into a datetime object (or None if the object is missing)"""
    if len(time_string) < 10:
        return None
    splits = time_string.split(" ")
    year = int("20" + splits[0][1:])
    month = int(splits[1][1:])
    day = int(splits[2][1:])
    hour = int(splits[3][1:])
    return datetime.datetime(year, month, day, hour) #, minutes)


def _get_header():
    """Prints the header of the CSV file without parsing the whole file"""
    with zipfile.ZipFile(ZIP_FILE) as zip_file:
        with zip_file.open(CSV_NAME, mode="r") as f:
            csv_reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8-sig'))
            print(next(csv_reader))  # print header with indices


def _read_file(path=None, from_parquet=True, store_parquet=None) -> pd.DataFrame:
    """Reads the dataframe in parquet format, with fallback to CSV (slower)"""
    start_time = time.time()

    if path is None:
        path = ZIP_FILE if not from_parquet else PARQUET_FILE

    if from_parquet and path.exists():
        logger.info(f"Reading PARQUET file {path}")
        df = pd.read_parquet(path)
    else:
        logger.info(f"Reading CSV file {path}")
        df = pd.read_csv(path, na_values={"action_serial": "0.00"}, header=0,
                         names=NAME_MAPPER.values(), dtype=TYPE_MAPPER, converters={"local_time": _parse_time})

    if store_parquet is not None:
        logger.info(f"Storing file as {store_parquet}")
        df.to_parquet(store_parquet)

    logger.info(f"Finished reading file in {time.time() - start_time:.2f}s")
    return df


def _filter_file(df: pd.DataFrame) -> pd.DataFrame:
    """Removes incorrect rows from the file. i.e. missing country, missing timestamp, missing url, missing action AND
    event."""
    start_time = time.time()
    logger.info("Filtering dataframe...")

    # drop missing country
    logger.info("Dropping missing countries...")
    df = df.loc[pd.notna(df["country"])]

    # drop missing timestamp
    logger.info("Dropping missing timestamps...")
    df = df.loc[pd.notna(df["local_time"])]

    # drop missing url
    logger.info("Dropping missing urls...")
    df = df.loc[pd.notna(df["url"])]

    # drop missing action AND event
    logger.info("Dropping missing event and action...")
    df = df.loc[
        ~(pd.isna(df["events"])
          & pd.isna(df["action"]))
    ]

    logger.info(f"Finished filtering df in {time.time() - start_time:.2f}s")
    return df


def _parse_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parses a dataframe into an eventlog (with only events of interest) and case table"""
    logger.info("Setting up parsing function...")
    start_time = time.time()
    action_time = start_time

    eventlog_header = ["user_id", "timestamp", "activity", "sorting"]
    case_table_header = src.data.UserParser.get_header()
    case_table = []
    eventlog = []
    sorting = 0

    visitor, timestamp = None, None
    user_id = None

    cols = df.columns  # for re-use
    vals = df.values  # takes a while
    num = len(df)
    del df

    logger.info(f"Finished setting up parsing function in {time.time() - action_time:.2f}")
    logger.info("Parsing rows... (this might take a while)")
    action_time = time.time()
    for index, row in enumerate(vals):
        # update percentage progress
        if index % (num // 1000)  == 0:
            print(f"{int(index) / num:.2%}", end="\r")

        row = dict(zip(cols, row))  # add names to row

        new_visitor = row["visitor_id"]
        new_timestamp = row["local_time"]

        if new_visitor != visitor:
            # new/first visitor
            if visitor is not None:
                # append end of session marker and reset sorting
                eventlog.append([user_id, timestamp, "end of session", sorting])
                sorting = 0
            # parse new visitor
            visitor = new_visitor
            timestamp = new_timestamp
            user_id, user_row = up.parse_user(row)
            case_table.append(user_row)
        elif new_visitor == visitor and new_timestamp != timestamp:
            # new session
            eventlog.append([user_id, timestamp, "end of session", sorting])
            sorting = 0
            timestamp = new_timestamp

        activity, cleaned_activity = ap.parse_activity(row)
        if cleaned_activity is None:
            # uninteresting activity -> don't store
            continue

        eventlog.append([user_id, timestamp, cleaned_activity, sorting])
        sorting += 10

    # last user -> manually add end of session event
    eventlog.append([user_id, timestamp, "end of session", sorting])

    print()  # add newline to progress report
    logger.info(f"Finished parsing in {time.time() - action_time:.2f}s")
    action_time = time.time()
    logger.info(f"Converting lists to data_frames...")
    case_table = pd.DataFrame(data=case_table, columns=case_table_header)
    case_table = case_table.astype(dtype={
        "user_id": np.int,
        "visitor_id": str,
        "city": "category",
        "area": "category",
        "country": "category",
        "device": "category",
        "os": "category",
        "ltt": "category",
    })
    eventlog = pd.DataFrame(data=eventlog, columns=eventlog_header)
    eventlog = eventlog.astype(dtype={
        "user_id": np.int,
        "activity": "category",
        "sorting": np.int
    })
    logger.info(f"Finished converting lists to data frames in {time.time() - action_time:.2f}s")
    logger.info(f"Finished parsing {num} rows into {len(eventlog)} events in {time.time() - start_time:2f}s")
    return case_table, eventlog


def create_eventlog(input = None, output_prefix="sept", read_parquet=True, store_parquet=None):
    """"Creates the eventlog and case table in the interim folder, with <output_prefix> as a a prefix"""
    df = _read_file(input, read_parquet, store_parquet=store_parquet)
    df = _filter_file(df)
    df = df.sort_values(["visitor_id", "local_time", "action_serial"])

    case_table, eventlog = _parse_rows(df)

    logger.info("Writing case_table to parquet and csv...")
    case_table.to_parquet(config.DATA_FOLDER_INTERIM / f"{output_prefix}_case_table.parquet")
    case_table.to_csv(config.DATA_FOLDER_INTERIM / f"{output_prefix}_case_table.csv", index=False, date_format="%d/%m/%Y %H:%M:%S")

    logger.info("Writing eventlog to parquet and csv...")
    eventlog.to_parquet(config.DATA_FOLDER_INTERIM / f"{output_prefix}_eventlog.parquet")
    eventlog.to_csv(config.DATA_FOLDER_INTERIM / f"{output_prefix}_eventlog.csv", index=False, date_format="%d/%m/%Y %H:%M:%S")
    logger.info("Done creating case table and eventlog")


if __name__ == '__main__':
    create_eventlog()
