from pathlib import Path

import pandas as pd

from src.data import _02_convert_eventlog_to_sequences


def get_case_table(path: Path = None) -> pd.DataFrame:
    """Retrieves the case table from the interim folder."""
    if path is None:
        path = _02_convert_eventlog_to_sequences.CASE_TABLE_FILE
    return pd.read_parquet(path)


def merge_behavioural(df: pd.DataFrame, case_table: pd.DataFrame = None) -> pd.DataFrame:
    """(Right) Merges the behavioural clusters with the case table"""
    if case_table is None:
        case_table = get_case_table()
    result = df.join(case_table)
    return result
