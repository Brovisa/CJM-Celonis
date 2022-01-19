"""
Contains the configuration and settings for the project and parses the environment variables, so that they are
accessible via settings.<VAR>.
"""
import logging
import os
import time
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BASE_PATH = Path(os.path.dirname(__file__)).resolve()

ZIP_FILE = 'Luxor_SGH_w42_export_v4.zip' # 'SGH_sept.zip'
CSV_NAME = "Luxor_SGH_w42_export_v4.csv" # "Luxor_SGH_US_-_Sept2020.csv"
PARQUET_FILE = 'SGH_w42_sept.parquet'
PARQUET_FILE_FILTERED = 'SGH_w42_filtered.parquet'

# # Environment data
# Data folders
DATA_FOLDER_RAW = Path(os.environ.get("DATA_FOLDER_RAW"))
DATA_FOLDER_INTERIM = Path(os.environ.get("DATA_FOLDER_INTERIM"))
DATA_FOLDER_PROCESSED = Path(os.environ.get("DATA_FOLDER_PROCESSED"))
DATA_FOLDER_EXTERNAL = Path(os.environ.get("DATA_FOLDER_EXTERNAL"))
REPORT_FOLDER = Path(os.environ.get("REPORT_FOLDER"))


# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", default="INFO").upper()
LOG_FORMAT = os.environ.get("LOG_FMT", None)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
LOGGER = logging.getLogger(__name__)


def timed(func):
    """
    A decorator that wraps the passed in function and logs the execution time.
    """

    @wraps(func)
    def wrap(*args, **kwargs):
        start = time.time()
        LOGGER.info(f"Running {func.__name__}()...")
        result = func(*args, **kwargs)
        LOGGER.info(f"{func.__name__}() finished in {time.time() - start:.2f}s")
        return result

    return wrap
