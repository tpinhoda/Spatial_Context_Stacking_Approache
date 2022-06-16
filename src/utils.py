"""Utility functions"""
import os
import json
import logging
from typing import Dict
from dotenv import load_dotenv
from coloredlogs import install as coloredlogs_install
from rich.traceback import install as rich_install


def initialize_coloredlog():
    """Install the colored log package"""
    coloredlogs_install()


def initialize_logging():
    """Initialize the format of the logger messages"""
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


def initialize_rich_tracerback():
    """Install the rich tracerback"""
    rich_install()


def load_env_variables(project_dir) -> Dict[str, str]:
    """Loads enviromental variables in the .env file."""
    dotenv_path = os.path.join(project_dir, ".env")
    load_dotenv(dotenv_path, override=True)
    return {
        "root_path": os.environ.get("ROOT_DATA"),
        "census_filepath": os.environ.get("CENSUS_FILEPATH"),
        "meshblock_filepath": os.environ.get("MESHBLOCK_FILEPATH"),
        "other_filepath": os.environ.get("OTHER_FILEPATH"),
    }


def load_json(json_path):
    """Load json file."""
    with open(json_path, encoding="utf-8") as file:
        return json.load(file)
