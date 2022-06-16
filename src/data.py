# -*- coding: utf-8 -*-
"""Abstract class to represent the file data."""
import logging
import shutil
from dataclasses import dataclass
from os import mkdir, listdir, remove, rename
from os.path import join, isfile
from abc import ABC
from typing import List


@dataclass
class Data(ABC):
    """Represents file data.

    This class is responsible to organize the data in folders.

    Attributes
    ----------
        root_path : str
            Root path
        cur_dir: str
            Currenti working directory
        logger_name: str
            Name of the logger

    """

    root_path: str = None
    cur_dir: str = None
    logger_name: str = None

    def logger_info(self, message: str):
        """Print logger info message"""
        logger = logging.getLogger(self.logger_name)
        logger.info(message)

    def logger_warning(self, message: str):
        """Print logger info message"""
        logger = logging.getLogger(self.logger_name)
        logger.warning(message)

    def logger_error(self, message: str):
        """Print logger info message"""
        logger = logging.getLogger(self.logger_name)
        logger.error(message)

    def set_logger_to_crit(self, module):
        """Set module logger to critical"""
        logging.getLogger(module).setLevel(logging.CRITICAL)

    def _mkdir(self, folder_name: str) -> None:
        """Creates a folder at current path"""
        # logger = logging.getLogger(self.logger_name)
        self.cur_dir = join(self.cur_dir, folder_name)
        try:
            mkdir(self.cur_dir)
            # logger.info(f"Creating folder: /{folder_name}")
        except FileExistsError:
            pass
            # logger.info(f"Entering folder: /{folder_name}")

    def _set_cur_dir(self) -> None:
        """Creates the initial folds to store the dataset"""
        # self.logger_info(f"Root: {self.root_path}")
        # self.logger_info("Creating or Entering dataset folders.")
        self.cur_dir = self.root_path

    def _get_root_path(self) -> str:
        """Returns the initial folders path"""
        return self.root_path

    def _get_files_in_cur_dir(self) -> List[str]:
        """Returns a list of filesnames in the current directory"""
        return [
            filename
            for filename in listdir(self.cur_dir)
            if isfile(join(self.cur_dir, filename))
        ]

    def _get_folders_in_cur_dir(self) -> List[str]:
        """Returns a list of folders in the current directory"""
        return [
            filename
            for filename in listdir(self.cur_dir)
            if not isfile(join(self.cur_dir, filename))
        ]

    def _get_files_in_dir(self, directory: str) -> List[str]:
        """Returns a list of filename in directory"""
        return [
            filename
            for filename in listdir(directory)
            if isfile(join(directory, filename))
        ]

    def _get_folders_in_dir(self, directory: str) -> List[str]:
        """Returns a list of filename in directory"""
        return [
            filename
            for filename in listdir(directory)
            if not isfile(join(directory, filename))
        ]

    def _remove_file_from_cur_dir(self, filename: str) -> None:
        """Remvoves a filename from the current directory"""
        remove(join(self.cur_dir, filename))

    def _remove_folders_from_cur_dir(self):
        folders = self._get_folders_in_cur_dir()
        for folder in folders:
            shutil.rmtree(join(self.cur_dir, folder), ignore_errors=True)

    def _rename_file_from_cur_dir(self, old_filename: str, new_filename: str) -> None:
        """Rename a file from the current dir"""
        try:
            rename(join(self.cur_dir, old_filename), join(self.cur_dir, new_filename))
        except FileExistsError:
            pass

    def init_logger_name(self, msg: str):
        """Initialize the logger name"""
        self.logger_name = msg

    def _make_folders(self, folders: List[str]):
        """Make the initial folders"""
        self._set_cur_dir()
        for folder in folders:
            self._mkdir(folder)
