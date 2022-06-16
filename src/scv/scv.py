"""Generate spatial folds"""
from abc import ABC, abstractmethod
import os
import json
from dataclasses import dataclass, field
from typing import List
import pandas as pd
from src.data import Data


@dataclass
class SpatialCV(Data, ABC):
    """Represents the Spatial Cross Validation.

     Attributes
    ----------
        data: pd.Dataframe
            The spatial dataset to generate the folds
        fold_col: str
            The fold column name
    """

    scv_method: str = "No_Buffer"
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_col: str = "FOLD_INDEX"
    train_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _get_index_train(self, index_test) -> List:
        """Return the train set indexes based on the test set indexes"""
        return [idx for idx in self.data.index if idx not in index_test]

    def _split_data_test_train(self, test_data) -> pd.DataFrame:
        """Split the data into train and test set, based on a given teste set"""
        index_test = test_data.index
        index_train = self._get_index_train(index_test)
        self.test_data = self.data.loc[index_test].copy()
        self.train_data = self.data.loc[index_train].copy()

    def _clean_data(self, cols_drop: List):
        """Clean the dataset to present only attributes of interest"""
        self.train_data.drop(columns=cols_drop, inplace=True)
        self.test_data.drop(columns=cols_drop, inplace=True)

    def _save_fold_by_index_training(self):
        """Save the fold index relation table"""
        self.train_data[self.fold_col].to_csv(
            os.path.join(self.cur_dir, "fold_by_idx.csv")
        )

    def _save_data(self):
        """Save the train and test set using feather"""
        self.train_data.reset_index(inplace=True)
        self.train_data.to_feather(os.path.join(self.cur_dir, "train.ftr"))
        self.test_data.reset_index(inplace=True)
        self.test_data.to_feather(os.path.join(self.cur_dir, "test.ftr"))

    def _save_buffered_indexes(self, removing_buffer):
        """Save the indexes of the buffers"""
        train_test_idx = (
            self.train_data.index.values.tolist() + self.test_data.index.values.tolist()
        )
        discarded_idx = [
            _ for _ in self.data.index if _ not in train_test_idx + removing_buffer
        ]
        split_data = {
            "train": self.train_data.index.values.tolist(),
            "test": self.test_data.index.values.tolist(),
            "removing_buffer": removing_buffer,
            "discarded": discarded_idx,
        }
        path_to_save = os.path.join(self.cur_dir, "split_data.json")
        with open(path_to_save, "w", encoding="utf-8") as file:
            json.dump(split_data, file, indent=4)

    def _save_time(self, end, start):
        time = end - start
        filepath = os.path.join(self.cur_dir, "execution_time.txt")
        with open(filepath, "w", encoding="utf-8") as file:
            msg = f"Execution time \n seconds: {time} \n minutes: {time/60} \n hours: {time/3600}"
            file.write(msg)

    @abstractmethod
    def run(self):
        """Generate graph-based spatial folds"""
