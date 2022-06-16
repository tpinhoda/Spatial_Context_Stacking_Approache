"""Generate ultra-conservative spatial folds"""
import os
import time
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.scv.scv import SpatialCV

ULTRACONSERVATIVE = "UltraConservative"


@dataclass
class UltraConservative(SpatialCV):
    """Represents the Ultra-Conservative Spatial Cross-Validation.

    Attributes
    ----------
        data: pd.Dataframe
            The spatial dataset to generate the folds
        fold_col: str
            The fold column name
        target_col: str
            The targer attribute column name
        adj_matrix: pd.Dataframe
            The adjacency matrix regarding the spatial objects in the data
        fast: bool
            Whether to skip the semivariogram process and run with the ICMLA21 paper results
        root_path : str
            Root path

    """

    target_col: str = "TARGET"
    adj_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    fast: bool = False
    sill_target: np.float64 = None

    def _calculate_sill(self):
        # Calculates sill, variance of the target variable
        self.sill_target = self.data[self.target_col].var()

    def _get_lag_neighbors(self, indexes, lag):
        # Return neighbors at a given lag neighborhood
        for _ in range(lag):
            area_matrix = self.adj_matrix.loc[indexes]
            neighbors = area_matrix.sum(axis=0) > 0
            neighbors = neighbors[neighbors].index
            neighbors_index = list({n for n in neighbors if n not in indexes})
            indexes += neighbors_index
        return neighbors_index

    def _calculate_buffer_size(self):
        # Calculate the size of the removing buffer
        lag = 0
        gamma = -np.inf
        self._calculate_sill()
        while gamma < self.sill_target:
            lag += 1
            sum_similarity = 0
            total_pairs = 0
            for index, row in self.data.iterrows():
                target = row[self.target_col]
                neighbors = self._get_lag_neighbors([index], lag)
                neighbors = [n for n in neighbors if n in self.data.index]
                neighbors_target = self.data.loc[neighbors, self.target_col]
                diffs = [(target - x) ** 2 for x in neighbors_target]
                sum_similarity += sum(diffs)
                total_pairs += len(diffs)
            gamma = sum_similarity / (2 * total_pairs)
            print(f"sill: {self.sill_target} - lag: {lag} - gamma: {gamma}")
        return lag

    def _convert_adj_matrix_index_types(self):
        # Convert adjacency matrix index types
        self.adj_matrix.index = self.adj_matrix.index.astype(self.data.index.dtype)
        self.adj_matrix.columns = self.adj_matrix.columns.astype(self.data.index.dtype)

    def _calculate_buffer(self, buffer_size):
        indexes = self.test_data.index.values.tolist()
        for _ in range(buffer_size):
            area_matrix = self.adj_matrix.loc[indexes]
            neighbors = area_matrix.sum(axis=0) > 0
            neighbors = neighbors[neighbors].index
            neighbors_index = list({n for n in neighbors if n not in indexes})
            indexes += neighbors_index
        buffer_index = indexes + neighbors_index

        return [
            idx
            for idx in buffer_index
            if idx in self.data.index and idx not in self.test_data.index
        ]

    def run(self) -> None:
        """Generate ultra-conservartive spatial folds"""
        # Create folder folds
        start_time = time.time()
        name_folds = ULTRACONSERVATIVE
        self._make_folders(["folds", name_folds])
        self._convert_adj_matrix_index_types()
        buffer_size = 27 if self.fast else self._calculate_buffer_size()
        for fold_name, test_data in tqdm(
            self.data.groupby(by=self.fold_col), desc="Creating folds"
        ):
            # Cread fold folder
            self._mkdir(str(fold_name))
            # Initialize x , y and reduce
            self._split_data_test_train(test_data)
            # Calculate removing buffer
            removing_buffer = self._calculate_buffer(buffer_size)
            self.train_data.drop(index=removing_buffer, inplace=True)
            # Save buffered data indexes
            self._save_buffered_indexes(removing_buffer)
            # Save fold index relation table
            self._save_fold_by_index_training()
            # Clean data
            self._clean_data(cols_drop=[self.fold_col])
            # Save data
            # self._save_data()
            # Update cur dir
            self.cur_dir = os.path.join(self._get_root_path(), "folds", name_folds)
        # Save execution time
        end_time = time.time()
        self._save_time(end_time, start_time)
        print(f"Execution time: {end_time-start_time} seconds")
