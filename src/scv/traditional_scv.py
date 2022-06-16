"""Generate traditional spatial folds"""
import os
import time
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from tqdm import tqdm
import geostatspy.geostats as geostats
from scipy.spatial.distance import cdist
from src.scv.scv import SpatialCV

ULTRACONSERVATIVE = "TraditionalSCV"


@dataclass
class TraditionalSCV(SpatialCV):
    """Represents the Traditional Spatial Cross-Validation.

    Attributes
    ----------
        data: pd.Dataframe
            The spatial dataset to generate the folds
        fold_col: str
            The fold column name
        target_col: str
            The targer attribute column name
        meshblocks: pd.Dataframe
            The meshblocks regarding the spatial objects in the data
        fast: bool
            Whether to skip the semivariogram process and run with the ICMLA21 paper results
        root_path : str
            Root path

    """

    target_col: str = "TARGET"
    index_col: str = "INDEX"
    meshblocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_meshblocks: str = None
    sill_target: np.float64 = None

    def _calculate_sill(self):
        # Calculates sill, variance of the target variable
        self.sill_target = self.data[self.target_col].var()

    def _calculate_buffer_size(self):
        # Calculate the size of the removing buffer
        tmin = -9999.0
        tmax = 9999.0  # no trimming
        lag_dist = 0.3
        lag_tol = 0.15
        nlag = 100
        # maximum lag is 700m and tolerance > 1/2 lag distance for smoothing
        bandh = 9999.9
        atol = 22.5  # no bandwidth, directional variograms
        isill = 0  # standardize sill
        # print(self.data[["x", "y", self.target_col]])
        lag, gamma, _ = geostats.gamv(
            self.data,
            "x",
            "y",
            self.target_col,
            tmin,
            tmax,
            lag_dist,
            lag_tol,
            nlag,
            360,
            atol,
            bandh,
            isill,
        )
        range = [(h, g) for h, g in zip(lag, gamma) if g > self.sill_target]
        return range[0][0]

    def _calculate_buffer(self, buffer_size):
        test = self.test_data[["x", "y"]].values.tolist()
        test = np.reshape(test, (-1, 2))
        training = self.train_data[["x", "y"]].values.tolist()
        training = np.reshape(training, (-1, 2))
        dist = cdist(training, test)
        max_dist = pd.Series(
            np.amin(dist, axis=1), index=self.train_data.index, name="max_dist"
        )
        return [idx for idx, value in max_dist.iteritems() if value < buffer_size]

    def _generate_x_y(self):
        self.meshblocks.index = self.meshblocks.index.astype(self.data.index.dtype)

        if not self.meshblocks.crs:
            self.meshblocks = self.meshblocks.set_crs(4326, allow_override=True)
        self.meshblocks["x"] = (
            self.meshblocks.to_crs("+proj=cea")
            .centroid.to_crs(self.meshblocks.crs)
            .apply(lambda p: p.x)
        )
        self.meshblocks["y"] = (
            self.meshblocks.to_crs("+proj=cea")
            .centroid.to_crs(self.meshblocks.crs)
            .apply(lambda p: p.y)
        )
        self.data = self.data.join(self.meshblocks[["x", "y"]])

    def run(self) -> None:
        """Generate ultra-conservartive spatial folds"""
        # Create folder folds
        start_time = time.time()
        name_folds = ULTRACONSERVATIVE
        self._make_folders(["folds", name_folds])
        self._generate_x_y()
        self._calculate_sill()
        buffer_size = self._calculate_buffer_size()

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
            self._clean_data(cols_drop=[self.fold_col, "x", "y"])
            # Save data
            # self._save_data()
            # Update cur dir
            self.cur_dir = os.path.join(self._get_root_path(), "folds", name_folds)
        # Save execution time
        end_time = time.time()
        self._save_time(end_time, start_time)
        print(f"Execution time: {end_time-start_time} seconds")
