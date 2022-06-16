"""Generate optmistic spatial folds"""
import os
import time
from dataclasses import dataclass
from tqdm import tqdm
from src.scv.scv import SpatialCV
from sklearn.model_selection import KFold

CV = "CrossValidation"


@dataclass
class CrossValidation(SpatialCV):
    """Represents the normal Cross-Validation, without considering
    the removing buffer.

     Attributes
    ----------
        data: pd.Dataframe
            The spatial dataset to generate the folds
        fold_col: str
            The fold column name
        root_path : str
            Root path
    """

    k: int = 10

    def run(self) -> None:
        """Generate merged data"""
        # Create folder folds
        start_time = time.time()
        name_folds = CV
        self._make_folders(["folds", name_folds])
        cv = KFold(n_splits=10, shuffle=True)
        for fold_name, (_, test_index) in enumerate(cv.split(self.data), start=1):
            self._mkdir(str(fold_name))
            # Initialize x , y and reduce
            test_data = self.data.iloc[test_index]
            self._split_data_test_train(test_data)
            # Save buffered data indexes
            self._save_buffered_indexes(removing_buffer=[])
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
