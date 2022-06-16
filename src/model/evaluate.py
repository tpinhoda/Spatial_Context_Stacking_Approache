"""Predict data process"""
import os
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from src.data import Data
import src.utils as utils

PRED_COL = "PREDICTIONS"
GROUND_TRUTH_COL = "GROUND_TRUTH"


@dataclass(init=True)
class Evaluate(Data):
    """Represents the predict data process.

     Attributes
    ----------
        ml_method:str
            The machine learning method name
        fs_method:str
            The feature selection method name
        scv_method:str
            The spatial cross-validation method name
        index_col: str
            The dataset´s index column name
        root_path : str
            Root path
    """

    ml_method: str = "LGBM"
    fs_method: str = "CFS"
    scv_method: str = "gbscv"
    index_col: str = "INDEX"
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    train: pd.DataFrame = field(default_factory=pd.DataFrame)
    test: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_idx: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_features: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)

    def _init_fields(self):
        self.metrics = {}

    def _read_predictions(self, data_path):
        """Read the prediction data"""
        self.predictions = pd.read_csv(data_path, index_col=self.index_col)

    def _read_train(self, json_path, data):
        """Read the train data"""
        split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
        self.train = data.loc[split_fold_idx["train"]].copy()

    def _read_test(self, json_path, data):
        """Read the test data"""
        split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
        self.test = data.loc[split_fold_idx["test"]].copy()

    def _read_fold_idx_table(self, data_path):
        """Read the fold_by_idx data"""
        self.fold_idx = pd.read_csv(
            os.path.join(data_path, "fold_by_idx.csv"), index_col=self.index_col
        )

    def _read_fs(self, data_path):
        """Read selected features json file"""
        self.selected_features = utils.load_json(data_path)

    @staticmethod
    def _init_metrics_dict():
        return {
            "FOLD": [],
            "TRAIN_N_FOLDS": [],
            "TRAIN_SIZE": [],
            "TEST_SIZE": [],
            "N_FEATURES": [],
            "RMSE": [],
        }

    def _initialize_data(self, folds_path, pred_path, fs_path, fold):
        """Load all data"""
        data = pd.read_csv(os.path.join(self.root_path, "data.csv"))
        data.set_index(self.index_col, inplace=True)
        self._read_predictions(os.path.join(pred_path, f"{fold}.csv"))
        self._read_train(os.path.join(folds_path, fold), data)
        self._read_test(os.path.join(folds_path, fold), data)
        self._read_fold_idx_table(os.path.join(folds_path, fold))
        self._read_fs(os.path.join(fs_path, f"{fold}.json"))

    def _get_fold_name(self, fold):
        self.metrics["FOLD"].append(fold)

    def _get_train_n_folds(self):
        fold_col = self.fold_idx.columns[0]
        self.metrics["TRAIN_N_FOLDS"].append(self.fold_idx[fold_col].nunique())

    def _get_train_size(self):
        self.metrics["TRAIN_SIZE"].append(self.train.shape[0])

    def _get_test_size(self):
        self.metrics["TEST_SIZE"].append(self.test.shape[0])

    def _get_n_features(self):
        self.metrics["N_FEATURES"].append(
            len(self.selected_features["selected_features"])
        )

    def _get_rmse(self):
        y_true = self.predictions[GROUND_TRUTH_COL]
        y_pred = self.predictions[PRED_COL]
        rmse = mean_squared_error(y_true, y_pred, squared=True)
        self.metrics["RMSE"].append(rmse)

    def _calculatemetrics(self, fold):
        self._get_fold_name(fold)
        self._get_train_n_folds()
        self._get_train_size()
        self._get_test_size()
        self._get_n_features()
        self._get_rmse()

    def _savemetrics(self):
        metrics = pd.DataFrame(self.metrics)
        metrics.to_csv(os.path.join(self.cur_dir, "metrics.csv"), index=False)

    def run(self):
        """Runs the predicting process per fold"""
        self._make_folders(
            ["results", self.scv_method, "evaluations", self.fs_method, self.ml_method,]
        )
        folds_path = os.path.join(self.root_path, "folds", self.scv_method)
        results_path = os.path.join(self.root_path, "results", self.scv_method)
        fs_path = os.path.join(results_path, "features_selected", self.fs_method)
        pred_path = os.path.join(
            results_path, "predictions", self.fs_method, self.ml_method
        )
        folds_name = self._get_folders_in_dir(folds_path)
        self.metrics = self._init_metrics_dict()
        for fold in tqdm(folds_name, desc="Evaluating predictions"):
            self._initialize_data(folds_path, pred_path, fs_path, fold)
            self._calculatemetrics(fold)
            self._savemetrics()
