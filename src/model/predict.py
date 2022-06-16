"""Predict data process"""
import os
import re
from dataclasses import dataclass, field
from typing import List
import joblib
import pandas as pd
from tqdm import tqdm
from src.data import Data
import src.utils as utils

PRED_COL = "PREDICTIONS"
GROUND_TRUTH_COL = "GROUND_TRUTH"


@dataclass
class Predict(Data):
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
        target_col: str
            The target column name
        root_path : str
            Root path
    """

    ml_method: str = "LGBM"
    fs_method: str = "CFS"
    scv_method: str = "gbscv"
    index_col: str = "INDEX"
    target_col: str = "TARGET"
    test_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: List = field(default_factory=list)

    def _read_test_data(self, json_path, data):
        """Read the training data"""
        split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
        self.test_data = data.loc[split_fold_idx["test"]].copy()

    def _selected_features_filtering(self, json_path):
        """Filter only the features selected"""
        selected_features = utils.load_json(json_path)
        selected_features["selected_features"].append(self.target_col)
        self.test_data = self.test_data[selected_features["selected_features"]]

    @staticmethod
    def load_model(filepath):
        """Load pickled models"""
        # return pickle.load(open(filepath, "rb"))
        return joblib.load(filepath)

    def _clean_train_data_col(self):
        clean_cols = [re.sub(r"\W+", "", col) for col in self.test_data.columns]
        self.test_data.columns = clean_cols

    def _split_data(self):
        """Split the data into explanatory and target features"""
        y_test = self.test_data[self.target_col]
        x_test = self.test_data.drop(columns=[self.target_col])
        return x_test, y_test

    def _predict(self, model):
        """make prediction"""
        self._clean_train_data_col()
        x_test, _ = self._split_data()
        self.predictions = model.predict(x_test)
        return self.predictions

    def save_prediction(self, fold):
        """Save the model's prediction"""
        self.test_data[PRED_COL] = self.predictions
        self.test_data[GROUND_TRUTH_COL] = self.test_data[self.target_col]
        pred_to_save = self.test_data[[PRED_COL, GROUND_TRUTH_COL]]
        pred_to_save.to_csv(os.path.join(self.cur_dir, f"{fold}.csv"))

    def run(self):
        """Runs the predicting process per fold"""
        data = pd.read_csv(os.path.join(self.root_path, "data.csv"))
        data.set_index(self.index_col, inplace=True)
        self._make_folders(
            ["results", self.scv_method, "predictions", self.fs_method, self.ml_method,]
        )
        folds_path = os.path.join(self.root_path, "folds", self.scv_method)
        results_path = os.path.join(self.root_path, "results", self.scv_method)
        fs_path = os.path.join(results_path, "features_selected", self.fs_method)
        ml_path = os.path.join(
            results_path, "trained_models", self.fs_method, self.ml_method
        )
        folds_name = self._get_folders_in_dir(folds_path)
        for fold in tqdm(folds_name, desc="Predicting test set"):
            self._read_test_data(os.path.join(folds_path, fold), data)
            self._selected_features_filtering(os.path.join(fs_path, f"{fold}.json"))
            model = self.load_model(os.path.join(ml_path, f"{fold}.pkl"))
            self._predict(model)
            self.save_prediction(fold)
