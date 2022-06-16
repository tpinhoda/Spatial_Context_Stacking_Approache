"""Training data process"""
import os
import re
import math
from dataclasses import dataclass, field
import pickle
import joblib
import lightgbm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR
import pandas as pd
from tqdm import tqdm
from src.data import Data
import src.utils as utils

MAP_MODELS = {
    "LGBM": lightgbm.LGBMRegressor,
    "DT": DecisionTreeRegressor,
    "SVM": SVR,
    "KNN": KNeighborsRegressor,
    "MLP": MLPRegressor,
    "RF": RandomForestRegressor,
    "Lasso": Lasso,
    "OLS": LinearRegression,
    "Ridge": Ridge,
    "ElasticNet": ElasticNet,
}


@dataclass
class Train(Data):
    """Represents the training data process.

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
    train_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _read_train_data(self, json_path, data):
        """Read the training data"""
        split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
        self.train_data = data.loc[split_fold_idx["train"]].copy()

    def _selected_features_filtering(self, json_path):
        """Filter only the features selected"""
        selected_features = utils.load_json(json_path)
        selected_features["selected_features"].append(self.target_col)
        self.train_data = self.train_data[selected_features["selected_features"]]

    def _get_model(self, params):
        """Get the models by name"""
        if self.ml_method == "KNN":
            return MAP_MODELS[self.ml_method](
                n_neighbors=math.floor(math.sqrt(self.train_data.shape[0]))
            )
        if self.ml_method == "MLP":
            return MAP_MODELS[self.ml_method](
                (math.floor(self.train_data.shape[1] / 2),),
                random_state=1,
                max_iter=50000,
                learning_rate_init=0.001,
                learning_rate="invscaling",
                shuffle=False,
                early_stopping=False,
                batch_size=100,
                tol=1e-2,
                activation="relu",
                solver="adam",
            )
        if self.ml_method == "RF":
            return MAP_MODELS[self.ml_method](n_estimators=200, random_state=1)
        if self.ml_method == "DT":
            return MAP_MODELS[self.ml_method](random_state=1)
        if self.ml_method == "Lasso":
            return MAP_MODELS[self.ml_method](alpha=0.001, random_state=1)
        if self.ml_method == "OLS":
            return MAP_MODELS[self.ml_method]()
        if self.ml_method == "Ridge":
            return MAP_MODELS[self.ml_method](alpha=0.001)
        if self.ml_method == "ElasticNet":
            return MAP_MODELS[self.ml_method](alpha=0.001)
        if self.ml_method == "SVM":
            return MAP_MODELS[self.ml_method]()
        return MAP_MODELS[self.ml_method](*params)

    def _split_data(self):
        """Split the data into explanatory and target features"""
        self._clean_train_data_col()
        y_train = self.train_data[self.target_col]
        x_train = self.train_data.drop(columns=[self.target_col])
        return x_train, y_train

    def _clean_train_data_col(self):
        clean_cols = [re.sub(r"\W+", "", col) for col in self.train_data.columns]
        self.train_data.columns = clean_cols

    def _fit(self, model):
        """Fit the model"""
        x_train, y_train = self._split_data()
        return model.fit(x_train, y_train)

    def save_model(self, model, fold):
        """Save the model using picke"""
        joblib.dump(
            model, open(os.path.join(self.cur_dir, f"{fold}.pkl"), "wb"), compress=9
        )

    def run(self):
        """Runs the training process per fold"""
        data = pd.read_csv(os.path.join(self.root_path, "data.csv"))
        data.set_index(self.index_col, inplace=True)

        self._make_folders(
            [
                "results",
                self.scv_method,
                "trained_models",
                self.fs_method,
                self.ml_method,
            ]
        )
        folds_path = os.path.join(self.root_path, "folds", self.scv_method)
        fs_path = os.path.join(
            self.root_path,
            "results",
            self.scv_method,
            "features_selected",
            self.fs_method,
        )
        folds_name = self._get_folders_in_dir(folds_path)

        for fold in tqdm(folds_name, desc="Training model"):
            params = {}
            self._read_train_data(os.path.join(folds_path, fold), data)
            original_train = self.train_data
            if "Local" in self.fs_method:
                context_list = self._get_files_in_dir(os.path.join(fs_path, fold))
                for context in context_list:
                    self.train_data = original_train
                    self._selected_features_filtering(
                        os.path.join(fs_path, fold, context)
                    )
                    model = self._get_model(params=params)
                    model = self._fit(model)
                    ml_path = self.cur_dir
                    self._mkdir(fold)
                    self.save_model(model, context.split(".")[0])
                    self.cur_dir = ml_path
            else:
                self._selected_features_filtering(os.path.join(fs_path, f"{fold}.json"))
                model = self._get_model(params=params)
                model = self._fit(model)
                self.save_model(model, fold)
