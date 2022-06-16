"""Predict data process"""
import os
import re
from dataclasses import dataclass, field
from typing import List
import pickle
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import joblib
import pandas as pd
import numpy as np
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
    train_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: List = field(default_factory=list)

    def _read_test_data(self, json_path, data):
        """Read the training data"""
        split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
        self.test_data = data.loc[split_fold_idx["test"]].copy()

    def _read_train_data(self, json_path, data):
        """Read the training data"""
        split_fold_idx = utils.load_json(os.path.join(json_path, "split_data.json"))
        self.train_data = data.loc[split_fold_idx["train"]].copy()

    def _selected_features_filtering(self, json_path):
        """Filter only the features selected"""

        selected_features = utils.load_json(json_path)
        selected_features["selected_features"].append(self.target_col)
        self.test_data = self.test_data[selected_features["selected_features"]]
        self.train_data = self.train_data[selected_features["selected_features"]]

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

    def _calculate_pca(self, data):
        """Return the PCA first component transformation on the traind data"""
        pca = PCA(n_components=1)
        data = data.drop(columns=[self.target_col])
        # For the IMCLA21 paper the PCA is executed only on the cennsus columns
        pca.fit(data)
        return pca.transform(data).flatten()

    def _calculate_similarity_matrix(self, pca_test, pca_train) -> np.ndarray:
        """Calculate the similarity matrix between test set and a given training
        fold set based on a given attribute"""
        return np.subtract.outer(pca_test, pca_train) ** 2

    @staticmethod
    def _calculate_gamma(similarity, weights) -> np.float64:
        """Calculate gamma or the semivariogram"""
        similarity = np.multiply(similarity, weights.to_numpy())
        gamma_dist = np.sum(similarity, axis=1)
        sum_diff = gamma_dist.sum()
        # sum_dist = similarity.size
        sum_dist = weights.to_numpy().sum()
        return sum_diff / (2 * sum_dist)

    def run(self):
        """Runs the predicting process per fold"""
        data = pd.read_csv(os.path.join(self.root_path, "data.csv"))
        geoweights = pd.read_csv(
            os.path.join(self.root_path, "normd_matrix.csv"), index_col="[GEO]_ID_CITY"
        )
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
        folds_name.remove("53")
        for fold in tqdm(folds_name, desc="Predicting test set"):
            self._read_test_data(os.path.join(folds_path, fold), data)
            self._read_train_data(os.path.join(folds_path, fold), data)
            original_test = self.test_data
            original_train = self.train_data
            if "Local" in self.fs_method:
                context_gamma = {}
                context_list = self._get_files_in_dir(os.path.join(fs_path, fold))
                test_var = {}
                geo_dist = {}
                fold_pred = pd.DataFrame()
                for context in context_list:
                    self.train_data = self.train_data[
                        self.train_data["INDEX_FOLDS"] == int(context.split(".")[0])
                    ]
                    self._selected_features_filtering(
                        os.path.join(fs_path, fold, context)
                    )
                    centroid_train = self.train_data.drop(
                        columns=[self.target_col]
                    ).mean(axis=0)
                    centroid_test = self.test_data.drop(columns=[self.target_col]).mean(
                        axis=0
                    )
                    try:
                        centroid_train = self.train_data.drop(
                            columns=[self.target_col, "INDEX_FOLDS"]
                        ).mean(axis=0)
                        centroid_test = self.test_data.drop(
                            columns=[self.target_col, "INDEX_FOLDS"]
                        ).mean(axis=0)
                    except KeyError:
                        pass
                    test_var[context.split(".")[0]] = (
                        (centroid_train - centroid_test) ** 2
                    ).sum()
                    # pca_test = self._calculate_pca(self.test_data)
                    # pca_train = self._calculate_pca(self.train_data)
                    # test_var[context.split(".")[0]] = np.var(pca_test)
                    # geo_weights_test = geoweights.loc[self.test_data.index, self.train_data.index.astype('str')]
                    # geo_dist[context.split(".")[0]] = geo_weights_test.to_numpy().mean()
                    # similarity = self._calculate_similarity_matrix(pca_test, pca_train)
                    # gamma = self._calculate_gamma(similarity, geo_weights_test)
                    # context_gamma[context.split(".")[0]] = gamma
                    model = self.load_model(
                        os.path.join(ml_path, fold, f"{context.split('.')[0]}.pkl")
                    )
                    fold_pred[context.split(".")[0]] = self._predict(model)
                    self.test_data = original_test
                    self.train_data = original_train

                # print(fold)
                # print(test_var)
                # print({k: v for k, v in sorted(test_var.items(), key=lambda item: item[1])})
                context_selected = [
                    k for k, _ in sorted(test_var.items(), key=lambda item: item[1])
                ][:1]
                # print(context_selected)
                # context_selected = [c for c, value in context_gamma.items() if value <= test_var[c]]
                # context_gamma = {c: context_gamma[c] for c in context_selected}
                # print({k: v for k, v in sorted(context_gamma.items(), key=lambda item: item[1])})
                # max_dist =max(context_gamma.values())
                # context_gamma_norm = {key: 1 - value/max_dist for key, value  in context_gamma.items()}
                # sum_context = 0
                # for key, value in context_gamma_norm.items():
                #    fold_pred[key] = fold_pred[key] * value
                #    sum_context += value

                # print(fold)
                # print(context_selected)
                # context_selected = context_gamma.keys()
                fold_pred["mean"] = fold_pred[context_selected].mean(axis=1)
                # fold_pred["mean_var"] = fold_pred[context_selected].sum(axis=1)/sum_context

                self.predictions = fold_pred["mean"].values
            else:
                self._selected_features_filtering(os.path.join(fs_path, f"{fold}.json"))
                model = self.load_model(os.path.join(ml_path, f"{fold}.pkl"))
                self._predict(model)
            self.save_prediction(fold)
