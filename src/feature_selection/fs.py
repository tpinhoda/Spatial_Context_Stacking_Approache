"""Feature selection process"""
import os
import json
from dataclasses import dataclass, field
from typing import List
import pandas as pd
from tqdm import tqdm
from weka.core import jvm
from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection
from weka.core.dataset import create_instances_from_matrices
from src.data import Data
from src import utils


@dataclass
class FeatureSelection(Data):
    """Represents the feature selection process.

     Attributes
    ----------
        fs_method:str
            The feature selection method name
        scv_method:str
            The spatial cross-validation method name
        index_col: str
            The dataset´s index column name
        fold_col: str
            The dataset´s folds column name
        target_col: str
            The target column name
        root_path : str
            Root path
    """

    fs_method: str = "CFS"
    scv_method: str = "gbscv"
    index_col: str = "INDEX"
    fold_col: str = "INDEX_FOLDS"
    target_col: str = "TARGET"
    cols_remove: List =  field(default_factory=list)
    _data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _reorganize_cols(self, data) -> pd.DataFrame:
        """Position the target column in the dataset last position"""
        if self.cols_remove:
            data.drop(columns=self.cols_remove, inplace=True)
        cols = [c for c in data.columns if c != self.target_col]
        
        return data[cols + [self.target_col]]

    def _cor_fs(self, data) -> List:
        """Runs correlation based feature selections"""
        cor = data.drop(columns=[self.target_col, self.fold_col]).corrwith(
            data[self.target_col]
        )
        features = cor.nlargest(100)
        return features.index.values.tolist()

    def _all_fs(self, data) -> List:
        """Select all data"""
        data.drop(columns=[self.target_col, self.fold_col], inplace=True)
        return data.columns.values.tolist()

    def _weka_cfs(self, data) -> List:
        """Runs the CFS method from WEKA"""
        data_weka = create_instances_from_matrices(data.to_numpy())
        data_weka.class_is_last()
        search = ASSearch(
            classname="weka.attributeSelection.BestFirst",
            options=["-D", "1", "-N", "5"],
        )
        evaluator = ASEvaluation(
            classname="weka.attributeSelection.CfsSubsetEval",
            options=["-P", "1", "-E", "1"],
        )
        attsel = AttributeSelection()
        attsel.search(search)
        attsel.evaluator(evaluator)
        attsel.select_attributes(data_weka)
        index_fs = [i - 1 for i in attsel.selected_attributes]
        return data.columns.values[index_fs].tolist()

    def _save_selected_features(self, features, fold):
        """Save the list of selected features in a json file"""
        json_features = {"selected_features": features}
        with open(
            os.path.join(self.cur_dir, f"{fold}.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(json_features, file, indent=4)

    def run(self):
        """Runs the feature selection per fold"""
        self._data = pd.read_csv(os.path.join(self.root_path, "data.csv"))
        self._data.set_index(self.index_col, inplace=True)
        self._data = self._reorganize_cols(self._data)

        self._make_folders(
            ["results", self.scv_method, "features_selected", self.fs_method]
        )
        folds_path = os.path.join(self.root_path, "folds", self.scv_method)
        folds_name = self._get_folders_in_dir(folds_path)
      
        for fold in tqdm(folds_name, desc="Selecting Features"):
            split_fold_idx = utils.load_json(
                os.path.join(folds_path, fold, "split_data.json")
            )
            training_data = self._data.loc[split_fold_idx["train"]].copy()
            if self.fs_method == "CFS":
                selected_features = self._weka_cfs(training_data)
            elif self.fs_method == "Pearson":
                selected_features = self._cor_fs(training_data)
            elif self.fs_method == "All":
                selected_features = self._all_fs(training_data)
            else:
                continue
            self._save_selected_features(selected_features, fold)

        if self.fs_method == "CFS":
            jvm.stop()
