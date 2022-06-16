import contextlib
import sys
import json
from os import mkdir
from os.path import join
import os
import time
import pandas as pd
import numpy as np
from weka.core import jvm
from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection
from weka.core.dataset import create_instances_from_matrices
from src import utils
from typing import List

os.system("taskset -p 0xff %d" % os.getpid())


def _mkdir(root_path, folder_name: str) -> None:
    """Creates a folder at current path"""
    # logger = logging.getLogger(self.logger_name)
    cur_dir = join(root_path, folder_name)
    with contextlib.suppress(FileExistsError):
        mkdir(cur_dir)
        # logger.info(f"Entering folder: /{folder_name}")


def _make_folders(root_path, folders: List[str]):
    """Make the initial folders"""
    for folder in folders:
        _mkdir(root_path, folder)
        root_path = join(root_path, folder)
    return root_path


def _target_as_last_col(data, target_col) -> pd.DataFrame:
    """Position the target column in the dataset last position"""
    cols = [c for c in data.columns if c != target_col]
    return data[cols + [target_col]]


def _weka_cfs(data, target_col):
    """Runs the CFS method from WEKA"""
    jvm.start()
    data = _target_as_last_col(data, target_col)
    data_weka = create_instances_from_matrices(data.to_numpy())
    data_weka.class_is_last()
    search = ASSearch(
        classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"],
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
    jvm.stop()
    return data.columns.values[index_fs].tolist()


def read_data(root_path, dataset_name, index_col, target_col, fold_col):
    """read data"""
    dataset = pd.read_csv(join(root_path, dataset_name, "data.csv"))
    reorganized_cols = [
        col for col in dataset.columns if col not in [target_col, fold_col]
    ]
    reorganized_cols = reorganized_cols[:4000]
    reorganized_cols.append(target_col)
    dataset = dataset[reorganized_cols]
    with contextlib.suppress(KeyError):
        dataset.drop(columns=["[GEO]_LATITUDE", "[GEO]_LONGITUDE"], inplace=True)
    with contextlib.suppress(KeyError):
        dataset.drop(columns=["[GEO]_DIVISIONNM"], inplace=True)
    dataset.set_index(index_col, inplace=True)
    return dataset


def _save_selected_features(output_path, features, fold):
    """Save the list of selected features in a json file"""
    json_features = {"selected_features": features}
    with open(join(output_path, f"{fold}.json"), "w", encoding="utf-8") as file:
        json.dump(json_features, file, indent=4)


def main(dataset_name, val_method, root_path, fold, index_col, target_col, fold_col):
    data = read_data(root_path, dataset_name, index_col, target_col, fold_col)
    output_path = _make_folders(
        join(root_path, dataset_name),
        ["results", val_method, "features_selected", "CFS"],
    )
    folds_path = join(root_path, dataset_name, "folds", val_method)

    split_fold_idx = utils.load_json(join(folds_path, fold, "split_data.json"))
    training_data = data.loc[split_fold_idx["train"]].copy()

    jvm.start()
    selected_features = _weka_cfs(training_data, target_col)
    _save_selected_features(output_path, selected_features, fold)
    jvm.stop()


if __name__ == "__main__":
    val_method, root_path, dataset_name, fold, index_col, target_col, fold_col = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        sys.argv[7],
    )
    main(dataset_name, val_method, root_path, fold, index_col, target_col, fold_col)
