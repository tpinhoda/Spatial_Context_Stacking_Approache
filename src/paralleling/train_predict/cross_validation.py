import contextlib
import os
import sys
import pandas as pd
import geopandas as gpd
from weka.core import jvm
from src import utils
from src.pipeline import Pipeline
from src.visualization.performance import VizMetrics
from src.visualization.dependence import VizDependence

# Set pipeline switchers
SWITCHERS = {
    "scv": False,
    "fs": False,
    "train": True,
    "predict": True,
    "evaluate": False,
}

ml_methods = [
    "KNN",
    "OLS",
    "Lasso",
    "Ridge",
    "ElasticNet",
    "DT",
    "LGBM",
    "RF",
    "MLP",
    "SVM",
]


def main(root_path, dataset, fs_method, index_col, index_fold, target_col, ml_method):
    """Runs main script"""
    utils.initialize_coloredlog()
    utils.initialize_rich_tracerback()
    utils.initialize_logging()

    data_path = os.path.join(root_path, dataset, "data.csv")
    # Load data
    data = pd.read_csv(data_path, index_col=index_col, low_memory=False)
    with contextlib.suppress(KeyError):
        data.drop(columns=["[GEO]_LATITUDE", "[GEO]_LONGITUDE"], inplace=True)
    # Run pipeline
    CrossValidation = Pipeline(
        root_path=os.path.join(root_path, dataset),
        data=data,
        meshblocks=None,
        index_col=index_col,
        fold_col=index_fold,
        target_col=target_col,
        scv_method="CrossValidation",
        fs_method=fs_method,
        ml_method=ml_method,
        switchers=SWITCHERS,
    )

    print(
        f"Running the CrossValidation SCV approach for dataset: {dataset} ML Method = {ml_method}"
    )
    CrossValidation.run()


if __name__ == "__main__":
    root_path = sys.argv[1]
    dataset = sys.argv[2]
    fs_method = sys.argv[3]
    index_col = sys.argv[4]
    fold_col = sys.argv[5]
    target_col = sys.argv[6]
    ml_method = sys.argv[7]
    print(dataset, fs_method, index_col, fold_col, target_col)
    main(root_path, dataset, fs_method, index_col, fold_col, target_col, ml_method)
