import os
import sys
import math
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

# from tables import Column
from weka.core import jvm
from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection
from weka.core.dataset import create_instances_from_matrices


def _target_as_last_col(data, target_col) -> pd.DataFrame:
    """Position the target column in the dataset last position"""
    cols = [c for c in data.columns if c != target_col]
    return data[cols[:4000] + [target_col]]


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


def all_features(data, target_col):
    return data.drop(target_col, axis=1).columns.values.tolist()


def main(
    root_path, data_sampled_path, data_path, fs_method, index_col, fold_col, target_col
):
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

    data_path = os.path.join(root_path, data_path, "data.csv")
    data_sampled_path = os.path.join(root_path, data_sampled_path)
    data = pd.read_csv(data_path, index_col=index_col)
    try:
        data.drop(columns=["[GEO]_DIVISIONNM"], inplace=True)
    except KeyError:
        pass
    try:
        data.drop(columns=["[GEO]_LATITUDE", "[GEO]_LONGITUDE"], inplace=True)
    except KeyError:
        pass
    data_sampled = pd.read_csv(
        os.path.join(data_sampled_path, "data.csv"), index_col=index_col
    )
    data_sampled.index = data_sampled.index.astype(data.index.dtype)
    data_sampled.drop(columns=[fold_col], inplace=True)
    try:
        data_sampled.drop(columns=["[GEO]_DIVISIONNM"], inplace=True)
    except KeyError:
        pass
    try:
        data_sampled.drop(columns=["[GEO]_LATITUDE", "[GEO]_LONGITUDE"], inplace=True)
    except KeyError:
        pass
    out_sample = data.drop(index=data_sampled.index).copy()
    columns_fold = out_sample[fold_col]
    if fs_method == "CFS":
        features = _weka_cfs(data_sampled, target_col)
    elif fs_method == "All":
        features = all_features(data_sampled, target_col)
    x = out_sample[features]
    y = out_sample[target_col]
    x.columns = [c.replace("[", "") for c in x.columns]
    x.columns = [c.replace("]", "") for c in x.columns]
    mean = []
    results_out = {}
    x_sampled = data_sampled[features]
    x_sampled.columns = [c.replace("[", "") for c in x.columns]
    x_sampled.columns = [c.replace("]", "") for c in x.columns]
    y_sampled = data_sampled[target_col]
    results_out["methods"] = ml_methods
    results_out["mean"] = []
    for ml_method in ml_methods:
        if ml_method == "KNN":
            model = KNeighborsRegressor(
                n_neighbors=math.floor(math.sqrt(x_sampled.shape[0]))
            )
        if ml_method == "OLS":
            model = LinearRegression()
        if ml_method == "Lasso":
            model = Lasso(alpha=0.001)
        if ml_method == "Ridge":
            model = Ridge(alpha=0.001)
        if ml_method == "ElasticNet":
            model = ElasticNet(alpha=0.001)
        if ml_method == "DT":
            model = DecisionTreeRegressor(random_state=1)
        if ml_method == "LGBM":
            model = LGBMRegressor()
        if ml_method == "RF":
            model = RandomForestRegressor(200)
        if ml_method == "SVM":
            model = SVR()
        if ml_method == "MLP":
            model = MLPRegressor(
                (math.floor((x_sampled.shape[1] + 1) / 2),),
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
        model.fit(x_sampled, y_sampled)
        # print(model.get_n_leaves())
        pred = model.predict(x)
        pred_full = pd.Series(pred, index=out_sample.index, name="pred_full")
        mean = mean_squared_error(y, pred)
        results_out["mean"].append(mean)

    results = pd.DataFrame(results_out)
    results["rank"] = results["mean"].rank()
    results.to_csv(
        os.path.join(data_sampled_path, "out_of_sample_error.csv"), index=False
    )


if __name__ == "__main__":
    root_path = sys.argv[1]
    sampled_data = sys.argv[2]
    data = sys.argv[3]
    fs_method = sys.argv[4]
    index_col = sys.argv[5]
    fold_col = sys.argv[6]
    target_col = sys.argv[7]
    main(root_path, sampled_data, data, fs_method, index_col, fold_col, target_col)
