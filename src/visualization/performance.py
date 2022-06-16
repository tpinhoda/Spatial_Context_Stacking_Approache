"""Performance visualization class"""
import os
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
import seaborn as sns
import scikit_posthocs as sp
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot
import matplotlib.pylab as plt
from tqdm import tqdm
from src.data import Data


@dataclass
class VizMetrics(Data):
    """Generates plots to visualize models performance.

    This object generates performance plots to compare different spatial
    cross-validation approaches.

    Attributes
    ----------
    root_path : str
        Root path
    index_col : str
        The metrics csv index column name
    fs_method : str
        The feature selection method used
    ml_method : str
        The machine learning method used
    """

    cv_methods: List = field(default_factory=list)
    index_col: str = None
    fs_method: str = None
    ml_method: str = None
    cv_methods_path: List = field(default_factory=list)
    cv_methods_results: Dict = field(default_factory=dict)

    def init_methods_path(self):
        """Initialize spatial cv folder paths"""
        self.cv_methods_path = [
            os.path.join(
                self.root_path,
                "results",
                method,
                "evaluations",
                self.fs_method,
                self.ml_method,
                "metrics.csv",
            )
            for method in self.cv_methods
        ]

    def load_cv_results(self):
        """Load metric results from each spatial cv being considered"""
        for data_path, method in zip(self.cv_methods_path, self.cv_methods):
            self.cv_methods_results[method] = pd.read_csv(
                data_path, index_col=self.index_col
            )

    def generate_metric_df(self, metric):
        """Generates a dataframe for a given metric"""
        index_fold = self.cv_methods_results["Optimistic"].index
        metric_df = pd.DataFrame(columns=self.cv_methods, index=index_fold)
        for cv_method, results in self.cv_methods_results.items():
            metric_df[cv_method] = results[metric]
        metric_df.index = metric_df.index.astype(str)
        return metric_df

    def generate_metric_plot(self):
        """Generates plots for the performance metrics"""
        sns.set(font_scale=2.2)
        sns.set_style("whitegrid", {"axes.grid": False})

        rmse = self.generate_metric_df(metric="RMSE")
        features = self.generate_metric_df(metric="N_FEATURES")
        instances = self.generate_metric_df(metric="TRAIN_SIZE")
        metrics = {"rmse": rmse, "features": features, "instances": instances}

        with PdfPages(os.path.join(self.cur_dir, "metrics.pdf")) as pdf_pages:
            for metric_name, metric in tqdm(metrics.items(), desc="Generating plots"):
                fig, fig_ax = pyplot.subplots(figsize=(20, 5))
                plt.xticks(rotation=45)
                fig_ax.set(ylabel="")
                fig_ax.set_title(metric_name.upper())
                sns.lineplot(
                    data=metric,
                    markers=True,
                    err_style="bars",
                    ax=fig_ax,
                    dashes=True,
                    linewidth=5,
                    palette=[
                        "#16b004",
                        "#6e1703",
                        "#6e1703",
                        "#6e1703",
                        "#6e1703",
                        "#6e1703",
                        "#6e1703",
                        "#f8ff0a",
                    ],
                    # palette="Set1"
                )

                fig_ax.legend(
                    bbox_to_anchor=(0.5, -1.0),
                    loc="lower center",
                    ncol=4,
                    borderaxespad=0.0,
                )

                pdf_pages.savefig(fig, bbox_inches="tight")

    def generate_mean_table(self):
        """Generates the mean performance for each spatial cv approache"""
        self.logger_info("Generating mean table.")
        columns = self.cv_methods_results["Optimistic"].columns.values.tolist()
        columns_std = [f"{col}_std" for col in columns]
        columns = columns + columns_std
        columns = columns.sort()
        mean_df = pd.DataFrame(columns=columns, index=self.cv_methods)
        for method, results in self.cv_methods_results.items():
            describe_df = results.describe()
            for col in results.columns:
                mean_df.loc[
                    method, col
                ] = f"{describe_df.loc['mean', col]} ({describe_df.loc['std', col]})"
        mean_df.T.to_csv(os.path.join(self.cur_dir, "mean_metrics.csv"))

    def tukey_post_hoc_test(self, metric):
        """Generate post hoc statistics"""
        metric_df = pd.DataFrame(columns=self.cv_methods)
        for method, results in self.cv_methods_results.items():
            metric_df[method] = results[metric]
        metric_df = metric_df.melt(var_name="groups", value_name="values")
        test = sp.posthoc_tukey(metric_df, val_col="values", group_col="groups")
        print(test)

    def run(self):
        """Runs the visualization step"""
        self._make_folders(["comparison"])
        self.set_logger_to_crit("matplotlib")
        self.init_methods_path()
        self.load_cv_results()
        self.generate_mean_table()
        self.generate_metric_plot()
        self.tukey_post_hoc_test("RMSE")
