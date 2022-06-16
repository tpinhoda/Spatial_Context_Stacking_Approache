"""Pipeline to analyse electoral data"""
from dataclasses import dataclass, field
from re import L
from typing import Dict, List, Optional
import inspect
import pandas as pd
from src.scv.optimistic import Optimistic
from src.scv.gbscv import GraphBasedSCV
from src.scv.reg_gbscv import RegGraphBasedSCV
from src.scv.ultra_coservative import UltraConservative
from src.scv.traditional_scv import TraditionalSCV
from src.scv.cv import CrossValidation
from src.feature_selection.fs import FeatureSelection
from src.model.train import Train
from src.model.predict import Predict
from src.model.evaluate import Evaluate

PIPELINE_MAP = {
    "scv": {
        "UltraConservative": UltraConservative,
        "TraditionalSCV": TraditionalSCV,
        "RBuffer": GraphBasedSCV,
        "SRBuffer": GraphBasedSCV,
        "Optimistic": Optimistic,
        "RegGBSCV": RegGraphBasedSCV,
        "CrossValidation": CrossValidation,
    },
    "fs": FeatureSelection,
    "train": Train,
    "predict": Predict,
    "evaluate": Evaluate,
}


@dataclass(init=True)
class Pipeline:
    """Represents a pipeline to evaluate data.

    This object evaluate spatial data.

    Attributes
    ----------
    root_path : str
        Root path
    data: pd.Dataframe
        The spatial dataset to generate the folds
    adj_matrix: pd.Dataframe
        The adjacency matrix regarding the spatial objects in the data
    index_col: str
        The dataset´s index column name
    fold_col: str
        The dataset´s folds column name
    target_col: str
        The target column name
    scv_method: str
        The spatial cross-validation method
    run_selection: bool
        Whether to run or not the selection step
    kappa: float
        Graph-Based SCV kappa paramenter
    fs_method: str
        The feature selection method
    ml_method: str
        The machine learning method
    paper: bool
        Whether to run the spatial-cross validation according to the ICMLA21 paper
    switchers: Dict[str, int]
        Dictionary of switchers to generate the pipeline
    """

    root_path: str = None
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    adj_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    meshblocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_col: str = None
    index_meshblocks: str = None
    fold_col: str = None
    target_col: str = None
    scv_method: str = None
    run_selection: Optional[bool] = None
    kappa: Optional[float] = None
    fs_method: str = None
    ml_method: str = None
    paper: bool = False
    fast: bool = False
    type_graph: str = None
    switchers: Dict[str, str] = field(default_factory=dict)
    pipeline: List[str] = field(default_factory=list)
    cols_remove: List[str] = field(default_factory=list)

    @staticmethod
    def _get_class_attributes(class_process):
        """Returns the attributes required to instanciate a class"""
        attributes = inspect.getmembers(
            class_process, lambda a: not inspect.isroutine(a)
        )
        attributes = [
            a[0]
            for a in attributes
            if not (a[0].startswith("__") and a[0].endswith("__"))
        ]
        return [attr for attr in attributes if not attr.startswith("_")]

    def _get_parameter_value(self, attributes):
        """Get parameter values"""
        params = {
            "root_path": self.root_path,
            "data": self.data,
            "adj_matrix": self.adj_matrix,
            "meshblocks": self.meshblocks,
            "w_matrix": self.w_matrix,
            "index_col": self.index_col,
            "index_meshblocks": self.index_meshblocks,
            "fold_col": self.fold_col,
            "target_col": self.target_col,
            "scv_method": self.scv_method,
            "run_selection": self.run_selection,
            "kappa": self.kappa,
            "fs_method": self.fs_method,
            "ml_method": self.ml_method,
            "paper": self.paper,
            "fast": self.fast,
            "type_graph": self.type_graph,
            "cols_remove": self.cols_remove
        }
        if params["scv_method"] == "RegGBSCV":
            if params["run_selection"]:
                params["scv_method"] = f"RegGBSCV_SR_Kappa_{self.kappa}"
            else:
                params["scv_method"] = f"RegGBSCV_R_Kappa_{self.kappa}"

        return {attr: params.get(attr) for attr in attributes}

    def _generate_parameters(self, process):
        """Generate parameters dict"""
        attributes = self._get_class_attributes(process)
        return self._get_parameter_value(attributes)

    def _get_init_function(self, process):
        """Return the initialization fucntion"""
        return PIPELINE_MAP[process]

    def _init_class(self, process):
        """Initialize a generic class"""
        if process == "scv":
            data_class = self._get_init_function("scv")[self.scv_method]
        else:
            data_class = self._get_init_function(process)
        parameters = self._generate_parameters(data_class())
        return data_class(**parameters)

    def _init_evaluate(self, process):
        """Initialize evaluate class"""
        eval_class = self._get_init_function(process)
        parameters = self._generate_parameters(eval_class())
        return eval_class(**parameters)

    def get_pipeline_order(self):
        """Return pipeline order"""
        return [process for process in self.switchers if self.switchers[process]]

    def map_pipeline_process(self, process):
        """Map the process initialization functions"""
        processes = {
            "scv": self._init_class,
            "fs": self._init_class,
            "train": self._init_class,
            "predict": self._init_class,
            "evaluate": self._init_evaluate,
        }
        return processes[process](process)

    def generate_pipeline(self):
        """Generate pipeline to process data"""
        pipeline_order = self.get_pipeline_order()
        for process in pipeline_order:
            self.pipeline.append(self.map_pipeline_process(process))

    def run(self):
        """Run pipeline"""
        self.generate_pipeline()
        for process in self.pipeline:
            process.run()
