"""Generate graph-based cross-validation spatial folds"""
import os
import time
from typing import Dict, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from src.scv.scv import SpatialCV


X_1DIM_COL = "X_1DIM"


@dataclass
class RegGraphBasedSCV(SpatialCV):
    """Generates the Regularization Graph Based Spatial Cross-Validation folds
    Attributes
    ----------
        data: pd.Dataframe
            The spatial dataset to generate the folds
        fold_col: str
            The fold column name
        target_col: str
            The targer attribute column name
        adj_matrix: pd.Dataframe
            The adjacency matrix regarding the spatial objects in the data
        paper: bool
            Whether to run experiments according to ICMLA21 paper
        root_path : str
            Root path
    """

    kappa: float = 0.5
    run_selection: bool = False
    target_col: str = "TARGET"
    adj_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    paper: bool = False
    type_graph: str = "Sparse"
    sill_target: Dict = field(default_factory=dict)
    sill_reduced: Dict = field(default_factory=dict)
    sill_max_reduced: Dict = field(default_factory=dict)
    w_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)

    def _init_fields(self):
        if self.type_graph == "Sparse":
            self.w_matrix = pd.DataFrame(
                index=self.adj_matrix.index, columns=self.adj_matrix.columns
            )
            self.w_matrix.fillna(1, inplace=True)
        self.sill_target = {}
        self.sill_reduced = {}
        self.sill_max_reduced = {}

    def _calculate_train_pca(self) -> np.array:
        """Return the PCA first component transformation on the traind data"""
        pca = PCA(n_components=1)

        train = self.data.drop(columns=[self.fold_col, self.target_col])
        # For the IMCLA21 paper the PCA is executed only on the cennsus columns
        if self.paper:
            cols = [c for c in train.columns if "CENSUS" in c]
            train = train[cols]
        pca.fit(train)
        return pca.transform(train).flatten()

    def _calculate_removing_buffer_sill(self, fold_name, fold_data, global_var) -> Dict:
        """Calculate the sill for each fold to be used on the removing buffer process"""
        fold_target = fold_data[self.target_col]
        test_target = self.test_data[self.target_col]
        target_var = fold_target.append(test_target).var()
        self.sill_target[fold_name] = (target_var + global_var) / 2

    def _calculate_selection_buffer_sill(
        self, fold_name, fold_data, global_var
    ) -> Dict:
        """Calculate the sill for each fold to be used on the selection buffer process"""
        reduced_var = fold_data[X_1DIM_COL].append(self.test_data[X_1DIM_COL]).var()
        # self.sill_reduced[fold_name] = (reduced_var + global_var) / 2
        self.sill_reduced[fold_name] = reduced_var
        max_var_train = max(self.sill_reduced, key=self.sill_reduced.get)
        for _ in self.sill_reduced:
            self.sill_max_reduced[_] = self.sill_reduced[max_var_train]

    def _initiate_buffers_sills(self) -> Dict:
        """Initialize and calculate the sills for the removing and selectiont procedures"""
        global_target_var = self.data[self.target_col].var()
        global_reduced_var = self.data[X_1DIM_COL].var()
        self.sill_target = {}
        self.sill_reduced = {}
        for fold_name, fold_data in self.train_data.groupby(by=self.fold_col):
            self._calculate_selection_buffer_sill(
                fold_name, fold_data, global_reduced_var
            )
            self._calculate_removing_buffer_sill(
                fold_name, fold_data, global_target_var
            )

    def _convert_adj_matrix_index_types(self) -> pd.DataFrame:
        """Convert adjacenty matrixy index and columns types to the same as in the data"""
        self.adj_matrix.index = self.adj_matrix.index.astype(self.data.index.dtype)
        self.adj_matrix.columns = self.adj_matrix.columns.astype(self.data.index.dtype)
        self.w_matrix.index = self.w_matrix.index.astype(self.data.index.dtype)
        self.w_matrix.columns = self.w_matrix.columns.astype(self.data.index.dtype)

    @staticmethod
    def _get_neighbors(indexes, adj_matrix) -> List:
        """Return the 1-degree neighborhood from a given sub-graph formed by indexes"""
        area_matrix = adj_matrix.loc[indexes]
        neighbors = area_matrix.sum(axis=0) > 0
        neighbors = neighbors[neighbors].index
        neighbors = [n for n in neighbors if n not in indexes]
        return neighbors

    def _calculate_longest_path(self) -> int:
        """Calculate the longest_path from a BFS tree taking the test set as root"""
        path_indexes = self.test_data.index.values.tolist()
        local_data_idx = (
            self.test_data.index.values.tolist() + self.train_data.index.values.tolist()
        )
        matrix = self.adj_matrix.loc[local_data_idx, local_data_idx]
        neighbors = self._get_neighbors(path_indexes, matrix)
        size_tree = 0
        while len(neighbors) > 0:
            size_tree += 1
            neighbors = self._get_neighbors(path_indexes, matrix)
            path_indexes = path_indexes + neighbors
        return size_tree

    def _calculate_similarity_matrix(self, fold_data, attribute) -> np.ndarray:
        """Calculate the similarity matrix between test set and a given training
        fold set based on a given attribute"""
        test_values = self.test_data[attribute].to_numpy()
        node_values = fold_data[attribute]
        return (test_values - node_values) ** 2

    @staticmethod
    def _calculate_gamma(similarity, geo_weights, kappa) -> np.float64:
        """Calculate gamma or the semivariogram"""
        gamma_dist = similarity - (kappa * (1 - geo_weights) * similarity)
        sum_diff = np.sum(gamma_dist)
        sum_dist = len((similarity))
        return sum_diff / (2 * sum_dist)

    def _get_neighbors_weights(self, index):
        """Return the matrix weights test set x neighbors"""
        return self.w_matrix.loc[self.test_data.index, index]

    def _calculate_gamma_by_node(self, neighbors, attribute, kappa) -> Dict:
        """Calculate the semivariogram by folds"""
        nodes_gamma = {}
        neighbors = [n for n in neighbors if n in self.train_data.index]
        neighbors_data = self.train_data.loc[neighbors]
        for index, node_data in neighbors_data.iterrows():
            similarity = self._calculate_similarity_matrix(node_data, attribute)
            geo_weights = self._get_neighbors_weights(index)
            gamma = self._calculate_gamma(similarity, geo_weights, kappa)
            nodes_gamma[index] = gamma
        return nodes_gamma

    def _get_n_fold_neighbohood(self) -> int:
        """Get ne number of folds neighbors from the test set"""
        neighbors_idx = self._get_neighbors(self.test_data.index, self.adj_matrix)
        neighbors_idx = [n for n in neighbors_idx if n in self.data.index]
        return len(self.data.loc[neighbors_idx].groupby(self.fold_col))

    @staticmethod
    def _calculate_exponent(size_tree, count_n) -> np.float64:
        """Caclulate the decay exponent"""
        return np.log(1 * size_tree - count_n) / np.log(1 * size_tree)

    def _propagate_variance(self, attribute, kappa) -> List:
        """Calculate propagate variance"""
        # Initialize variables
        buffer = []  # containg the index of instaces buffered
        nodes_gamma = {}
        # Start creating the buffer
        while len(buffer) < self.train_data.shape[0]:
            # Get the instance indexes from te test set + the indexes buffer
            growing_graph_idx = self.test_data.index.values.tolist() + buffer
            # Get the neighbor
            h_neighbors = self._get_neighbors(growing_graph_idx, self.adj_matrix)
            # Calculate the semivariogram for each fold in the neighborhood
            nodes_gamma.update(
                self._calculate_gamma_by_node(h_neighbors, attribute, kappa)
            )
            buffer += h_neighbors
        return nodes_gamma

    def _calculate_selection_buffer(self, nodes_propagated, attribute):
        """Calculate buffer nodes"""
        buffered_nodes = []
        sill = self.data[attribute].var()
        buffered_nodes = [
            node for node, gamma in nodes_propagated.items() if gamma < sill
        ]
        return buffered_nodes

    def _calculate_removing_buffer(self, nodes_propagated, nodes_reduced, attribute):
        """Calculate buffer nodes"""
        sill_target = self.test_data[attribute].var()
        # sill_w_matrix = self.w_matrix.to_numpy().var()
        sill_reduced = self.test_data[X_1DIM_COL].var()

        # sill_target = self.kappa * sill_target + (1 - self.kappa) * sill_w_matrix

        buffered_nodes_target = [
            node for node, gamma in nodes_propagated.items() if gamma < sill_target
        ]
        buffered_nodes_reduced = [
            node for node, gamma in nodes_reduced.items() if gamma < sill_reduced
        ]
        # return [node for node in buffered_nodes_target if node in buffered_nodes_reduced]
        return buffered_nodes_target

    def run(self):
        """Generate graph-based spatial folds"""
        # Create folder folds
        start_time = time.time()
        self._init_fields()
        self._make_folders(["folds", self.scv_method])
        self.data[X_1DIM_COL] = self._calculate_train_pca()
        for fold_name, test_data in tqdm(
            self.data.groupby(by=self.fold_col), desc="Creating folds"
        ):
            if fold_name != -1:
                # Cread fold folder
                self._mkdir(str(fold_name))
                # Initialize x , y and reduce
                self._split_data_test_train(test_data)
                # Calculate local sill
                self._initiate_buffers_sills()
                # Ensure indexes and columns compatibility
                self._convert_adj_matrix_index_types()
                # Calculate selection buffer
                nodes_prop_reduced = self._propagate_variance(X_1DIM_COL, self.kappa)
                selection_buffer = self._calculate_selection_buffer(
                    nodes_prop_reduced, X_1DIM_COL
                )
                if self.run_selection:
                    self.train_data = self.train_data.loc[selection_buffer]
                # The train data is used to calcualte the buffer. Thus, the size tree,
                # and the gamma calculation will be influenced by the selection buffer.
                # Calculate removing buffer
                nodes_prop_target = self._propagate_variance(
                    self.target_col, self.kappa
                )
                removing_buffer = self._calculate_removing_buffer(
                    nodes_prop_target, nodes_prop_reduced, self.target_col
                )
                # removing_buffer = [node for node in removing_buffer if node in selection_buffer]
                # removing_buffer = selection_buffer
                self.train_data.drop(index=removing_buffer, inplace=True)
                # Save buffered data indexes
                self._save_buffered_indexes(removing_buffer)
                # Save fold index relation table
                self._save_fold_by_index_training()
                # Clean data
                self._clean_data(cols_drop=[X_1DIM_COL, self.fold_col])
                # Save data
                # self._save_data()
                # Update cur dir
                self.cur_dir = os.path.join(
                    self._get_root_path(), "folds", self.scv_method
                )
        # Save execution time
        end_time = time.time()
        self._save_time(end_time, start_time)
        print(f"Execution time: {end_time-start_time} seconds")
