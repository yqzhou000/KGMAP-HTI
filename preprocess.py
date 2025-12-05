import numpy as np 
import scipy.sparse as sp
import pandas as pd
import pickle
import os
from utils import normalize_sym, normalize_row
from node2vec import node2vec_embedding


class HerbKnowledgeGraph:

    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def build_graph(self):
        print("Building herb-target knowledge graph...")

        data_files = {
            'herb_target': pd.read_csv(os.path.join(self.data_dir, "herb_target.dat"),
                                       delimiter=',', names=['hid', 'tid', 'rating']),
            'herb_herb': pd.read_csv(os.path.join(self.data_dir, "herb_herb.dat"),
                                     delimiter=',', names=['h1', 'h2', 'weight']),
            'sim_herb': pd.read_csv(os.path.join(self.data_dir, "sim_herbs.dat"),
                                    delimiter=',', names=['h1', 'h2', 'weight']),
            'herb_disease': pd.read_csv(os.path.join(self.data_dir, "herb_disease.dat"),
                                        delimiter=',', names=['hid', 'did', 'weight']),
            'herb_ingredient': pd.read_csv(os.path.join(self.data_dir, "herb_ingredient.dat"),
                                           delimiter=',', names=['hid', 'iid', 'weight']),
            'ingredient_target': pd.read_csv(os.path.join(self.data_dir, "ingredient_target.dat"),
                                             delimiter=',', names=['iid', 'tid', 'weight']),
            'target_target': pd.read_csv(os.path.join(self.data_dir, "target_target.dat"),
                                         delimiter=',', names=['t1', 't2', 'weight']),
            'sim_target': pd.read_csv(os.path.join(self.data_dir, "sim_targets.dat"),
                                      delimiter=',', names=['t1', 't2', 'weight']),
            'target_disease': pd.read_csv(os.path.join(self.data_dir, "target_disease.dat"),
                                          delimiter=',', names=['tid', 'did', 'weight'])
        }

        all_herb_ids = pd.concat([
            data_files['herb_target']['hid'],
            data_files['herb_disease']['hid'],
            data_files['herb_ingredient']['hid'],
            data_files['herb_herb']['h1'], data_files['herb_herb']['h2'],
            data_files['sim_herb']['h1'], data_files['sim_herb']['h2']
        ]).unique()
        self.herb_ids = sorted(all_herb_ids)
        self.num_herb = len(self.herb_ids)

        all_target_ids = pd.concat([
            data_files['herb_target']['tid'],
            data_files['ingredient_target']['tid'],
            data_files['target_target']['t1'], data_files['target_target']['t2'],
            data_files['sim_target']['t1'], data_files['sim_target']['t2'],
            data_files['target_disease']['tid']
        ]).unique()
        self.target_ids = sorted(all_target_ids)
        self.num_target = len(self.target_ids)

        all_ingredient_ids = pd.concat([
            data_files['herb_ingredient']['iid'],
            data_files['ingredient_target']['iid']
        ]).unique()
        self.ingredient_ids = sorted(all_ingredient_ids)
        self.num_ingredient = len(self.ingredient_ids)

        all_disease_ids = pd.concat([
            data_files['herb_disease']['did'],
            data_files['target_disease']['did']
        ]).unique()
        self.disease_ids = sorted(all_disease_ids)
        self.num_disease = len(self.disease_ids)

        self.herb_id_map = {id_: idx for idx, id_ in enumerate(self.herb_ids)}
        self.target_id_map = {id_: idx for idx, id_ in enumerate(self.target_ids)}
        self.ingredient_id_map = {id_: idx for idx, id_ in enumerate(self.ingredient_ids)}
        self.disease_id_map = {id_: idx for idx, id_ in enumerate(self.disease_ids)}

        self.offsets = {
            'herb': 0,
            'target': self.num_herb,
            'ingredient': self.num_herb + self.num_target,
            'disease': self.num_herb + self.num_target + self.num_ingredient
        }
        self.total_nodes = self.offsets['disease'] + self.num_disease

        print(f"Node statistics - Herbs: {self.num_herb}, Targets: {self.num_target}, "
              f"Ingredients: {self.num_ingredient}, Diseases: {self.num_disease}")
        print(f"Total nodes: {self.total_nodes}")

        self.node_types = self._create_node_types()

        self.adjs_offset = self._build_adjacency_matrices(data_files)

        self.feature_matrices = self._prepare_feature_matrices(data_files)

        self._save_preprocessed_data()

        return self

    def _create_node_types(self):
        node_types = np.zeros((self.total_nodes,), dtype=np.int32)
        node_types[self.offsets['herb']:self.offsets['target']] = 0
        node_types[self.offsets['target']:self.offsets['ingredient']] = 1
        node_types[self.offsets['ingredient']:self.offsets['disease']] = 2
        node_types[self.offsets['disease']:] = 3
        return node_types

    def _build_adjacency_matrices(self, data_files):
        adjs_offset = {}

        ht_pos = data_files['herb_target'][data_files['herb_target']['rating'] == 1].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for h_id, t_id in ht_pos:
            h_idx = self.herb_id_map[h_id]
            t_idx = self.target_id_map[t_id]
            adj[h_idx, t_idx + self.offsets['target']] = 1
        adjs_offset['0'] = sp.coo_matrix(adj)
        print(f"herb-target edges: {len(ht_pos)}")

        hd = data_files['herb_disease'].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for h_id, d_id in hd:
            h_idx = self.herb_id_map[h_id]
            d_idx = self.disease_id_map[d_id]
            adj[h_idx, d_idx + self.offsets['disease']] = 1
        adjs_offset['1'] = sp.coo_matrix(adj)
        print(f"herb-disease edges: {len(hd)}")

        hi = data_files['herb_ingredient'].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for h_id, i_id in hi:
            h_idx = self.herb_id_map[h_id]
            i_idx = self.ingredient_id_map[i_id]
            adj[h_idx, i_idx + self.offsets['ingredient']] = 1
        adjs_offset['2'] = sp.coo_matrix(adj)
        print(f"herb-ingredient edges: {len(hi)}")

        it = data_files['ingredient_target'].to_numpy()[:, :2].astype(int)
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for i_id, t_id in it:
            i_idx = self.ingredient_id_map[i_id]
            t_idx = self.target_id_map[t_id]
            adj[i_idx + self.offsets['ingredient'], t_idx + self.offsets['target']] = 1
        adjs_offset['3'] = sp.coo_matrix(adj)
        print(f"ingredient-target edges: {len(it)}")

        td = data_files['target_disease'].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for t_id, d_id in td:
            t_idx = self.target_id_map[t_id]
            d_idx = self.disease_id_map[d_id]
            adj[t_idx + self.offsets['target'], d_idx + self.offsets['disease']] = 1
        adjs_offset['4'] = sp.coo_matrix(adj)
        print(f"target-disease edges: {len(td)}")

        if 'sim_herb' in data_files:
            simhh = data_files['sim_herb']
            adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
            for _, row in simhh.iterrows():
                h1_idx = self.herb_id_map[int(row['h1'])]
                h2_idx = self.herb_id_map[int(row['h2'])]
                adj[h1_idx, h2_idx] = row['weight']
                adj[h2_idx, h1_idx] = row['weight']
            print(f"herb-herb edges (weighted): {len(simhh)}")
        else:
            hh = data_files['herb_herb'].to_numpy()[:, :2]
            adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
            for h1_id, h2_id in hh:
                h1_idx = self.herb_id_map[h1_id]
                h2_idx = self.herb_id_map[h2_id]
                adj[h1_idx, h2_idx] = 1
                adj[h2_idx, h1_idx] = 1
            print(f"herb-herb edges (binary): {len(hh)}")
        adjs_offset['5'] = sp.coo_matrix(adj)

        if 'sim_target' in data_files:
            simtt = data_files['sim_target']
            adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
            for _, row in simtt.iterrows():
                t1_idx = self.target_id_map[int(row['t1'])]
                t2_idx = self.target_id_map[int(row['t2'])]
                adj[t1_idx + self.offsets['target'], t2_idx + self.offsets['target']] = row['weight']
                adj[t2_idx + self.offsets['target'], t1_idx + self.offsets['target']] = row['weight']
            print(f"target-target edges (weighted): {len(simtt)}")
        else:
            tt = data_files['target_target'].to_numpy()[:, :2]
            adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
            for t1_id, t2_id in tt:
                t1_idx = self.target_id_map[t1_id]
                t2_idx = self.target_id_map[t2_id]
                adj[t1_idx + self.offsets['target'], t2_idx + self.offsets['target']] = 1
                adj[t2_idx + self.offsets['target'], t1_idx + self.offsets['target']] = 1
            print(f"target-target edges (binary): {len(tt)}")
        adjs_offset['6'] = sp.coo_matrix(adj)

        return adjs_offset 

    def _prepare_feature_matrices(self, data_files):
        matrices = {}

        hd = data_files['herb_disease'].to_numpy()[:, :2]
        hd_matrix = np.zeros((self.num_disease, self.num_herb), dtype=int)
        for h_id, d_id in hd:
            h_idx = self.herb_id_map[h_id]
            d_idx = self.disease_id_map[d_id]
            hd_matrix[d_idx, h_idx] = 1
        matrices['hd_matrix'] = hd_matrix

        hi = data_files['herb_ingredient'].to_numpy()[:, :2]
        ih_matrix = np.zeros((self.num_ingredient, self.num_herb), dtype=int)
        for h_id, i_id in hi:
            h_idx = self.herb_id_map[h_id]
            i_idx = self.ingredient_id_map[i_id]
            ih_matrix[i_idx, h_idx] = 1
        matrices['ih_matrix'] = ih_matrix

        it = data_files['ingredient_target'].to_numpy()[:, :2].astype(int)
        ti_matrix = np.zeros((self.num_target, self.num_ingredient), dtype=int)
        for i_id, t_id in it:
            i_idx = self.ingredient_id_map[i_id]
            t_idx = self.target_id_map[t_id]
            ti_matrix[t_idx, i_idx] = 1
        matrices['ti_matrix'] = ti_matrix

        td = data_files['target_disease'].to_numpy()[:, :2]
        dt_matrix = np.zeros((self.num_disease, self.num_target), dtype=int)
        for t_id, d_id in td:
            t_idx = self.target_id_map[t_id]
            d_idx = self.disease_id_map[d_id]
            dt_matrix[d_idx, t_idx] = 1
        matrices['dt_matrix'] = dt_matrix

        return matrices

    def _save_preprocessed_data(self):
        np.save(os.path.join(self.save_dir, "node_types.npy"), self.node_types)

        with open(os.path.join(self.save_dir, "adjs_offset.pkl"), "wb") as f:
            pickle.dump(self.adjs_offset, f)

        np.savez(os.path.join(self.save_dir, 'combined_matrices.npz'),
                 **self.feature_matrices)

        meta_info = {
            'num_herb': self.num_herb,
            'num_target': self.num_target,
            'num_ingredient': self.num_ingredient,
            'num_disease': self.num_disease,
            'offsets': self.offsets,
            'total_nodes': self.total_nodes
        }
        with open(os.path.join(self.save_dir, "meta_info.pkl"), "wb") as f:
            pickle.dump(meta_info, f)

        print(f"Preprocessed data saved to: {self.save_dir}")
