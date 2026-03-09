import numpy as np 
import scipy.sparse as sp
import pandas as pd
import pickle
import os
from utils import normalize_sym, normalize_row
from node2vec import node2vec_embedding


class HerbKnowledgeGraph:
    """草药知识图谱构建"""

    def __init__(self, data_dir, save_dir):
        self.data_dir = data_dir
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def build_graph(self):
        """构建知识图谱"""
        print("构建草药-靶标知识图谱...")

        # 读取所有数据文件
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

        # data_files = self._load_data_luo()

        # ---------- 节点唯一ID统计 -------------
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

        # 映射字典
        self.herb_id_map = {id_: idx for idx, id_ in enumerate(self.herb_ids)}
        self.target_id_map = {id_: idx for idx, id_ in enumerate(self.target_ids)}
        self.ingredient_id_map = {id_: idx for idx, id_ in enumerate(self.ingredient_ids)}
        self.disease_id_map = {id_: idx for idx, id_ in enumerate(self.disease_ids)}

        # 节点偏移
        self.offsets = {
            'herb': 0,
            'target': self.num_herb,
            'ingredient': self.num_herb + self.num_target,
            'disease': self.num_herb + self.num_target + self.num_ingredient
        }
        self.total_nodes = self.offsets['disease'] + self.num_disease

        print(f"节点统计 - 草药: {self.num_herb}, 靶标: {self.num_target}, "
              f"成分: {self.num_ingredient}, 疾病: {self.num_disease}")
        print(f"总节点数: {self.total_nodes}")

        # 创建节点类型
        self.node_types = self._create_node_types()

        # 构建邻接矩阵
        self.adjs_offset = self._build_adjacency_matrices(data_files)

        # 准备特征矩阵
        self.feature_matrices = self._prepare_feature_matrices(data_files)

        # 保存预处理数据
        self._save_preprocessed_data()

        return self

    # data_luo 
    # def _read_matrix_txt(self, filename, dtype=float):
    #     path = os.path.join(self.data_dir, filename)
    #     return np.loadtxt(path, dtype=dtype)
    #
    # def _matrix_to_edges(self, mat, col_names, weight_col):
    #     rows, cols = np.where(mat > 0)
    #     weights = mat[rows, cols]
    #     df = pd.DataFrame({
    #         col_names[0]: rows.astype(int),
    #         col_names[1]: cols.astype(int),
    #         weight_col: weights.astype(float)
    #     })
    #     return df
    #
    # def _load_data_luo(self):
    #     print("检测到 data_luo 格式数据，转换为边列表...")
    #
    #     drug_protein = self._read_matrix_txt("mat_drug_protein.txt", dtype=float)
    #     drug_disease = self._read_matrix_txt("mat_drug_disease.txt", dtype=float)
    #     drug_drug = self._read_matrix_txt("mat_drug_drug.txt", dtype=float)
    #     protein_protein = self._read_matrix_txt("mat_protein_protein.txt", dtype=float)
    #     protein_disease = self._read_matrix_txt("mat_protein_disease.txt", dtype=float)
    #     drug_se = self._read_matrix_txt("mat_drug_se.txt", dtype=float)
    #     sim_drug = self._read_matrix_txt("Similarity_Matrix_Drugs.txt", dtype=float)
    #     sim_protein = self._read_matrix_txt("Similarity_Matrix_Proteins.txt", dtype=float)
    #
    #     herb_target = self._matrix_to_edges(drug_protein, ['hid', 'tid'], 'rating')
    #     herb_target['rating'] = 1
    #     herb_disease = self._matrix_to_edges(drug_disease, ['hid', 'did'], 'weight')
    #     herb_herb = self._matrix_to_edges(drug_drug, ['h1', 'h2'], 'weight')
    #     target_target = self._matrix_to_edges(protein_protein, ['t1', 't2'], 'weight')
    #     target_disease = self._matrix_to_edges(protein_disease, ['tid', 'did'], 'weight')
    #     herb_ingredient = self._matrix_to_edges(drug_se, ['hid', 'iid'], 'weight')
    #     sim_herb = self._matrix_to_edges(sim_drug, ['h1', 'h2'], 'weight')
    #     sim_target = self._matrix_to_edges(sim_protein, ['t1', 't2'], 'weight')
    #
    #     ingredient_target = pd.DataFrame(columns=['iid', 'tid', 'weight'])
    #
    #     return {
    #         'herb_target': herb_target,
    #         'herb_herb': herb_herb,
    #         'sim_herb': sim_herb,
    #         'herb_disease': herb_disease,
    #         'herb_ingredient': herb_ingredient,
    #         'ingredient_target': ingredient_target,
    #         'target_target': target_target,
    #         'sim_target': sim_target,
    #         'target_disease': target_disease
    #     }

    def _create_node_types(self):
        """创建节点类型数组"""
        node_types = np.zeros((self.total_nodes,), dtype=np.int32)
        node_types[self.offsets['herb']:self.offsets['target']] = 0
        node_types[self.offsets['target']:self.offsets['ingredient']] = 1
        node_types[self.offsets['ingredient']:self.offsets['disease']] = 2
        node_types[self.offsets['disease']:] = 3
        return node_types

    def _build_adjacency_matrices(self, data_files):
        """构建邻接矩阵，所有ID映射到新的索引"""
        adjs_offset = {}

        # 0: herb-target
        ht_pos = data_files['herb_target'][data_files['herb_target']['rating'] == 1].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for h_id, t_id in ht_pos:
            h_idx = self.herb_id_map[h_id]
            t_idx = self.target_id_map[t_id]
            adj[h_idx, t_idx + self.offsets['target']] = 1
        adjs_offset['0'] = sp.coo_matrix(adj)
        print(f"herb-target edges: {len(ht_pos)}")

        # 1: herb-disease
        hd = data_files['herb_disease'].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for h_id, d_id in hd:
            h_idx = self.herb_id_map[h_id]
            d_idx = self.disease_id_map[d_id]
            adj[h_idx, d_idx + self.offsets['disease']] = 1
        adjs_offset['1'] = sp.coo_matrix(adj)
        print(f"herb-disease edges: {len(hd)}")

        # 2: herb-ingredient
        hi = data_files['herb_ingredient'].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for h_id, i_id in hi:
            h_idx = self.herb_id_map[h_id]
            i_idx = self.ingredient_id_map[i_id]
            adj[h_idx, i_idx + self.offsets['ingredient']] = 1
        adjs_offset['2'] = sp.coo_matrix(adj)
        print(f"herb-ingredient edges: {len(hi)}")

        # 3: ingredient-target
        it = data_files['ingredient_target'].to_numpy()[:, :2].astype(int)
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for i_id, t_id in it:
            i_idx = self.ingredient_id_map[i_id]
            t_idx = self.target_id_map[t_id]
            adj[i_idx + self.offsets['ingredient'], t_idx + self.offsets['target']] = 1
        adjs_offset['3'] = sp.coo_matrix(adj)
        print(f"ingredient-target edges: {len(it)}")

        # 4: target-disease
        td = data_files['target_disease'].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for t_id, d_id in td:
            t_idx = self.target_id_map[t_id]
            d_idx = self.disease_id_map[d_id]
            adj[t_idx + self.offsets['target'], d_idx + self.offsets['disease']] = 1
        adjs_offset['4'] = sp.coo_matrix(adj)
        print(f"target-disease edges: {len(td)}")

        # 5: herb-herb
        hh = data_files['herb_herb'].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for h1_id, h2_id in hh:
            h1_idx = self.herb_id_map[h1_id]
            h2_idx = self.herb_id_map[h2_id]
            adj[h1_idx, h2_idx] = 1
        adjs_offset['5'] = sp.coo_matrix(adj)
        print(f"herb-herb edges: {len(hh)}")

        # 6: sim_herb (weighted)
        simhh = data_files['sim_herb']
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for _, row in simhh.iterrows():
            h1_idx = self.herb_id_map[int(row['h1'])]
            h2_idx = self.herb_id_map[int(row['h2'])]
            adj[h1_idx, h2_idx] = row['weight']
        adjs_offset['6'] = sp.coo_matrix(adj)
        print(f"sim_herb edges: {len(simhh)}")

        # 7: target-target
        tt = data_files['target_target'].to_numpy()[:, :2]
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for t1_id, t2_id in tt:
            t1_idx = self.target_id_map[t1_id]
            t2_idx = self.target_id_map[t2_id]
            adj[t1_idx + self.offsets['target'], t2_idx + self.offsets['target']] = 1
        adjs_offset['7'] = sp.coo_matrix(adj)
        print(f"target-target edges: {len(tt)}")

        # 8: sim_target (weighted)
        simtt = data_files['sim_target']
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for _, row in simtt.iterrows():
            t1_idx = self.target_id_map[int(row['t1'])]
            t2_idx = self.target_id_map[int(row['t2'])]
            adj[t1_idx + self.offsets['target'], t2_idx + self.offsets['target']] = row['weight']
        adjs_offset['8'] = sp.coo_matrix(adj)
        print(f"sim_target edges: {len(simtt)}")

        return adjs_offset

    def _prepare_feature_matrices(self, data_files):
        """准备特征矩阵"""
        matrices = {}

        # herb-disease matrix
        hd = data_files['herb_disease'].to_numpy()[:, :2]
        hd_matrix = np.zeros((self.num_disease, self.num_herb), dtype=int)
        for h_id, d_id in hd:
            h_idx = self.herb_id_map[h_id]
            d_idx = self.disease_id_map[d_id]
            hd_matrix[d_idx, h_idx] = 1
        matrices['hd_matrix'] = hd_matrix

        # ingredient-herb matrix
        hi = data_files['herb_ingredient'].to_numpy()[:, :2]
        ih_matrix = np.zeros((self.num_ingredient, self.num_herb), dtype=int)
        for h_id, i_id in hi:
            h_idx = self.herb_id_map[h_id]
            i_idx = self.ingredient_id_map[i_id]
            ih_matrix[i_idx, h_idx] = 1
        matrices['ih_matrix'] = ih_matrix

        # target-ingredient matrix
        it = data_files['ingredient_target'].to_numpy()[:, :2].astype(int)
        ti_matrix = np.zeros((self.num_target, self.num_ingredient), dtype=int)
        for i_id, t_id in it:
            i_idx = self.ingredient_id_map[i_id]
            t_idx = self.target_id_map[t_id]
            ti_matrix[t_idx, i_idx] = 1
        matrices['ti_matrix'] = ti_matrix

        # disease-target matrix
        td = data_files['target_disease'].to_numpy()[:, :2]
        dt_matrix = np.zeros((self.num_disease, self.num_target), dtype=int)
        for t_id, d_id in td:
            t_idx = self.target_id_map[t_id]
            d_idx = self.disease_id_map[d_id]
            dt_matrix[d_idx, t_idx] = 1
        matrices['dt_matrix'] = dt_matrix

        return matrices

    def _save_preprocessed_data(self):
        """保存预处理数据"""
        # 保存节点类型
        np.save(os.path.join(self.save_dir, "node_types.npy"), self.node_types)

        # 保存邻接矩阵
        with open(os.path.join(self.save_dir, "adjs_offset.pkl"), "wb") as f:
            pickle.dump(self.adjs_offset, f)

        # 保存特征矩阵
        np.savez(os.path.join(self.save_dir, 'combined_matrices.npz'),
                 **self.feature_matrices)

        # 保存元信息
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

        print(f"预处理数据已保存到: {self.save_dir}")

