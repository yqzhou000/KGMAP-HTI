import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class HerbTargetDataset:
    def __init__(self, data_path, herb_ingredient_path, ingredient_target_path, target_disease_path, 
                 num_herb, num_ingredient, num_target, num_disease, num_folds=5, seed=1, neg_ratio=1.0):
        self.data_path = data_path
        self.herb_ingredient_path = herb_ingredient_path
        self.ingredient_target_path = ingredient_target_path
        self.target_disease_path = target_disease_path
        self.num_herb = num_herb
        self.num_ingredient = num_ingredient
        self.num_target = num_target
        self.num_disease = num_disease
        self.num_folds = num_folds
        self.seed = seed
        self.neg_ratio = neg_ratio

    def generate_folds(self):
        np.random.seed(self.seed)
        
        # 读取原始草药-靶标数据
        herb_target_df = pd.read_csv(self.data_path, encoding='utf-8', delimiter=',',
                                   names=['hid', 'tid', 'rating'])
        
        # 读取路径数据
        herb_ingredient_df = pd.read_csv(self.herb_ingredient_path, encoding='utf-8', delimiter=',',
                                       names=['hid', 'iid', 'weight'])
        ingredient_target_df = pd.read_csv(self.ingredient_target_path, encoding='utf-8', delimiter=',',
                                         names=['iid', 'tid', 'weight'])
        target_disease_df = pd.read_csv(self.target_disease_path, encoding='utf-8', delimiter=',',
                                      names=['tid', 'did', 'weight'])

        # 构建草药→成分→靶标的路径映射
        herb_target_paths = self._build_herb_target_paths(
            herb_ingredient_df, ingredient_target_df, target_disease_df
        )
        
        # 正样本：仍然是原始的草药-靶标对
        dp_pos_all = herb_target_df[herb_target_df['rating'] == 1].to_numpy()[:, :2]
        dp_pos_all[:, 1] += self.num_herb  # 偏移靶标ID
        
        # 为每个草药-靶标对找到对应的路径信息
        path_mapping = self._create_path_mapping(herb_target_paths, dp_pos_all)
        
        all_herbs = herb_target_df['hid'].unique()
        all_targets = herb_target_df['tid'].unique()
        pos_set_all = set((h, t) for h, t in dp_pos_all[:, :2] - np.array([0, self.num_herb]))

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        pos_train_fold, pos_val_fold, pos_test_fold = [], [], []
        neg_train_fold, neg_val_fold, neg_test_fold = [], [], []
        path_train_fold, path_val_fold, path_test_fold = [], [], []

        for fold, (train_idx, test_idx) in enumerate(kf.split(dp_pos_all)):
            pos_train = dp_pos_all[train_idx]
            pos_test = dp_pos_all[test_idx]
            val_size = len(pos_test)
            pos_val = pos_train[:val_size]
            pos_train = pos_train[val_size:]

            # 对应的路径信息
            path_train = [path_mapping.get(tuple(pair), []) for pair in pos_train]
            path_val = [path_mapping.get(tuple(pair), []) for pair in pos_val]
            path_test = [path_mapping.get(tuple(pair), []) for pair in pos_test]

            pos_train_fold.append(pos_train)
            pos_val_fold.append(pos_val)
            pos_test_fold.append(pos_test)
            path_train_fold.append(path_train)
            path_val_fold.append(path_val)
            path_test_fold.append(path_test)

            # 生成负样本
            train_pos_raw = pos_train - np.array([0, self.num_herb])
            val_pos_raw = pos_val - np.array([0, self.num_herb])
            test_pos_raw = pos_test - np.array([0, self.num_herb])

            train_set = set(map(tuple, train_pos_raw))
            val_set = set(map(tuple, val_pos_raw))
            test_set = set(map(tuple, test_pos_raw))

            neg_train = self._gen_neg_samples(train_set, len(pos_train), all_herbs, all_targets)
            neg_val = self._gen_neg_samples(val_set, len(pos_val), all_herbs, all_targets)
            neg_test = self._gen_neg_samples(test_set, len(pos_test), all_herbs, all_targets)

            # 偏移靶标ID
            neg_train[:, 1] += self.num_herb
            neg_val[:, 1] += self.num_herb
            neg_test[:, 1] += self.num_herb

            neg_train_fold.append(neg_train)
            neg_val_fold.append(neg_val)
            neg_test_fold.append(neg_test)

        return (pos_train_fold, pos_val_fold, pos_test_fold,
                neg_train_fold, neg_val_fold, neg_test_fold,
                path_train_fold, path_val_fold, path_test_fold)

    def _build_herb_target_paths(self, herb_ingredient_df, ingredient_target_df, target_disease_df):
        """构建草药→成分→靶标的路径，包含疾病上下文信息"""
        paths = []
        
        # 构建映射字典
        herb_to_ingredients = herb_ingredient_df.groupby('hid')['iid'].apply(list).to_dict()
        ingredient_to_targets = ingredient_target_df.groupby('iid')['tid'].apply(list).to_dict()
        target_to_diseases = target_disease_df.groupby('tid')['did'].apply(list).to_dict()
        
        # 构建所有可能的草药→成分→靶标路径
        for herb_id in herb_to_ingredients:
            for ingredient_id in herb_to_ingredients.get(herb_id, []):
                for target_id in ingredient_to_targets.get(ingredient_id, []):
                    # 获取该靶标相关的疾病（作为上下文）
                    related_diseases = target_to_diseases.get(target_id, [])
                    
                    path_info = {
                        'herb_id': herb_id,
                        'ingredient_id': ingredient_id,
                        'target_id': target_id,
                        'related_diseases': related_diseases,
                        'path': [herb_id, ingredient_id, target_id],
                        'full_context': [herb_id, ingredient_id, target_id] + related_diseases
                    }
                    paths.append(path_info)
        
        return paths

    def _create_path_mapping(self, herb_target_paths, herb_target_pairs):
        """为每个草药-靶标对创建路径映射"""
        path_mapping = {}
        
        # 将路径按照草药-靶标对进行分组
        for path_info in herb_target_paths:
            herb_id = path_info['herb_id']
            target_id = path_info['target_id']
            
            # 构建与原始数据格式一致的key（包含偏移）
            key = (herb_id, target_id + self.num_herb)
            
            if key not in path_mapping:
                path_mapping[key] = []
            path_mapping[key].append(path_info)
        
        return path_mapping

    def _gen_neg_samples(self, pos_set, n_samples, all_herbs, all_targets):
        """生成负样本"""
        neg_set = set()
        while len(neg_set) < n_samples:
            h = np.random.choice(all_herbs)
            t = np.random.choice(all_targets)
            if (h, t) not in pos_set:
                neg_set.add((h, t))
        return np.array(list(neg_set), dtype=int)

    def get_path_statistics(self):
        """获取路径统计信息"""
        herb_ingredient_df = pd.read_csv(self.herb_ingredient_path, encoding='utf-8', delimiter=',',
                                       names=['hid', 'iid', 'weight'])
        ingredient_target_df = pd.read_csv(self.ingredient_target_path, encoding='utf-8', delimiter=',',
                                         names=['iid', 'tid', 'weight'])
        
        herb_target_paths = self._build_herb_target_paths(
            herb_ingredient_df, ingredient_target_df, 
            pd.read_csv(self.target_disease_path, encoding='utf-8', delimiter=',',
                       names=['tid', 'did', 'weight'])
        )
        
        print(f"总共找到 {len(herb_target_paths)} 条草药→成分→靶标路径")
        
        # 统计每个草药-靶标对的路径数量
        path_counts = {}
        for path_info in herb_target_paths:
            key = (path_info['herb_id'], path_info['target_id'])
            path_counts[key] = path_counts.get(key, 0) + 1
        
        path_count_values = list(path_counts.values())
        print(f"平均每个草药-靶标对有 {np.mean(path_count_values):.2f} 条路径")
        print(f"最多路径数: {max(path_count_values)}")
        print(f"最少路径数: {min(path_count_values)}")
        
        return herb_target_paths, path_counts