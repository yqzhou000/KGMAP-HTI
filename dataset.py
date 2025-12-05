# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class HerbTargetDataset:
    def __init__(self, data_path, herb_ingredient_path, ingredient_target_path,
                 target_disease_path, herb_disease_path, num_herb, num_ingredient, 
                 num_target, num_disease, num_folds=5, seed=2023):

        self.num_herb = num_herb
        self.num_ingredient = num_ingredient
        self.num_target = num_target
        self.num_disease = num_disease
        self.num_folds = num_folds
        self.seed = seed

        self.load_data(data_path, herb_ingredient_path, 
                      ingredient_target_path, target_disease_path, herb_disease_path)

        self.build_pathways()
        
        self.generate_cv_splits()
    
    def load_data(self, data_path, herb_ingredient_path, 
                  ingredient_target_path, target_disease_path, herb_disease_path):

        df = pd.read_csv(data_path, names=['herb_id', 'target_id', 'rating'])
        self.pos_pairs = df[df['rating'] == 1][['herb_id', 'target_id']].values
        self.neg_pairs = df[df['rating'] == 0][['herb_id', 'target_id']].values
        
        hi_df = pd.read_csv(herb_ingredient_path,
                           names=['herb_id', 'ingredient_id', 'rating'])
        self.herb_ingredients = {}
        for _, row in hi_df[hi_df['rating'] == 1].iterrows():
            herb_id = row['herb_id']
            ingredient_id = row['ingredient_id']
            if herb_id not in self.herb_ingredients:
                self.herb_ingredients[herb_id] = set()
            self.herb_ingredients[herb_id].add(ingredient_id)
        
        it_df = pd.read_csv(ingredient_target_path,
                           names=['ingredient_id', 'target_id', 'rating'])
        self.ingredient_targets = {}
        for _, row in it_df[it_df['rating'] == 1].iterrows():
            ingredient_id = row['ingredient_id']
            target_id = row['target_id']
            if ingredient_id not in self.ingredient_targets:
                self.ingredient_targets[ingredient_id] = set()
            self.ingredient_targets[ingredient_id].add(target_id)
        
        td_df = pd.read_csv(target_disease_path,
                           names=['target_id', 'disease_id', 'rating'])
        self.target_diseases = {}
        for _, row in td_df[td_df['rating'] == 1].iterrows():
            target_id = row['target_id']
            disease_id = row['disease_id']
            if target_id not in self.target_diseases:
                self.target_diseases[target_id] = set()
            self.target_diseases[target_id].add(disease_id)
        
        hd_df = pd.read_csv(herb_disease_path,
                           names=['herb_id', 'disease_id', 'rating'])
        self.herb_diseases = {}
        for _, row in hd_df[hd_df['rating'] == 1].iterrows():
            herb_id = row['herb_id']
            disease_id = row['disease_id']
            if herb_id not in self.herb_diseases:
                self.herb_diseases[herb_id] = set()
            self.herb_diseases[herb_id].add(disease_id)
        
        print(f"Data loaded:")
        print(f"  Positive herb-target pairs: {len(self.pos_pairs)}")
        print(f"  Negative herb-target pairs: {len(self.neg_pairs)}")
        print(f"  Herb-ingredient mappings: {len(self.herb_ingredients)}")
        print(f"  Ingredient-target mappings: {len(self.ingredient_targets)}")
        print(f"  Target-disease mappings: {len(self.target_diseases)}")
        print(f"  Herb-disease mappings: {len(self.herb_diseases)}")
    
    def build_pathways(self):
        self.herb_target_paths = []
        self.path_counts = {}
        self.complete_paths = []
        
        for herb_id, target_id in self.pos_pairs:
            if herb_id in self.herb_ingredients:
                for ingredient_id in self.herb_ingredients[herb_id]:
                    if ingredient_id in self.ingredient_targets:
                        if target_id in self.ingredient_targets[ingredient_id]:

                            related_diseases = []
                            if target_id in self.target_diseases:
                                related_diseases = list(self.target_diseases[target_id])
                            
                            path_info = {
                                'herb_id': herb_id,
                                'ingredient_id': ingredient_id,
                                'target_id': target_id,
                                'related_diseases': related_diseases
                            }
                            
                            self.herb_target_paths.append(path_info)
                            
                            key = (herb_id, target_id)
                            self.path_counts[key] = self.path_counts.get(key, 0) + 1
                            
                            for disease_id in related_diseases:
                                complete_path = {
                                    'herb_id': herb_id,
                                    'ingredient_id': ingredient_id,
                                    'target_id': target_id,
                                    'disease_id': disease_id
                                }
                                self.complete_paths.append(complete_path)
        
        print(f"Pathways built:")
        print(f"  Total paths: {len(self.herb_target_paths)}")
        print(f"  Unique herb-target pairs with paths: {len(self.path_counts)}")
        print(f"  Complete herb-ingredient-target-disease paths: {len(self.complete_paths)}")
    
    def generate_cv_splits(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
        
        self.pos_splits = list(kf.split(self.pos_pairs))
        
        self.neg_splits = list(kf.split(self.neg_pairs))
        
        print(f"Generated {self.num_folds}-fold cross-validation splits")
    
    def get_fold_data(self, fold_idx):

        train_idx, test_idx = self.pos_splits[fold_idx]
        
        train_pos = self.pos_pairs[train_idx]
        test_pos = self.pos_pairs[test_idx]
        
        train_idx_neg, test_idx_neg = self.neg_splits[fold_idx]
        train_neg = self.neg_pairs[train_idx_neg]
        test_neg = self.neg_pairs[test_idx_neg]
        
        return train_pos, train_neg, test_pos, test_neg
    
    def get_path_statistics(self):
        return self.herb_target_paths, self.path_counts
    
    def get_complete_paths(self):
        return self.complete_paths
    
    def generate_semantic_negative_samples(self, pos_pairs, neg_ratio=1):
        neg_samples = []
        pos_set = set(map(tuple, pos_pairs))
        
        for herb_id, target_id in pos_pairs:
            for _ in range(neg_ratio):
                if herb_id in self.herb_ingredients:
                    ingredients = list(self.herb_ingredients[herb_id])
                    if len(ingredients) > 0:
                        ingredient = np.random.choice(ingredients)
                        if ingredient in self.ingredient_targets:
                            candidate_targets = list(
                                self.ingredient_targets[ingredient]
                            )
                            candidate_targets = [
                                t for t in candidate_targets 
                                if (herb_id, t) not in pos_set
                            ]
                            if len(candidate_targets) > 0:
                                neg_target = np.random.choice(candidate_targets)
                                neg_samples.append([herb_id, neg_target])
                                continue
                
                neg_target = np.random.randint(0, self.num_target)
                while (herb_id, neg_target) in pos_set:
                    neg_target = np.random.randint(0, self.num_target)
                neg_samples.append([herb_id, neg_target])
        
        return np.array(neg_samples)
