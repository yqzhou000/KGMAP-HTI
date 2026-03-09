# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import datetime
import time
from tqdm import tqdm

import scipy.sparse as sp

from utils import sparse_mx_to_torch_sparse_tensor, normalize_row, normalize_sym
from dataset import HerbTargetDataset
from model import KGMAPHTIModel


class HerbTargetPredictor:
    """Herb-Ingredient-Target-Disease Predictor"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu}")
        self.use_basic_prediction = False
        
        # Load meta information
        self.load_meta_info()
        
        # Load path information
        self.load_path_info()

        # Load graph tensors for KGMAP model
        self.load_graph_tensors()
        
        # Load trained models
        self.load_trained_models()
        
        print("Predictor initialized successfully!")
        print(f"Number of herbs: {self.meta_info['num_herb']}")
        print(f"Number of targets: {self.meta_info['num_target']}")
        print(f"Number of ingredients: {self.meta_info['num_ingredient']}")
        print(f"Number of diseases: {self.meta_info['num_disease']}")
        print(f"Available paths: {len(self.path_counts)}")
        if self.use_basic_prediction:
            print("Note: Using basic prediction method (no trained models)")
    
    def load_meta_info(self):
        """Load meta information"""
        with open(os.path.join(self.config.preprocessed_dir, "meta_info.pkl"), "rb") as f:
            self.meta_info = pickle.load(f)
            
        # Set node offsets
        self.herb_offset = 0
        self.target_offset = self.meta_info['num_herb']
        self.ingredient_offset = self.meta_info['num_herb'] + self.meta_info['num_target']
        self.disease_offset = (self.meta_info['num_herb'] + self.meta_info['num_target'] + 
                              self.meta_info['num_ingredient'])
    
    def load_path_info(self):
        """Load path information"""
        print("Loading path information...")
        
        # Create dataset instance to get path information
        dataset = HerbTargetDataset(
            data_path=os.path.join(self.config.data_dir, "herb_target.dat"),
            herb_ingredient_path=os.path.join(self.config.data_dir, "herb_ingredient.dat"),
            ingredient_target_path=os.path.join(self.config.data_dir, "ingredient_target.dat"),
            target_disease_path=os.path.join(self.config.data_dir, "target_disease.dat"),
            num_herb=self.meta_info['num_herb'],
            num_ingredient=self.meta_info['num_ingredient'],
            num_target=self.meta_info['num_target'],
            num_disease=self.meta_info['num_disease'],
            num_folds=5,
            seed=self.config.seed
        )
        
        # Get path statistics
        self.herb_target_paths, self.path_counts = dataset.get_path_statistics()
        
        # Build path mappings
        self.build_path_mappings()
        
        # Load raw data mappings for path expansion
        self.load_raw_data_mappings()

    def load_graph_tensors(self):
        """Load graph tensors for KGMAP-HTI model"""
        with open(os.path.join(self.config.preprocessed_dir, "adjs_offset.pkl"), "rb") as f:
            adjs_offset = pickle.load(f)

        self.adjs_pt = []

        for i in range(5):
            self.adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_row(adjs_offset[str(i)] +
                              sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(self.device))
            self.adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_row(adjs_offset[str(i)].T +
                              sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(self.device))

        for i in range(5, 9):
            self.adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_sym(adjs_offset[str(i)] +
                              sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(self.device))

        self.adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            sp.eye(adjs_offset['0'].shape[0], dtype=np.float32).tocoo()).to(self.device))
        self.adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['0'].shape).to(self.device))
    
    def load_raw_data_mappings(self):
        """Load raw data mappings for path expansion prediction"""
        print("Building complete data mappings...")
        
        # Read all raw data
        hi_df = pd.read_csv(os.path.join(self.config.data_dir, "herb_ingredient.dat"), names=['hid', 'iid', 'rating'])
        it_df = pd.read_csv(os.path.join(self.config.data_dir, "ingredient_target.dat"), names=['iid', 'tid', 'rating'])
        td_df = pd.read_csv(os.path.join(self.config.data_dir, "target_disease.dat"), names=['tid', 'did', 'rating'])
        
        # Build complete mapping relationships (including all positive relationships)
        self.all_herb_ingredients = {}
        self.all_ingredient_targets = {}
        self.all_target_diseases = {}
        
        # Herb-ingredient mapping
        for _, row in hi_df[hi_df['rating'] == 1].iterrows():
            herb_id = row['hid']
            ingredient_id = row['iid']
            if herb_id not in self.all_herb_ingredients:
                self.all_herb_ingredients[herb_id] = set()
            self.all_herb_ingredients[herb_id].add(ingredient_id)
        
        # Ingredient-target mapping
        for _, row in it_df[it_df['rating'] == 1].iterrows():
            ingredient_id = row['iid']
            target_id = row['tid']
            if ingredient_id not in self.all_ingredient_targets:
                self.all_ingredient_targets[ingredient_id] = set()
            self.all_ingredient_targets[ingredient_id].add(target_id)
        
        # Target-disease mapping
        for _, row in td_df[td_df['rating'] == 1].iterrows():
            target_id = row['tid']
            disease_id = row['did']
            if target_id not in self.all_target_diseases:
                self.all_target_diseases[target_id] = set()
            self.all_target_diseases[target_id].add(disease_id)
        
        print(f"Complete mappings built:")
        print(f"  Herb-ingredient relationships: {len(self.all_herb_ingredients)} herbs")
        print(f"  Ingredient-target relationships: {len(self.all_ingredient_targets)} ingredients")
        print(f"  Target-disease relationships: {len(self.all_target_diseases)} targets")
    
    def build_path_mappings(self):
        """Build path mapping relationships"""
        print("Building path mappings...")
        
        # Herb->ingredient mapping
        self.herb_to_ingredients = {}
        # Ingredient->target mapping
        self.ingredient_to_targets = {}
        # Target->disease mapping
        self.target_to_diseases = {}
        # Complete path mapping
        self.herb_target_to_paths = {}
        
        for path_info in self.herb_target_paths:
            herb_id = path_info['herb_id']
            ingredient_id = path_info['ingredient_id']
            target_id = path_info['target_id']
            diseases = path_info.get('related_diseases', [])
            
            # Build mappings at each level
            if herb_id not in self.herb_to_ingredients:
                self.herb_to_ingredients[herb_id] = set()
            self.herb_to_ingredients[herb_id].add(ingredient_id)
            
            if ingredient_id not in self.ingredient_to_targets:
                self.ingredient_to_targets[ingredient_id] = set()
            self.ingredient_to_targets[ingredient_id].add(target_id)
            
            for disease_id in diseases:
                if target_id not in self.target_to_diseases:
                    self.target_to_diseases[target_id] = set()
                self.target_to_diseases[target_id].add(disease_id)
            
            # Build herb-target path mapping
            if (herb_id, target_id) not in self.herb_target_to_paths:
                self.herb_target_to_paths[(herb_id, target_id)] = []
            
            self.herb_target_to_paths[(herb_id, target_id)].append({
                'ingredient_id': ingredient_id,
                'diseases': diseases,
                'path': [herb_id, ingredient_id, target_id] + diseases
            })
    
    def find_possible_pathways(self, herb_id, target_id):
        """Find all possible pathways for a given herb-target pair"""
        possible_paths = []
        
        # 1. First check known paths
        known_paths = self.herb_target_to_paths.get((herb_id, target_id), [])
        for path in known_paths:
            possible_paths.append({
                'ingredient_id': path['ingredient_id'],
                'diseases': path['diseases'],
                'path_type': 'known',
                'confidence': 1.0
            })
        
        # 2. If no known paths, try to build possible paths
        if not known_paths:
            # Get all ingredients of this herb
            herb_ingredients = self.all_herb_ingredients.get(herb_id, set())
            
            for ingredient_id in herb_ingredients:
                # Check if this ingredient can act on the target
                ingredient_targets = self.all_ingredient_targets.get(ingredient_id, set())
                
                if target_id in ingredient_targets:
                    # Found potential path, get diseases of this target
                    target_diseases = list(self.all_target_diseases.get(target_id, set()))
                    
                    possible_paths.append({
                        'ingredient_id': ingredient_id,
                        'diseases': target_diseases,
                        'path_type': 'predicted',
                        'confidence': 0.8  # Predicted paths have slightly lower confidence
                    })
        
        # 3. If still no ingredient paths found, but target has disease information
        if not possible_paths:
            target_diseases = list(self.all_target_diseases.get(target_id, set()))
            if target_diseases:
                possible_paths.append({
                    'ingredient_id': None,  # Unknown ingredient
                    'diseases': target_diseases,
                    'path_type': 'partial',
                    'confidence': 0.6
                })
        
        # 4. If this herb has ingredients but no direct connection to target
        if not possible_paths:
            herb_ingredients = self.all_herb_ingredients.get(herb_id, set())
            if herb_ingredients:
                # Randomly select an ingredient as possible path
                sample_ingredient = next(iter(herb_ingredients))
                possible_paths.append({
                    'ingredient_id': sample_ingredient,
                    'diseases': [],
                    'path_type': 'novel_mechanism',
                    'confidence': 0.4
                })
        
        return possible_paths
    
    def load_trained_models(self):
        """Load trained models"""
        print("Loading trained models...")

        self.kgmap_model = None

        model_candidates = [
            os.path.join(self.config.model_dir, "kgmap_model.pth"),
            os.path.join(self.config.model_dir, "best_kgmap_model.pth"),
            os.path.join(self.config.model_dir, "best_herb_target_path_models.pkl")
        ]

        state_dict = None
        for path in model_candidates:
            if os.path.exists(path):
                try:
                    if path.endswith(".pth"):
                        data = torch.load(path, map_location="cpu")
                    else:
                        with open(path, "rb") as f:
                            data = pickle.load(f)
                    if isinstance(data, dict) and "state_dict" in data:
                        state_dict = data["state_dict"]
                    elif isinstance(data, dict):
                        state_dict = data
                    break
                except Exception as e:
                    print(f"Failed to load {path}: {e}")

        num_entities = self.meta_info["total_nodes"]
        num_relations = len(self.adjs_pt)
        kge_dim = getattr(self.config, "kge_dim", 128)
        num_layers = getattr(self.config, "gcn_layers", 3)

        self.kgmap_model = KGMAPHTIModel(
            num_entities=num_entities,
            num_relations=num_relations,
            n_hid=self.config.n_hid,
            num_layers=num_layers,
            kge_dim=kge_dim,
            dropout=self.config.dropout
        ).to(self.device)

        if state_dict is not None:
            try:
                self.kgmap_model.load_state_dict(state_dict, strict=False)
                self.kgmap_model.eval()
                print("Loaded KGMAP-HTI model parameters.")
                return
            except Exception as e:
                print(f"Warning: Failed to load KGMAP-HTI model params: {e}")

        print("Warning: No valid KGMAP-HTI model parameters found, will use basic prediction.")
        self.use_basic_prediction = True
    
    def try_load_alternative_models(self):
        """Try to load other possible model files"""
        model_dir = self.config.model_dir
        
        # Possible model file names
        possible_files = [
            "full_training_results.pkl",
            "best_models.pkl",
            "fold_models.pkl", 
            "herb_target_models.pkl",
            "trained_models.pkl"
        ]
        
        # Also check .pth files
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.pth'):
                    possible_files.append(file)
        
        for filename in possible_files:
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                print(f"Trying to load: {filepath}")
                try:
                    if filename.endswith('.pth'):
                        data = torch.load(filepath, map_location='cpu')
                        # Single .pth file, convert to list format
                        self.trained_models = [data]
                    else:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                        self.trained_models = data
                    
                    # Check if contains model parameters
                    if self.check_model_params(self.trained_models):
                        print(f"Successfully found file with model parameters: {filepath}")
                        self.prepare_features()
                        return
                        
                except Exception as e:
                    print(f"  Loading failed: {e}")
        
        # If no suitable model file found, create basic predictor
        print("No valid trained models found, will use basic prediction method")
        self.use_basic_prediction = True
        self.prepare_features()
    
    def check_model_params(self, data):
        """Check if data contains model parameters"""
        if isinstance(data, dict):
            # Check direct tensor parameters
            tensor_keys = [k for k, v in data.items() if isinstance(v, torch.Tensor)]
            if tensor_keys:
                return True
            
            # Check nested dictionaries
            for key, value in data.items():
                if isinstance(value, dict):
                    nested_tensors = [k for k, v in value.items() if isinstance(v, torch.Tensor)]
                    if nested_tensors:
                        return True
        
        elif isinstance(data, list):
            for item in data:
                if self.check_model_params(item):
                    return True
        
        elif isinstance(data, torch.Tensor):
            return True
        
        return False
    
    def final_score(self, y_hat, confidence, path_weight):
        """Compute final score using KGMAP-HTI formula"""
        y = torch.tensor(float(y_hat), device=self.device)
        c = torch.tensor(float(confidence), device=self.device)
        w = torch.tensor(float(path_weight), device=self.device)
        if self.kgmap_model is not None:
            return float(self.kgmap_model.final_score(y, c, w).item())
        return float((y * c * (1.0 + w)).item())

    def compute_confidence(self, herb_id, target_id, ingredient_id, path_type_confidence, path_weight):
        """融合路径注意力与路径数量的置信度"""
        if self.kgmap_model is None or self.h_final is None:
            return float(path_type_confidence)

        path_count = self.path_counts.get((herb_id, target_id), 0)
        path_count_norm = min(float(path_count) / 50.0, 1.0) if path_count > 0 else 1.0

        if ingredient_id is None or ingredient_id == '':
            beta_mean = 1.0 if path_count == 0 else 0.0
        else:
            path_info_list = [{
                'herb_id': herb_id,
                'ingredient_id': ingredient_id,
                'target_id': target_id
            }]
            beta_mean = self.kgmap_model.path_attention_confidence(
                self.h_final,
                path_info_list,
                {
                    'herb': self.herb_offset,
                    'target': self.target_offset,
                    'ingredient': self.ingredient_offset,
                    'disease': self.disease_offset
                }
            )
            if beta_mean == 0.0:
                beta_mean = 1.0

        confidence = float(path_type_confidence) * float(beta_mean) * float(path_count_norm)
        return confidence
    
    def predict_herb_targets(self, herb_ids=None, target_ids=None, threshold=0.5):
        """Predict herb-target interactions and build complete pathways"""
        print("Starting herb-target prediction...")
        
        # Determine prediction range
        if herb_ids is None:
            herb_ids = list(range(self.meta_info['num_herb']))
        if target_ids is None:
            target_ids = list(range(self.meta_info['num_target']))
        
        print(f"Prediction range: {len(herb_ids)} herbs × {len(target_ids)} targets")
        
        # Create prediction pairs
        prediction_pairs = []
        path_weights = []
        
        for herb_id in herb_ids:
            for target_id in target_ids:
                prediction_pairs.append([herb_id, target_id + self.target_offset])
                
                # Calculate path weight
                pair_key = (herb_id, target_id)
                weight = min(float(self.path_counts.get(pair_key, 0)) / 50.0, 0.5)
                path_weights.append(weight)
        
        prediction_pairs = np.array(prediction_pairs)
        path_weights = np.array(path_weights)
        
        print(f"Total prediction pairs: {len(prediction_pairs):,}")
        
        # Batch prediction
        all_predictions = self.batch_predict(prediction_pairs, path_weights)
        
        # Filter results and build complete pathways
        pathway_results = []
        processed_pairs = 0
        
        print("Building prediction pathways...")
        for i, (herb_id, target_id_with_offset) in enumerate(tqdm(prediction_pairs, desc="Building pathways")):
            target_id = target_id_with_offset - self.target_offset
            score = all_predictions[i]

            # Skip pairs that cannot exceed threshold after final score adjustment
            if score * 1.5 < threshold:
                continue

            # Find all possible pathways
            possible_paths = self.find_possible_pathways(herb_id, target_id)
                
            if possible_paths:
                # Create result row for each possible pathway
                for path_info in possible_paths:
                    ingredient_id = path_info['ingredient_id']
                    diseases = path_info['diseases']
                    path_type = path_info['path_type']
                    path_confidence = path_info['confidence']
                    
                    if diseases:
                        # Create one row for each disease
                        for disease_id in diseases:
                            confidence = self.compute_confidence(
                                herb_id, target_id, ingredient_id, path_confidence, path_weights[i]
                            )
                            final_score = self.final_score(score, confidence, path_weights[i])
                            if final_score <= threshold:
                                continue
                            pathway_result = {
                                'herb_id': herb_id,
                                'ingredient_id': ingredient_id if ingredient_id is not None else '',
                                'target_id': target_id,
                                'disease_id': disease_id,
                                'prediction_score': final_score,
                                'path_confidence': path_confidence,
                                'path_weight': path_weights[i],
                                'path_type': path_type,
                                'pathway': self.format_pathway(herb_id, ingredient_id, target_id, disease_id),
                                'predicted_interaction': True
                            }
                            pathway_results.append(pathway_result)
                    else:
                            confidence = self.compute_confidence(
                                herb_id, target_id, ingredient_id, path_confidence, path_weights[i]
                            )
                            final_score = self.final_score(score, confidence, path_weights[i])
                            if final_score <= threshold:
                                continue
                        # Pathway without disease information
                    pathway_result = {
                            'herb_id': herb_id,
                            'ingredient_id': ingredient_id if ingredient_id is not None else '',
                            'target_id': target_id,
                            'disease_id': '',
                            'prediction_score': final_score,
                            'path_confidence': path_confidence,
                            'path_weight': path_weights[i],
                            'path_type': path_type,
                            'pathway': self.format_pathway(herb_id, ingredient_id, target_id, None),
                            'predicted_interaction': True
                        }
                    pathway_results.append(pathway_result)
            else:
                confidence = self.compute_confidence(
                    herb_id, target_id, '', 0.3, path_weights[i]
                )
                final_score = self.final_score(score, confidence, path_weights[i])
                if final_score <= threshold:
                    continue
                # Completely new association with no pathway information
                pathway_result = {
                    'herb_id': herb_id,
                    'ingredient_id': '',
                    'target_id': target_id,
                    'disease_id': '',
                    'prediction_score': final_score,
                    'path_confidence': 0.3,  # Lowest confidence
                    'path_weight': path_weights[i],
                    'path_type': 'novel_association',
                    'pathway': f"{herb_id}→?→{target_id}→?",
                    'predicted_interaction': True
                }
                pathway_results.append(pathway_result)
            
            processed_pairs += 1
        
        print(f"Generated {len(pathway_results)} specific pathway predictions (threshold: {threshold})")
        print(f"Involving {processed_pairs} high-confidence herb-target pairs")
        return pathway_results
    
    def format_pathway(self, herb_id, ingredient_id, target_id, disease_id):
        """Format pathway display"""
        pathway_parts = [str(herb_id)]
        
        if ingredient_id is not None:
            pathway_parts.append(str(ingredient_id))
        else:
            pathway_parts.append("?")
        
        pathway_parts.append(str(target_id))
        
        if disease_id is not None and disease_id != '':
            pathway_parts.append(str(disease_id))
        else:
            pathway_parts.append("?")
        
        return "→".join(pathway_parts)
    
    def batch_predict(self, prediction_pairs, path_weights, batch_size=4096):
        """Batch prediction"""
        if self.use_basic_prediction:
            return self.basic_batch_predict(prediction_pairs, path_weights)

        dataset = TensorDataset(
            torch.LongTensor(prediction_pairs),
            torch.FloatTensor(path_weights)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if self.kgmap_model is None:
            print("Warning: KGMAP model not available, using basic prediction method")
            return self.basic_batch_predict(prediction_pairs, path_weights)

        all_predictions = []
        with torch.no_grad():
            self.h_final = self.kgmap_model.encode(self.adjs_pt)
            for batch_pairs, batch_weights in tqdm(dataloader, desc="Predicting with KGMAP-HTI"):
                batch_pairs = batch_pairs.to(self.device)
                batch_preds = self.kgmap_model.predict_pairs(
                    self.h_final,
                    batch_pairs,
                    [self.herb_target_to_paths.get(
                        (int(h), int(t - self.target_offset)), []
                    ) for h, t in batch_pairs.cpu().numpy()],
                    batch_weights.to(self.device),
                    {
                        'herb': self.herb_offset,
                        'target': self.target_offset,
                        'ingredient': self.ingredient_offset,
                        'disease': self.disease_offset
                    }
                )
                all_predictions.extend(batch_preds.cpu().numpy())

        return np.array(all_predictions)
    
    def extract_model_state(self, model_info, fold_idx):
        """Extract clean model state dictionary from saved model information"""
        if isinstance(model_info, torch.Tensor):
            return None  # Single tensor is not a complete state dictionary
            
        if not isinstance(model_info, dict):
            return model_info
        
        # Known non-model parameter keys
        non_model_keys = {'fold', 'best_val_auc', 'test_metrics', 'training_time', 'epoch', 
                         'val_auc', 'test_auc', 'train_loss', 'val_loss'}
        
        # Check common model state keys
        if 'model_state_dict' in model_info:
            return model_info['model_state_dict']
        elif 'state_dict' in model_info:
            return model_info['state_dict']
        
        # Filter non-model parameters
        model_state = {k: v for k, v in model_info.items() if k not in non_model_keys}
        
        # Check if contains expected model parameter keys
        expected_keys = ['herb_encoder.0.weight', 'target_encoder.0.weight', 'predictor.0.weight']
        has_model_params = any(key in model_state for key in expected_keys)
        
        if not has_model_params:
            return None
        
        return model_state
    
    def basic_batch_predict(self, prediction_pairs, path_weights):
        """Basic prediction method (used when no trained models available)"""
        print("Using basic prediction method...")
        
        predictions = []
        
        for i, (herb_id, target_id_with_offset) in enumerate(prediction_pairs):
            target_id = target_id_with_offset - self.target_offset
            path_weight = path_weights[i]
            
            # Basic prediction score = path weight + random noise
            base_score = float(min(path_weight * 2.0, 0.8))  # Based on path weight
            
            # If has path information, increase score
            if (herb_id, target_id) in self.herb_target_to_paths:
                num_paths = len(self.herb_target_to_paths[(herb_id, target_id)])
                path_bonus = min(num_paths * 0.1, 0.3)
                base_score += path_bonus
            
            # Add some randomness to simulate prediction uncertainty
            import random
            noise = random.uniform(-0.1, 0.1)
            y_hat = max(0.0, min(1.0, base_score + noise))
            predictions.append(y_hat)
        
        return np.array(predictions)
    
    def save_predictions(self, predictions, output_path, prediction_type="pathway"):
        """Save prediction results (highlight new discoveries)"""
        print(f"Saving prediction results to: {output_path}")
        
        if prediction_type == "pathway":
            # Save pathway prediction results
            df_data = []
            for pred in predictions:
                # Determine prediction type and discovery type
                has_ingredient = pred.get('ingredient_id', '') != ''
                has_disease = pred.get('disease_id', '') != ''
                path_type = pred.get('path_type', 'unknown')
                
                # Determine discovery type
                if path_type == 'known':
                    prediction_type_label = "Known Pathway"
                    discovery_type = "known_pathway"
                elif path_type == 'predicted':
                    prediction_type_label = "Predicted Pathway"
                    discovery_type = "predicted_pathway"
                elif path_type == 'partial':
                    prediction_type_label = "Novel Ingredient"
                    discovery_type = "novel_ingredient"
                elif path_type == 'novel_mechanism':
                    prediction_type_label = "Novel Mechanism"
                    discovery_type = "novel_mechanism"
                elif path_type == 'novel_association':
                    prediction_type_label = "Novel Association"
                    discovery_type = "novel_association"
                else:
                    if has_ingredient and has_disease:
                        prediction_type_label = "Complete Pathway"
                        discovery_type = "complete_pathway"
                    elif has_ingredient and not has_disease:
                        prediction_type_label = "Novel Disease"
                        discovery_type = "novel_disease"
                    elif not has_ingredient and has_disease:
                        prediction_type_label = "Novel Ingredient"
                        discovery_type = "novel_ingredient"
                    else:
                        prediction_type_label = "Novel Association"
                        discovery_type = "novel_association"
                
                row = {
                    'herb_id': pred['herb_id'],
                    'ingredient_id': pred.get('ingredient_id', ''),
                    'target_id': pred['target_id'],
                    'disease_id': pred.get('disease_id', ''),
                    'prediction_score': pred['prediction_score'],
                    'path_confidence': pred.get('path_confidence', 1.0),
                    'path_weight': pred['path_weight'],
                    'pathway': pred.get('pathway', ''),
                    'prediction_type': prediction_type_label,
                    'discovery_type': discovery_type,
                    'path_type': path_type,
                    'predicted_interaction': pred['predicted_interaction']
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
        
        # Save as CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Pathway prediction results saved: {len(predictions)} pathway records")
        
        # Display discovery type statistics
        if len(df_data) > 0:
            discovery_stats = {}
            for row in df_data:
                discovery_type = row['discovery_type']
                discovery_stats[discovery_type] = discovery_stats.get(discovery_type, 0) + 1
            
            print("\nModel Discovery Type Statistics:")
            for discovery_type, count in discovery_stats.items():
                percentage = count / len(df_data) * 100
                print(f"  {discovery_type}: {count} records ({percentage:.1f}%)")
            
            # Calculate novel discovery ratio
            known_types = {'known_pathway', 'complete_pathway'}
            novel_count = sum(count for dtype, count in discovery_stats.items() if dtype not in known_types)
            
            print(f"\nTotal Novel Discoveries: {novel_count} records ({novel_count/len(df_data)*100:.1f}%)")
            print("These are new drug-target associations predicted by the model!")
            
            # Save classified files by prediction type
            base_path = output_path.replace('.csv', '')
            
            # Save novel discoveries
            novel_predictions = df[~df['discovery_type'].isin(known_types)]
            if len(novel_predictions) > 0:
                novel_path = f"{base_path}_novel_discoveries.csv"
                novel_predictions.to_csv(novel_path, index=False, encoding='utf-8')
                print(f"Novel discoveries saved separately to: {novel_path}")
            
            # Save known pathway validation
            known_predictions = df[df['discovery_type'].isin(known_types)]
            if len(known_predictions) > 0:
                known_path = f"{base_path}_known_pathways.csv"
                known_predictions.to_csv(known_path, index=False, encoding='utf-8')
                print(f"Known pathway validation saved to: {known_path}")


def run_predictions(config):
    """Run prediction pipeline - highlight new discoveries"""
    print("="*60)
    print("Herb-Ingredient-Target-Disease Prediction Pipeline")
    print("Discovering New Drug-Target Associations")
    print("="*60)
    
    # Create predictor
    predictor = HerbTargetPredictor(config)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.result_dir, f"predictions_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict herb-target interactions and generate pathway format results
    print("\nStarting prediction and discovery of new associations...")
    print(f"Prediction range: all {predictor.meta_info['num_herb']} herbs")
    
    pathway_predictions = predictor.predict_herb_targets(
        herb_ids=None,  # None means predict all herbs (0-131)
        target_ids=None,  # All targets
        threshold=0.5
    )
    
    # Save pathway prediction results
    pathway_output_path = os.path.join(output_dir, "herb_pathway_predictions.csv")
    predictor.save_predictions(pathway_predictions, pathway_output_path, "pathway")
    
    # Generate discovery report
    summary_path = os.path.join(output_dir, "discovery_report.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Herb-Target New Discovery Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Prediction time: {datetime.datetime.now()}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Data Scale:\n")
        f.write(f"  Number of herbs: {predictor.meta_info['num_herb']} (all)\n")
        f.write(f"  Number of targets: {predictor.meta_info['num_target']}\n")
        f.write(f"  Number of ingredients: {predictor.meta_info['num_ingredient']}\n")
        f.write(f"  Number of diseases: {predictor.meta_info['num_disease']}\n\n")
        
        f.write("Discovery Results:\n")
        f.write(f"  Total predicted pathways: {len(pathway_predictions)}\n\n")
        
        f.write("Output Files:\n")
        f.write("  1. herb_pathway_predictions.csv - Complete prediction results\n")
        f.write("  2. herb_pathway_predictions_novel_discoveries.csv - Novel discoveries\n")
        f.write("  3. herb_pathway_predictions_known_pathways.csv - Known pathway validation\n")
        f.write("  4. discovery_report.txt - Discovery report\n\n")
        
        f.write("Discovery Type Descriptions:\n")
        f.write("  • Known Pathway: Complete pathway exists in data, for model validation\n")
        f.write("  • Predicted Pathway: Based on known ingredient-target relationships\n")
        f.write("  • Novel Ingredient: Predict herb acts through unknown ingredients\n")
        f.write("  • Novel Mechanism: Discover new mechanisms of herb action\n")
        f.write("  • Novel Association: Completely new herb-target interactions\n\n")
        
        f.write("Application Value:\n")
        f.write("  • Drug Repositioning: Discover new indications for known drugs\n")
        f.write("  • Mechanism Research: Reveal unknown mechanisms of drug action\n")
        f.write("  • Drug Development: Provide candidate targets for new drug development\n")
        
        # Add prediction quality statistics
        if pathway_predictions:
            scores = [p['prediction_score'] for p in pathway_predictions]
            f.write(f"\nPrediction Quality Statistics:\n")
            f.write(f"  Average confidence: {np.mean(scores):.4f}\n")
            f.write(f"  Highest confidence: {np.max(scores):.4f}\n")
            f.write(f"  Lowest confidence: {np.min(scores):.4f}\n")
    
    print(f"\nPrediction complete! All results saved in: {output_dir}")
    print("Output files:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")
    
    return output_dir


if __name__ == "__main__":
    # Example usage
    from config import Config
    
    config = Config()
    output_dir = run_predictions(config)
    print(f"\nPrediction results saved in: {output_dir}")
