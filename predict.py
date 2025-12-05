# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import datetime

from config import Config
from model import KGMAP_HTI
from dataset import HerbTargetDataset
from utils import sparse_mx_to_torch_sparse_tensor
from train import load_preprocessed_data, generate_node_features


class HerbTargetPredictor:
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
        
        print("Initializing KGMAP-HTI Predictor...")
        
        self.load_data()

        self.load_model(model_path)
        
        print("Predictor initialized successfully!")
    
    def load_data(self):
        print("Loading data...")
        
        self.meta_info, self.node_types, self.adjs, self.matrices = \
            load_preprocessed_data(self.config)
        
        self.node_types = self.node_types.to(self.device)
        self.adjs = [adj.to(self.device) for adj in self.adjs]
        
        self.node_feats = generate_node_features(
            self.meta_info, self.config, self.device
        )
        
        self.dataset = HerbTargetDataset(
            data_path=os.path.join(self.config.data_dir, "herb_target.dat"),
            herb_ingredient_path=os.path.join(self.config.data_dir, "herb_ingredient.dat"),
            ingredient_target_path=os.path.join(self.config.data_dir, "ingredient_target.dat"),
            target_disease_path=os.path.join(self.config.data_dir, "target_disease.dat"),
            herb_disease_path=os.path.join(self.config.data_dir, "herb_disease.dat"),
            num_herb=self.meta_info['num_herb'],
            num_ingredient=self.meta_info['num_ingredient'],
            num_target=self.meta_info['num_target'],
            num_disease=self.meta_info['num_disease'],
            num_folds=5,
            seed=self.config.seed
        )
        
        self.herb_target_paths, self.path_counts = self.dataset.get_path_statistics()
        self.complete_paths = self.dataset.get_complete_paths()
        
        print(f"  Total herbs: {self.meta_info['num_herb']}")
        print(f"  Total targets: {self.meta_info['num_target']}")
        print(f"  Total diseases: {self.meta_info['num_disease']}")
        print(f"  Known pathways: {len(self.herb_target_paths)}")
        print(f"  Complete pathways: {len(self.complete_paths)}")
    
    def load_model(self, model_path=None):
        print("Loading model...")
        
        in_dims = [self.config.embedding_dim] * 4
        self.model = KGMAP_HTI(
            in_dims=in_dims,
            n_hid=self.config.n_hid,
            n_steps=self.config.n_steps,
            dropout=0.0,
            attn_dim=self.config.attn_dim,
            use_path_enhancement=self.config.use_path_enhancement
        ).to(self.device)
        
        if model_path is None:
            model_files = [f for f in os.listdir(self.config.model_dir)
                          if f.startswith('best_model_fold') and f.endswith('.pth')]
            if len(model_files) == 0:
                print("Warning: No trained model found. Using random initialization.")
                return
            
            model_path = os.path.join(self.config.model_dir, 'best_model_fold1.pth')
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"  Model loaded from: {model_path}")
        else:
            print(f"Warning: Model file not found: {model_path}")
    
    def predict_herb_target_pairs(self, herb_target_pairs):
        self.model.eval()
        
        num_layers = len(self.model.gcn_layers)
        idxes_seq = [[0] * self.model.gcn_layers[i].n_step for i in range(num_layers)]
        idxes_res = [[0] * sum(range(self.model.gcn_layers[i].n_step)) 
                     for i in range(num_layers)]
        
        predictions = []
        
        with torch.no_grad():
            batch_size = 512
            num_batches = (len(herb_target_pairs) + batch_size - 1) // batch_size
            
            for i in tqdm(range(num_batches), desc="Predicting"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(herb_target_pairs))
                batch_pairs = herb_target_pairs[start_idx:end_idx]
                
                batch_pairs_tensor = torch.from_numpy(batch_pairs).long().to(self.device)
                
                paths_info = []
                for herb_id, target_id in batch_pairs:
                    key = (herb_id, target_id)
                    if key in self.path_counts:
                        paths_info.append([{'ingredient_id': 0, 'related_diseases': []}])
                    else:
                        paths_info.append([])
                
                outputs = self.model(
                    self.node_feats, self.node_types, self.adjs,
                    idxes_seq, idxes_res, batch_pairs_tensor, paths_info
                )
                
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_complete_paths(self, threshold=0.5):
        print(f"\nPredicting complete herb-ingredient-target-disease paths...")
        
        self.model.eval()
        
        num_layers = len(self.model.gcn_layers)
        idxes_seq = [[0] * self.model.gcn_layers[i].n_step for i in range(num_layers)]
        idxes_res = [[0] * sum(range(self.model.gcn_layers[i].n_step)) 
                     for i in range(num_layers)]
        
        with torch.no_grad():
            out = self.model(
                self.node_feats, self.node_types, self.adjs,
                idxes_seq, idxes_res
            )
        
        herb_offset = 0
        target_offset = len(self.node_feats[0])
        ingredient_offset = target_offset + len(self.node_feats[1])
        disease_offset = ingredient_offset + len(self.node_feats[2])
        
        path_predictions = []
        
        with torch.no_grad():
            for path in tqdm(self.complete_paths, desc="Predicting complete paths"):
                herb_id = path['herb_id']
                ingredient_id = path['ingredient_id']
                target_id = path['target_id']
                disease_id = path['disease_id']
                
                herb_node_id = herb_offset + (herb_id % len(self.node_feats[0]))
                target_node_id = target_offset + (target_id % len(self.node_feats[1]))
                ingredient_node_id = ingredient_offset + (ingredient_id % len(self.node_feats[2]))
                disease_node_id = disease_offset + (disease_id % len(self.node_feats[3]))
                
                if (herb_node_id < out.size(0) and target_node_id < out.size(0) and 
                    ingredient_node_id < out.size(0) and disease_node_id < out.size(0)):
                    
                    herb_emb = out[herb_node_id]
                    ingredient_emb = out[ingredient_node_id]
                    target_emb = out[target_node_id]
                    disease_emb = out[disease_node_id]
                    
                    if hasattr(self.model, 'complete_path_encoder'):
                        path_emb = self.model.complete_path_encoder(
                            herb_emb, ingredient_emb, target_emb, disease_emb
                        )
                        score = self.model.predictor(path_emb).squeeze().item()
                    else:
                        combined_emb = (herb_emb + ingredient_emb + target_emb + disease_emb) / 4
                        score = self.model.predictor(combined_emb.unsqueeze(0)).squeeze().item()
                    
                    path_predictions.append({
                        'herb_id': herb_id,
                        'ingredient_id': ingredient_id,
                        'target_id': target_id,
                        'disease_id': disease_id,
                        'prediction_score': score,
                        'predicted_interaction': score > threshold
                    })
        
        results = pd.DataFrame(path_predictions)
        
        if len(results) > 0:
            results = results.sort_values('prediction_score', ascending=False)
        
        return results
    
    def predict_all_pairs(self, threshold=0.5):
        print(f"\nPredicting all herb-target pairs...")
        print(f"  Total predictions: {self.meta_info['num_herb'] * self.meta_info['num_target']:,}")
        
        all_pairs = []
        for herb_id in range(self.meta_info['num_herb']):
            for target_id in range(self.meta_info['num_target']):
                all_pairs.append([herb_id, target_id])
        
        all_pairs = np.array(all_pairs)
        
        predictions = self.predict_herb_target_pairs(all_pairs)
        
        results = pd.DataFrame({
            'herb_id': all_pairs[:, 0],
            'target_id': all_pairs[:, 1],
            'prediction_score': predictions,
            'predicted_interaction': predictions > threshold
        })
        
        results['has_known_pathway'] = results.apply(
            lambda row: (row['herb_id'], row['target_id']) in self.path_counts,
            axis=1
        )
        
        results['path_count'] = results.apply(
            lambda row: self.path_counts.get((row['herb_id'], row['target_id']), 0),
            axis=1
        )
        
        results['final_score'] = results['prediction_score'] * (
            1 + np.minimum(results['path_count'] / 50, 0.5)
        )
        
        results = results.sort_values('final_score', ascending=False)
        
        return results
    
    def predict_for_herb(self, herb_id, top_k=10):
        print(f"\nPredicting targets for herb {herb_id}...")
        
        pairs = np.array([[herb_id, target_id]
                         for target_id in range(self.meta_info['num_target'])])

        predictions = self.predict_herb_target_pairs(pairs)
        
        results = pd.DataFrame({
            'herb_id': herb_id,
            'target_id': pairs[:, 1],
            'prediction_score': predictions
        })
        
        results['has_known_pathway'] = results.apply(
            lambda row: (row['herb_id'], row['target_id']) in self.path_counts,
            axis=1
        )
        
        results = results.sort_values('prediction_score', ascending=False)
        
        return results.head(top_k)
    
    def save_predictions(self, predictions, output_path):
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        
        print(f"\nPrediction Statistics:")
        print(f"  Total predictions: {len(predictions)}")
        
        if 'prediction_score' in predictions.columns:
            print(f"  Predicted interactions (score > 0.5): "
                  f"{(predictions['prediction_score'] > 0.5).sum()}")
        
        if 'has_known_pathway' in predictions.columns:
            print(f"  Known pathways: {predictions['has_known_pathway'].sum()}")
            print(f"  Novel predictions: "
                  f"{(~predictions['has_known_pathway'] & (predictions['prediction_score'] > 0.5)).sum()}")


def run_prediction(config):
    print("="*70)
    print("KGMAP-HTI Prediction Pipeline")
    print("="*70)
    
    predictor = HerbTargetPredictor(config)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.result_dir, f"predictions_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Predicting all herb-target interactions...")
    print("="*70)
    
    all_predictions = predictor.predict_all_pairs(threshold=0.5)
    
    output_path = os.path.join(output_dir, "all_predictions.csv")
    predictor.save_predictions(all_predictions, output_path)
    
    novel_predictions = all_predictions[
        (~all_predictions['has_known_pathway']) & 
        (all_predictions['prediction_score'] > 0.7)
    ]
    
    novel_output_path = os.path.join(output_dir, "novel_predictions.csv")
    predictor.save_predictions(novel_predictions, novel_output_path)
    
    print("\n" + "="*70)
    print("Predicting complete herb-ingredient-target-disease paths...")
    print("="*70)
    
    complete_path_predictions = predictor.predict_complete_paths(threshold=0.5)
    
    if len(complete_path_predictions) > 0:
        complete_path_output = os.path.join(output_dir, "complete_path_predictions.csv")
        predictor.save_predictions(complete_path_predictions, complete_path_output)
        
        high_confidence_paths = complete_path_predictions[
            complete_path_predictions['prediction_score'] > 0.7
        ]
        
        if len(high_confidence_paths) > 0:
            high_conf_output = os.path.join(output_dir, "high_confidence_complete_paths.csv")
            predictor.save_predictions(high_confidence_paths, high_conf_output)
    
    print(f"\nAll results saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    config = Config()
    output_dir = run_prediction(config)
