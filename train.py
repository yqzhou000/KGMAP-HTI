# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    precision_score, recall_score, f1_score, mean_absolute_error
)
import time
import datetime
from tqdm import tqdm

from config import Config
from model import KGMAP_HTI
from dataset import HerbTargetDataset
from utils import sparse_mx_to_torch_sparse_tensor, get_metrics, set_random_seed
from node2vec import node2vec_embedding


class TransE(nn.Module):

    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(TransE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        self.relation_embeddings.weight.data = F.normalize(
            self.relation_embeddings.weight.data, p=2, dim=1
        )
    
    def forward(self, heads, relations, tails):
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        score = torch.norm(h + r - t, p=2, dim=1)
        
        return score
    
    def loss(self, pos_heads, pos_relations, pos_tails, 
             neg_heads, neg_relations, neg_tails):
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        
        neg_score = self.forward(neg_heads, neg_relations, neg_tails)
        
        loss = torch.relu(self.margin + pos_score - neg_score).mean()
        
        return loss


def train_transe(triples, num_entities, num_relations, config, device):
    print("Training TransE for node initialization...")
    
    transe = TransE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=config.embedding_dim,
        margin=config.transe_margin
    ).to(device)
    
    optimizer = torch.optim.Adam(transe.parameters(), lr=config.transe_lr)
    
    triples = np.array(triples)
    num_triples = len(triples)
    
    for epoch in range(config.transe_epochs):
        indices = np.random.permutation(num_triples)
        triples_shuffled = triples[indices]
        
        total_loss = 0
        batch_size = 512
        num_batches = (num_triples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_triples)
            batch_triples = triples_shuffled[start_idx:end_idx]
            
            pos_heads = torch.LongTensor(batch_triples[:, 0]).to(device)
            pos_relations = torch.LongTensor(batch_triples[:, 1]).to(device)
            pos_tails = torch.LongTensor(batch_triples[:, 2]).to(device)
            
            neg_triples = []
            for head, rel, tail in batch_triples:
                if np.random.rand() < 0.5:
                    neg_head = np.random.randint(0, num_entities)
                    neg_triples.append([neg_head, rel, tail])
                else:
                    neg_tail = np.random.randint(0, num_entities)
                    neg_triples.append([head, rel, neg_tail])
            
            neg_triples = np.array(neg_triples)
            neg_heads = torch.LongTensor(neg_triples[:, 0]).to(device)
            neg_relations = torch.LongTensor(neg_triples[:, 1]).to(device)
            neg_tails = torch.LongTensor(neg_triples[:, 2]).to(device)
            
            loss = transe.loss(pos_heads, pos_relations, pos_tails,
                             neg_heads, neg_relations, neg_tails)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                transe.entity_embeddings.weight.data = F.normalize(
                    transe.entity_embeddings.weight.data, p=2, dim=1
                )
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch+1}/{config.transe_epochs}, Loss: {avg_loss:.4f}")
    
    entity_embeddings = transe.entity_embeddings.weight.data.clone()
    
    print("TransE training completed!")
    
    return entity_embeddings


def build_triples_from_kg(meta_info, adjs):

    print("Building triples from knowledge graph...")
    print(f"Total adjacency matrices: {len(adjs)}")

    triples = []

    relation_names = [
        'herb-target',
        'herb-disease',
        'herb-ingredient',
        'ingredient-target',
        'target-disease',
        'herb-herb',
        'target-target'
    ]

    for rel_id in range(len(adjs)):
        adj = adjs[rel_id].coalesce()
        indices = adj.indices().cpu().numpy()
        values = adj.values().cpu().numpy()

        for i in range(indices.shape[1]):
            head = indices[0, i]
            tail = indices[1, i]
            triples.append([head, rel_id, tail])

        print(f"  Relation {rel_id} ({relation_names[rel_id]}): {indices.shape[1]} edges")

    num_relations = 7

    print(f"Built {len(triples)} triples with {num_relations} relation types")

    return triples, num_relations


def generate_node_features_with_transe(meta_info, config, device, adjs):
    print("Generating node features with TransE...")
    
    if config.use_transe_init:
        triples, num_relations = build_triples_from_kg(meta_info, adjs)
        
        total_nodes = meta_info['total_nodes']
        
        entity_embeddings = train_transe(
            triples, total_nodes, num_relations, config, device
        )
        
        herb_feats = entity_embeddings[:meta_info['num_herb']].cpu().numpy()
        target_feats = entity_embeddings[
            meta_info['offsets']['target']:meta_info['offsets']['ingredient']
        ].cpu().numpy()
        ingredient_feats = entity_embeddings[
            meta_info['offsets']['ingredient']:meta_info['offsets']['disease']
        ].cpu().numpy()
        disease_feats = entity_embeddings[
            meta_info['offsets']['disease']:
        ].cpu().numpy()
    else:
        herb_feats = np.random.randn(meta_info['num_herb'], config.embedding_dim).astype(np.float32)
        target_feats = np.random.randn(meta_info['num_target'], config.embedding_dim).astype(np.float32)
        ingredient_feats = np.random.randn(meta_info['num_ingredient'], config.embedding_dim).astype(np.float32)
        disease_feats = np.random.randn(meta_info['num_disease'], config.embedding_dim).astype(np.float32)
    
    herb_feats = torch.FloatTensor(herb_feats).to(device)
    target_feats = torch.FloatTensor(target_feats).to(device)
    ingredient_feats = torch.FloatTensor(ingredient_feats).to(device)
    disease_feats = torch.FloatTensor(disease_feats).to(device)
    
    node_feats = [herb_feats, target_feats, ingredient_feats, disease_feats]
    
    print(f"Node features generated:")
    print(f"  Herbs: {herb_feats.shape}")
    print(f"  Targets: {target_feats.shape}")
    print(f"  Ingredients: {ingredient_feats.shape}")
    print(f"  Diseases: {disease_feats.shape}")
    
    return node_feats


def load_preprocessed_data(config):
    print("Loading preprocessed data...")
    
    with open(os.path.join(config.preprocessed_dir, "meta_info.pkl"), "rb") as f:
        meta_info = pickle.load(f)
    
    node_types = np.load(os.path.join(config.preprocessed_dir, "node_types.npy"))
    node_types = torch.LongTensor(node_types)
    
    with open(os.path.join(config.preprocessed_dir, "adjs_offset.pkl"), "rb") as f:
        adjs_dict = pickle.load(f)
    
    adjs = []
    for i in range(7):
        adj_sp = adjs_dict[str(i)]
        adj_torch = sparse_mx_to_torch_sparse_tensor(adj_sp)
        adjs.append(adj_torch)
    
    matrices = np.load(os.path.join(config.preprocessed_dir, 'combined_matrices.npz'))
    
    print(f"Loaded data:")
    print(f"  Total nodes: {meta_info['total_nodes']}")
    print(f"  Herbs: {meta_info['num_herb']}")
    print(f"  Targets: {meta_info['num_target']}")
    print(f"  Ingredients: {meta_info['num_ingredient']}")
    print(f"  Diseases: {meta_info['num_disease']}")
    print(f"  Adjacency matrices: {len(adjs)}")
    
    return meta_info, node_types, adjs, matrices


def generate_node_features(meta_info, config, device):
    print("Generating node features...")
    
    herb_feats = np.random.randn(meta_info['num_herb'], config.embedding_dim).astype(np.float32)
    target_feats = np.random.randn(meta_info['num_target'], config.embedding_dim).astype(np.float32)
    ingredient_feats = np.random.randn(meta_info['num_ingredient'], config.embedding_dim).astype(np.float32)
    disease_feats = np.random.randn(meta_info['num_disease'], config.embedding_dim).astype(np.float32)
    
    herb_feats = torch.FloatTensor(herb_feats).to(device)
    target_feats = torch.FloatTensor(target_feats).to(device)
    ingredient_feats = torch.FloatTensor(ingredient_feats).to(device)
    disease_feats = torch.FloatTensor(disease_feats).to(device)
    
    return [herb_feats, target_feats, ingredient_feats, disease_feats]


def generate_adaptive_search_indices(model, adjs, n_steps):
    num_layers = len(model.gcn_layers)
    num_relations = len(adjs)
    
    idxes_seq = []
    idxes_res = []
    
    for layer_idx in range(num_layers):
        n_step = n_steps[layer_idx]
        
        seq_indices = [np.random.randint(0, num_relations) for _ in range(n_step)]
        idxes_seq.append(seq_indices)
        
        res_count = sum(range(n_step))
        res_indices = [np.random.randint(0, num_relations + 1) for _ in range(res_count)]
        idxes_res.append(res_indices)
    
    return idxes_seq, idxes_res


def get_complete_path_info(herb_id, target_id, dataset):
    path_info_list = []
    
    if herb_id in dataset.herb_ingredients:
        for ingredient_id in dataset.herb_ingredients[herb_id]:
            if ingredient_id in dataset.ingredient_targets:
                if target_id in dataset.ingredient_targets[ingredient_id]:
                    related_diseases = []
                    if target_id in dataset.target_diseases:
                        related_diseases = list(dataset.target_diseases[target_id])
                    
                    path_info = {
                        'ingredient_id': ingredient_id,
                        'related_diseases': related_diseases
                    }
                    path_info_list.append(path_info)
    
    return path_info_list


def create_data_loaders(dataset, fold, config):
    train_pos, train_neg, test_pos, test_neg = dataset.get_fold_data(fold)
    
    train_pairs = np.vstack([train_pos, train_neg])
    train_labels = np.hstack([
        np.ones(len(train_pos)),
        np.zeros(len(train_neg))
    ])
    
    test_pairs = np.vstack([test_pos, test_neg])
    test_labels = np.hstack([
        np.ones(len(test_pos)),
        np.zeros(len(test_neg))
    ])
    
    shuffle_idx = np.random.permutation(len(train_pairs))
    train_pairs = train_pairs[shuffle_idx]
    train_labels = train_labels[shuffle_idx]
    
    train_dataset = TensorDataset(
        torch.LongTensor(train_pairs),
        torch.FloatTensor(train_labels)
    )
    
    test_dataset = TensorDataset(
        torch.LongTensor(test_pairs),
        torch.FloatTensor(test_labels)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader


def train_epoch(model, node_feats, node_types, adjs, train_loader,
               optimizer, device, dataset, idxes_seq, idxes_res):
    model.train()
    
    total_loss = 0
    all_labels = []
    all_preds = []
    all_scores = []
    
    for batch_pairs, batch_labels in train_loader:
        batch_pairs = batch_pairs.to(device)
        batch_labels = batch_labels.to(device)
        
        paths_info = []
        for herb_id, target_id in batch_pairs:
            herb_id = herb_id.item()
            target_id = target_id.item()
            path_info = get_complete_path_info(herb_id, target_id, dataset)
            paths_info.append(path_info)
        
        outputs = model(node_feats, node_types, adjs, idxes_seq, idxes_res,
                       batch_pairs, paths_info)
        
        loss = F.binary_cross_entropy(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        all_labels.extend(batch_labels.cpu().numpy())
        all_scores.extend(outputs.detach().cpu().numpy())
        all_preds.extend((outputs > 0.5).float().cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    metrics = get_metrics(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_scores)
    )
    
    return avg_loss, metrics


def evaluate(model, node_feats, node_types, adjs, test_loader, 
            device, dataset, idxes_seq, idxes_res):
    model.eval()
    
    total_loss = 0
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for batch_pairs, batch_labels in test_loader:
            batch_pairs = batch_pairs.to(device)
            batch_labels = batch_labels.to(device)
            
            paths_info = []
            for herb_id, target_id in batch_pairs:
                herb_id = herb_id.item()
                target_id = target_id.item()
                path_info = get_complete_path_info(herb_id, target_id, dataset)
                paths_info.append(path_info)
            
            outputs = model(node_feats, node_types, adjs, idxes_seq, idxes_res,
                           batch_pairs, paths_info)
            
            loss = F.binary_cross_entropy(outputs, batch_labels)
            total_loss += loss.item()
            
            all_labels.extend(batch_labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    metrics = get_metrics(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_scores)
    )
    
    return avg_loss, metrics


def train_kgmap_hti(config):
    print("="*70)
    print("KGMAP-HTI Training (Improved Version)")
    print("="*70)
    
    set_random_seed(config.seed)
    
    device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    meta_info, node_types, adjs, matrices = load_preprocessed_data(config)
    node_types = node_types.to(device)
    adjs = [adj.to(device) for adj in adjs]
    
    print("\n" + "="*70)
    print("Step 1: Initialize node features with TransE")
    print("="*70)
    node_feats = generate_node_features_with_transe(meta_info, config, device, adjs)
    
    dataset = HerbTargetDataset(
        data_path=os.path.join(config.data_dir, "herb_target.dat"),
        herb_ingredient_path=os.path.join(config.data_dir, "herb_ingredient.dat"),
        ingredient_target_path=os.path.join(config.data_dir, "ingredient_target.dat"),
        target_disease_path=os.path.join(config.data_dir, "target_disease.dat"),
        herb_disease_path=os.path.join(config.data_dir, "herb_disease.dat"),
        num_herb=meta_info['num_herb'],
        num_ingredient=meta_info['num_ingredient'],
        num_target=meta_info['num_target'],
        num_disease=meta_info['num_disease'],
        num_folds=config.num_folds,
        seed=config.seed
    )
    
    fold_results = []
    
    for fold in range(config.num_folds):
        print(f"\n{'='*70}")
        print(f"Fold {fold+1}/{config.num_folds}")
        print(f"{'='*70}")
        
        train_loader, test_loader = create_data_loaders(dataset, fold, config)
        
        in_dims = [config.embedding_dim] * 4
        model = KGMAP_HTI(
            in_dims=in_dims,
            n_hid=config.n_hid,
            n_steps=config.n_steps,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            use_path_enhancement=config.use_path_enhancement
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\nStep 2: Generate adaptive search indices")
        idxes_seq, idxes_res = generate_adaptive_search_indices(
            model, adjs, config.n_steps
        )
        print(f"  Sequential indices: {idxes_seq}")
        print(f"  Residual indices (first layer): {idxes_res[0][:5]}...")
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        best_auc = 0
        best_metrics = None
        patience_counter = 0
        
        print("\nStep 3: Training with complete path information")
        for epoch in range(config.max_epochs):
            epoch_start = time.time()
            
            train_loss, train_metrics = train_epoch(
                model, node_feats, node_types, adjs, train_loader,
                optimizer, device, dataset, idxes_seq, idxes_res
            )
            
            test_loss, test_metrics = evaluate(
                model, node_feats, node_types, adjs, test_loader,
                device, dataset, idxes_seq, idxes_res
            )
            
            epoch_time = time.time() - epoch_start
            
            if epoch % 10 == 0 or epoch == config.max_epochs - 1:
                print(f"[Epoch {epoch:3d}] ({epoch_time:.1f}s)")
                print(f"  Train - Loss: {train_loss:.4f}, AUC: {train_metrics['auc']:.4f}, "
                      f"AUPR: {train_metrics['aupr']:.4f}, F1: {train_metrics['f1']:.4f}")
                print(f"  Test  - Loss: {test_loss:.4f}, AUC: {test_metrics['auc']:.4f}, "
                      f"AUPR: {test_metrics['aupr']:.4f}, F1: {test_metrics['f1']:.4f}")
            
            if test_metrics['auc'] > best_auc:
                best_auc = test_metrics['auc']
                best_metrics = test_metrics.copy()
                patience_counter = 0
                
                torch.save(model.state_dict(), 
                          os.path.join(config.model_dir, f'best_model_fold{fold+1}.pth'))
            else:
                patience_counter += 1
            
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print(f"\nFold {fold+1} Best Results:")
        print(f"  AUC: {best_metrics['auc']:.4f}")
        print(f"  AUPR: {best_metrics['aupr']:.4f}")
        print(f"  ACC: {best_metrics['acc']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        print(f"  F1: {best_metrics['f1']:.4f}")
        print(f"  MAE: {best_metrics['mae']:.4f}")
        print(f"  ME: {best_metrics['me']:.4f}")
        
        fold_results.append(best_metrics)
    
    print(f"\n{'='*70}")
    print("5-Fold Cross-Validation Results (Improved Version)")
    print(f"{'='*70}")
    
    metrics_names = ['auc', 'aupr', 'acc', 'precision', 'recall', 'f1', 'mae', 'me']
    
    for metric in metrics_names:
        values = [r[metric] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config.result_dir, f'training_results_improved_{timestamp}.pkl')
    
    with open(results_file, 'wb') as f:
        pickle.dump({
            'fold_results': fold_results,
            'config': config,
            'improvements': [
                'TransE initialization',
                'Dynamic search strategy',
                'Complete path information'
            ]
        }, f)
    
    print(f"\nResults saved to: {results_file}")
    print("\nImprovements applied:")
    print("  ✓ TransE initialization for node embeddings")
    print("  ✓ Dynamic generation of adaptive search indices")
    print("  ✓ Complete path information passing")
    
    return fold_results


if __name__ == "__main__":
    config = Config()
    results = train_kgmap_hti(config)
