import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import scipy.sparse as sp

from dataset import HerbTargetDataset
from model import KGMAPHTIModel
from utils import sparse_mx_to_torch_sparse_tensor, normalize_row, normalize_sym


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_graph_tensors(preprocessed_dir, device, use_semantic_edges=True):
    with open(os.path.join(preprocessed_dir, "meta_info.pkl"), "rb") as f:
        meta_info = pickle.load(f)

    node_types = np.load(os.path.join(preprocessed_dir, "node_types.npy"))
    node_types = torch.from_numpy(node_types).to(device)

    with open(os.path.join(preprocessed_dir, "adjs_offset.pkl"), "rb") as f:
        adjs_offset = pickle.load(f)

    adjs_pt = []
    for i in range(5):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_row(adjs_offset[str(i)] +
                          sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(device))
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_row(adjs_offset[str(i)].T +
                          sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(device))

    for i in range(5, 9):
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_sym(adjs_offset[str(i)] +
                          sp.eye(adjs_offset[str(i)].shape[0], dtype=np.float32))).to(device))

    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
        sp.eye(adjs_offset['0'].shape[0], dtype=np.float32).tocoo()).to(device))
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset['0'].shape).to(device))

    if not use_semantic_edges:
        merged = None
        for adj in adjs_pt[:-2]:
            merged = adj if merged is None else merged + adj
        if merged is not None:
            adjs_pt = [merged for _ in adjs_pt[:-2]] + adjs_pt[-2:]

    return meta_info, node_types, adjs_pt


def _build_relation_triples(data_dir, meta_info):
    offsets = meta_info['offsets']
    triples = []

    rel_map = {
        'ht': 0, 'th': 1,
        'hd': 2, 'dh': 3,
        'hi': 4, 'ih': 5,
        'it': 6, 'ti': 7,
        'td': 8, 'dt': 9,
        'hh': 10, 'sim_hh': 11,
        'tt': 12, 'sim_tt': 13,
    }

    ht = pd.read_csv(os.path.join(data_dir, "herb_target.dat"),
                     delimiter=',', names=['hid', 'tid', 'rating'])
    ht = ht[ht['rating'] == 1][['hid', 'tid']].to_numpy()
    for h_id, t_id in ht:
        h = int(h_id) + offsets['herb']
        t = int(t_id) + offsets['target']
        triples.append([h, rel_map['ht'], t])
        triples.append([t, rel_map['th'], h])

    hd = pd.read_csv(os.path.join(data_dir, "herb_disease.dat"),
                     delimiter=',', names=['hid', 'did', 'weight']).to_numpy()
    for h_id, d_id, _ in hd:
        h = int(h_id) + offsets['herb']
        d = int(d_id) + offsets['disease']
        triples.append([h, rel_map['hd'], d])
        triples.append([d, rel_map['dh'], h])

    hi = pd.read_csv(os.path.join(data_dir, "herb_ingredient.dat"),
                     delimiter=',', names=['hid', 'iid', 'weight']).to_numpy()
    for h_id, i_id, _ in hi:
        h = int(h_id) + offsets['herb']
        i = int(i_id) + offsets['ingredient']
        triples.append([h, rel_map['hi'], i])
        triples.append([i, rel_map['ih'], h])

    it = pd.read_csv(os.path.join(data_dir, "ingredient_target.dat"),
                     delimiter=',', names=['iid', 'tid', 'weight']).to_numpy()
    for i_id, t_id, _ in it:
        i = int(i_id) + offsets['ingredient']
        t = int(t_id) + offsets['target']
        triples.append([i, rel_map['it'], t])
        triples.append([t, rel_map['ti'], i])

    td = pd.read_csv(os.path.join(data_dir, "target_disease.dat"),
                     delimiter=',', names=['tid', 'did', 'weight']).to_numpy()
    for t_id, d_id, _ in td:
        t = int(t_id) + offsets['target']
        d = int(d_id) + offsets['disease']
        triples.append([t, rel_map['td'], d])
        triples.append([d, rel_map['dt'], t])

    hh = pd.read_csv(os.path.join(data_dir, "herb_herb.dat"),
                     delimiter=',', names=['h1', 'h2', 'weight']).to_numpy()
    for h1_id, h2_id, _ in hh:
        h1 = int(h1_id) + offsets['herb']
        h2 = int(h2_id) + offsets['herb']
        triples.append([h1, rel_map['hh'], h2])
        triples.append([h2, rel_map['hh'], h1])

    simhh = pd.read_csv(os.path.join(data_dir, "sim_herbs.dat"),
                        delimiter=',', names=['h1', 'h2', 'weight']).to_numpy()
    for h1_id, h2_id, _ in simhh:
        h1 = int(h1_id) + offsets['herb']
        h2 = int(h2_id) + offsets['herb']
        triples.append([h1, rel_map['sim_hh'], h2])
        triples.append([h2, rel_map['sim_hh'], h1])

    tt = pd.read_csv(os.path.join(data_dir, "target_target.dat"),
                     delimiter=',', names=['t1', 't2', 'weight']).to_numpy()
    for t1_id, t2_id, _ in tt:
        t1 = int(t1_id) + offsets['target']
        t2 = int(t2_id) + offsets['target']
        triples.append([t1, rel_map['tt'], t2])
        triples.append([t2, rel_map['tt'], t1])

    simtt = pd.read_csv(os.path.join(data_dir, "sim_targets.dat"),
                        delimiter=',', names=['t1', 't2', 'weight']).to_numpy()
    for t1_id, t2_id, _ in simtt:
        t1 = int(t1_id) + offsets['target']
        t2 = int(t2_id) + offsets['target']
        triples.append([t1, rel_map['sim_tt'], t2])
        triples.append([t2, rel_map['sim_tt'], t1])

    triples = np.array(triples, dtype=np.int64)
    return torch.from_numpy(triples), rel_map


def _build_path_maps(dataset):
    herb_target_paths, path_counts = dataset.get_path_statistics()
    pair_to_paths = {}
    pair_to_ingredients = {}
    ingredient_to_herbs = {}
    ingredient_to_targets = {}

    for path_info in herb_target_paths:
        h = path_info['herb_id']
        t = path_info['target_id']
        ing = path_info['ingredient_id']

        pair_to_paths.setdefault((h, t), []).append(path_info)
        pair_to_ingredients.setdefault((h, t), []).append(ing)
        ingredient_to_herbs.setdefault(ing, set()).add(h)
        ingredient_to_targets.setdefault(ing, set()).add(t)

    return herb_target_paths, path_counts, pair_to_paths, pair_to_ingredients, ingredient_to_herbs, ingredient_to_targets


def _semantic_negative_sampling(pos_pairs, pos_set, pair_to_ingredients,
                                ingredient_to_herbs, ingredient_to_targets,
                                num_herb, num_target, max_attempts=20):
    neg_samples = []
    for h, t in pos_pairs:
        sampled = False
        ingredients = pair_to_ingredients.get((h, t), [])
        for _ in range(max_attempts):
            if ingredients:
                ing = np.random.choice(ingredients)
                if np.random.rand() < 0.5 and len(ingredient_to_herbs.get(ing, [])) > 1:
                    h_candidates = list(ingredient_to_herbs[ing] - {h})
                    if h_candidates:
                        h_new = int(np.random.choice(h_candidates))
                        candidate = (h_new, t)
                    else:
                        candidate = None
                else:
                    t_candidates = list(ingredient_to_targets.get(ing, set()) - {t})
                    if t_candidates:
                        t_new = int(np.random.choice(t_candidates))
                        candidate = (h, t_new)
                    else:
                        candidate = None
            else:
                candidate = None

            if candidate and candidate not in pos_set:
                neg_samples.append(candidate)
                sampled = True
                break

        if not sampled:
            # fallback random
            while True:
                h_new = np.random.randint(0, num_herb)
                t_new = np.random.randint(0, num_target)
                if (h_new, t_new) not in pos_set:
                    neg_samples.append((h_new, t_new))
                    break

    return np.array(neg_samples, dtype=np.int64)


def _random_negative_sampling(pos_pairs, pos_set, num_herb, num_target):
    neg_samples = []
    for _ in pos_pairs:
        while True:
            h_new = np.random.randint(0, num_herb)
            t_new = np.random.randint(0, num_target)
            if (h_new, t_new) not in pos_set:
                neg_samples.append((h_new, t_new))
                break
    return np.array(neg_samples, dtype=np.int64)


def _sample_kge_negatives(pos_triples, rel_schemas, meta_info, pair_to_ingredients,
                          ingredient_to_herbs, ingredient_to_targets, pos_ht_set):
    offsets = meta_info['offsets']
    num_herb = meta_info['num_herb']
    num_target = meta_info['num_target']
    num_ingredient = meta_info['num_ingredient']
    num_disease = meta_info['num_disease']

    def _rand_entity(entity_type):
        if entity_type == 'herb':
            return np.random.randint(0, num_herb) + offsets['herb']
        if entity_type == 'target':
            return np.random.randint(0, num_target) + offsets['target']
        if entity_type == 'ingredient':
            return np.random.randint(0, num_ingredient) + offsets['ingredient']
        if entity_type == 'disease':
            return np.random.randint(0, num_disease) + offsets['disease']
        return np.random.randint(0, num_herb) + offsets['herb']

    neg_triples = []
    for h, r, t in pos_triples:
        r = int(r)
        schema = rel_schemas.get(r, ('herb', 'target'))

        if r in (0, 1):
            # semantic path constrained negative for herb-target
            if r == 0:
                h_raw = int(h - offsets['herb'])
                t_raw = int(t - offsets['target'])
            else:
                h_raw = int(t - offsets['herb'])
                t_raw = int(h - offsets['target'])

            neg_pair = _semantic_negative_sampling(
                np.array([[h_raw, t_raw]]), pos_ht_set, pair_to_ingredients,
                ingredient_to_herbs, ingredient_to_targets, num_herb, num_target
            )[0]

            if r == 0:
                h_new = neg_pair[0] + offsets['herb']
                t_new = neg_pair[1] + offsets['target']
            else:
                h_new = neg_pair[1] + offsets['target']
                t_new = neg_pair[0] + offsets['herb']

            neg_triples.append([h_new, r, t_new])
            continue

        if np.random.rand() < 0.5:
            h_new = _rand_entity(schema[0])
            neg_triples.append([h_new, r, t])
        else:
            t_new = _rand_entity(schema[1])
            neg_triples.append([h, r, t_new])

    return torch.tensor(neg_triples, dtype=torch.long)


def _compute_metrics(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    return {
        'auc': roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5,
        'aupr': average_precision_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5,
        'acc': accuracy_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mae': mean_absolute_error(y_true, y_score),
        'me': float(np.mean(y_score - y_true))
    }


def train_herb_target_model_with_paths(config):
    print("开始按照论文描述训练KGMAP-HTI模型...")
    _set_seed(config.seed)

    torch.cuda.set_device(config.gpu)
    device = torch.device(f"cuda:{config.gpu}")

    meta_info, node_types, adjs_pt = _load_graph_tensors(
        config.preprocessed_dir,
        device,
        use_semantic_edges=getattr(config, "use_semantic_edges", True)
    )
    offsets = meta_info['offsets']

    dataset = HerbTargetDataset(
        data_path=os.path.join(config.data_dir, "herb_target.dat"),
        herb_ingredient_path=os.path.join(config.data_dir, "herb_ingredient.dat"),
        ingredient_target_path=os.path.join(config.data_dir, "ingredient_target.dat"),
        target_disease_path=os.path.join(config.data_dir, "target_disease.dat"),
        num_herb=meta_info['num_herb'],
        num_ingredient=meta_info['num_ingredient'],
        num_target=meta_info['num_target'],
        num_disease=meta_info['num_disease'],
        num_folds=5,
        seed=config.seed
    )

    herb_target_paths, path_counts, pair_to_paths, pair_to_ingredients, ingredient_to_herbs, ingredient_to_targets = _build_path_maps(dataset)

    pos_df = pd.read_csv(os.path.join(config.data_dir, "herb_target.dat"),
                         delimiter=',', names=['hid', 'tid', 'rating'])
    pos_pairs_all = pos_df[pos_df['rating'] == 1][['hid', 'tid']].to_numpy()
    pos_set_all = set((int(h), int(t)) for h, t in pos_pairs_all)

    triples, rel_map = _build_relation_triples(config.data_dir, meta_info)
    triples = triples.to(device)

    rel_schemas = {
        rel_map['ht']: ('herb', 'target'),
        rel_map['th']: ('target', 'herb'),
        rel_map['hd']: ('herb', 'disease'),
        rel_map['dh']: ('disease', 'herb'),
        rel_map['hi']: ('herb', 'ingredient'),
        rel_map['ih']: ('ingredient', 'herb'),
        rel_map['it']: ('ingredient', 'target'),
        rel_map['ti']: ('target', 'ingredient'),
        rel_map['td']: ('target', 'disease'),
        rel_map['dt']: ('disease', 'target'),
        rel_map['hh']: ('herb', 'herb'),
        rel_map['sim_hh']: ('herb', 'herb'),
        rel_map['tt']: ('target', 'target'),
        rel_map['sim_tt']: ('target', 'target'),
    }

    num_entities = meta_info['total_nodes']
    num_relations = len(adjs_pt)

    kge_dim = getattr(config, 'kge_dim', 128)
    num_layers = getattr(config, 'gcn_layers', 3)
    kge_margin = getattr(config, 'kge_margin', 1.0)
    kge_weight = getattr(config, 'kge_weight', 0.1)
    if not getattr(config, "use_transe_pretrain", True):
        kge_weight = 0.0
    kge_batch_size = getattr(config, 'kge_batch_size', 4096)
    batch_size = getattr(config, 'batch_size', 2048)

    model = KGMAPHTIModel(
        num_entities=num_entities,
        num_relations=num_relations,
        n_hid=config.n_hid,
        num_layers=num_layers,
        kge_dim=kge_dim,
        dropout=config.dropout,
        use_adaptive_search=getattr(config, "use_adaptive_search", True),
        use_semantic_edges=getattr(config, "use_semantic_edges", True),
        use_residual=getattr(config, "use_residual", True),
        use_relation_attention=getattr(config, "use_relation_attention", True),
        use_metapath_attention=getattr(config, "use_metapath_attention", True),
        use_transe_pretrain=getattr(config, "use_transe_pretrain", True)
    ).to(device)

    if not getattr(config, "use_transe_pretrain", True):
        model.reset_transe_embeddings()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)

    pos_train_fold, pos_val_fold, pos_test_fold, _, _, _, _, _, _ = dataset.generate_folds()

    all_results = []
    best_global_auc = 0.0
    best_global_state = None
    for fold in range(5):
        print(f"\n>>> Fold {fold + 1}/5")
        model.train()

        pos_train = pos_train_fold[fold]
        pos_val = pos_val_fold[fold]
        pos_test = pos_test_fold[fold]

        pos_train_raw = pos_train - np.array([0, meta_info['num_herb']])
        pos_val_raw = pos_val - np.array([0, meta_info['num_herb']])
        pos_test_raw = pos_test - np.array([0, meta_info['num_herb']])

        if getattr(config, "use_semantic_neg_sampling", True):
            neg_train_raw = _semantic_negative_sampling(
                pos_train_raw, pos_set_all, pair_to_ingredients,
                ingredient_to_herbs, ingredient_to_targets,
                meta_info['num_herb'], meta_info['num_target']
            )
            neg_val_raw = _semantic_negative_sampling(
                pos_val_raw, pos_set_all, pair_to_ingredients,
                ingredient_to_herbs, ingredient_to_targets,
                meta_info['num_herb'], meta_info['num_target']
            )
            neg_test_raw = _semantic_negative_sampling(
                pos_test_raw, pos_set_all, pair_to_ingredients,
                ingredient_to_herbs, ingredient_to_targets,
                meta_info['num_herb'], meta_info['num_target']
            )
        else:
            neg_train_raw = _random_negative_sampling(
                pos_train_raw, pos_set_all, meta_info['num_herb'], meta_info['num_target']
            )
            neg_val_raw = _random_negative_sampling(
                pos_val_raw, pos_set_all, meta_info['num_herb'], meta_info['num_target']
            )
            neg_test_raw = _random_negative_sampling(
                pos_test_raw, pos_set_all, meta_info['num_herb'], meta_info['num_target']
            )

        neg_train = neg_train_raw + np.array([0, meta_info['num_herb']])
        neg_val = neg_val_raw + np.array([0, meta_info['num_herb']])
        neg_test = neg_test_raw + np.array([0, meta_info['num_herb']])

        train_samples = np.vstack([pos_train, neg_train])
        train_labels = np.hstack([np.ones(len(pos_train)), np.zeros(len(neg_train))])
        val_samples = np.vstack([pos_val, neg_val])
        val_labels = np.hstack([np.ones(len(pos_val)), np.zeros(len(neg_val))])
        test_samples = np.vstack([pos_test, neg_test])
        test_labels = np.hstack([np.ones(len(pos_test)), np.zeros(len(neg_test))])

        def _build_weights(samples):
            weights = []
            for h_id, t_id_offset in samples:
                t_id = t_id_offset - meta_info['num_herb']
                weights.append(min(path_counts.get((int(h_id), int(t_id)), 0) / 50.0, 0.5))
            return np.array(weights, dtype=np.float32)

        train_weights = _build_weights(train_samples)
        val_weights = _build_weights(val_samples)
        test_weights = _build_weights(test_samples)

        best_val_auc = 0.0
        best_state = None
        patience = 10
        patience_counter = 0

        for epoch in range(config.epochs):
            model.train()

            # HTI loss (batch by batch)
            epoch_preds = []
            epoch_labels = []

            for start in range(0, len(train_samples), batch_size):
                end = start + batch_size
                batch_pairs = train_samples[start:end]
                batch_labels = train_labels[start:end]
                batch_weights = train_weights[start:end]

                batch_paths = [pair_to_paths.get((int(h), int(t - meta_info['num_herb'])), [])
                               for h, t in batch_pairs]

                if triples.size(0) > kge_batch_size:
                    idx = torch.randperm(triples.size(0))[:kge_batch_size]
                    pos_kge = triples[idx]
                else:
                    pos_kge = triples

                neg_kge = _sample_kge_negatives(
                    pos_kge.cpu().numpy(),
                    rel_schemas,
                    meta_info,
                    pair_to_ingredients,
                    ingredient_to_herbs,
                    ingredient_to_targets,
                    pos_set_all
                ).to(device)

                kge_loss = model.kge_loss(pos_kge, neg_kge, margin=kge_margin)

                optimizer.zero_grad()
                h_final = model.encode(adjs_pt)
                preds = model.predict_pairs(
                    h_final,
                    torch.LongTensor(batch_pairs).to(device),
                    batch_paths,
                    torch.FloatTensor(batch_weights).to(device),
                    offsets
                )

                labels = torch.FloatTensor(batch_labels).to(device)
                loss_hti = F.binary_cross_entropy(preds, labels)

                loss = loss_hti + kge_weight * kge_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_preds.extend(preds.detach().cpu().numpy())
                epoch_labels.extend(batch_labels.tolist())

            train_metrics = _compute_metrics(np.array(epoch_labels), np.array(epoch_preds))

            # validation
            model.eval()
            with torch.no_grad():
                h_final = model.encode(adjs_pt)
                val_preds = []
                for start in range(0, len(val_samples), batch_size):
                    end = start + batch_size
                    batch_pairs = val_samples[start:end]
                    batch_weights = val_weights[start:end]
                    batch_paths = [pair_to_paths.get((int(h), int(t - meta_info['num_herb'])), [])
                                   for h, t in batch_pairs]
                    preds = model.predict_pairs(
                        h_final,
                        torch.LongTensor(batch_pairs).to(device),
                        batch_paths,
                        torch.FloatTensor(batch_weights).to(device),
                        offsets
                    )
                    val_preds.extend(preds.detach().cpu().numpy())

            val_metrics = _compute_metrics(val_labels, np.array(val_preds))

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"   [STOP] 早停于 epoch {epoch} (patience={patience})")
                break

            if epoch % 5 == 0 or epoch == config.epochs - 1:
                print(f"[Fold {fold + 1}][Epoch {epoch}] "
                      f"Train AUC {train_metrics['auc']:.4f} | "
                      f"Val AUC {val_metrics['auc']:.4f} | "
                      f"KGE {kge_loss.item():.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            h_final = model.encode(adjs_pt)
            test_preds = []
            for start in range(0, len(test_samples), batch_size):
                end = start + batch_size
                batch_pairs = test_samples[start:end]
                batch_weights = test_weights[start:end]
                batch_paths = [pair_to_paths.get((int(h), int(t - meta_info['num_herb'])), [])
                               for h, t in batch_pairs]
                preds = model.predict_pairs(
                    h_final,
                    torch.LongTensor(batch_pairs).to(device),
                    batch_paths,
                    torch.FloatTensor(batch_weights).to(device),
                    offsets
                )
                test_preds.extend(preds.detach().cpu().numpy())

        test_metrics = _compute_metrics(test_labels, np.array(test_preds))
        all_results.append({'fold': fold + 1, 'best_val_auc': best_val_auc, 'test_metrics': test_metrics})
        print(f"Fold {fold + 1} Test AUC: {test_metrics['auc']:.4f}, AUPR: {test_metrics['aupr']:.4f}")

        if best_val_auc > best_global_auc and best_state is not None:
            best_global_auc = best_val_auc
            best_global_state = best_state

    metric_names = list(all_results[0]['test_metrics'].keys())
    mean_metrics = np.array([[np.mean([r['test_metrics'][m] for r in all_results]) for m in metric_names]])
    baseline_metrics = np.array([[0.5, 0.5, 0.5, 0, 0, 0, 0.5, 0.0]])
    improvements = mean_metrics - baseline_metrics

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.result_dir, exist_ok=True)

    # 保存详细结果（对齐历史输出格式）
    results_table = []
    for metric in metric_names:
        values = [r['test_metrics'][metric] for r in all_results]
        results_table.append({
            'Metric': metric.upper(),
            'Mean': f"{np.mean(values):.6f}",
            'Std': f"{np.std(values):.6f}",
            'Min': f"{np.min(values):.6f}",
            'Max': f"{np.max(values):.6f}",
        })

    results_df = pd.DataFrame(results_table)
    results_csv_path = os.path.join(config.result_dir, f'detailed_results_{timestamp}.csv')
    results_df.to_csv(results_csv_path, index=False)

    _save_simple_visualizations(all_results, results_table, config.result_dir, timestamp)

    if best_global_state is not None:
        os.makedirs(config.model_dir, exist_ok=True)
        save_path = os.path.join(config.model_dir, "kgmap_model.pth")
        torch.save({"state_dict": best_global_state, "best_val_auc": best_global_auc}, save_path)

    return all_results, (mean_metrics, baseline_metrics, improvements)


def _save_simple_visualizations(all_results, results_table, result_dir, timestamp):
    """保存简化版可视化图表，保持与历史目录结构一致"""
    plot_dir = os.path.join(result_dir, f"training_plots_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)

    metrics = [row['Metric'] for row in results_table]
    means = [float(row['Mean']) for row in results_table]
    stds = [float(row['Std']) for row in results_table]

    # 1) 指标均值条形图
    plt.figure(figsize=(12, 7))
    plt.bar(metrics, means, yerr=stds, capsize=4, color="#4C78A8")
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    plt.title("Metrics Mean ± Std")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"metrics_comparison_{timestamp}.png"), dpi=300)
    plt.close()

    # 2) 详细统计表图
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table_data = [[r['Metric'], r['Mean'], r['Std'], r['Min'], r['Max']] for r in results_table]
    col_labels = ["Metric", "Mean", "Std", "Min", "Max"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center')
    table.scale(1, 1.4)
    plt.title("Detailed Statistics")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"detailed_statistics_{timestamp}.png"), dpi=300)
    plt.close()

    # 3) 各折指标箱线图
    fig, ax = plt.subplots(figsize=(12, 7))
    data = []
    labels = []
    for m in metrics:
        vals = [r['test_metrics'][m.lower()] for r in all_results]
        data.append(vals)
        labels.append(m)
    ax.boxplot(data, labels=labels)
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    plt.title("Metrics Distribution Across Folds")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"training_history_{timestamp}.png"), dpi=300)
    plt.close()


def run_ablation_experiments(config):
    """运行消融实验并保存结果"""
    ablations = [
        ("A1", {"use_semantic_neg_sampling": False}),
        ("A2", {"use_adaptive_search": False}),
        ("A3", {"use_semantic_edges": False}),
        ("A4", {"use_residual": False}),
        ("A5", {"use_relation_attention": False}),
        ("A6", {"use_metapath_attention": False}),
        ("A7", {"use_transe_pretrain": False}),
        ("KGMAP-HTI", {}),
    ]

    metric_order = ["AUC", "AUPR", "ACC", "PREC", "REC", "F1", "MAE", "ME"]
    rows = []

    default_flags = {
        "use_semantic_neg_sampling": True,
        "use_adaptive_search": True,
        "use_semantic_edges": True,
        "use_residual": True,
        "use_relation_attention": True,
        "use_metapath_attention": True,
        "use_transe_pretrain": True,
    }

    for name, overrides in ablations:
        for k, v in default_flags.items():
            setattr(config, k, v)
        for k, v in overrides.items():
            setattr(config, k, v)
        print(f"\n>>> Running {name}")
        all_results, (mean_metrics, _, _) = train_herb_target_model_with_paths(config)
        metric_names = [m.upper() for m in list(all_results[0]['test_metrics'].keys())]
        values = mean_metrics[0].tolist()
        metric_map = {m: float(v) for m, v in zip(metric_names, values)}
        row = {"Metric": name}
        for m in metric_order:
            row[m] = metric_map.get(m, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows, columns=["Metric"] + metric_order)
    os.makedirs(config.result_dir, exist_ok=True)
    out_path = os.path.join(config.result_dir, "results.csv")
    df.to_csv(out_path, index=False)
    print(f"Ablation results saved to: {out_path}")
    return df
