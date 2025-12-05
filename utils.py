# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch


def normalize_sym(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_row(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(adj).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(y_true, y_pred):
    return np.mean((y_pred > 0.5) == y_true)


def calculate_mean_error(y_true, y_pred):
    return np.mean(y_pred - y_true)


def get_metrics(y_true, y_pred, y_score):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, precision_score, 
        recall_score, f1_score, mean_absolute_error
    )
    
    metrics = {}
    
    try:
        metrics['auc'] = roc_auc_score(y_true, y_score)
    except:
        metrics['auc'] = 0.0
    
    try:
        metrics['aupr'] = average_precision_score(y_true, y_score)
    except:
        metrics['aupr'] = 0.0
    
    metrics['acc'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mae'] = mean_absolute_error(y_true, y_score)
    metrics['me'] = calculate_mean_error(y_true, y_score)
    
    return metrics


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
