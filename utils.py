import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, mean_absolute_error
)


def normalize_sym(adj):
    """对称归一化"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_row(mx):
    """行归一化"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def evaluate_metrics(y_true, y_score):
    """计算评估指标"""
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        'auc': roc_auc_score(y_true, y_score),
        'aupr': average_precision_score(y_true, y_score),
        'acc': accuracy_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'me': np.mean(y_score - y_true),
        'mae': mean_absolute_error(y_true, y_score)
    }

    return metrics


import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_metrics(metric_history, save_path):
    # 强制更新所有字体大小
    mpl.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12
    })

    for metric_name, fold_data in metric_history.items():
        plt.figure(figsize=(10, 6))
        for i, fold_metric in enumerate(fold_data):
            plt.plot(fold_metric, label=f'Fold {i+1}', linewidth=2)

        plt.xlabel('Epoch')
        plt.ylabel(metric_name.upper())
        plt.title(f'{metric_name.upper()} over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_path}/metric_{metric_name}.png', dpi=300)
        plt.close()
