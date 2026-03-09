import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import os

# 在类外设置全局字体大小（避免 seaborn 覆盖 matplotlib）
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12
})

class ResultAnalyzer:
    """结果分析器"""

    def __init__(self, result_dir='./results'):
        self.result_dir = result_dir

    def load_results(self, result_file='kgmap_training_results.pkl'):
        """加载训练结果"""
        with open(os.path.join(self.result_dir, result_file), 'rb') as f:
            self.results = pickle.load(f)
        return self.results

    def plot_training_curves(self, save_dir=None):
        """当前训练流程未保存逐epoch曲线，保留接口以兼容"""
        if save_dir is None:
            save_dir = self.result_dir
        print("当前未保存逐epoch训练曲线，无法绘制。")

    def plot_final_metrics(self, save_dir=None):
        """绘制最终指标箱线图（字体增强版）"""
        if save_dir is None:
            save_dir = self.result_dir

        all_results = self.results['all_results']
        metric_names = list(all_results[0]['test_metrics'].keys())
        all_metrics = {m: [r['test_metrics'][m] for r in all_results] for m in metric_names}

        data = []
        for metric, values in all_metrics.items():
            for value in values:
                data.append({'Metric': metric.upper(), 'Value': value})
        df = pd.DataFrame(data)

        plt.figure(figsize=(14, 7))
        sns.boxplot(x='Metric', y='Value', data=df)

        plt.title('Final Metrics Distribution Across Folds', fontsize=20)
        plt.ylabel('Value', fontsize=18)
        plt.xlabel('Metric', fontsize=18)
        plt.xticks(rotation=45, fontsize=16)
        plt.yticks(fontsize=16)

        for i, metric in enumerate(all_metrics.keys()):
            mean_val = np.mean(all_metrics[metric])
            plt.text(i, mean_val, f'{mean_val:.3f}',
                     ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'final_metrics_boxplot.png'), dpi=300)
        plt.close()

    def plot_prediction_distribution(self, predictions, labels, save_path=None):
        """预测分布图（字体增强版）"""
        if save_path is None:
            save_path = os.path.join(self.result_dir, 'prediction_distribution.png')

        plt.figure(figsize=(12, 7))

        pos_preds = predictions[labels == 1]
        neg_preds = predictions[labels == 0]

        plt.hist(pos_preds, bins=50, alpha=0.5, label='Positive')
        plt.hist(neg_preds, bins=50, alpha=0.5, label='Negative')

        plt.xlabel('Prediction Score', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        plt.title('Distribution of Prediction Scores', fontsize=20)
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
