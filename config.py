import os
import argparse
import torch

class Config:
    """配置类"""
    def __init__(self):
        self.data_dir = './data/herb'
        # 对比用公共数据集（示例路径）
        # D:\A学习文件\课题\KGMAP-HTI\KGMAP-HTI\KGMAP-HTI\data\data_luo
        self.preprocessed_dir = './preprocessed_herb'
        self.log_dir = './logs'
        self.model_dir = './models'
        self.result_dir = './results'
        # 对比用公共数据集结果（示例文件）
        # D:\A学习文件\课题\KGMAP-HTI\KGMAP-HTI\KGMAP-HTI\result_1.csv
        
        # 创建必要的目录
        for dir_path in [self.preprocessed_dir, self.log_dir, self.model_dir, self.result_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 模型参数
        self.lr = 0.005
        self.wd = 0.001
        self.n_hid = 64
        self.dropout = 0.2
        self.epochs = 200
        self.gpu = 2
        self.seed = 1
        
        # 架构搜索参数
        self.alr = 3e-4
        self.eps = 0.0
        self.decay = 0.8
        self.search_epochs = 200
        
        # Node2Vec参数
        self.node2vec_dim = 64
        self.walk_length = 200
        self.num_walks = 10
        self.p = 1
        self.q = 1
        self.workers = 2
        
        # 预测参数
        self.prediction_threshold = 0.5
        self.max_predictions_per_herb = 100
        self.batch_size_predict = 4096
        
        self.device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='Herb-Target Interaction Prediction')
    
    # 基本参数
    parser.add_argument('--mode', 
                        choices=['preprocess', 'train', 'predict', 'all', 'ablation'],
                        default='all', 
                        help='运行模式')
    parser.add_argument('--data_dir', type=str, default='./data/herb', help='数据目录')
    parser.add_argument('--gpu', type=int, default=2, help='GPU设备ID')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--wd', type=float, default=0.001, help='权重衰减')
    parser.add_argument('--n_hid', type=int, default=64, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    
    # 预测参数
    parser.add_argument('--prediction_threshold', type=float, default=0.5, 
                        help='预测阈值')
    parser.add_argument('--max_herbs', type=int, default=50, 
                        help='预测的最大草药数量')
    parser.add_argument('--max_targets', type=int, default=None, 
                        help='预测的最大靶标数量')

    # Ablation flags
    parser.add_argument('--use_semantic_neg_sampling', action='store_true', default=True,
                        help='使用语义路径负采样')
    parser.add_argument('--use_adaptive_search', action='store_true', default=True,
                        help='使用自适应结构搜索')
    parser.add_argument('--use_semantic_edges', action='store_true', default=True,
                        help='使用语义关系边')
    parser.add_argument('--use_residual', action='store_true', default=True,
                        help='使用残差连接')
    parser.add_argument('--use_relation_attention', action='store_true', default=True,
                        help='使用关系注意力')
    parser.add_argument('--use_metapath_attention', action='store_true', default=True,
                        help='使用元路径注意力')
    parser.add_argument('--use_transe_pretrain', action='store_true', default=True,
                        help='使用TransE预训练')
    
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    return args
