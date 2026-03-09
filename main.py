import os
import sys
import torch
import numpy as np
from config import Config, get_args
from preprocess import HerbKnowledgeGraph
from train_kgmap import train_herb_target_model_with_paths
import pickle


def run_preprocessing(config):
    """运行数据预处理"""
    print("=" * 50)
    print("开始数据预处理...")
    print("=" * 50)
    
    # 构建知识图谱
    kg_builder = HerbKnowledgeGraph(config.data_dir, config.preprocessed_dir)
    kg_builder.build_graph()
    
    print("数据预处理完成！")


def run_training(config):
    """运行模型训练"""
    print("=" * 50)
    print("开始基于路径的草药-靶标预测模型训练...")
    print("=" * 50)
    
    # 训练基于路径的草药-靶标预测模型
    all_results, (mean_path, mean_baseline, mean_improvement) = train_herb_target_model_with_paths(config)
    
    # 保存性能总结
    results_summary = {
        'path_enhanced_metrics': mean_path,
        'baseline_metrics': mean_baseline,
        'improvements': mean_improvement,
        'metric_names': ['AUC', 'AUPR', 'ACC', 'Precision', 'Recall', 'F1', 'MAE', 'ME']
    }
    
    summary_path = os.path.join(config.result_dir, "training_results_summary.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"性能总结已保存到: {summary_path}")

    detailed_path = os.path.join(config.result_dir, "kgmap_training_results.pkl")
    with open(detailed_path, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'summary': results_summary
        }, f)
    print(f"详细训练结果已保存到: {detailed_path}")
    return all_results, results_summary


def main():
    """主函数"""
    # 解析命令行参数
    args = get_args()
    
    # 创建配置对象
    config = Config()
    
    # 更新配置参数
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    print("草药-成分-靶标-疾病预测系统")
    print(f"运行模式: {args.mode}")
    print(f"数据目录: {config.data_dir}")
    print(f"预处理目录: {config.preprocessed_dir}")
    print(f"模型目录: {config.model_dir}")
    print(f"结果目录: {config.result_dir}")
    print(f"使用设备: {config.device}")
    
    try:
        if args.mode == 'preprocess':
            run_preprocessing(config)
            
        elif args.mode == 'train':
            run_training(config)
            
        elif args.mode == 'predict':
            from predict import run_predictions
            run_predictions(config)
            
        elif args.mode == 'ablation':
            from train_kgmap import run_ablation_experiments
            run_ablation_experiments(config)
            
        elif args.mode == 'all':
            run_preprocessing(config)
            run_training(config)
            from predict import run_predictions
            run_predictions(config)
            
        else:
            print(f"未知的运行模式: {args.mode}")
            print("可用模式:")
            print("  preprocess - 数据预处理")
            print("  train      - 模型训练")
            print("  predict    - 批量预测")
            print("  ablation   - 消融实验")
            print("  all        - 完整流程")
            sys.exit(1)
            
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 50)
    print("程序执行完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
