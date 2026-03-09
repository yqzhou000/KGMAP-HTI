# generate_example_data.py
"""
生成示例数据用于测试草药-靶标相互作用预测模型
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager


def generate_example_data(output_dir='./data/herb',
                          n_herbs=200, n_targets=300,
                          n_ingredients=150, n_diseases=100,
                          seed=42):
    """
    生成示例数据集

    Args:
        output_dir: 输出目录
        n_herbs: 草药数量
        n_targets: 靶标数量
        n_ingredients: 成分数量
        n_diseases: 疾病数量
        seed: 随机种子
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)

    print("生成示例数据...")
    print(f"草药数量: {n_herbs}")
    print(f"靶标数量: {n_targets}")
    print(f"成分数量: {n_ingredients}")
    print(f"疾病数量: {n_diseases}")

    # 1. herb_target.dat - 主要预测目标
    print("\n1. 生成herb_target.dat...")
    n_interactions = int(n_herbs * n_targets * 0.01)  # 1%稀疏度
    herb_ids = np.random.randint(0, n_herbs, n_interactions)
    target_ids = np.random.randint(0, n_targets, n_interactions)
    # 70%正样本，30%负样本
    ratings = np.random.choice([0, 1], n_interactions, p=[0.3, 0.7])

    df_ht = pd.DataFrame({
        'herb': herb_ids,
        'target': target_ids,
        'rating': ratings
    })
    df_ht = df_ht.drop_duplicates(subset=['herb', 'target'])
    df_ht.to_csv(os.path.join(output_dir, 'herb_target.dat'),
                 index=False, header=False)
    print(f"   生成 {len(df_ht)} 条herb-target关系")

    # 2. herb_herb.dat
    print("\n2. 生成herb_herb.dat...")
    n_hh = int(n_herbs * 3)  # 平均每个草药3个关联
    h1 = np.random.randint(0, n_herbs, n_hh)
    h2 = np.random.randint(0, n_herbs, n_hh)
    df_hh = pd.DataFrame({
        'h1': h1,
        'h2': h2,
        'weight': np.ones(n_hh)
    })
    df_hh = df_hh[df_hh['h1'] != df_hh['h2']].drop_duplicates()
    df_hh.to_csv(os.path.join(output_dir, 'herb_herb.dat'),
                 index=False, header=False)
    print(f"   生成 {len(df_hh)} 条herb-herb关系")

    # 3. sim_herb.dat
    print("\n3. 生成sim_herb.dat...")
    n_sim_h = int(n_herbs * 5)  # 平均每个草药5个相似草药
    h1 = np.random.randint(0, n_herbs, n_sim_h)
    h2 = np.random.randint(0, n_herbs, n_sim_h)
    # 生成相似度分数（0.5-1.0之间）
    sim_scores = np.random.uniform(0.5, 1.0, n_sim_h)
    df_sim_h = pd.DataFrame({
        'h1': h1,
        'h2': h2,
        'weight': sim_scores
    })
    df_sim_h = df_sim_h[df_sim_h['h1'] != df_sim_h['h2']].drop_duplicates()
    df_sim_h.to_csv(os.path.join(output_dir, 'sim_herb.dat'),
                    index=False, header=False)
    print(f"   生成 {len(df_sim_h)} 条herb相似性关系")

    # 4. herb_disease.dat
    print("\n4. 生成herb_disease.dat...")
    n_hd = int(n_herbs * 2.5)  # 平均每个草药治疗2.5种疾病
    herb_ids = np.random.randint(0, n_herbs, n_hd)
    disease_ids = np.random.randint(0, n_diseases, n_hd)
    df_hd = pd.DataFrame({
        'herb': herb_ids,
        'disease': disease_ids,
        'weight': np.ones(n_hd)
    })
    df_hd = df_hd.drop_duplicates()
    df_hd.to_csv(os.path.join(output_dir, 'herb_disease.dat'),
                 index=False, header=False)
    print(f"   生成 {len(df_hd)} 条herb-disease关系")

    # 5. herb_ingredient.dat
    print("\n5. 生成herb_ingredient.dat...")
    n_hi = int(n_herbs * 3)  # 平均每个草药3种成分
    herb_ids = np.random.randint(0, n_herbs, n_hi)
    ingredient_ids = np.random.randint(0, n_ingredients, n_hi)
    df_hi = pd.DataFrame({
        'herb': herb_ids,
        'ingredient': ingredient_ids,
        'weight': np.ones(n_hi)
    })
    df_hi = df_hi.drop_duplicates()
    df_hi.to_csv(os.path.join(output_dir, 'herb_ingredient.dat'),
                 index=False, header=False)
    print(f"   生成 {len(df_hi)} 条herb-ingredient关系")

    # 6. ingredient_target.dat
    print("\n6. 生成ingredient_target.dat...")
    n_it = int(n_ingredients * 2.5)  # 平均每个成分作用于2.5个靶标
    ingredient_ids = np.random.randint(0, n_ingredients, n_it)
    target_ids = np.random.randint(0, n_targets, n_it)
    df_it = pd.DataFrame({
        'ingredient': ingredient_ids,
        'target': target_ids,
        'weight': np.ones(n_it)
    })
    df_it = df_it.drop_duplicates()
    df_it.to_csv(os.path.join(output_dir, 'ingredient_target.dat'),
                 index=False, header=False)
    print(f"   生成 {len(df_it)} 条ingredient-target关系")

    # 7. target_target.dat
    print("\n7. 生成target_target.dat...")
    n_tt = int(n_targets * 1.5)  # 平均每个靶标1.5个关联
    t1 = np.random.randint(0, n_targets, n_tt)
    t2 = np.random.randint(0, n_targets, n_tt)
    df_tt = pd.DataFrame({
        't1': t1,
        't2': t2,
        'weight': np.ones(n_tt)
    })
    df_tt = df_tt[df_tt['t1'] != df_tt['t2']].drop_duplicates()
    df_tt.to_csv(os.path.join(output_dir, 'target_target.dat'),
                 index=False, header=False)
    print(f"   生成 {len(df_tt)} 条target-target关系")

    # 8. sim_target.dat
    print("\n8. 生成sim_target.dat...")
    n_sim_t = int(n_targets * 3)  # 平均每个靶标3个相似靶标
    t1 = np.random.randint(0, n_targets, n_sim_t)
    t2 = np.random.randint(0, n_targets, n_sim_t)
    sim_scores = np.random.uniform(0.5, 1.0, n_sim_t)
    df_sim_t = pd.DataFrame({
        't1': t1,
        't2': t2,
        'weight': sim_scores
    })
    df_sim_t = df_sim_t[df_sim_t['t1'] != df_sim_t['t2']].drop_duplicates()
    df_sim_t.to_csv(os.path.join(output_dir, 'sim_target.dat'),
                    index=False, header=False)
    print(f"   生成 {len(df_sim_t)} 条target相似性关系")

    # 9. target_disease.dat
    print("\n9. 生成target_disease.dat...")
    n_td = int(n_targets * 1.5)  # 平均每个靶标关联1.5种疾病
    target_ids = np.random.randint(0, n_targets, n_td)
    disease_ids = np.random.randint(0, n_diseases, n_td)
    df_td = pd.DataFrame({
        'target': target_ids,
        'disease': disease_ids,
        'weight': np.ones(n_td)
    })
    df_td = df_td.drop_duplicates()
    df_td.to_csv(os.path.join(output_dir, 'target_disease.dat'),
                 index=False, header=False)
    print(f"   生成 {len(df_td)} 条target-disease关系")

    print(f"\n数据生成完成！保存在: {output_dir}")

    # 返回统计信息
    stats = {
        'n_herbs': n_herbs,
        'n_targets': n_targets,
        'n_ingredients': n_ingredients,
        'n_diseases': n_diseases,
        'n_herb_target': len(df_ht),
        'n_herb_herb': len(df_hh),
        'n_herb_ingredient': len(df_hi),
        'n_ingredient_target': len(df_it),
        'n_target_disease': len(df_td)
    }

    return stats


def visualize_knowledge_graph_sample(data_dir='./data/herb', save_path='kg_sample.png'):
    """
    可视化知识图谱的一个小样本
    """
    print("\n生成知识图谱可视化...")

    # 创建图
    G = nx.Graph()

    # 读取部分数据（只取前几个节点以便可视化）
    n_sample = 10

    # 添加节点
    herbs = [f'H{i}' for i in range(n_sample)]
    targets = [f'T{i}' for i in range(n_sample)]
    ingredients = [f'I{i}' for i in range(5)]
    diseases = [f'D{i}' for i in range(5)]

    G.add_nodes_from(herbs, node_type='herb')
    G.add_nodes_from(targets, node_type='target')
    G.add_nodes_from(ingredients, node_type='ingredient')
    G.add_nodes_from(diseases, node_type='disease')

    # 读取边（只取前几条）
    df_ht = pd.read_csv(os.path.join(data_dir, 'herb_target.dat'),
                        names=['h', 't', 'r'], nrows=20)
    df_ht = df_ht[df_ht['r'] == 1]  # 只取正样本
    for _, row in df_ht.iterrows():
        if row['h'] < n_sample and row['t'] < n_sample:
            G.add_edge(f'H{int(row["h"])}', f'T{int(row["t"])}', edge_type='h-t')

    df_hi = pd.read_csv(os.path.join(data_dir, 'herb_ingredient.dat'),
                        names=['h', 'i', 'w'], nrows=10)
    for _, row in df_hi.iterrows():
        if row['h'] < n_sample and row['i'] < 5:
            G.add_edge(f'H{int(row["h"])}', f'I{int(row["i"])}', edge_type='h-i')

    df_hd = pd.read_csv(os.path.join(data_dir, 'herb_disease.dat'),
                        names=['h', 'd', 'w'], nrows=10)
    for _, row in df_hd.iterrows():
        if row['h'] < n_sample and row['d'] < 5:
            G.add_edge(f'H{int(row["h"])}', f'D{int(row["d"])}', edge_type='h-d')

    # 绘制
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=3, iterations=50)

    # 设置节点颜色
    node_colors = []
    for node in G.nodes():
        if node.startswith('H'):
            node_colors.append('#90EE90')  # 草药 - 浅绿色
        elif node.startswith('T'):
            node_colors.append('#87CEEB')  # 靶标 - 天蓝色
        elif node.startswith('I'):
            node_colors.append('#FFD700')  # 成分 - 金色
        else:
            node_colors.append('#FFA07A')  # 疾病 - 浅珊瑚色

    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=1000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # 设置边的样式
    edge_styles = {
        'h-t': {'edge_color': 'red', 'style': 'solid', 'width': 2},
        'h-i': {'edge_color': 'green', 'style': 'dashed', 'width': 1.5},
        'h-d': {'edge_color': 'blue', 'style': 'dotted', 'width': 1.5}
    }

    for edge_type, style in edge_styles.items():
        edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get('edge_type') == edge_type]
        if edges:
            nx.draw_networkx_edges(G, pos, edges, **style, alpha=0.6)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='草药',
               markerfacecolor='#90EE90', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='靶标',
               markerfacecolor='#87CEEB', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='成分',
               markerfacecolor='#FFD700', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='疾病',
               markerfacecolor='#FFA07A', markersize=15),
        Line2D([0], [0], color='red', linewidth=2, label='草药-靶标'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='草药-成分'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle=':', label='草药-疾病')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.title('草药知识图谱示例', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"知识图谱可视化已保存: {save_path}")


def print_data_statistics(data_dir='./data/herb'):
    """
    打印数据统计信息
    """
    print("\n=== 数据统计信息 ===")

    files = [
        'herb_target.dat',
        'herb_herb.dat',
        'sim_herb.dat',
        'herb_disease.dat',
        'herb_ingredient.dat',
        'ingredient_target.dat',
        'target_target.dat',
        'sim_target.dat',
        'target_disease.dat'
    ]

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            print(f"{file}: {len(df)} 条记录")

            # 对于herb_target.dat，统计正负样本
            if file == 'herb_target.dat':
                df.columns = ['herb', 'target', 'rating']
                pos_count = len(df[df['rating'] == 1])
                neg_count = len(df[df['rating'] == 0])
                print(f"  - 正样本: {pos_count} ({pos_count / len(df) * 100:.1f}%)")
                print(f"  - 负样本: {neg_count} ({neg_count / len(df) * 100:.1f}%)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成草药-靶标模型的示例数据')
    parser.add_argument('--output_dir', type=str, default='./data/herb',
                        help='输出目录')
    parser.add_argument('--n_herbs', type=int, default=200,
                        help='草药数量')
    parser.add_argument('--n_targets', type=int, default=300,
                        help='靶标数量')
    parser.add_argument('--n_ingredients', type=int, default=150,
                        help='成分数量')
    parser.add_argument('--n_diseases', type=int, default=100,
                        help='疾病数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--visualize', action='store_true',
                        help='生成知识图谱可视化')
    parser.add_argument('--create_readme', action='store_true',
                        help='创建README文件')

    args = parser.parse_args()

    # 生成数据
    stats = generate_example_data(
        output_dir=args.output_dir,
        n_herbs=args.n_herbs,
        n_targets=args.n_targets,
        n_ingredients=args.n_ingredients,
        n_diseases=args.n_diseases,
        seed=args.seed
    )

    # 打印统计信息
    print_data_statistics(args.output_dir)

    # 可视化
    if args.visualize:
        visualize_knowledge_graph_sample(args.output_dir)

    # 创建README
    if args.create_readme:
        create_readme()

    print("\n完成！现在可以运行以下命令开始训练：")
    print("python main.py --mode all")