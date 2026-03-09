import pandas as pd
import os
from config import Config

def diagnose_data_ids():
    """诊断各个数据文件中的ID范围"""
    config = Config()
    
    print("诊断数据文件中的ID范围")
    print("="*50)
    
    # 1. 检查草药-靶标数据
    print("1. 草药-靶标数据 (herb_target.dat):")
    try:
        ht_df = pd.read_csv(os.path.join(config.data_dir, "herb_target.dat"), names=['hid', 'tid', 'rating'])
        print(f"   草药ID范围: {ht_df['hid'].min()} - {ht_df['hid'].max()}")
        print(f"   靶标ID范围: {ht_df['tid'].min()} - {ht_df['tid'].max()}")
        print(f"   数据条数: {len(ht_df)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 2. 检查草药-成分数据
    print("\n2. 草药-成分数据 (herb_ingredient.dat):")
    try:
        hi_df = pd.read_csv(os.path.join(config.data_dir, "herb_ingredient.dat"), names=['hid', 'iid', 'rating'])
        print(f"   草药ID范围: {hi_df['hid'].min()} - {hi_df['hid'].max()}")
        print(f"   成分ID范围: {hi_df['iid'].min()} - {hi_df['iid'].max()}")
        print(f"   数据条数: {len(hi_df)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 3. 检查成分-靶标数据
    print("\n3. 成分-靶标数据 (ingredient_target.dat):")
    try:
        it_df = pd.read_csv(os.path.join(config.data_dir, "ingredient_target.dat"), names=['iid', 'tid', 'rating'])
        print(f"   成分ID范围: {it_df['iid'].min()} - {it_df['iid'].max()}")
        print(f"   靶标ID范围: {it_df['tid'].min()} - {it_df['tid'].max()}")
        print(f"   数据条数: {len(it_df)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 4. 检查靶标-疾病数据
    print("\n4. 靶标-疾病数据 (target_disease.dat):")
    try:
        td_df = pd.read_csv(os.path.join(config.data_dir, "target_disease.dat"), names=['tid', 'did', 'rating'])
        print(f"   靶标ID范围: {td_df['tid'].min()} - {td_df['tid'].max()}")
        print(f"   疾病ID范围: {td_df['did'].min()} - {td_df['did'].max()}")
        print(f"   数据条数: {len(td_df)}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n" + "="*50)
    print("建议检查:")
    print("1. 所有ID应该从0开始连续编号")
    print("2. 各个文件中相同类型的ID范围应该一致")
    print("3. 如果ID不连续，可能需要重新映射")

if __name__ == "__main__":
    diagnose_data_ids()