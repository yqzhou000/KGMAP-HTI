import os
import pandas as pd
import pickle
from config import Config

def check_id_mapping_discrepancy():
    """检查预测ID与原始文件ID的差异"""
    config = Config()
    
    print("="*80)
    print("              ID MAPPING DISCREPANCY CHECK")
    print("="*80)
    
    # 1. 读取原始数据文件
    print("\n1. Reading Original Data Files")
    print("-"*50)
    
    original_data = {}
    
    try:
        # 读取草药-靶标数据
        ht_df = pd.read_csv(os.path.join(config.data_dir, "herb_target.dat"), names=['hid', 'tid', 'rating'])
        original_data['herb_target'] = ht_df
        print(f"herb_target.dat: {len(ht_df)} records")
        print(f"  Herb IDs: {ht_df['hid'].min()} - {ht_df['hid'].max()} ({ht_df['hid'].nunique()} unique)")
        print(f"  Target IDs: {ht_df['tid'].min()} - {ht_df['tid'].max()} ({ht_df['tid'].nunique()} unique)")
        
        # 读取草药-成分数据
        hi_df = pd.read_csv(os.path.join(config.data_dir, "herb_ingredient.dat"), names=['hid', 'iid', 'rating'])
        original_data['herb_ingredient'] = hi_df
        print(f"herb_ingredient.dat: {len(hi_df)} records")
        print(f"  Herb IDs: {hi_df['hid'].min()} - {hi_df['hid'].max()} ({hi_df['hid'].nunique()} unique)")
        print(f"  Ingredient IDs: {hi_df['iid'].min()} - {hi_df['iid'].max()} ({hi_df['iid'].nunique()} unique)")
        
        # 读取成分-靶标数据
        it_df = pd.read_csv(os.path.join(config.data_dir, "ingredient_target.dat"), names=['iid', 'tid', 'rating'])
        original_data['ingredient_target'] = it_df
        print(f"ingredient_target.dat: {len(it_df)} records")
        print(f"  Ingredient IDs: {it_df['iid'].min()} - {it_df['iid'].max()} ({it_df['iid'].nunique()} unique)")
        print(f"  Target IDs: {it_df['tid'].min()} - {it_df['tid'].max()} ({it_df['tid'].nunique()} unique)")
        
        # 读取靶标-疾病数据
        td_df = pd.read_csv(os.path.join(config.data_dir, "target_disease.dat"), names=['tid', 'did', 'rating'])
        original_data['target_disease'] = td_df
        print(f"target_disease.dat: {len(td_df)} records")
        print(f"  Target IDs: {td_df['tid'].min()} - {td_df['tid'].max()} ({td_df['tid'].nunique()} unique)")
        print(f"  Disease IDs: {td_df['did'].min()} - {td_df['did'].max()} ({td_df['did'].nunique()} unique)")
        
    except Exception as e:
        print(f"Error reading original data: {e}")
        return
    
    # 2. 读取预测结果
    print("\n\n2. Reading Prediction Results")
    print("-"*50)
    
    # 找到最新的预测结果
    result_dir = config.result_dir
    pred_file = None
    
    if os.path.exists(result_dir):
        pred_dirs = [d for d in os.listdir(result_dir) if d.startswith('predictions_')]
        if pred_dirs:
            latest_pred = sorted(pred_dirs)[-1]
            pred_file = os.path.join(result_dir, latest_pred, 'herb_target_predictions.csv')
    
    if pred_file and os.path.exists(pred_file):
        try:
            pred_df = pd.read_csv(pred_file)
            print(f"Prediction file: {pred_file}")
            print(f"Prediction records: {len(pred_df)}")
            print(f"  Predicted Herb IDs: {pred_df['herb_id'].min()} - {pred_df['herb_id'].max()} ({pred_df['herb_id'].nunique()} unique)")
            print(f"  Predicted Target IDs: {pred_df['target_id'].min()} - {pred_df['target_id'].max()} ({pred_df['target_id'].nunique()} unique)")
            
            # 显示几个预测样本
            print("\nPrediction samples:")
            print(pred_df[['herb_id', 'target_id', 'prediction_score', 'ingredients', 'related_diseases']].head(3).to_string(index=False))
            
        except Exception as e:
            print(f"Error reading prediction results: {e}")
            return
    else:
        print("No prediction results found")
        return
    
    # 3. 对比分析
    print("\n\n3. ID Mapping Analysis")
    print("-"*50)
    
    # 对比草药ID
    orig_herbs = set(original_data['herb_target']['hid'].unique())
    pred_herbs = set(pred_df['herb_id'].unique())
    
    print("HERB ID COMPARISON:")
    print(f"  Original herb IDs: {len(orig_herbs)} (range: {min(orig_herbs)} - {max(orig_herbs)})")
    print(f"  Predicted herb IDs: {len(pred_herbs)} (range: {min(pred_herbs)} - {max(pred_herbs)})")
    print(f"  Overlap: {len(orig_herbs & pred_herbs)} IDs")
    print(f"  Original only: {len(orig_herbs - pred_herbs)} IDs")
    print(f"  Predicted only: {len(pred_herbs - orig_herbs)} IDs")
    
    if orig_herbs != pred_herbs:
        print("  ? HERB IDs DO NOT MATCH!")
        print(f"  Sample original herbs: {sorted(list(orig_herbs))[:10]}")
        print(f"  Sample predicted herbs: {sorted(list(pred_herbs))[:10]}")
    else:
        print("  ? Herb IDs match")
    
    # 对比靶标ID
    orig_targets = set(original_data['herb_target']['tid'].unique())
    pred_targets = set(pred_df['target_id'].unique())
    
    print("\nTARGET ID COMPARISON:")
    print(f"  Original target IDs: {len(orig_targets)} (range: {min(orig_targets)} - {max(orig_targets)})")
    print(f"  Predicted target IDs: {len(pred_targets)} (range: {min(pred_targets)} - {max(pred_targets)})")
    print(f"  Overlap: {len(orig_targets & pred_targets)} IDs")
    print(f"  Original only: {len(orig_targets - pred_targets)} IDs")
    print(f"  Predicted only: {len(pred_targets - orig_targets)} IDs")
    
    if orig_targets != pred_targets:
        print("  ? TARGET IDs DO NOT MATCH!")
        print(f"  Sample original targets: {sorted(list(orig_targets))[:10]}")
        print(f"  Sample predicted targets: {sorted(list(pred_targets))[:10]}")
    else:
        print("  ? Target IDs match")
    
    # 4. 检查成分和疾病ID
    print("\n\n4. Ingredient and Disease ID Check")
    print("-"*50)
    
    # 从预测结果中提取成分ID
    pred_ingredients = set()
    pred_diseases = set()
    
    for idx, row in pred_df.iterrows():
        if pd.notna(row['ingredients']) and row['ingredients']:
            # 处理可能的分隔符（逗号或顿号）
            ingredients = str(row['ingredients']).replace('、', ',').split(',')
            for ing in ingredients:
                if ing.strip():
                    try:
                        pred_ingredients.add(int(ing.strip()))
                    except:
                        pass
        
        if pd.notna(row['related_diseases']) and row['related_diseases']:
            diseases = str(row['related_diseases']).replace('、', ',').split(',')
            for dis in diseases:
                if dis.strip():
                    try:
                        pred_diseases.add(int(dis.strip()))
                    except:
                        pass
    
    # 对比成分ID
    orig_ingredients = set(original_data['herb_ingredient']['iid'].unique())
    print("INGREDIENT ID COMPARISON:")
    print(f"  Original ingredient IDs: {len(orig_ingredients)} (range: {min(orig_ingredients)} - {max(orig_ingredients)})")
    print(f"  Predicted ingredient IDs: {len(pred_ingredients)} (range: {min(pred_ingredients) if pred_ingredients else 'N/A'} - {max(pred_ingredients) if pred_ingredients else 'N/A'})")
    
    if pred_ingredients:
        overlap_ing = orig_ingredients & pred_ingredients
        print(f"  Overlap: {len(overlap_ing)} IDs")
        if len(overlap_ing) == 0:
            print("  ? NO INGREDIENT ID OVERLAP!")
            print(f"  Sample original ingredients: {sorted(list(orig_ingredients))[:10]}")
            print(f"  Sample predicted ingredients: {sorted(list(pred_ingredients))[:10]}")
        else:
            print("  ? Some ingredient IDs match")
    
    # 对比疾病ID
    orig_diseases = set(original_data['target_disease']['did'].unique())
    print("\nDISEASE ID COMPARISON:")
    print(f"  Original disease IDs: {len(orig_diseases)} (range: {min(orig_diseases)} - {max(orig_diseases)})")
    print(f"  Predicted disease IDs: {len(pred_diseases)} (range: {min(pred_diseases) if pred_diseases else 'N/A'} - {max(pred_diseases) if pred_diseases else 'N/A'})")
    
    if pred_diseases:
        overlap_dis = orig_diseases & pred_diseases
        print(f"  Overlap: {len(overlap_dis)} IDs")
        if len(overlap_dis) == 0:
            print("  ? NO DISEASE ID OVERLAP!")
            print(f"  Sample original diseases: {sorted(list(orig_diseases))[:10]}")
            print(f"  Sample predicted diseases: {sorted(list(pred_diseases))[:10]}")
        else:
            print("  ? Some disease IDs match")
    
    # 5. 检查可能的ID映射文件
    print("\n\n5. Looking for ID Mapping Files")
    print("-"*50)
    
    mapping_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(keyword in file.lower() for keyword in ['mapping', 'map', 'index', 'id']):
                if file.endswith(('.pkl', '.csv', '.txt')):
                    mapping_files.append(os.path.join(root, file))
    
    if mapping_files:
        print("Potential mapping files found:")
        for f in mapping_files:
            print(f"  {f}")
    else:
        print("No obvious mapping files found")
    
    # 6. 检查数据预处理过程
    print("\n\n6. Data Preprocessing Check")
    print("-"*50)
    
    # 检查dataset.py中的ID处理
    dataset_file = "dataset.py"
    if os.path.exists(dataset_file):
        print("Checking dataset.py for ID processing...")
        with open(dataset_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 查找可能的ID重映射代码
        keywords = ['remap', 'mapping', 'reindex', 'offset', 'transform']
        found_keywords = []
        for keyword in keywords:
            if keyword in content.lower():
                found_keywords.append(keyword)
        
        if found_keywords:
            print(f"  Found potential ID processing keywords: {found_keywords}")
        else:
            print("  No obvious ID processing keywords found")
    
    # 7. 建议解决方案
    print("\n\n7. Recommended Solutions")
    print("-"*50)
    
    print("To resolve ID mapping discrepancies:")
    print("1. Check if HerbTargetDataset class performs ID remapping")
    print("2. Look for ID mapping tables in preprocessed files")
    print("3. Verify if the model expects 0-based consecutive IDs")
    print("4. Consider creating reverse mapping to convert predictions back to original IDs")
    
    print("\nImmediate investigation steps:")
    print("python -c \"from dataset import HerbTargetDataset; help(HerbTargetDataset.__init__)\"")
    print("grep -n \"remap\\|mapping\\|reindex\" dataset.py")
    print("ls -la ./preprocessed/")
    
    print("\n" + "="*80)
    
    return original_data, pred_df

if __name__ == "__main__":
    try:
        original_data, pred_df = check_id_mapping_discrepancy()
    except Exception as e:
        print(f"Error during ID mapping check: {e}")
        import traceback
        traceback.print_exc()