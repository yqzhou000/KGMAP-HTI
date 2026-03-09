import os
import pickle
import torch
from config import Config

def check_model_structure():
    config = Config()
    model_path = os.path.join(config.model_dir, "best_herb_target_path_models.pkl")
    
    print(f"Checking model file: {model_path}")
    print("="*50)
    
    if not os.path.exists(model_path):
        print(f"ERROR: File not found: {model_path}")
        return
    
    try:
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        
        print(f"File loaded successfully")
        print(f"Type: {type(models)}")
        print(f"Length: {len(models) if hasattr(models, '__len__') else 'N/A'}")
        print()
        
        if isinstance(models, list):
            for i, model_info in enumerate(models[:3]):  # Check first 3 only
                print(f"Model {i}:")
                print(f"  Type: {type(model_info)}")
                
                if isinstance(model_info, dict):
                    print(f"  Keys: {list(model_info.keys())}")
                    
                    # Look for model parameters
                    param_count = 0
                    for key, value in model_info.items():
                        if isinstance(value, torch.Tensor):
                            param_count += 1
                            if param_count <= 3:  # Show first 3 tensor keys
                                print(f"    {key}: Tensor {value.shape}")
                        elif isinstance(value, dict):
                            print(f"    {key}: Dict with {len(value)} keys")
                            # Check if this dict contains tensors
                            tensor_keys = [k for k, v in value.items() if isinstance(v, torch.Tensor)]
                            if tensor_keys:
                                print(f"      Contains tensors: {len(tensor_keys)} items")
                                print(f"      Sample keys: {tensor_keys[:3]}")
                        else:
                            print(f"    {key}: {type(value)}")
                    
                    if param_count > 3:
                        print(f"    ... and {param_count - 3} more tensor parameters")
                
                print()
        
    except Exception as e:
        print(f"ERROR loading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_structure()