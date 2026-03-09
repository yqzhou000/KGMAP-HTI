import pickle
import torch
import os

def check_models():
    model_path = "./models/best_herb_target_path_models.pkl"
    
    print(f"Checking model file: {model_path}")
    print(f"File size: {os.path.getsize(model_path)} bytes")
    print("="*50)
    
    try:
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        
        print(f"Data type: {type(models)}")
        print(f"Length: {len(models) if hasattr(models, '__len__') else 'N/A'}")
        print()
        
        if isinstance(models, list):
            for i, model_info in enumerate(models):
                print(f"Model {i}:")
                print(f"  Type: {type(model_info)}")
                
                if isinstance(model_info, dict):
                    print(f"  Keys: {list(model_info.keys())}")
                    
                    # Check tensor parameters
                    tensor_count = 0
                    for key, value in model_info.items():
                        if isinstance(value, torch.Tensor):
                            tensor_count += 1
                            print(f"    {key}: Tensor {value.shape}")
                        elif isinstance(value, dict):
                            nested_tensors = [k for k, v in value.items() if isinstance(v, torch.Tensor)]
                            if nested_tensors:
                                print(f"    {key}: Dict with {len(nested_tensors)} tensors")
                                for tensor_key in nested_tensors[:3]:  # Show first 3
                                    tensor_shape = value[tensor_key].shape
                                    print(f"      {tensor_key}: {tensor_shape}")
                            else:
                                print(f"    {key}: Dict with {len(value)} non-tensor items")
                        else:
                            print(f"    {key}: {type(value)} = {value}")
                    
                    if tensor_count == 0:
                        print("    WARNING: No tensor parameters found!")
                
                print()
                
                if i >= 2:  # Only show first 3
                    break
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_models()