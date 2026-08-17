import os
import sys
import glob
import numpy as np
import torch
from src.model import build_model
import yaml
import warnings

# Suppress warnings for clean stdout
warnings.filterwarnings('ignore')

def tta_forward(model, x):
    """8-fold Test-Time Augmentation for maximum PSNR"""
    predictions = []
    for flip in [False, True]:
        for rot in [0, 1, 2, 3]:
            aug = x
            if flip:
                aug = torch.flip(aug, dims=[-1])
            if rot > 0:
                aug = torch.rot90(aug, k=rot, dims=[-2, -1])
            
            out = model(aug)['restored']
            
            if rot > 0:
                out = torch.rot90(out, k=4 - rot, dims=[-2, -1])
            if flip:
                out = torch.flip(out, dims=[-1])
                
            predictions.append(out)
    return torch.stack(predictions, dim=0).mean(dim=0)

def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # 2. It creates the output directory if it does not already exist.
    os.makedirs(output_dir, exist_ok=True)
    
    # 11. The solution can run on an NVIDIA GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model configuration and weights
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'train.yaml')
    weights_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pt')
    
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
    try:
        model = build_model(config).to(device)
    except Exception as e:
        print(f"Failed to build model: {e}")
        sys.exit(1)
        
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('ema_state_dict', checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint)))
        model.load_state_dict(state_dict, strict=False)
        print(f"Model weights loaded successfully from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found. Ensure models/model.pt exists for actual submission.")
        
    model.eval()
    
    # 1. reads all .npy files from the input directory.
    npy_files = glob.glob(os.path.join(input_dir, '*.npy'))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return
        
    print(f"Found {len(npy_files)} .npy files to process.")
        
    for path in npy_files:
        filename = os.path.basename(path)
        
        try:
            img = np.load(path).astype(np.float32)
            
            # Robust Dynamic Normalization to [0, 1]
            img_min = img.min()
            img_max = img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = img - img_min
                
            tensor_img = torch.from_numpy(img).float()
            
            # Standardize shape to N, C, H, W
            if tensor_img.ndim == 2:
                tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)
            elif tensor_img.ndim == 3:
                if tensor_img.shape[-1] in [1, 3]:
                    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0)
                else:
                    tensor_img = tensor_img.unsqueeze(0)
                    
            tensor_img = tensor_img.to(device)
            
            # Forward pass
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    out = tta_forward(model, tensor_img)
            
            # 6. Output values are within [0,1]
            out = torch.clamp(out, 0.0, 1.0)
            
            # 5. Outputs are grayscale arrays with shape (H, W) or (H, W, 1).
            out_np = out.squeeze().cpu().numpy()
            if out_np.ndim == 3 and out_np.shape[0] == 1:
                out_np = out_np.squeeze(0) # Ensure (H, W)
                
            # 6. ...and contain no NaN or Inf values.
            out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)
            
            # 4. Each output has the same filename as its corresponding input.
            save_path = os.path.join(output_dir, filename)
            # 3. It generates one restored .npy file for every input file.
            np.save(save_path, out_np)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    main()
