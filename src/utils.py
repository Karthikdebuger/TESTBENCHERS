import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import hashlib
import yaml
import logging
import math
from typing import Dict, Any, Optional, Union

def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(config_device: str = 'auto') -> torch.device:
    """Returns the torch device based on availability and config."""
    if config_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(config_device)

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Computes the Peak Signal-to-Noise Ratio (PSNR) in dB."""
    pred = torch.clamp(pred, 0.0, data_range)
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(1)
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr.mean().item()

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 2D Gaussian window for SSIM computation."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, data_range: float = 1.0) -> float:
    """Computes the Structural Similarity Index Measure (SSIM)."""
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)
    
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

_lpips_models = {}

def compute_lpips_score(pred: torch.Tensor, target: torch.Tensor, net: str = 'vgg', device: Union[str, torch.device] = 'cpu') -> float:
    """Computes the LPIPS score."""
    import lpips
    if net not in _lpips_models:
        _lpips_models[net] = lpips.LPIPS(net=net).to(device)
        _lpips_models[net].eval()
    
    loss_fn = _lpips_models[net]
    
    if pred.size(1) == 1:
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        
    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0
    
    with torch.no_grad():
        score = loss_fn(pred, target)
        
    return score.mean().item()

def hash_config(config: Dict[str, Any]) -> str:
    """Computes the SHA256 hash of a configuration dictionary."""
    config_str = yaml.dump(config, sort_keys=True)
    return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, config: Dict[str, Any], path: str, ema_model: Optional[torch.nn.Module] = None) -> None:
    """Saves a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'config': config,
        'config_hash': hash_config(config)
    }
    if ema_model is not None:
        checkpoint['ema_model_state'] = ema_model.state_dict()
    torch.save(checkpoint, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, ema_model: Optional[torch.nn.Module] = None, device: Union[str, torch.device] = 'cpu') -> int:
    """Loads a training checkpoint."""
    if not os.path.exists(path):
        logging.warning(f"Checkpoint not found at {path}")
        return 0
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state', {}), strict=False)
    
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    if ema_model is not None and 'ema_model_state' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state'], strict=False)
        
    return checkpoint.get('epoch', 0)

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self) -> None:
        self.reset()
        
    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EMA:
    """Exponential Moving Average (EMA) of model parameters."""
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def setup_logging(log_file: Optional[str] = None) -> None:
    """Configures Python logging to both console and file."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
