"""
Dataset module for SemiCon AI Hackathon image restoration project.
Provides dataloaders for paired noisy and ground truth (GT) images.
Supports multiple image formats and dynamic per-image normalization.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Configure module logger
logger = logging.getLogger(__name__)

class PairedImageDataset(Dataset):
    """
    Dataset for paired noisy and ground truth (GT) images.
    
    Args:
        noisy_dir (str or Path): Directory containing noisy images.
        gt_dir (str or Path): Directory containing ground truth images.
        extensions (List[str]): List of file extensions to include (e.g., ['.png', '.tif']).
        patch_size (Optional[int]): If provided, takes random crops of size patch_size x patch_size.
        augment_fn (Optional[Callable]): Optional augmentation function to apply to both images.
    """
    
    def __init__(self, 
                 noisy_dir: str, 
                 gt_dir: str, 
                 extensions: List[str], 
                 patch_size: Optional[int] = None, 
                 augment_fn: Optional[Callable] = None):
        super().__init__()
        
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)
        self.extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
        self.patch_size = patch_size
        self.augment_fn = augment_fn
        
        self.channels = None
        self.samples = self._find_pairs()
        
    def _find_pairs(self) -> List[Tuple[Path, Path]]:
        """Finds matching pairs of noisy and GT images."""
        if not self.noisy_dir.exists() or not self.gt_dir.exists():
            logger.warning(f"Directories do not exist: {self.noisy_dir} or {self.gt_dir}")
            return []
            
        noisy_files = []
        for ext in self.extensions:
            # Use glob to find files matching extensions
            pattern = str(self.noisy_dir / f"**/*{ext}")
            noisy_files.extend(glob.glob(pattern, recursive=True))
            
        noisy_files = sorted([Path(f) for f in noisy_files])
        
        paired_samples = []
        for noisy_path in noisy_files:
            # Assuming matching filename in gt_dir (same name and extension)
            gt_path = self.gt_dir / noisy_path.relative_to(self.noisy_dir)
            
            if gt_path.exists():
                paired_samples.append((noisy_path, gt_path))
            else:
                logger.warning(f"Skipping {noisy_path}: No matching GT file found at {gt_path}")
                
        logger.info(f"Found {len(paired_samples)} matching image pairs out of {len(noisy_files)} noisy images.")
        return paired_samples

    def _load_image(self, path: Path) -> np.ndarray:
        """Loads an image from path into a numpy array."""
        ext = path.suffix.lower()
        
        if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            img = Image.open(path)
            img = np.array(img)
            # Add channel dim if grayscale
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        elif ext in ['.tif', '.tiff']:
            import tifffile
            img = tifffile.imread(str(path))
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        elif ext == '.npy':
            img = np.load(str(path))
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
        elif ext == '.pt':
            tensor = torch.load(str(path), map_location='cpu')
            if tensor.ndim == 3 and tensor.shape[0] in [1, 3, 4]:
                # Assuming CHW, convert to HWC for consistency before crop
                img = tensor.permute(1, 2, 0).numpy()
            elif tensor.ndim == 2:
                img = tensor.unsqueeze(-1).numpy()
            else:
                img = tensor.numpy()
        else:
            raise ValueError(f"Unsupported extension: {ext}")
            
        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads and returns a paired image sample."""
        # Try loading, if fails, log and return random next valid sample
        max_attempts = 10
        
        for attempt in range(max_attempts):
            current_idx = (idx + attempt) % len(self.samples)
            noisy_path, gt_path = self.samples[current_idx]
            
            try:
                # Load images as HWC numpy arrays
                noisy_img = self._load_image(noisy_path)
                gt_img = self._load_image(gt_path)
                
                # Check shapes: GT may be larger by upscale factor
                noisy_h, noisy_w = noisy_img.shape[:2]
                gt_h, gt_w = gt_img.shape[:2]
                
                # Determine scale factor
                scale_h = gt_h // noisy_h if noisy_h > 0 else 1
                scale_w = gt_w // noisy_w if noisy_w > 0 else 1
                if scale_h != scale_w:
                    raise ValueError(f"Non-uniform scale: H={scale_h}x, W={scale_w}x")
                scale = scale_h
                    
                # Convert to float32
                noisy_img = noisy_img.astype(np.float32)
                gt_img = gt_img.astype(np.float32)
                
                # Process GT: Normalize to [0, 1] if uint8 or similar
                if gt_img.max() > 1.0:
                    gt_img = gt_img / 255.0
                    
                # Process Noisy: Dynamic per-image normalization to [0, 1]
                noisy_min = float(noisy_img.min())
                noisy_max = float(noisy_img.max())
                
                # Avoid division by zero
                range_val = max(noisy_max - noisy_min, 1e-8)
                noisy_img = (noisy_img - noisy_min) / range_val
                
                # Apply random crop (accounting for scale factor)
                if self.patch_size is not None:
                    ps = self.patch_size
                    gt_ps = ps * scale
                    
                    if noisy_h < ps or noisy_w < ps:
                        # Skip crop if image is already smaller
                        pass
                    else:
                        top = np.random.randint(0, noisy_h - ps + 1)
                        left = np.random.randint(0, noisy_w - ps + 1)
                        
                        noisy_img = noisy_img[top:top + ps, left:left + ps, :]
                        # GT crop at scaled coordinates
                        gt_top = top * scale
                        gt_left = left * scale
                        gt_img = gt_img[gt_top:gt_top + gt_ps, gt_left:gt_left + gt_ps, :]
                
                # Convert HWC to CHW tensors
                noisy_tensor = torch.from_numpy(noisy_img).permute(2, 0, 1).contiguous()
                gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).contiguous()
                
                # Initialize channels on first successful load
                if self.channels is None:
                    self.channels = noisy_tensor.shape[0]
                
                # Apply augmentations (assumes augment_fn takes and returns dict)
                sample = {
                    'noisy': noisy_tensor,
                    'gt': gt_tensor,
                    'filename': noisy_path.name,
                    'noisy_min': noisy_min,
                    'noisy_max': noisy_max
                }
                
                if self.augment_fn is not None:
                    try:
                        # augment_fn now returns (noisy, gt, deg_params_dict)
                        aug_result = self.augment_fn(sample['noisy'], sample['gt'])
                        
                        if len(aug_result) == 3:
                            aug_noisy, aug_gt, deg_params = aug_result
                            sample['deg_params'] = deg_params
                        else:
                            # Fallback in case a different augment function is passed
                            aug_noisy, aug_gt = aug_result
                            sample['deg_params'] = {}
                            
                        sample['noisy'] = aug_noisy
                        sample['gt'] = aug_gt
                    except Exception as e:
                        # If augmentation fails, use unaugmented data
                        logger.warning(f"Augmentation failed: {e}. Using raw data.")
                        sample['deg_params'] = {}
                    
                return sample
                
            except Exception as e:
                logger.error(f"Error loading {noisy_path}: {str(e)}. Skipping.")
                
        raise RuntimeError("Failed to load any valid images after multiple attempts.")


def get_dataloader(config: Dict[str, Any], split: str = 'train', augment_fn: Optional[Callable] = None) -> DataLoader:
    """
    Creates and returns a dataloader based on the provided config.
    
    Args:
        config: Full config dictionary.
        split: 'train', 'val', or 'test'.
        augment_fn: Optional augmentation function.
        
    Returns:
        DataLoader for the specified split.
    """
    data_cfg = config.get('data', {})
    
    # Map split to config keys
    noisy_key = f'{split}_noisy_dir'
    gt_key = f'{split}_gt_dir'
    
    noisy_dir = data_cfg.get(noisy_key)
    gt_dir = data_cfg.get(gt_key)
    extensions = data_cfg.get('extensions', ['.png', '.jpg', '.tif', '.npy', '.pt'])
    patch_size = data_cfg.get('patch_size') if split == 'train' else None
    
    if noisy_dir is None:
        raise ValueError(f"'{noisy_key}' not found in config['data']. Check your train.yaml.")
    if gt_dir is None:
        # For test split, gt may not exist
        if split == 'test':
            logger.warning(f"No GT directory for test split. Metrics will not be computed.")
        else:
            raise ValueError(f"'{gt_key}' not found in config['data']. Check your train.yaml.")
    
    dataset = PairedImageDataset(
        noisy_dir=noisy_dir,
        gt_dir=gt_dir,
        extensions=extensions,
        patch_size=patch_size,
        augment_fn=augment_fn
    )
    
    training_cfg = config.get('training', {})
    batch_size = training_cfg.get('batch_size', 4)
    num_workers = data_cfg.get('num_workers', 4)
    pin_memory = data_cfg.get('pin_memory', True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size if split == 'train' else max(1, batch_size),
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train')
    )
    
    logger.info(f"Created {split} dataloader: {len(dataset)} samples, batch_size={batch_size}")
    return dataloader


class RangeLogger:
    """
    Logs and tracks the range (min, max, mean) of images across batches.
    Useful for monitoring dynamic normalization ranges.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.running_mean = 0.0
        self.count = 0
        
    def update(self, noisy_batch: torch.Tensor):
        """
        Updates statistics based on a batch of tensors.
        
        Args:
            noisy_batch: Tensor of shape (B, C, H, W)
        """
        batch_min = float(noisy_batch.min())
        batch_max = float(noisy_batch.max())
        batch_mean = float(noisy_batch.mean())
        
        self.global_min = min(self.global_min, batch_min)
        self.global_max = max(self.global_max, batch_max)
        
        # Incremental mean update
        b = noisy_batch.size(0)
        self.running_mean = (self.running_mean * self.count + batch_mean * b) / (self.count + b)
        self.count += b
        
    def report(self) -> Dict[str, float]:
        """Returns the current statistics."""
        return {
            'min': self.global_min,
            'max': self.global_max,
            'mean': self.running_mean
        }
