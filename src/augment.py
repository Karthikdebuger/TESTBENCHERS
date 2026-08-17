import os
import random
import math
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Callable, Optional, List

def add_speckle_noise(image: torch.Tensor, var_range: Tuple[float, float] = (0.01, 0.15)) -> torch.Tensor:
    """
    Adds multiplicative speckle noise: image * (1 + noise) where noise ~ N(0, var).
    
    Args:
        image (torch.Tensor): Input image tensor, expected in range [0, 1] (..., C, H, W).
        var_range (Tuple[float, float]): Range of variance for the normal distribution.
        
    Returns:
        torch.Tensor: Image with speckle noise applied, clamped to [0, 1].
    """
    var = random.uniform(var_range[0], var_range[1])
    sigma = math.sqrt(var)
    noise = torch.randn_like(image) * sigma
    noisy_image = image * (1 + noise)
    return torch.clamp(noisy_image, 0.0, 1.0)

def add_gaussian_noise(image: torch.Tensor, sigma_range: Tuple[float, float] = (5.0, 50.0)) -> torch.Tensor:
    """
    Adds additive Gaussian noise. 
    Noise is simulated on [0, 255] scale then normalized back for [0, 1] images.
    
    Args:
        image (torch.Tensor): Input image tensor, expected in range [0, 1] (..., C, H, W).
        sigma_range (Tuple[float, float]): Range of standard deviation on [0, 255] scale.
        
    Returns:
        torch.Tensor: Image with Gaussian noise applied, clamped to [0, 1].
    """
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    sigma_norm = sigma / 255.0
    noise = torch.randn_like(image) * sigma_norm
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0.0, 1.0)

def add_downsample(image: torch.Tensor, scale_range: Tuple[float, float] = (2.0, 4.0)) -> torch.Tensor:
    """
    Downsamples then upsamples back to original size using bicubic interpolation 
    to simulate resolution loss.
    
    Args:
        image (torch.Tensor): Input image tensor (..., C, H, W).
        scale_range (Tuple[float, float]): Range of scale factor.
        
    Returns:
        torch.Tensor: Image with simulated resolution loss.
    """
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[-2:]
    
    # Calculate new dimensions
    new_h = max(1, int(h / scale))
    new_w = max(1, int(w / scale))
    
    # Needs to be at least 4D for interpolate (B, C, H, W)
    was_3d = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        was_3d = True
        
    # Downsample
    downsampled = F.interpolate(image, size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True)
    # Upsample back
    upsampled = F.interpolate(downsampled, size=(h, w), mode='bicubic', align_corners=False, antialias=True)
    
    if was_3d:
        upsampled = upsampled.squeeze(0)
        
    return torch.clamp(upsampled, 0.0, 1.0)


class GeometricAugment:
    """
    Applies the SAME geometric transform to both noisy and gt pairs.
    """
    def __init__(self, random_flip: bool = True, random_rotate: bool = True, random_crop: bool = True, crop_size: int = 256):
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_crop = random_crop
        self.crop_size = crop_size

    def __call__(self, noisy: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_h, noisy_w = noisy.shape[-2:]
        gt_h, gt_w = gt.shape[-2:]
        
        scale_h = gt_h // noisy_h if noisy_h > 0 else 1
        scale_w = gt_w // noisy_w if noisy_w > 0 else 1
        if scale_h != scale_w:
            raise ValueError(f"Non-uniform scale: H={scale_h}x, W={scale_w}x")
        scale = scale_h

        # Random Crop
        if self.random_crop:
            ps = self.crop_size
            gt_ps = ps * scale
            
            # Pad if smaller than crop_size
            if noisy_h < ps or noisy_w < ps:
                pad_h = max(0, ps - noisy_h)
                pad_w = max(0, ps - noisy_w)
                noisy = F.pad(noisy, (0, pad_w, 0, pad_h), mode='reflect')
                gt = F.pad(gt, (0, pad_w * scale, 0, pad_h * scale), mode='reflect')
                noisy_h, noisy_w = noisy.shape[-2:]
            
            top = random.randint(0, noisy_h - ps)
            left = random.randint(0, noisy_w - ps)
            
            noisy = noisy[..., top:top+ps, left:left+ps]
            
            gt_top = top * scale
            gt_left = left * scale
            gt = gt[..., gt_top:gt_top+gt_ps, gt_left:gt_left+gt_ps]

        # Random Flip
        if self.random_flip:
            if random.random() < 0.5: # Horizontal flip
                noisy = torch.flip(noisy, [-1])
                gt = torch.flip(gt, [-1])
            if random.random() < 0.5: # Vertical flip
                noisy = torch.flip(noisy, [-2])
                gt = torch.flip(gt, [-2])

        # Random Rotate (90 degrees)
        if self.random_rotate:
            k = random.randint(0, 3) # 0, 90, 180, 270 degrees
            if k > 0:
                noisy = torch.rot90(noisy, k, [-2, -1])
                gt = torch.rot90(gt, k, [-2, -1])

        return noisy.contiguous(), gt.contiguous()


class PhysicsAwareDegradation:
    """
    Applies physics-aware degradations in random order.
    Tracks actual degradation parameters for auxiliary loss supervision.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_degradations: List[str] = ['gaussian', 'downsample', 'speckle']
        # Track last-applied parameters
        self.last_params = {
            'gaussian_sigma': 0.0,
            'speckle_var': 0.0,
            'scale_factor': 1.0,
        }

    def set_active_degradations(self, degradations: List[str]):
        """Allows CurriculumScheduler to filter active degradations."""
        self.active_degradations = degradations

    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Applies enabled degradations in RANDOM ORDER.
        Returns (degraded_image, deg_params_dict)."""
        # Reset params
        self.last_params = {
            'gaussian_sigma': 0.0,
            'speckle_var': 0.0,
            'scale_factor': 1.0,
        }

        degradations = []
        if 'speckle' in self.active_degradations:
            var_range = self.config.get('speckle_var_range', (0.01, 0.15))
            var = random.uniform(var_range[0], var_range[1])
            sigma = math.sqrt(var)
            self.last_params['speckle_var'] = var
            degradations.append(lambda img, s=sigma: self._apply_speckle(img, s))
            
        if 'gaussian' in self.active_degradations:
            sigma_range = self.config.get('gaussian_sigma_range', (5.0, 50.0))
            sigma = random.uniform(sigma_range[0], sigma_range[1])
            sigma_norm = sigma / 255.0
            self.last_params['gaussian_sigma'] = sigma_norm
            degradations.append(lambda img, s=sigma_norm: self._apply_gaussian(img, s))
            
        if 'downsample' in self.active_degradations:
            scale_range = self.config.get('downsample_scale_range', (2.0, 4.0))
            scale = random.uniform(scale_range[0], scale_range[1])
            self.last_params['scale_factor'] = scale
            degradations.append(lambda img, s=scale: self._apply_downsample(img, s))

        # Apply in random permutation each call
        random.shuffle(degradations)
        
        degraded = image
        for func in degradations:
            degraded = func(degraded)
            
        return degraded, dict(self.last_params)

    @staticmethod
    def _apply_speckle(image: torch.Tensor, sigma: float) -> torch.Tensor:
        noise = torch.randn_like(image) * sigma
        return torch.clamp(image * (1 + noise), 0.0, 1.0)

    @staticmethod
    def _apply_gaussian(image: torch.Tensor, sigma_norm: float) -> torch.Tensor:
        noise = torch.randn_like(image) * sigma_norm
        return torch.clamp(image + noise, 0.0, 1.0)

    @staticmethod
    def _apply_downsample(image: torch.Tensor, scale: float) -> torch.Tensor:
        h, w = image.shape[-2:]
        new_h = max(1, int(h / scale))
        new_w = max(1, int(w / scale))
        was_3d = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            was_3d = True
        downsampled = F.interpolate(image, size=(new_h, new_w), mode='bicubic',
                                     align_corners=False, antialias=True)
        upsampled = F.interpolate(downsampled, size=(h, w), mode='bicubic',
                                   align_corners=False, antialias=True)
        if was_3d:
            upsampled = upsampled.squeeze(0)
        return torch.clamp(upsampled, 0.0, 1.0)


class CurriculumScheduler:
    """
    Curriculum scheduling for degradation complexity.
    """
    def __init__(self, warmup_epochs: int, full_epochs: int, degradation_model: Optional[PhysicsAwareDegradation] = None):
        self.warmup_epochs = warmup_epochs
        self.full_epochs = full_epochs
        self.degradation_model = degradation_model if degradation_model else PhysicsAwareDegradation({})
        
        # Define difficulty levels
        self.levels = [
            ['gaussian'],                               # Level 1: Mild, single degradation
            ['gaussian', 'downsample'],                 # Level 2: Two degradations
            ['gaussian', 'downsample', 'speckle']       # Level 3: Full complexity
        ]

    def get_augment_fn(self, epoch: int) -> Callable:
        """Returns an augmentation function appropriate for the current epoch."""
        if epoch < self.warmup_epochs:
            active_degs = self.levels[0]
        elif epoch < self.full_epochs:
            progress = (epoch - self.warmup_epochs) / max(1, (self.full_epochs - self.warmup_epochs))
            if progress < 0.5:
                active_degs = self.levels[1]
            else:
                active_degs = self.levels[2]
        else:
            active_degs = self.levels[2]
            
        def augment_fn(image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            self.degradation_model.set_active_degradations(active_degs)
            return self.degradation_model(image)
            
        return augment_fn

    def __call__(self, noisy: torch.Tensor, gt: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Convenience wrapper for the curriculum scheduling."""
        aug_fn = self.get_augment_fn(epoch)
        degraded_noisy, deg_params = aug_fn(noisy)
        return degraded_noisy, gt, deg_params


class TrainAugmentation:
    """
    Combines GeometricAugment and Curriculum-scheduled PhysicsAwareDegradation.
    """
    def __init__(self, config: Dict[str, Any]):
        self.geometric = GeometricAugment(
            random_flip=config.get('random_flip', True),
            random_rotate=config.get('random_rotate', True),
            random_crop=config.get('random_crop', True),
            crop_size=config.get('crop_size', 256)
        )
        self.degradation = PhysicsAwareDegradation(config)
        self.scheduler = CurriculumScheduler(
            warmup_epochs=config.get('warmup_epochs', 5),
            full_epochs=config.get('full_epochs', 20),
            degradation_model=self.degradation
        )

    def __call__(self, noisy: torch.Tensor, gt: torch.Tensor, epoch: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Applies geometric augment to pair, then physics degradation to noisy only.
        
        Args:
            noisy: Input noisy/degraded image tensor.
            gt: Ground truth image tensor.
            epoch: Current training epoch for curriculum.
            
        Returns:
            Tuple of (augmented_noisy, augmented_gt, deg_params_dict)
        """
        # Apply identical geometric transformations
        noisy, gt = self.geometric(noisy, gt)
        
        deg_params = {}
        # Apply physics-aware degradation to noisy only
        if epoch is not None:
            noisy, gt, deg_params = self.scheduler(noisy, gt, epoch)
        else:
            # Full degradation if no epoch provided
            self.degradation.set_active_degradations(['gaussian', 'downsample', 'speckle'])
            noisy, deg_params = self.degradation(noisy)
            
        return noisy, gt, deg_params
