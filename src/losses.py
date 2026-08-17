"""
losses.py

Loss module for SemiCon AI Hackathon image restoration project.
Contains PyTorch differentiable losses: Charbonnier, SSIM, FFT, Edge (Sobel),
Range Consistency, and LPIPS, along with a CompositeLoss wrapper.
"""
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss.
    A differentiable approximation to L1 loss that is more robust to heavy-tailed outliers,
    such as those from speckle noise in semiconductor imagery.
    Formula: loss = sqrt((pred - target)^2 + eps^2)
    """
    def __init__(self, eps: float = 1e-2):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Ground truth tensor (B, C, H, W)
        """
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel."""
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    """Creates a 2D Gaussian window for SSIM."""
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int, channel: int, size_average: bool = True) -> torch.Tensor:
    """Computes the SSIM metric."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    Loss = 1 - SSIM.
    Works for grayscale and RGB by using depthwise convolutions over the channels.
    """
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = _create_window(window_size, self.channel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = pred.size()
        if channel == self.channel and self.window.data.type() == pred.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel).to(pred.device).type_as(pred)
            self.window = window
            self.channel = channel
        return 1.0 - _ssim(pred, target, window, self.window_size, channel, self.size_average)


class FFTLoss(nn.Module):
    """
    Frequency-domain L1 Loss.
    Computes L1 loss on the magnitude of the 2D Fast Fourier Transform.
    Useful for capturing fine texture and high-frequency differences.
    """
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm="ortho")
        target_fft = torch.fft.fft2(target, dim=(-2, -1), norm="ortho")
        
        # Use magnitude for the loss
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return self.criterion(pred_mag, target_mag)


class RangeConsistencyLoss(nn.Module):
    """
    Range Consistency Loss.
    Directly penalizes output values outside the [0, 1] range.
    Helps enforce ground truth data contract without hard clipping.
    """
    def __init__(self):
        super(RangeConsistencyLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ignore target, evaluate pred against [0, 1] bounds
        # ReLU(x - 1) is positive if x > 1
        # ReLU(-x) is positive if x < 0
        upper_violation = F.relu(pred - 1.0)
        lower_violation = F.relu(-pred)
        return torch.mean(upper_violation + lower_violation)


class DegradationAuxLoss(nn.Module):
    """
    Auxiliary degradation parameter regression loss.
    Computes MSE between predicted and ground-truth degradation parameters.
    Only active when synthetic augmentation provides ground-truth params.

    L_deg = (1/3) * [ (pred_gauss - gt_gauss)^2 +
                       (pred_speckle - gt_speckle)^2 +
                       (pred_scale - gt_scale)^2 ]
    """
    def __init__(self):
        super(DegradationAuxLoss, self).__init__()

    def forward(self, deg_predictions: dict, deg_targets: dict) -> torch.Tensor:
        """
        Args:
            deg_predictions: dict with 'gaussian_sigma', 'speckle_var', 'scale_factor'
            deg_targets: dict with same keys, ground-truth values from augmentation
        Returns:
            Scalar MSE loss
        """
        loss = torch.tensor(0.0, device=next(iter(deg_predictions.values())).device)
        count = 0
        for key in ['gaussian_sigma', 'speckle_var', 'scale_factor']:
            if key in deg_predictions and key in deg_targets:
                pred = deg_predictions[key].view(-1)
                target = deg_targets[key].view(-1).to(pred.device)
                loss = loss + torch.mean((pred - target) ** 2)
                count += 1
        if count > 0:
            loss = loss / count
        return loss


class EdgeLoss(nn.Module):
    """
    Gradient-domain loss using Sobel filters.
    L_edge = mean(|Sobel_x(pred) - Sobel_x(gt)|) + mean(|Sobel_y(pred) - Sobel_y(gt)|)
    Preserves sharp edges critical in semiconductor die imagery.
    Register Sobel kernels as buffers (not parameters).
    """
    def __init__(self):
        super(EdgeLoss, self).__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.size(1)
        weight_x = self.sobel_x.expand(C, 1, 3, 3)
        weight_y = self.sobel_y.expand(C, 1, 3, 3)

        pred_x = F.conv2d(pred, weight_x, padding=1, groups=C)
        pred_y = F.conv2d(pred, weight_y, padding=1, groups=C)
        gt_x = F.conv2d(target, weight_x, padding=1, groups=C)
        gt_y = F.conv2d(target, weight_y, padding=1, groups=C)

        return torch.mean(torch.abs(pred_x - gt_x)) + torch.mean(torch.abs(pred_y - gt_y))


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) Loss.
    Wraps the lpips library and lazy-loads the backbone.
    Logs a warning about potential domain mismatch for non-natural images (like semiconductor).
    """
    def __init__(self, net: str = 'vgg'):
        super(LPIPSLoss, self).__init__()
        self.net = net
        self.loss_fn = None
        self._warned = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is None:
            try:
                import lpips
            except ImportError:
                raise ImportError("lpips package is required for LPIPSLoss. Run `pip install lpips`.")
            
            logger.info(f"Initializing LPIPS with '{self.net}' backbone.")
            # Move to same device as pred
            self.loss_fn = lpips.LPIPS(net=self.net).to(pred.device)
            self.loss_fn.eval()
            # No need to train LPIPS feature extractor
            for param in self.loss_fn.parameters():
                param.requires_grad = False
                
        if not self._warned:
            logger.warning("LPIPS is trained on natural images. Domain mismatch for semiconductor imagery may lead to suboptimal perceptual guidance.")
            self._warned = True

        # Handle grayscale inputs (C=1) by repeating to 3 channels, since LPIPS expects 3 channels
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
            
        # LPIPS expects inputs in range [-1, 1], assuming pred/target are in [0, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0

        loss = self.loss_fn(pred_scaled, target_scaled)
        return torch.mean(loss)


class CompositeLoss(nn.Module):
    """
    Composite Loss for the architecture.
    Combines Charbonnier, SSIM, FFT, Edge, Range Consistency, and LPIPS losses based on config.
    
    Supports both nested config format (from train.yaml):
        loss:
          charbonnier:
            enabled: true
            epsilon: 1e-3
          ssim:
            enabled: true
            weight: 0.2
    
    And flat config format:
        loss:
          charbonnier_weight: 1.0
          charbonnier_eps: 1e-3
    """
    def __init__(self, config: dict):
        super(CompositeLoss, self).__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

        loss_cfg = config.get('loss', {})

        # --- Charbonnier ---
        charb = loss_cfg.get('charbonnier', {})
        if isinstance(charb, dict):
            # Nested format from train.yaml
            if charb.get('enabled', False):
                eps = charb.get('epsilon', 1e-3)
                self.losses['charbonnier'] = CharbonnierLoss(eps=eps)
                # Charbonnier is the primary loss, weight=1.0 by default
                self.weights['charbonnier'] = charb.get('weight', 1.0)
        elif loss_cfg.get('charbonnier_weight', 0.0) > 0:
            # Flat format
            self.losses['charbonnier'] = CharbonnierLoss(eps=loss_cfg.get('charbonnier_eps', 1e-3))
            self.weights['charbonnier'] = loss_cfg['charbonnier_weight']

        # --- SSIM ---
        ssim_cfg = loss_cfg.get('ssim', {})
        if isinstance(ssim_cfg, dict):
            if ssim_cfg.get('enabled', False):
                ws = ssim_cfg.get('window_size', 11)
                self.losses['ssim'] = SSIMLoss(window_size=ws)
                self.weights['ssim'] = ssim_cfg.get('weight', 0.2)
        elif loss_cfg.get('ssim_weight', 0.0) > 0:
            self.losses['ssim'] = SSIMLoss(window_size=loss_cfg.get('ssim_window_size', 11))
            self.weights['ssim'] = loss_cfg['ssim_weight']

        # --- FFT ---
        fft_cfg = loss_cfg.get('fft', {})
        if isinstance(fft_cfg, dict):
            if fft_cfg.get('enabled', False):
                self.losses['fft'] = FFTLoss()
                self.weights['fft'] = fft_cfg.get('weight', 0.1)
        elif loss_cfg.get('fft_weight', 0.0) > 0:
            self.losses['fft'] = FFTLoss()
            self.weights['fft'] = loss_cfg['fft_weight']

        # --- Range Consistency ---
        rc_cfg = loss_cfg.get('range_consistency', {})
        if isinstance(rc_cfg, dict):
            if rc_cfg.get('enabled', False):
                self.losses['range_consistency'] = RangeConsistencyLoss()
                self.weights['range_consistency'] = rc_cfg.get('weight', 0.05)
        elif loss_cfg.get('range_consistency_weight', 0.0) > 0:
            self.losses['range_consistency'] = RangeConsistencyLoss()
            self.weights['range_consistency'] = loss_cfg['range_consistency_weight']

        # --- Edge (Sobel gradient-domain — semiconductor-critical) ---
        edge_cfg = loss_cfg.get('edge', {})
        if isinstance(edge_cfg, dict):
            if edge_cfg.get('enabled', False):
                self.losses['edge'] = EdgeLoss()
                self.weights['edge'] = edge_cfg.get('weight', 0.1)
        elif loss_cfg.get('edge_weight', 0.0) > 0:
            self.losses['edge'] = EdgeLoss()
            self.weights['edge'] = loss_cfg['edge_weight']

        # --- LPIPS (optional, gated — may be domain-mismatched) ---
        lpips_cfg = loss_cfg.get('lpips', {})
        if isinstance(lpips_cfg, dict):
            if lpips_cfg.get('enabled', False):
                net = lpips_cfg.get('net', 'vgg')
                self.losses['lpips'] = LPIPSLoss(net=net)
                self.weights['lpips'] = lpips_cfg.get('weight', 0.1)
        elif loss_cfg.get('lpips_weight', 0.0) > 0:
            self.losses['lpips'] = LPIPSLoss(net=loss_cfg.get('lpips_net', 'vgg'))
            self.weights['lpips'] = loss_cfg['lpips_weight']

        # --- Degradation Auxiliary Loss (v2 upgrade) ---
        deg_cfg = loss_cfg.get('degradation', {})
        self.use_deg_loss = False
        if isinstance(deg_cfg, dict) and deg_cfg.get('enabled', False):
            self.deg_loss_fn = DegradationAuxLoss()
            self.deg_weight = deg_cfg.get('weight', 0.1)
            self.use_deg_loss = True
            logger.info(f"Degradation auxiliary loss enabled with weight={self.deg_weight}")

        if len(self.losses) == 0:
            logger.warning("No losses enabled in CompositeLoss. Check your configuration.")
        else:
            enabled = {k: self.weights[k] for k in self.losses}
            logger.info(f"CompositeLoss initialized with: {enabled}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                model_output: dict = None, deg_params: dict = None):
        """
        Args:
            pred: Predicted tensor (B, C, H, W)
            target: Ground truth tensor (B, C, H, W)
            model_output: Full model output dict (for degradation loss)
            deg_params: Ground-truth degradation parameters from augmentation
            
        Returns:
            total_loss: Scalar tensor representing the weighted sum of all enabled losses.
            loss_dict: Dictionary containing the individual unweighted loss values for logging.
        """
        # CAST TO FP32 to prevent AMP NaN explosion in loss functions (FFT, Edge, SSIM)
        pred = pred.float()
        target = target.float()

        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_dict = {}

        for name, loss_fn in self.losses.items():
            try:
                l_val = loss_fn(pred, target)
                weighted_l_val = l_val * self.weights[name]
                total_loss = total_loss + weighted_l_val
                loss_dict[name] = l_val.item()
            except Exception as e:
                logger.error(f"Error computing {name} loss: {e}")
                loss_dict[name] = float('nan')

        # Degradation auxiliary loss (only when synthetic aug params are available)
        if (self.use_deg_loss and model_output is not None
                and deg_params is not None
                and 'deg_predictions' in model_output):
            try:
                l_deg = self.deg_loss_fn(
                    model_output['deg_predictions'], deg_params
                )
                total_loss = total_loss + l_deg * self.deg_weight
                loss_dict['degradation'] = l_deg.item()
            except Exception as e:
                logger.error(f"Error computing degradation loss: {e}")
                loss_dict['degradation'] = float('nan')

        return total_loss, loss_dict
