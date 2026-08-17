"""
SemiCon AI Hackathon — Unified Restoration Model
=================================================
Architecture: NAFNet backbone + self-supervised degradation conditioning
             + unrolled-stage structure + PixelShuffle SR head + global residual learning

Reference: Architecture spec §3.1–§3.6
- §3.1 Backbone: NAFNet blocks (LayerNorm, depthwise conv, SimpleGate, SCA)
- §3.2 Conditioning: Self-supervised branch detecting degradation composition
- §3.3 Unrolled stages: 2–3 repeated denoise→deconvolve→re-inject stages
- §3.4 SR head: PixelShuffle, gated by conditioning
- §3.5 Global residual: output = bilinear_upsample(input) + residual
- §3.6 Range handling: soft-clamped output (sigmoid-scaled)
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# §3.1 — NAFNet Building Blocks
# ============================================================================

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → normalize over C, H, W
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleGate(nn.Module):
    """
    SimpleGate: splits features along channel dim and multiplies halves.
    Replaces nonlinear activations (ReLU/GELU) — core NAFNet innovation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA): global average pool → 1x1 conv → scale.
    Targets speckle's multiplicative structure by channel-wise recalibration.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1,
                              padding=0, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pool(x)
        attn = self.conv(attn)
        return x * attn


class NAFBlock(nn.Module):
    """
    NAFNet block: LayerNorm → 1x1 expand → depthwise 3x3 → SimpleGate → SCA → 1x1 compress
                + LayerNorm → 1x1 expand → SimpleGate → 1x1 compress (FFN)
    Each component has a specific justification per §3.1.
    """

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2,
                 drop_out_rate: float = 0.0):
        super().__init__()
        dw_channels = channels * dw_expand

        # Spatial mixing branch
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3,
                               padding=1, stride=1, groups=dw_channels, bias=True)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)

        # Channel mixing branch (FFN)
        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        # Learnable scaling factors (beta, gamma) for residual connections
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # Channel mixing (FFN)
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg2(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class ConditionedNAFBlock(nn.Module):
    """
    NAFBlock with degradation conditioning injection.
    The conditioning vector modulates features via learned affine transform.
    """

    def __init__(self, channels: int, cond_dim: int, dw_expand: int = 2,
                 ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        self.naf_block = NAFBlock(channels, dw_expand, ffn_expand, drop_out_rate)
        # Conditioning: project cond vector to channel-wise scale and shift
        self.cond_scale = nn.Linear(cond_dim, channels)
        self.cond_shift = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
            cond: (B, cond_dim) conditioning vector
        """
        # Apply NAFBlock
        out = self.naf_block(x)
        # Modulate with conditioning (FiLM-style)
        scale = self.cond_scale(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        shift = self.cond_shift(cond).unsqueeze(-1).unsqueeze(-1)
        return out * (1 + scale) + shift


# ============================================================================
# §3.2 — Multi-Scale Pyramidal Degradation Conditioning Branch (v2)
# ============================================================================

class DegradationConditioningBranch(nn.Module):
    """
    Multi-scale pyramidal degradation encoder (upgraded from AAR).

    Extracts features at 3 spatial scales to capture:
    - Level 1 (64ch, 64x64): High-freq localized noise (Gaussian)
    - Level 2 (128ch, 32x32): Mid-freq speckle clustering & texture
    - Level 3 (256ch, 16x16): Low-freq structural attenuation & blur

    Each level has its own AdaptiveAvgPool, then all are concatenated
    to a 448-dim vector before MLP projection to cond_vector.

    Also produces 3 supervised auxiliary heads for degradation loss.
    """

    def __init__(self, in_channels: int, num_features: int = 6,
                 embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Level 1: 32 -> 64 channels, stride-2 downsample
        self.level1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Level 2: 64 -> 128 channels, stride-2 downsample
        self.level2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Level 3: 128 -> 256 channels, stride-2 downsample
        self.level3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Per-level global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP: 448 -> 128 -> embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(64 + 128 + 256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, embed_dim),
        )

        # Supervised auxiliary heads (for degradation loss during training)
        self.head_gaussian = nn.Sequential(
            nn.Linear(embed_dim, 1), nn.Sigmoid()
        )
        self.head_speckle = nn.Sequential(
            nn.Linear(embed_dim, 1), nn.Sigmoid()
        )
        self.head_scale = nn.Sequential(
            nn.Linear(embed_dim, 1), nn.Sigmoid()
        )

        # SR gate: predicts whether downsampling degradation is present
        self.sr_gate = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, H, W) NoisyLR input

        Returns:
            cond_vector: (B, embed_dim) conditioning vector
            sr_gate_value: (B, 1) gate for SR head [0, 1]
            deg_predictions: dict with 'gaussian_sigma', 'speckle_var',
                             'scale_factor' — each (B, 1)
        """
        # Multi-scale feature extraction
        s = self.stem(x)

        l1 = self.level1(s)
        f1 = self.pool(l1).flatten(1)  # (B, 64)

        l2 = self.level2(l1)
        f2 = self.pool(l2).flatten(1)  # (B, 128)

        l3 = self.level3(l2)
        f3 = self.pool(l3).flatten(1)  # (B, 256)

        # Concatenate multi-scale features
        cat = torch.cat([f1, f2, f3], dim=1)  # (B, 448)

        # Project to conditioning vector
        cond_vector = self.mlp(cat)  # (B, embed_dim)

        # Supervised degradation predictions
        deg_predictions = {
            'gaussian_sigma': self.head_gaussian(cond_vector),
            'speckle_var': self.head_speckle(cond_vector),
            'scale_factor': self.head_scale(cond_vector) * 3.0 + 1.0,  # map [0,1] -> [1,4]
        }

        # SR gating
        sr_gate_value = self.sr_gate(cond_vector)  # (B, 1)

        return cond_vector, sr_gate_value, deg_predictions


# ============================================================================
# §3.4 — PixelShuffle SR Head
# ============================================================================

class PixelShuffleSRHead(nn.Module):
    """
    Super-resolution head using PixelShuffle upsampling.
    Active only when conditioning branch signals downsampling is present (§3.4).
    Gated by sr_gate_value from DegradationConditioningBranch.
    """

    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int = 2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels,
                              out_channels * (upscale_factor ** 2),
                              kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor, sr_gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
            sr_gate: (B, 1) gating value from conditioning branch

        Returns:
            Upsampled features, gated by sr_gate
        """
        upsampled = self.pixel_shuffle(self.conv(x))
        # Gate: when sr_gate≈0 (no downsampling detected), pass through identity
        # When sr_gate≈1 (downsampling detected), use full upsampled output
        gate = sr_gate.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1, 1)
        # Identity path: bilinear upsample of input
        identity = F.interpolate(x, scale_factor=self.upscale_factor,
                                 mode='bilinear', align_corners=False)
        # Trim identity channels to match upsampled if needed
        if identity.shape[1] != upsampled.shape[1]:
            identity = identity[:, :upsampled.shape[1], :, :]
        return gate * upsampled + (1 - gate) * identity


# ============================================================================
# Encoder & Decoder with Conditioning
# ============================================================================

class Encoder(nn.Module):
    """Multi-scale encoder with NAFBlocks and downsampling."""

    def __init__(self, in_channels: int, width: int,
                 block_counts: list, middle_block_count: int):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, kernel_size=3,
                               padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, num_blocks in enumerate(block_counts):
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num_blocks)])
            )
            self.downs.append(
                nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2)
            )
            chan *= 2

        self.middle = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_block_count)]
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            bottleneck: (B, C, H/16, W/16) bottleneck features
            skips: list of skip connections from each encoder level
        """
        x = self.intro(x)
        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)
        return x, skips


class ConditionedDecoder(nn.Module):
    """Multi-scale decoder with conditioning injection at each level."""

    def __init__(self, out_channels: int, width: int,
                 block_counts: list, cond_dim: int):
        super().__init__()

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Compute channel sizes for decoder levels (reverse of encoder)
        num_levels = len(block_counts)
        channels = [width * (2 ** i) for i in range(num_levels)]
        channels.reverse()

        # Start from bottleneck channels
        chan = channels[0] * 2  # bottleneck has 2x the last encoder level

        for i, num_blocks in enumerate(block_counts):
            self.ups.append(
                nn.ConvTranspose2d(chan, chan // 2, kernel_size=2, stride=2)
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[ConditionedNAFBlock(chan, cond_dim) for _ in range(num_blocks)]
                )
            )

        self.ending = nn.Conv2d(width, out_channels, kernel_size=3,
                                padding=1, bias=True)

    def forward(self, x: torch.Tensor, skips: list,
                cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: bottleneck features
            skips: skip connections from encoder (deepest first after reversal)
            cond: (B, cond_dim) conditioning vector
        """
        skips = skips[::-1]  # reverse to match decoder order

        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            # Handle size mismatch from non-power-of-2 inputs
            if x.shape != skips[i].shape:
                x = F.interpolate(x, size=skips[i].shape[2:],
                                  mode='bilinear', align_corners=False)
            x = x + skips[i]
            for block in decoder:
                x = block(x, cond)

        x = self.ending(x)
        return x


# ============================================================================
# §3.3 — Single Restoration Stage (for Unrolling)
# ============================================================================

class RestorationStage(nn.Module):
    """
    One stage of the unrolled restoration pipeline (§3.3):
    denoise → deconvolve/upsample → re-inject conditioning.

    Lightweight version of the full encoder-decoder, designed for weight sharing.
    """

    def __init__(self, channels: int, cond_dim: int, num_blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConditionedNAFBlock(channels, cond_dim) for _ in range(num_blocks)
        ])
        self.refine = nn.Conv2d(channels, channels, kernel_size=3,
                                padding=1, bias=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        for block in self.blocks:
            x = block(x, cond)
        x = self.refine(x)
        return x + residual  # stage-level residual


# ============================================================================
# Full Unified Restoration Model
# ============================================================================

class UnifiedRestorationModel(nn.Module):
    """
    Unified restoration model combining all components (§3):
    - NAFNet backbone (encoder-decoder)
    - Self-supervised degradation conditioning branch
    - Unrolled-stage structure (2-3 stages, weight-shared)
    - PixelShuffle SR head (gated by conditioning)
    - Global residual learning

    One model, not an ensemble. Conditions itself on detected degradation.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config['model']

        self.in_channels = model_cfg.get('in_channels', 1)
        self.out_channels = model_cfg.get('out_channels', 1)
        self.width = model_cfg.get('width', 64)
        self.global_residual = model_cfg.get('global_residual', True)
        self.num_unrolled_stages = model_cfg.get('num_unrolled_stages', 2)
        self.share_weights = model_cfg.get('share_weights', True)
        self.upscale_factor = model_cfg.get('upscale_factor', 2)
        cond_embed_dim = model_cfg.get('cond_embed_dim', 64)
        cond_num_features = model_cfg.get('cond_num_features', 6)

        enc_block_counts = model_cfg.get('enc_block_counts', [2, 2, 4, 8])
        dec_block_counts = model_cfg.get('dec_block_counts', [2, 2, 2, 2])
        middle_block_count = model_cfg.get('middle_block_count', 12)

        # §3.2 — Degradation conditioning branch
        self.cond_branch = DegradationConditioningBranch(
            in_channels=self.in_channels,
            num_features=cond_num_features,
            embed_dim=cond_embed_dim,
        )

        # §3.1 — NAFNet Encoder
        self.encoder = Encoder(
            in_channels=self.in_channels,
            width=self.width,
            block_counts=enc_block_counts,
            middle_block_count=middle_block_count,
        )

        # §3.1 — Conditioned Decoder
        self.decoder = ConditionedDecoder(
            out_channels=self.out_channels,
            width=self.width,
            block_counts=dec_block_counts,
            cond_dim=cond_embed_dim,
        )

        # §3.3 — Unrolled stages
        if self.num_unrolled_stages > 1:
            if self.share_weights:
                # Share a single stage module across all iterations
                self.unrolled_stage = RestorationStage(
                    channels=self.out_channels,
                    cond_dim=cond_embed_dim,
                    num_blocks=4,
                )
            else:
                self.unrolled_stages = nn.ModuleList([
                    RestorationStage(
                        channels=self.out_channels,
                        cond_dim=cond_embed_dim,
                        num_blocks=4,
                    )
                    for _ in range(self.num_unrolled_stages - 1)
                ])

        # §3.4 — SR Head (gated by conditioning)
        self.sr_head = PixelShuffleSRHead(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            upscale_factor=self.upscale_factor,
        )

        # §3.6 — Range enforcement via loss (no sigmoid — avoids gradient saturation)
        # Range consistency loss softly penalizes outputs outside [0, 1]
        # Hard clamping is applied only at inference time
        self.output_activation = nn.Identity()

        logger.info(
            f"UnifiedRestorationModel initialized: "
            f"in={self.in_channels}, out={self.out_channels}, "
            f"width={self.width}, stages={self.num_unrolled_stages}, "
            f"upscale={self.upscale_factor}, "
            f"global_residual={self.global_residual}"
        )

    def _global_residual_base(self, x: torch.Tensor,
                              target_size: tuple = None) -> torch.Tensor:
        """
        Bilinearly upsample input for global residual connection (§3.5).
        If no upsampling needed, returns identity.
        """
        if target_size is not None and (x.shape[2] != target_size[0] or
                                         x.shape[3] != target_size[1]):
            return F.interpolate(x, size=target_size,
                                 mode='bilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the unified restoration model.

        Args:
            x: (B, C, H, W) NoisyLR input

        Returns:
            dict with:
                'restored': (B, C, H', W') restored image
                'cond_vector': conditioning vector
                'sr_gate': SR gate value
                'deg_predictions': dict of degradation parameter predictions
        """
        # §3.2 — Extract degradation conditioning
        cond_vector, sr_gate, deg_predictions = self.cond_branch(x)

        # §3.1 — Encode
        bottleneck, skips = self.encoder(x)

        # §3.1 — Decode with conditioning
        decoded = self.decoder(bottleneck, skips, cond_vector)

        # §3.3 — Unrolled stages (re-inject conditioning each stage)
        if self.num_unrolled_stages > 1:
            for stage_idx in range(self.num_unrolled_stages - 1):
                if self.share_weights:
                    decoded = self.unrolled_stage(decoded, cond_vector)
                else:
                    decoded = self.unrolled_stages[stage_idx](decoded, cond_vector)

        # §3.4 — SR head (gated: active only when downsampling detected)
        restored = self.sr_head(decoded, sr_gate)

        # §3.5 — Global residual learning
        if self.global_residual:
            base = self._global_residual_base(x, target_size=restored.shape[2:])
            # Ensure channel count matches
            if base.shape[1] != restored.shape[1]:
                base = base[:, :restored.shape[1], :, :]
            restored = base + restored

        # §3.6 — Output activation (Identity during training, clamp at eval)
        restored = self.output_activation(restored)

        return {
            'restored': restored,
            'cond_vector': cond_vector,
            'sr_gate': sr_gate,
            'deg_predictions': deg_predictions,
        }


# ============================================================================
# Model Construction Helper
# ============================================================================

def build_model(config: dict, device: torch.device = None) -> UnifiedRestorationModel:
    """
    Construct the unified restoration model from config.

    Args:
        config: full training config dict
        device: target device (auto-detected if None)

    Returns:
        UnifiedRestorationModel on the specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UnifiedRestorationModel(config)
    model = model.to(device)

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    # Load config
    config_path = "configs/train.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback minimal config for testing
        config = {
            'model': {
                'in_channels': 1,
                'out_channels': 1,
                'width': 32,
                'enc_block_counts': [1, 1, 1, 1],
                'dec_block_counts': [1, 1, 1, 1],
                'middle_block_count': 1,
                'cond_embed_dim': 32,
                'cond_num_features': 6,
                'num_unrolled_stages': 2,
                'share_weights': True,
                'upscale_factor': 2,
                'global_residual': True,
            }
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config, device)

    # Test forward pass
    batch_size = 2
    channels = config['model']['in_channels']
    x = torch.randn(batch_size, channels, 64, 64).to(device)

    with torch.no_grad():
        output = model(x)

    print(f"\nSmoke Test Results:")
    print(f"  Input shape:     {x.shape}")
    print(f"  Output shape:    {output['restored'].shape}")
    print(f"  Output range:    [{output['restored'].min():.4f}, {output['restored'].max():.4f}]")
    print(f"  Cond vector:     {output['cond_vector'].shape}")
    print(f"  SR gate:         {output['sr_gate'].squeeze().tolist()}")
    print(f"  Deg features:    {output['degradation_features'].shape}")
    print(f"\n  [OK] Model forward pass successful!")
