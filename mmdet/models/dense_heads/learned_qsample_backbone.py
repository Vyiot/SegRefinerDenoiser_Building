# -----------------------------------------------------------------------------------
# LearnedQSampleBackbone: ResNet-50 + FPN + Conv Decoder for learned q-sample.
#
# Uses a pre-trained ResNet-50 backbone (with first conv extended from 3→4 channels)
# to extract multi-scale features, an FPN neck for feature fusion, and a lightweight
# convolutional decoder head with FiLM timestep conditioning.
#
# Architecture:
#   ┌────────────────────────────────────────────────────────────────────┐
#   │  Input: cat(RGB, GT mask) = (B, 4, 256, 256) + timestep t        │
#   ├────────────────────────────────────────────────────────────────────┤
#   │  ResNet-50 Backbone (in_channels=4, pre-trained + zero-padded)    │
#   │    C2: (B, 256,  64, 64)                                         │
#   │    C3: (B, 512,  32, 32)                                         │
#   │    C4: (B, 1024, 16, 16)                                         │
#   │    C5: (B, 2048,  8,  8)                                         │
#   ├────────────────────────────────────────────────────────────────────┤
#   │  FPN Neck (out_channels=256)                                      │
#   │    P2: (B, 256, 64, 64)                                          │
#   │    P3: (B, 256, 32, 32)                                          │
#   │    P4: (B, 256, 16, 16)                                          │
#   │    P5: (B, 256,  8,  8)                                          │
#   ├────────────────────────────────────────────────────────────────────┤
#   │  Conv Decoder Head                                                │
#   │    Upsample all levels to P2 (64×64) → Concat → (B, 1024, 64,64)│
#   │    Conv 1024→256 → FiLM(t) → Conv blocks → Upsample to 256×256  │
#   │    Conv 256→64→1 → Sigmoid → STE binarize                        │
#   └────────────────────────────────────────────────────────────────────┘
#
# Inputs:
#   - object_img:      (B, 3, 256, 256) RGB image
#   - object_gt_masks: (B, 1, 256, 256) ground-truth binary mask
#   - t:               (B,) discrete timestep indices
#
# Output:
#   - object_noisy_masks: (B, 1, 256, 256) discrete binary mask
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import HEADS, build_backbone, build_neck


# ============================================================================
# Timestep Embedding
# ============================================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embedding. Returns (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ============================================================================
# STE Binarization
# ============================================================================

class StraightThroughBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_binarize(x):
    return StraightThroughBinarize.apply(x)


# ============================================================================
# FiLM Modulation Layer
# ============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for timestep conditioning.

    Transforms timestep embedding into per-channel scale (gamma) and shift (beta),
    then modulates spatial features: out = x * gamma + beta.
    """

    def __init__(self, t_dim, feat_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(t_dim, feat_dim * 2),
            nn.SiLU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
        )

    def forward(self, x, t_emb):
        """
        Args:
            x:     (B, C, H, W) spatial features
            t_emb: (B, t_dim) timestep embedding
        Returns:
            (B, C, H, W) modulated features
        """
        params = self.fc(t_emb)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, C)
        gamma = gamma[:, :, None, None]  # (B, C, 1, 1)
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta  # residual-style: scale = 1 + gamma


# ============================================================================
# Convolutional Refinement Block
# ============================================================================

class ConvBlock(nn.Module):
    """Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU with residual skip."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


# ============================================================================
# Main Model: LearnedQSampleBackbone
# ============================================================================

@HEADS.register_module()
class LearnedQSampleBackbone(BaseModule):
    """
    Backbone-based learned q-sample model.

    Uses a pre-trained ResNet-50 backbone with FPN neck and a lightweight
    convolutional decoder with FiLM timestep conditioning.

    The first convolution of ResNet-50 is extended from 3→4 input channels
    by zero-initializing the weight for the 4th channel (GT mask), preserving
    all pre-trained ImageNet features while allowing the mask signal to be
    learned from scratch.

    Args:
        backbone (dict): Config for the backbone (ResNet-50).
        neck (dict): Config for the FPN neck.
        fpn_channels (int): FPN output channels. Default: 256.
        decoder_channels (int): Internal decoder channel count. Default: 256.
        num_decoder_blocks (int): Number of ConvBlock refinement stages. Default: 4.
        t_dim (int): Timestep embedding dimension. Default: 256.
        use_gumbel (bool): Use Gumbel-Softmax instead of STE. Default: False.
    """

    def __init__(
        self,
        backbone=None,
        neck=None,
        backbone_cfg=None,
        neck_cfg=None,
        fpn_channels=256,
        decoder_channels=256,
        num_decoder_blocks=4,
        t_dim=256,
        num_timesteps=6,
        use_gumbel=False,
        **kwargs,
    ):
        super().__init__()

        # Handle naming variations from config
        if backbone is None:
            backbone = backbone_cfg
        if neck is None:
            neck = neck_cfg

        self.fpn_channels = fpn_channels
        self.decoder_channels = decoder_channels
        self.t_dim = t_dim
        self.num_timesteps = num_timesteps
        self.use_gumbel = use_gumbel

        # ---- Backbone (e.g., ResNet-50 or MiT-B5) ----
        if backbone is not None:
            self.backbone = build_backbone(backbone)
            self._extend_first_conv()
        else:
            self.backbone = None

        # ---- FPN Neck ----
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None

        # ---- Timestep Embedding MLP ----
        self.time_embed = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # ---- Multi-scale Fusion ----
        # FPN produces 4 scales; we fuse by upsampling all to the largest
        # resolution and concatenating along channels
        num_fpn_levels = 4
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * num_fpn_levels, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )

        # ---- FiLM Conditioning ----
        self.film = FiLMLayer(t_dim, decoder_channels)

        # ---- Decoder Refinement Blocks ----
        self.decoder_blocks = nn.ModuleList([
            ConvBlock(decoder_channels) for _ in range(num_decoder_blocks)
        ])

        # ---- Output Head ----
        # After decoder blocks at 1/4 resolution (64×64), upsample to full
        self.output_head = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def init_weights(self):
        """Initialize weights.

        Handles the 4th input channel extension for the backbone:
        loads pre-trained 3-channel weights, then zero-pads the first
        conv to accept 4 channels.
        """
        super().init_weights()

        # After backbone.init_weights() loads pretrained weights,
        # extend first conv from 3→4 channels if needed
        self._extend_first_conv()

    def _extend_first_conv(self):
        """Extend the backbone's first conv layer from 3 to 4 input channels.

        Copies the pre-trained 3-channel weights and appends a zero-initialized
        4th channel. This preserves ImageNet features while allowing the mask
        channel to be learned from scratch.
        """
        # ResNet stores the first conv in self.backbone.conv1
        # (or self.backbone.stem for deep_stem variants)
        if hasattr(self.backbone, 'conv1'):
            old_conv = self.backbone.conv1
        elif hasattr(self.backbone, 'stem'):
            # deep_stem: first layer of the stem sequential
            old_conv = self.backbone.stem[0]
        elif hasattr(self.backbone, 'layers') and len(self.backbone.layers) > 0:
            # PVT/ViT: First layer of first stage's PatchEmbed
            stage0 = self.backbone.layers[0]
            if hasattr(stage0[0], 'projection'):
                old_conv = stage0[0].projection
            else:
                return
        else:
            return

        if old_conv.in_channels == 4:
            # Already 4 channels (e.g., loaded from a 4-channel checkpoint)
            return

        if old_conv.in_channels != 3:
            return

        # Create new conv with 4 input channels
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Copy pre-trained weights for the first 3 channels
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0.0  # zero-init mask channel
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias

        # Replace the conv layer
        if hasattr(self.backbone, 'conv1'):
            self.backbone.conv1 = new_conv
        elif hasattr(self.backbone, 'stem'):
            self.backbone.stem[0] = new_conv
        elif hasattr(self.backbone, 'layers') and len(self.backbone.layers) > 0:
            stage0 = self.backbone.layers[0]
            if hasattr(stage0[0], 'projection'):
                stage0[0].projection = new_conv

    def forward(self, object_img, object_gt_masks=None, t=None, return_feats=False):
        """
        Forward pass.

        Args:
            object_img:      (B, 3, 256, 256) normalized RGB image or (B, 4, 256, 256) if used as backbone
            object_gt_masks: (B, 1, 256, 256) ground-truth binary mask (optional if img is 4ch)
            t:               (B,) integer timestep indices
            return_feats:    If True, returns the FPN features instead of the binary mask.

        Returns:
            (B, 1, 256, 256) discrete binary noisy mask OR list of FPN tensors
        """
        if object_img.shape[1] == 4:
            x = object_img
        else:
            # ---- Concatenate image and GT mask ----
            x = torch.cat([object_img, object_gt_masks], dim=1)  # (B, 4, 256, 256)

        # ---- Timestep embedding ----
        if t is not None:
            t_emb = self.time_embed(timestep_embedding(t, self.t_dim))  # (B, t_dim)
        else:
            t_emb = None

        # ---- Backbone: extract multi-scale features ----
        feats = self.backbone(x)  # tuple of (C2, C3, C4, C5)

        # ---- FPN: produce multi-scale pyramid ----
        if self.neck is not None:
            fpn_feats = self.neck(feats)  # tuple of (P2, P3, P4, P5)
        else:
            fpn_feats = feats

        if return_feats:
            return fpn_feats

        # ---- Multi-scale Fusion ----
        # Upsample all FPN levels to P2 resolution and concatenate
        p2_size = fpn_feats[0].shape[2:]  # (64, 64) for 256 input
        fused = []
        for feat in fpn_feats[:4]:  # use only 4 levels
            if feat.shape[2:] != p2_size:
                feat = F.interpolate(feat, size=p2_size, mode='bilinear', align_corners=False)
            fused.append(feat)
        fused = torch.cat(fused, dim=1)  # (B, fpn_channels*4, 64, 64)
        fused = self.fusion_conv(fused)   # (B, decoder_channels, 64, 64)

        # ---- FiLM Timestep Conditioning ----
        fused = self.film(fused, t_emb)

        # ---- Decoder Refinement ----
        h = fused
        for block in self.decoder_blocks:
            h = block(h)

        # ---- Upsample to full resolution ----
        h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)

        # ---- Output Head ----
        logits = self.output_head(h)  # (B, 1, 256, 256)
        soft_mask = logits.sigmoid()

        # ---- Discrete Binarization ----
        if self.use_gumbel and self.training:
            logits_2class = torch.cat([logits, -logits], dim=1)
            B_val, _, H, W = logits_2class.shape
            logits_flat = logits_2class.permute(0, 2, 3, 1).reshape(-1, 2)
            gumbel_out = F.gumbel_softmax(logits_flat, tau=1.0, hard=True)
            binary_mask = gumbel_out[:, 0].reshape(B_val, 1, H, W)
        else:
            binary_mask = ste_binarize(soft_mask)

        return binary_mask
