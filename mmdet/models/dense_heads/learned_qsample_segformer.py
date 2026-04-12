# -----------------------------------------------------------------------------------
# LearnedQSampleSegFormer: MiT-B5 + SegFormer MLP Decode Head for learned q-sample.
#
# Reuses PyramidVisionTransformerV2 (PVTv2) as MiT-B5 backbone — they are
# architecturally identical: overlapping patch embeddings, Efficient Self-Attention
# with spatial reduction, and Mix-FFN with depth-wise convolution.
#
# The SegFormer MLP decode head follows the official design:
#   1. Each scale → Linear(C_i, embed_dim) to unify channel dimensions
#   2. Upsample all to 1/4 resolution (64×64 for 256×256 input)
#   3. Concatenate → Linear(4*embed_dim, embed_dim)
#   4. FiLM timestep conditioning
#   5. Linear(embed_dim, 1) → Upsample → Sigmoid → STE
#
# Architecture:
#   ┌────────────────────────────────────────────────────────────────────┐
#   │  Input: cat(RGB, GT mask) = (B, 4, 256, 256) + timestep t        │
#   ├────────────────────────────────────────────────────────────────────┤
#   │  MiT-B5 Backbone (PVTv2 with MiT-B5 hyperparameters)             │
#   │    F1: (B,  64, 64, 64)  ← 1/4                                   │
#   │    F2: (B, 128, 32, 32)  ← 1/8                                   │
#   │    F3: (B, 320, 16, 16)  ← 1/16                                  │
#   │    F4: (B, 512,  8,  8)  ← 1/32                                  │
#   ├────────────────────────────────────────────────────────────────────┤
#   │  SegFormer MLP Decode Head                                        │
#   │    Linear proj each scale → embed_dim=768                         │
#   │    Upsample all to F1 size (64×64)                                │
#   │    Concat → (B, 768*4, 64, 64) → Linear → (B, 768, 64, 64)      │
#   │    FiLM timestep conditioning                                     │
#   │    MLP → (B, 1, 64, 64) → Upsample → (B, 1, 256, 256)           │
#   │    Sigmoid → STE binarize                                         │
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
from ..builder import HEADS, build_backbone


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
# FiLM Modulation
# ============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for timestep conditioning."""

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
        """
        params = self.fc(t_emb)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, C)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


# ============================================================================
# SegFormer MLP Decode Head
# ============================================================================

class SegFormerMLPHead(nn.Module):
    """
    Lightweight All-MLP Decode Head from SegFormer.

    Takes multi-scale features from the backbone, projects each to a unified
    embedding dimension, upsamples to the largest feature map size, concatenates,
    and fuses through a linear layer.

    This follows the official SegFormer implementation:
      Xie et al., "SegFormer: Simple and Efficient Design for Semantic
      Segmentation with Transformers", NeurIPS 2021.

    Args:
        in_channels (list[int]): Input channel dimensions from each backbone stage.
        embed_dim (int): Unified embedding dimension for all scales. Default: 768.
        out_channels (int): Output channels. Default: 1.
    """

    def __init__(self, in_channels, embed_dim=768, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_scales = len(in_channels)

        # Per-scale linear projection: Conv1x1 to unify channels
        self.linear_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for ch in in_channels
        ])

        # Fusion: concatenated multi-scale → unified embedding
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_scales, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, 1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, out_channels, 1),
        )

    def forward(self, features):
        """
        Args:
            features: list of (B, C_i, H_i, W_i) multi-scale feature maps

        Returns:
            (B, out_channels, H_0, W_0) fused output at largest scale resolution
        """
        target_size = features[0].shape[2:]  # largest feature map (1/4 resolution)

        projected = []
        for i, feat in enumerate(features):
            p = self.linear_projections[i](feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            projected.append(p)

        # Concatenate and fuse
        fused = torch.cat(projected, dim=1)  # (B, embed_dim * num_scales, H, W)
        fused = self.fusion(fused)            # (B, embed_dim, H, W)

        # Output
        out = self.output_conv(fused)  # (B, out_channels, H, W)
        return out, fused


# ============================================================================
# Main Model: LearnedQSampleSegFormerHeader
# ============================================================================

@HEADS.register_module()
class LearnedQSampleSegFormerHeader(BaseModule):
    """
    MiT-B5 backbone + SegFormer MLP Decode Head for learned q-sample.

    Uses PyramidVisionTransformerV2 as the MiT-B5 backbone (they share the
    same architecture: overlapping patch embeddings, efficient self-attention
    with spatial reduction, and Mix-FFN with depth-wise convolution).

    The first patch embedding is extended from 3→4 input channels to accept
    the concatenated (RGB, GT mask) input.

    Args:
        backbone (dict): Config for PVTv2 backbone with MiT-B5 parameters.
        in_channels_list (list[int]): Per-stage output channels of the backbone.
            Default for MiT-B5: [64, 128, 320, 512].
        embed_dim (int): Unified embedding dimension in MLP head. Default: 768.
        t_dim (int): Timestep embedding dimension. Default: 256.
        use_gumbel (bool): Use Gumbel-Softmax instead of STE. Default: False.
    """

    def __init__(
        self,
        backbone,
        in_channels_list=(64, 128, 320, 512),
        embed_dim=768,
        t_dim=256,
        num_timesteps=6,
        use_gumbel=False,
        intermediate_channels=None,
        **kwargs,
    ):
        super().__init__()

        self.t_dim = t_dim
        self.num_timesteps = num_timesteps
        if intermediate_channels is not None:
            embed_dim = intermediate_channels
        self.embed_dim = embed_dim
        self.use_gumbel = use_gumbel

        # ---- MiT-B5 Backbone (PVTv2) ----
        self.backbone = build_backbone(backbone)

        # ---- Timestep Embedding MLP ----
        self.time_embed = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # ---- SegFormer MLP Decode Head ----
        self.decode_head = SegFormerMLPHead(
            in_channels=list(in_channels_list),
            embed_dim=embed_dim,
            out_channels=1,
        )

        # ---- FiLM Conditioning (applied on fused features) ----
        self.film = FiLMLayer(t_dim, embed_dim)

        # ---- Post-FiLM refinement ----
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, 1),
        )

    def init_weights(self):
        """Initialize weights and extend first patch embedding from 3→4 channels."""
        super().init_weights()
        self._extend_first_patch_embed()

    def _extend_first_patch_embed(self):
        """Extend the first PatchEmbed conv from 3→4 input channels.

        Copies pre-trained 3-channel weights and zero-initializes the 4th
        channel (GT mask), preserving all pre-trained features.
        """
        # PVT stores stages in self.backbone.layers
        # Each stage is [patch_embed, encoder_layers, norm]
        # The first stage's patch_embed has a .projection Conv2d
        if not hasattr(self.backbone, 'layers') or len(self.backbone.layers) == 0:
            return

        patch_embed = self.backbone.layers[0][0]  # first stage's PatchEmbed
        if not hasattr(patch_embed, 'projection'):
            return

        old_conv = patch_embed.projection
        if old_conv.in_channels == 4:
            return  # already extended
        if old_conv.in_channels != 3:
            return  # unexpected

        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0.0
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        patch_embed.projection = new_conv

    def forward(self, object_img, object_gt_masks, t):
        """
        Forward pass.

        Args:
            object_img:      (B, 3, 256, 256) normalized RGB image
            object_gt_masks: (B, 1, 256, 256) ground-truth binary mask
            t:               (B,) integer timestep indices

        Returns:
            (B, 1, 256, 256) discrete binary noisy mask
        """
        target_size = object_img.shape[2:]  # (256, 256)

        # ---- Concatenate image and GT mask ----
        x = torch.cat([object_img, object_gt_masks], dim=1)  # (B, 4, 256, 256)

        # ---- Timestep embedding ----
        t_emb = self.time_embed(timestep_embedding(t, self.t_dim))  # (B, t_dim)

        # ---- Backbone: extract multi-scale features ----
        if hasattr(self.backbone, 'forward') and 'return_feats' in self.backbone.forward.__code__.co_varnames:
            feats = self.backbone(x, return_feats=True)
        else:
            feats = self.backbone(x)
        # MiT-B5 outputs: [(B,64,64,64), (B,128,32,32), (B,320,16,16), (B,512,8,8)]

        # ---- SegFormer MLP Decode Head ----
        # Returns logits at 1/4 resolution and the fused feature map
        coarse_logits, fused_features = self.decode_head(feats)
        # coarse_logits: (B, 1, 64, 64), fused_features: (B, embed_dim, 64, 64)

        # ---- FiLM Timestep Conditioning ----
        conditioned = self.film(fused_features, t_emb)  # (B, embed_dim, 64, 64)

        # ---- Post-FiLM refinement ----
        refined_logits = self.refine(conditioned)  # (B, 1, 64, 64)

        # ---- Combine decode head output and refined output ----
        logits = coarse_logits + refined_logits  # residual addition

        # ---- Upsample to full resolution ----
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

        # ---- Sigmoid + Binarization ----
        soft_mask = logits.sigmoid()

        if self.use_gumbel and self.training:
            logits_2class = torch.cat([logits, -logits], dim=1)
            B_val, _, H, W = logits_2class.shape
            logits_flat = logits_2class.permute(0, 2, 3, 1).reshape(-1, 2)
            gumbel_out = F.gumbel_softmax(logits_flat, tau=1.0, hard=True)
            binary_mask = gumbel_out[:, 0].reshape(B_val, 1, H, W)
        else:
            binary_mask = ste_binarize(soft_mask)

        return binary_mask
