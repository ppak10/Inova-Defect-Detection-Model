"""
Detection model for v05 (7ch photometrics) = v03 architecture + two probe-driven changes:
- SegHead consumes a raw PHOTOMETRIC stack (standardized recoat/scan
  frames + prev-diffs at output resolution) alongside backbone
  features — pixel statistics beat DINOv2 61x-random on short feed
  because self-supervised backbones are invariant to the brightness
  cues that class lives on (runs/v02/probe_shortfeed.txt).
- A dedicated REGION head (MLP on region-pooled fused features) — a
  linear probe on pooled frozen features hit 0.336 AP on swelling
  where the dense pipeline scored ~0.06 (runs/v02/probe_curl.py).

Original v01 notes:

DefectSegmenter: frozen DINOv2 encoder applied per frame -> per-frame
metadata embeddings (phase, dt, lighting) added to patch tokens ->
per-patch attention pooling across the frame sequence (padding-masked,
so uninformative/missing frames are discounted) -> conv seg head ->
dense per-class probability maps. Region scores are derived
deterministically (masked mean over the region map), matching the
per-region pixel-fraction labels in the prepare cache.

Output contract (see CLAUDE.md "Model plan"): the dense 2D map is the
ONLY served output, delivered to the agentic system in bed/galvo
coordinates (inverse homography applied service-side on Inova); the
agent does its own map->part attribution from prescribed geometry.
region_scores() is internal — a training loss term and eval view.

Inference target: torch on a reserved RTX 6000 Ada on this machine
(GPU-primary, advisory-availability — see CLAUDE.md "Deployment").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .constants import PEREGRINE_ALL_CLASSES

BACKBONES = {
    "small": ("facebook/dinov2-small", 384),
    "base": ("facebook/dinov2-base", 768),
    "large": ("facebook/dinov2-large", 1024),
}

# Frame phase vocabulary (index = embedding id)
PHASES = ["recoat", "scan", "prev_scan"]

# ImageNet grayscale-replicated normalization (DINOv2 expects it)
IMAGENET_MEAN = 0.449
IMAGENET_STD = 0.226


@dataclass
class ModelConfig:
    backbone: str = "base"
    input_size: int = 518  # multiple of 14 (ViT patch); 518 -> 37x37 patches
    logits_size: int = 256  # dense-map output resolution (loss + contract)
    n_classes: int = len(PEREGRINE_ALL_CLASSES)
    meta_dim: int = 3  # (dt_seconds_normalized, lighting_state, valid_flag)
    head_channels: int = 256
    freeze_backbone: bool = True


class MetaEmbedding(nn.Module):
    """(phase id, continuous meta) -> token-space embedding."""

    def __init__(self, meta_dim: int, embed_dim: int):
        super().__init__()
        self.phase = nn.Embedding(len(PHASES), embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(meta_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, phase_ids: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.phase(phase_ids) + self.proj(meta)


class FrameFusion(nn.Module):
    """Per-patch attention pooling across the K-frame sequence."""

    def __init__(self, embed_dim: int, n_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor, frame_valid: torch.Tensor) -> torch.Tensor:
        """tokens (B, K, P, D); frame_valid (B, K) bool -> fused (B, P, D)."""
        b, k, p, d = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(b * p, k, d)
        pad = ~frame_valid.repeat_interleave(p, dim=0)  # (B*P, K)
        q = self.query.expand(b * p, 1, d)
        fused, _ = self.attn(q, x, x, key_padding_mask=pad)
        # residual toward the valid-frame mean keeps gradients healthy
        mean = (tokens * frame_valid[:, :, None, None]).sum(1) / frame_valid.sum(
            1
        ).clamp(min=1)[:, None, None]
        return self.norm(fused.reshape(b, p, d) + mean)


N_PHOTOMETRIC = 7  # std recoat/scan, 2 diffs, box-std, 2 box-|diff| maps


class SegHead(nn.Module):
    """Patch grid (+ part mask) -> features; fuse full-res photometric
    stack; -> dense class logits."""

    def __init__(self, embed_dim: int, channels: int, n_classes: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim + 1, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GroupNorm(16, channels // 2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels // 2 + N_PHOTOMETRIC + 1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, n_classes, 1),
        )

    def forward(
        self, grid: torch.Tensor, part_mask: torch.Tensor, photometric: torch.Tensor
    ) -> torch.Tensor:
        m = F.interpolate(part_mask[:, None].float(), size=grid.shape[-2:], mode="nearest")
        feat = self.net(torch.cat([grid, m], dim=1))
        feat = F.interpolate(
            feat, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
        )
        m_out = F.interpolate(
            part_mask[:, None].float(), size=(self.out_size, self.out_size), mode="nearest"
        )
        return self.fuse(torch.cat([feat, photometric, m_out], dim=1))


class RegionHead(nn.Module):
    """MLP on region-pooled fused features -> per-region class logits."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, n_classes)
        )

    def forward(self, grid: torch.Tensor, region_map: torch.Tensor, n_regions: int) -> torch.Tensor:
        """grid (B,D,s,s); region_map (B,L,L) -> (B, n_regions, n_classes)."""
        b, d, s, _ = grid.shape
        onehot = F.one_hot(region_map.clamp(min=0), n_regions).float()
        onehot = onehot * (region_map >= 0)[..., None]
        w = F.adaptive_avg_pool2d(onehot.permute(0, 3, 1, 2), s)  # (B,R,s,s)
        w = w / w.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        pooled = torch.einsum("brhw,bdhw->brd", w, grid)
        return self.mlp(pooled)


class DefectSegmenter(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        from transformers import AutoModel

        self.cfg = cfg = cfg or ModelConfig()
        name, embed_dim = BACKBONES[cfg.backbone]
        self.backbone = AutoModel.from_pretrained(name)
        if cfg.freeze_backbone:
            self.backbone.requires_grad_(False)
        self.meta_embed = MetaEmbedding(cfg.meta_dim, embed_dim)
        self.fusion = FrameFusion(embed_dim)
        self.head = SegHead(embed_dim, cfg.head_channels, cfg.n_classes, cfg.logits_size)
        self.region_head = RegionHead(embed_dim, cfg.n_classes)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, K, H, W) standardized grayscale -> tokens (B, K, P, D)."""
        b, k, h, w = frames.shape
        # replicate to 3ch and renormalize from per-image-standardized to
        # ImageNet stats (standardized input ~N(0,1) -> shift/scale)
        x = frames.reshape(b * k, 1, h, w).repeat(1, 3, 1, 1)
        x = x * IMAGENET_STD + (0.5 - IMAGENET_MEAN)  # keep contrast, recenter
        if (h, w) != (self.cfg.input_size, self.cfg.input_size):
            x = F.interpolate(
                x,
                size=(self.cfg.input_size, self.cfg.input_size),
                mode="bilinear",
                align_corners=False,
            )
        ctx = torch.no_grad() if self.cfg.freeze_backbone else torch.enable_grad()
        with ctx:
            out = self.backbone(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS
        return out.reshape(b, k, out.shape[1], out.shape[2])

    def forward(
        self,
        frames: torch.Tensor,  # (B, K, H, W) per-image standardized
        phase_ids: torch.Tensor,  # (B, K) int in range(len(PHASES))
        meta: torch.Tensor,  # (B, K, meta_dim)
        frame_valid: torch.Tensor,  # (B, K) bool
        part_mask: torch.Tensor,  # (B, H', W') bool
        photometric: torch.Tensor,  # (B, 4, logits_size, logits_size)
        region_map: torch.Tensor | None = None,  # (B, L, L) for region head
        n_regions: int = 0,
    ):
        tokens = self.encode_frames(frames)
        tokens = tokens + self.meta_embed(phase_ids, meta)[:, :, None, :]
        fused = self.fusion(tokens, frame_valid)  # (B, P, D)
        p = fused.shape[1]
        side = int(p**0.5)
        grid = fused.permute(0, 2, 1).reshape(-1, fused.shape[2], side, side)
        dense = self.head(grid, part_mask, photometric)
        if region_map is None:
            return dense
        return dense, self.region_head(grid, region_map, n_regions)


def region_scores(
    probs: torch.Tensor, region_map: torch.Tensor, n_regions: int
) -> torch.Tensor:
    """Masked mean of per-pixel probs per region — the predicted pixel
    fraction, directly comparable to the frac_<class> labels.

    probs (B, C, H, W); region_map (B, H, W) int, -1 = none ->
    (B, n_regions, C). Implemented as normalized one-hot matmul (ONNX-
    friendly, no scatter).
    """
    b, c, h, w = probs.shape
    rm = F.interpolate(
        region_map[:, None].float(), size=(h, w), mode="nearest"
    ).long()[:, 0]
    onehot = F.one_hot(rm.clamp(min=0), n_regions).float()  # (B, H, W, R)
    onehot = onehot * (rm >= 0)[..., None]
    flat = onehot.reshape(b, h * w, n_regions)
    denom = flat.sum(dim=1).clamp(min=1.0)  # (B, R)
    pooled = torch.bmm(flat.transpose(1, 2), probs.reshape(b, c, h * w).transpose(1, 2))
    return pooled / denom[..., None]
