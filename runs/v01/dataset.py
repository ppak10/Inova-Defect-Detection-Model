"""
Datasets for v01.

PeregrineSequenceDataset serves layer samples from the prepare.py cache
as synthetic K-frame sequences — the halogen-state augmentation that
makes the frame-fusion module trainable on Peregrine's clean single
images (see CLAUDE.md "Measured imaging reality"). Each frame is a
degraded view of after_powder (phase=recoat) or after_melt (phase=scan)
with a sampled lighting state; the sampled state is passed as the
frame's metadata, so the model learns the wattage->appearance
association that is real telemetry on the Inova.

Item contents:
    frames      float32 (K, S, S)   — per-image standardized AFTER
                                      degradation (S = frame_size)
    phase_ids   int64   (K,)        — index into model.PHASES
    meta        float32 (K, 3)      — (dt_norm, lighting_state, valid)
    frame_valid bool    (K,)        — False = missing frame (padding)
    target      float32 (C, L, L)   — dense multi-label maps at
                                      logits_size L (max-pool "any
                                      coverage" downsample of class_bits)
    part_mask   bool    (S, S)
    region_map  int64   (L, L)      — for the region-fraction loss term
    labels      float32 (R, 10)     — per-region anomaly pixel fractions
    region_valid bool   (R,)

R varies per layer — use collate_layers as the DataLoader collate_fn.
Validation uses a clean, deterministic 2-frame sequence (no
degradation) so val metrics are comparable across epochs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .constants import (
    DATA_DIR,
    PEREGRINE_ALL_CLASSES,
    PEREGRINE_CLASSES,
)

PEREGRINE_CACHE = DATA_DIR / "peregrine"
FRAC_COLS = [f"frac_{c}" for c in PEREGRINE_CLASSES]
PHASE_RECOAT_ID, PHASE_SCAN_ID = 0, 1  # must match model.PHASES order


def standardize(img: np.ndarray) -> np.ndarray:
    """Per-image standardization — the cross-domain normalizer."""
    std = float(img.std())
    return (img - float(img.mean())) / (std if std > 1e-6 else 1.0)


def synth_halogen_frame(
    img: np.ndarray, rng: np.random.Generator, add_laser_spot: bool
) -> tuple[np.ndarray, float]:
    """Degrade a clean 0-255 grayscale frame toward measured Inova
    chamber conditions (CLAUDE.md "Measured imaging reality"): lighting
    swings black <-> hazy <-> blown-out, smooth glare gradients, haze
    (contrast loss), sensor noise, and the glowing laser spot during
    scanning. Returns (frame, lighting_state in [0,1])."""
    h, w = img.shape
    # lighting state: 0 = halogens off (black) ... 1 = full glare
    state = float(rng.beta(1.2, 1.2))
    gain = 0.02 + state * 1.9  # ~black at 0, blown-out beyond ~1.3
    out = img.astype(np.float32) * gain

    # smooth glare/illumination gradient (halogens are off-axis)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    ramp = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    out += state * rng.uniform(0, 120) * (1 - ramp)

    # haze: contrast compression toward a gray level
    haze = rng.uniform(0, 0.6)
    out = out * (1 - haze) + haze * rng.uniform(80, 180)

    if add_laser_spot and rng.random() < 0.5:
        sx, sy = rng.uniform(0, w), rng.uniform(0, h)
        spot = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * rng.uniform(2, 6) ** 2)))
        out += 255.0 * spot

    out += rng.normal(0, rng.uniform(1, 8), out.shape).astype(np.float32)
    return np.clip(out, 0, 255), state


class PeregrineSequenceDataset(Dataset):
    """One item per cached layer; K synthetic frames + dense targets."""

    def __init__(
        self,
        builds: list[int] | None = None,
        k_frames: int = 4,
        frame_size: int = 1024,
        logits_size: int = 256,
        augment: bool = True,
        seed: int = 0,
    ):
        regions = pd.read_parquet(PEREGRINE_CACHE / "regions.parquet")
        if builds is not None:
            regions = regions[regions["build"].isin(builds)]
        self.layers = regions.sort_values(
            ["build", "layer", "region_idx"]
        ).groupby(["build", "layer"])
        self.keys = sorted(self.layers.groups.keys())
        self.k = k_frames
        self.frame_size = frame_size
        self.logits_size = logits_size
        self.augment = augment
        self.seed = seed
        self.n_classes = len(PEREGRINE_ALL_CLASSES)

    def __len__(self) -> int:
        return len(self.keys)

    def _load(self, build: int, layer: int):
        with np.load(PEREGRINE_CACHE / f"build_{build}_layer_{layer:04d}.npz") as z:
            return (
                z["images"].astype(np.float32),
                z["part_mask"],
                z["region_map"].astype(np.int64),
                z["class_bits"],
            )

    def _dense_target(self, class_bits: np.ndarray) -> torch.Tensor:
        bits = torch.from_numpy(class_bits.astype(np.int64))
        maps = torch.stack(
            [(bits >> i) & 1 for i in range(self.n_classes)]
        ).float()  # (C, 1024, 1024)
        # "any coverage" downsample = max-pool, keeps thin streaks
        return F.adaptive_max_pool2d(maps[None], self.logits_size)[0]

    def __getitem__(self, i: int) -> dict:
        build, layer = self.keys[i]
        g = self.layers.get_group((build, layer))
        images, part_mask, region_map, class_bits = self._load(build, layer)
        rng = np.random.default_rng(
            self.seed + i if not self.augment else None
        )

        s = self.frame_size
        if images.shape[-1] != s:
            t = torch.from_numpy(images)[None]
            images = F.interpolate(t, size=(s, s), mode="bilinear")[0].numpy()

        frames, phases, meta, valid = [], [], [], []
        if self.augment:
            # random flips shared by frames/targets (streaks stay axis-aligned)
            flips = (bool(rng.random() < 0.5), bool(rng.random() < 0.5))
            axes = [ax for ax, f in zip((0, 1), flips) if f]
            if axes:
                images = np.flip(images, axis=[a + 1 for a in axes]).copy()
                part_mask = np.flip(part_mask, axis=axes).copy()
                region_map = np.flip(region_map, axis=axes).copy()
                class_bits = np.flip(class_bits, axis=axes).copy()
            for j in range(self.k):
                phase = PHASE_RECOAT_ID if j < self.k // 2 else PHASE_SCAN_ID
                is_valid = bool(rng.random() > 0.15) or j == self.k - 1
                src = images[0 if phase == PHASE_RECOAT_ID else 1]
                frame, state = synth_halogen_frame(
                    src, rng, add_laser_spot=phase == PHASE_SCAN_ID
                )
                dt = float(rng.uniform(0, 1)) if phase == PHASE_RECOAT_ID else float(
                    rng.uniform(-0.3, 0)
                )
                frames.append(standardize(frame) if is_valid else np.zeros_like(frame))
                phases.append(phase)
                meta.append((dt, state, float(is_valid)))
                valid.append(is_valid)
        else:
            # deterministic clean pair, padded to K
            for j in range(self.k):
                if j < 2:
                    frames.append(standardize(images[j]))
                    phases.append((PHASE_RECOAT_ID, PHASE_SCAN_ID)[j])
                    meta.append((0.5 - 0.5 * j, 0.6, 1.0))
                    valid.append(True)
                else:
                    frames.append(np.zeros_like(images[0]))
                    phases.append(PHASE_SCAN_ID)
                    meta.append((0.0, 0.0, 0.0))
                    valid.append(False)

        n_regions = int(g["region_idx"].max()) + 1
        labels = np.zeros((n_regions, len(PEREGRINE_CLASSES)), dtype=np.float32)
        region_valid = np.zeros(n_regions, dtype=bool)
        idx = g["region_idx"].to_numpy()
        labels[idx] = g[FRAC_COLS].to_numpy(dtype=np.float32)
        region_valid[idx] = True

        rm = torch.from_numpy(region_map)[None, None].float()
        rm = F.interpolate(rm, size=(self.logits_size,) * 2, mode="nearest")[0, 0].long()

        return {
            "frames": torch.from_numpy(np.stack(frames)),
            "phase_ids": torch.tensor(phases, dtype=torch.int64),
            "meta": torch.tensor(meta, dtype=torch.float32),
            "frame_valid": torch.tensor(valid, dtype=torch.bool),
            "target": self._dense_target(class_bits),
            "part_mask": torch.from_numpy(part_mask),
            "region_map": rm,
            "labels": torch.from_numpy(labels),
            "region_valid": torch.from_numpy(region_valid),
            "build": build,
            "layer": layer,
        }


def collate_layers(items: list[dict]) -> dict:
    """Pad the region dimension to the batch max; stack the rest."""
    r_max = max(i["labels"].shape[0] for i in items)
    n_cls = items[0]["labels"].shape[1]
    b = len(items)
    labels = torch.zeros(b, r_max, n_cls)
    region_valid = torch.zeros(b, r_max, dtype=torch.bool)
    for j, it in enumerate(items):
        r = it["labels"].shape[0]
        labels[j, :r] = it["labels"]
        region_valid[j, :r] = it["region_valid"]
    out = {
        k: torch.stack([i[k] for i in items])
        for k in ["frames", "phase_ids", "meta", "frame_valid", "target", "part_mask", "region_map"]
    }
    out.update(
        labels=labels,
        region_valid=region_valid,
        build=[i["build"] for i in items],
        layer=[i["layer"] for i in items],
    )
    return out
