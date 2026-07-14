"""
Datasets for v02 = v01 + the PREV_SCAN temporal frame.

Elevation-type defects (swelling, super_elevation ≈ curl) are
persistent physical states, not per-layer events — v01's single-layer
sequences gave the model no way to see "this part has been proud of the
powder plane for N layers". The full-build cache has consecutive
layers, so each sequence now leads with the PREVIOUS layer's after_melt
view (phase id 2 = prev_scan in model.PHASES), degraded like every
other frame at train time, marked invalid where no previous layer is
cached (layer 0 / gaps).

Also: the dataset now exposes `self.regions` so the trainer can compute
per-class REGION loss weights (v01 weighted only the pixel term).

Frames are produced at `frame_size` (default 518 in v02 train.py — the
model resizes to its input anyway; augmenting at 1024 was pure CPU
waste, ~4x the dataloader cost for zero quality gain).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .constants import (
    PEREGRINE_ALL_CLASSES,
    PEREGRINE_CACHE,
    PEREGRINE_CLASSES,
)

FRAC_COLS = [f"frac_{c}" for c in PEREGRINE_CLASSES]
PHASE_RECOAT_ID, PHASE_SCAN_ID, PHASE_PREV_SCAN_ID = 0, 1, 2


def standardize(img: np.ndarray) -> np.ndarray:
    std = float(img.std())
    return (img - float(img.mean())) / (std if std > 1e-6 else 1.0)


def synth_halogen_frame(
    img: np.ndarray, rng: np.random.Generator, add_laser_spot: bool
) -> tuple[np.ndarray, float]:
    """Degrade a clean 0-255 grayscale frame toward measured Inova
    chamber conditions (see repo CLAUDE.md). Returns (frame, state)."""
    h, w = img.shape
    state = float(rng.beta(1.2, 1.2))
    gain = 0.02 + state * 1.9
    out = img.astype(np.float32) * gain

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    ramp = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    out += state * rng.uniform(0, 120) * (1 - ramp)

    haze = rng.uniform(0, 0.6)
    out = out * (1 - haze) + haze * rng.uniform(80, 180)

    if add_laser_spot and rng.random() < 0.5:
        sx, sy = rng.uniform(0, w), rng.uniform(0, h)
        spot = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * rng.uniform(2, 6) ** 2)))
        out += 255.0 * spot

    out += rng.normal(0, rng.uniform(1, 8), out.shape).astype(np.float32)
    return np.clip(out, 0, 255), state


class PeregrineSequenceDataset(Dataset):
    """One item per cached layer; prev_scan frame + K-1 current frames."""

    def __init__(
        self,
        builds: list[int] | None = None,
        k_frames: int = 5,
        frame_size: int = 518,
        logits_size: int = 256,
        augment: bool = True,
        seed: int = 0,
    ):
        regions = pd.read_parquet(PEREGRINE_CACHE / "regions.parquet")
        if builds is not None:
            regions = regions[regions["build"].isin(builds)]
        self.regions = regions
        self.layers = regions.sort_values(
            ["build", "layer", "region_idx"]
        ).groupby(["build", "layer"])
        self.keys = sorted(self.layers.groups.keys())
        self.key_set = set(self.keys)
        self.k = k_frames
        self.frame_size = frame_size
        self.logits_size = logits_size
        self.augment = augment
        self.seed = seed
        self.n_classes = len(PEREGRINE_ALL_CLASSES)

    def __len__(self) -> int:
        return len(self.keys)

    def _npz(self, build: int, layer: int):
        return np.load(PEREGRINE_CACHE / f"build_{build}_layer_{layer:04d}.npz")

    def _resize(self, img: np.ndarray) -> np.ndarray:
        s = self.frame_size
        if img.shape[-1] == s:
            return img
        t = torch.from_numpy(img[None, None].astype(np.float32))
        return F.interpolate(t, size=(s, s), mode="bilinear")[0, 0].numpy()

    def _dense_target(self, class_bits: np.ndarray) -> torch.Tensor:
        bits = torch.from_numpy(class_bits.astype(np.int64))
        maps = torch.stack(
            [(bits >> i) & 1 for i in range(self.n_classes)]
        ).float()
        return F.adaptive_max_pool2d(maps[None], self.logits_size)[0]

    def __getitem__(self, i: int) -> dict:
        build, layer = self.keys[i]
        g = self.layers.get_group((build, layer))
        with self._npz(build, layer) as z:
            images = z["images"].astype(np.float32)
            part_mask = z["part_mask"]
            region_map = z["region_map"].astype(np.int64)
            class_bits = z["class_bits"]

        prev_melt = None
        prev_path = PEREGRINE_CACHE / f"build_{build}_layer_{layer - 1:04d}.npz"
        if layer > 0 and prev_path.exists():
            with np.load(prev_path) as zp:
                prev_melt = zp["images"][1].astype(np.float32)

        rng = np.random.default_rng(self.seed + i if not self.augment else None)

        recoat = self._resize(images[0])
        melt = self._resize(images[1])
        prev = self._resize(prev_melt) if prev_melt is not None else None

        flips: tuple[bool, bool] = (False, False)
        if self.augment:
            flips = (bool(rng.random() < 0.5), bool(rng.random() < 0.5))
            axes = [ax for ax, f in zip((0, 1), flips) if f]
            if axes:
                recoat = np.flip(recoat, axis=axes).copy()
                melt = np.flip(melt, axis=axes).copy()
                if prev is not None:
                    prev = np.flip(prev, axis=axes).copy()
                part_mask = np.flip(part_mask, axis=axes).copy()
                region_map = np.flip(region_map, axis=axes).copy()
                class_bits = np.flip(class_bits, axis=axes).copy()

        frames, phases, meta, valid = [], [], [], []

        # frame 0: previous layer's after_melt (temporal context)
        if prev is not None:
            if self.augment:
                f0, state = synth_halogen_frame(prev, rng, add_laser_spot=False)
            else:
                f0, state = prev, 0.6
            frames.append(standardize(f0))
            meta.append((-1.0, state, 1.0))
            valid.append(True)
        else:
            frames.append(np.zeros_like(recoat))
            meta.append((0.0, 0.0, 0.0))
            valid.append(False)
        phases.append(PHASE_PREV_SCAN_ID)

        # frames 1..K-1: current layer, recoat then scan
        for j in range(1, self.k):
            phase = PHASE_RECOAT_ID if j <= (self.k - 1) // 2 else PHASE_SCAN_ID
            src = recoat if phase == PHASE_RECOAT_ID else melt
            if self.augment:
                is_valid = bool(rng.random() > 0.15) or j == self.k - 1
                frame, state = synth_halogen_frame(
                    src, rng, add_laser_spot=phase == PHASE_SCAN_ID
                )
                dt = float(rng.uniform(0, 1)) if phase == PHASE_RECOAT_ID else float(
                    rng.uniform(-0.3, 0)
                )
                frames.append(standardize(frame) if is_valid else np.zeros_like(frame))
                meta.append((dt, state, float(is_valid)))
                valid.append(is_valid)
            else:
                if j <= 2:
                    frames.append(standardize(src))
                    meta.append((0.5 if phase == PHASE_RECOAT_ID else 0.0, 0.6, 1.0))
                    valid.append(True)
                else:
                    frames.append(np.zeros_like(src))
                    meta.append((0.0, 0.0, 0.0))
                    valid.append(False)
            phases.append(phase)

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
