"""
Fits and saves the pixstat short-feed detector — the 6-statistic
logistic that beats every neural attempt on incomplete_spreading
(probe: 0.578 AP vs 0.386-0.543; see runs/v05/README.md).

Per region (part or powder tile): [mean, std of standardized post-recoat
frame; mean, std of (recoat - prev_scan); mean, max of |diff|], fit on
ALL builds (deployment artifact — the honest eval number is the
build-3 probe). Stats computed at cache-native 1024 px — serve must match (upscale Inova frames to ~1024).

Writes runs/v05/pixstat_shortfeed.npz {w, b, mu, sd, train_ap}.

Usage: uv run python -m runs.v05.fit_pixstat [--max-layers 3000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .constants import PEREGRINE_CACHE
from .dataset import standardize
from .trainer import average_precision

OUT = Path(__file__).parent / "pixstat_shortfeed.npz"
COL = "frac_incomplete_spreading"
THR = 0.05
SIZE = 1024  # cache-native: the show-through cue is high-frequency; downsampling kills it (518 fit scored 0.145)


def region_stats(cur: np.ndarray, prev: np.ndarray, rmap: np.ndarray, n: int) -> np.ndarray:
    """(n, 6) stats via vectorized per-region reductions; NaN rows for
    empty regions."""
    d = cur - prev
    flat_r = rmap.reshape(-1)
    ok = flat_r >= 0
    r = flat_r[ok]
    vals = {k: v.reshape(-1)[ok] for k, v in
            dict(c=cur, c2=cur**2, d=d, d2=d**2, ad=np.abs(d)).items()}
    cnt = np.bincount(r, minlength=n).astype(np.float64)
    s = {k: np.bincount(r, weights=v, minlength=n) for k, v in vals.items()}
    admax = np.zeros(n)
    np.maximum.at(admax, r, vals["ad"])
    with np.errstate(invalid="ignore", divide="ignore"):
        m_c, m_d, m_ad = s["c"] / cnt, s["d"] / cnt, s["ad"] / cnt
        v_c = np.sqrt(np.clip(s["c2"] / cnt - m_c**2, 0, None))
        v_d = np.sqrt(np.clip(s["d2"] / cnt - m_d**2, 0, None))
    out = np.stack([m_c, v_c, m_d, v_d, m_ad, admax], axis=1)
    out[cnt < 16] = np.nan
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-layers", type=int, default=3000)
    args = p.parse_args()

    r = pd.read_parquet(PEREGRINE_CACHE / "regions.parquet")
    by_layer = r.groupby(["build", "layer"])[COL].max()
    pos = [k for k, v in by_layer.items() if v > THR]
    neg = [k for k, v in by_layer.items() if v <= THR]
    rng = np.random.default_rng(0)
    rng.shuffle(neg)
    layers = pos + neg[: max(args.max_layers - len(pos), len(pos))]
    layers = [
        (b, l) for b, l in layers
        if l > 0 and (PEREGRINE_CACHE / f"build_{b}_layer_{l - 1:04d}.npz").exists()
    ]
    print(f"{len(layers)} layers ({len(pos)} short-feed-positive)")

    grouped = r.set_index(["build", "layer", "region_idx"])[COL]
    X, y = [], []
    for i, (b, l) in enumerate(layers):
        with np.load(PEREGRINE_CACHE / f"build_{b}_layer_{l:04d}.npz") as z:
            cur = z["images"][0].astype(np.float32)
            rmap = z["region_map"].astype(np.int64)
        with np.load(PEREGRINE_CACHE / f"build_{b}_layer_{l - 1:04d}.npz") as zp:
            prev = zp["images"][1].astype(np.float32)
        n = int(rmap.max()) + 1
        if n <= 0:
            continue
        stats = region_stats(standardize(cur), standardize(prev), rmap, n)
        fr = grouped.loc[b, l]
        for ridx, frac in fr.items():
            if ridx < n and np.isfinite(stats[ridx]).all():
                X.append(stats[ridx])
                y.append(float(frac > THR))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(layers)} layers, {len(y)} regions")

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y_t = torch.tensor(np.array(y), dtype=torch.float32)
    print(f"regions {len(y_t)}, positives {int(y_t.sum())}")

    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    Xn = (X - mu) / sd
    lin = torch.nn.Linear(6, 1)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    pos_w = torch.tensor([(len(y_t) - y_t.sum()) / y_t.sum()])
    for _ in range(3000):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lin(Xn)[:, 0], y_t, pos_weight=pos_w)
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = lin(Xn)[:, 0].numpy()
    ap = average_precision(scores, y_t.numpy() > 0.5)
    print(f"train AP {ap:.4f} (honest build-3 probe number: 0.578)")

    np.savez(
        OUT,
        w=lin.weight.detach().numpy()[0],
        b=float(lin.bias.detach()),
        mu=mu.numpy(),
        sd=sd.numpy(),
        train_ap=ap,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
