"""
Diagnostic probe (v02 post-mortem): is the curl signal present in clean
Peregrine optics at all?

Linear probes on frozen DINOv2-base features of CLEAN (unaugmented)
imagery, region-pooled over part regions, ranked by region-level AP:

    cur        — current after_melt features only
    cur+diff   — plus (current − previous-layer after_melt) features
    pixstats   — 6 raw pixel statistics per region (no backbone)

Class: swelling (the trainable curl class — super_elevation has ~10
training examples in builds 1-4 and is untrainable under this split;
see runs/v02/README.md results discussion). Split: train builds 2+4
(~42k positive part regions), TEST BUILD 3 (1,929 positives — build 5's
91 make it useless for this class). Interpretation:
- cur+diff >> cur      -> temporal difference carries the cue; v03's
                          correlated-augmentation fix is justified.
- both ~ pixstats ~ 0  -> cue not in top-down optics at this res; go
                          to higher input res or thermal-side transfer.

Usage: uv run python -m runs.v02.probe_curl [--max-layers 2400]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .constants import PEREGRINE_CACHE
from .dataset import standardize
from .trainer import average_precision

THR = 0.05

# per-class probe config: label col, current-image index (0=after_powder
# post-recoat, 1=after_melt post-scan), region kinds, split.
# short feed: the physical cue is prev raster showing through fresh
# powder -> cur = post-RECOAT view, positives live in POWDER regions.
CONFIGS = {
    "swelling": dict(col="frac_swelling", cur_idx=1, kinds=("part",),
                     train_builds=[2, 4], test_build=3),
    "incomplete_spreading": dict(col="frac_incomplete_spreading", cur_idx=0,
                                 kinds=("part", "powder"),
                                 train_builds=[1, 2, 4], test_build=3),
}


def pick_layers(regions: pd.DataFrame, builds: list[int], n: int, rng, cfg) -> list[tuple[int, int]]:
    parts = regions[regions["kind"].isin(cfg["kinds"]) & regions["build"].isin(builds)]
    by_layer = parts.groupby(["build", "layer"])[cfg["col"]].max()
    pos = [k for k, v in by_layer.items() if v > THR]
    neg = [k for k, v in by_layer.items() if v <= THR]
    rng.shuffle(pos), rng.shuffle(neg)
    keep = pos[: n // 2] + neg[: n // 2]
    # prev layer must be cached
    return [
        (b, l) for b, l in keep
        if l > 0 and (PEREGRINE_CACHE / f"build_{b}_layer_{l - 1:04d}.npz").exists()
    ]


@torch.no_grad()
def extract(layers, backbone, device, cfg, batch: int = 16):
    """Per part region: pooled cur features, diff features, pixel stats,
    label fraction."""
    feats_cur, feats_diff, pixstats, fracs = [], [], [], []
    regions = pd.read_parquet(PEREGRINE_CACHE / "regions.parquet")
    regions = regions[regions["kind"].isin(cfg["kinds"])].set_index(["build", "layer", "region_idx"])

    for i in range(0, len(layers), batch):
        chunk = layers[i : i + batch]
        imgs, prevs, rmaps, keys = [], [], [], []
        for b, l in chunk:
            with np.load(PEREGRINE_CACHE / f"build_{b}_layer_{l:04d}.npz") as z:
                cur = z["images"][cfg["cur_idx"]].astype(np.float32)
                rmap = z["region_map"].astype(np.int64)
            with np.load(PEREGRINE_CACHE / f"build_{b}_layer_{l - 1:04d}.npz") as zp:
                prev = zp["images"][1].astype(np.float32)
            imgs.append(standardize(cur))
            prevs.append(standardize(prev))
            rmaps.append(rmap)
            keys.append((b, l))

        x = torch.tensor(np.stack(imgs + prevs))[:, None].repeat(1, 3, 1, 1).to(device)
        x = x * 0.226 + (0.5 - 0.449)
        x = F.interpolate(x, size=(518, 518), mode="bilinear")
        tok = backbone(pixel_values=x).last_hidden_state[:, 1:]  # (2B, P, D)
        side = int(tok.shape[1] ** 0.5)
        grid = tok.permute(0, 2, 1).reshape(len(x), -1, side, side)
        g_cur, g_prev = grid[: len(chunk)], grid[len(chunk) :]
        g_diff = g_cur - g_prev

        for j, (b, l) in enumerate(keys):
            rmap = torch.from_numpy(rmaps[j]).to(device)
            cur_t = torch.tensor(imgs[j], device=device)
            diff_t = cur_t - torch.tensor(prevs[j], device=device)
            for ridx in torch.unique(rmap):
                r = int(ridx)
                if r < 0:
                    continue
                try:
                    frac = float(regions.loc[(b, l, r), cfg["col"]])
                except KeyError:
                    continue  # powder tile index or filtered region
                m = (rmap == r).float()[None, None]
                w = F.adaptive_avg_pool2d(m, side)[0, 0].reshape(-1)  # fractional coverage
                if w.sum() < 1e-6:
                    continue
                w = w / w.sum()
                feats_cur.append((g_cur[j].reshape(g_cur.shape[1], -1) @ w).cpu())
                feats_diff.append((g_diff[j].reshape(g_diff.shape[1], -1) @ w).cpu())
                mm = m[0, 0].bool()
                cpx, dpx = cur_t[mm], diff_t[mm]
                pixstats.append(
                    torch.tensor(
                        [cpx.mean(), cpx.std(), dpx.mean(), dpx.std(), dpx.abs().mean(), dpx.abs().max()]
                    )
                )
                fracs.append(frac)
        if (i // batch) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(layers)} layers, {len(fracs)} regions")

    return (
        torch.stack(feats_cur).float(),
        torch.stack(feats_diff).float(),
        torch.stack(pixstats).float(),
        np.array(fracs),
    )


def fit_probe(x_tr, y_tr, x_te, device, steps: int = 2000) -> np.ndarray:
    """Logistic regression, class-balanced, returns test scores."""
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp(min=1e-6)
    x_tr, x_te = ((x_tr - mu) / sd).to(device), ((x_te - mu) / sd).to(device)
    y = torch.tensor(y_tr, dtype=torch.float32, device=device)
    pos_w = torch.tensor([(len(y) - y.sum()) / y.sum()], device=device)
    lin = torch.nn.Linear(x_tr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lin(x_tr)[:, 0], y, pos_weight=pos_w)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return lin(x_te)[:, 0].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cls", default="swelling", choices=list(CONFIGS))
    p.add_argument("--max-layers", type=int, default=2400)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from transformers import AutoModel

    backbone = AutoModel.from_pretrained("facebook/dinov2-base").to(args.device).eval()

    cfg = CONFIGS[args.cls]
    rng = np.random.default_rng(0)
    regions = pd.read_parquet(PEREGRINE_CACHE / "regions.parquet")
    tr_layers = pick_layers(regions, cfg["train_builds"], args.max_layers * 2 // 3, rng, cfg)
    te_layers = pick_layers(regions, [cfg["test_build"]], args.max_layers // 3, rng, cfg)
    print(f"cls {args.cls}: train layers {len(tr_layers)} (builds {cfg['train_builds']}), test layers {len(te_layers)} (build {cfg['test_build']})")

    print("extracting train features...")
    c_tr, d_tr, s_tr, f_tr = extract(tr_layers, backbone, args.device, cfg)
    print("extracting test features...")
    c_te, d_te, s_te, f_te = extract(te_layers, backbone, args.device, cfg)
    y_tr, y_te = (f_tr > THR).astype(float), f_te > THR
    print(f"train regions {len(y_tr)} ({int(y_tr.sum())} pos), test regions {len(y_te)} ({int(y_te.sum())} pos)")
    print(f"test prevalence (random-baseline AP): {y_te.mean():.4f}")

    probes = {
        "pixstats (no backbone)": (s_tr, s_te),
        "cur features": (c_tr, c_te),
        "cur+diff features": (torch.cat([c_tr, d_tr], 1), torch.cat([c_te, d_te], 1)),
        "diff features only": (d_tr, d_te),
    }
    for name, (xtr, xte) in probes.items():
        scores = fit_probe(xtr, y_tr, xte, args.device)
        print(f"  {name:26s} AP {average_precision(scores, y_te):.4f}")


if __name__ == "__main__":
    main()
