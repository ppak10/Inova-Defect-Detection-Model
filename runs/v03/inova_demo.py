"""
Entry point: uv run python -m runs.v03.inova_demo [--build 032] [--n 8]

Zero-shot transfer demo: runs the v03 checkpoint on real Inova tick
data. Deliberately "registration-lite" (no fisheye correction; coarse
similarity fit galvo->chamber by correlating the accumulated galvo
trace with post-scan darkening) — qualitative sanity check, not the
real registration.py.

Per layer (z2 step boundaries):
- accumulate galvo trace (binarize + max) -> part mask
- pick best post-recoat / post-scan chamber frames by detail quality
  (gradient energy with saturation/black penalties — CLAUDE.md
  "Measured imaging reality"), rotated 90 deg CW to galvo orientation
- assemble the v03 input sequence (prev_scan, recoat, scan) on a 518
  grid and run the model
- save overlay figures to runs/v03/figures/inova_<build>/
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .constants import INOVA_DIR, PEREGRINE_ALL_CLASSES
from .dataset import standardize
from .model import DefectSegmenter, ModelConfig

FIGURES_DIR = Path(__file__).parent / "figures"
SHOW_CLASSES = ["incomplete_spreading", "recoater_streaking", "debris", "swelling"]


def rot_cw90(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def decode_gray(frame: dict) -> np.ndarray:
    g = np.asarray(Image.open(io.BytesIO(frame["bytes"])).convert("L"), dtype=np.float32)
    return rot_cw90(g)


def quality(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (160, 120))
    c = small[20:100, 30:130]
    gx, gy = np.gradient(c)
    detail = float(np.sqrt(gx**2 + gy**2).mean())
    return detail * (1 - float((c > 245).mean())) * (1 - float((c < 15).mean()))


def galvo_mask(frame: dict) -> np.ndarray:
    a = np.asarray(Image.open(io.BytesIO(frame["bytes"])).convert("L"))
    return a > 40


def fit_similarity(dark: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float]:
    """Coarse grid search: warp mask (galvo space) into the chamber
    frame maximizing correlation with the darkening map."""
    h, w = dark.shape
    d = cv2.resize(dark, (w // 4, h // 4))
    d = (d - d.mean()) / (d.std() + 1e-6)
    m0 = mask.astype(np.float32)
    best = (-1e9, 1.0, 0.0, 0.0)
    for s in np.arange(0.5, 1.45, 0.05):
        mm = cv2.resize(m0, None, fx=s / 4 * (w / mask.shape[1]), fy=s / 4 * (w / mask.shape[1]))
        for ty in range(-24, 25, 4):
            for tx in range(-24, 25, 4):
                canvas = np.zeros_like(d)
                y0, x0 = (d.shape[0] - mm.shape[0]) // 2 + ty, (d.shape[1] - mm.shape[1]) // 2 + tx
                ys, xs = slice(max(y0, 0), max(min(y0 + mm.shape[0], d.shape[0]), 0)), slice(
                    max(x0, 0), max(min(x0 + mm.shape[1], d.shape[1]), 0)
                )
                sub = mm[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
                if sub.size == 0:
                    continue
                canvas[ys, xs] = sub
                if canvas.sum() < 10:
                    continue
                score = float((d * (canvas - canvas.mean())).mean())
                if score > best[0]:
                    best = (score, s, tx * 4.0, ty * 4.0)
    return best


def warp_mask(mask: np.ndarray, shape: tuple[int, int], s: float, tx: float, ty: float) -> np.ndarray:
    h, w = shape
    scale = s * (w / mask.shape[1])
    mm = cv2.resize(mask.astype(np.float32), None, fx=scale, fy=scale)
    canvas = np.zeros(shape, dtype=np.float32)
    y0 = int((h - mm.shape[0]) // 2 + ty)
    x0 = int((w - mm.shape[1]) // 2 + tx)
    ys, xs = slice(max(y0, 0), min(y0 + mm.shape[0], h)), slice(max(x0, 0), min(x0 + mm.shape[1], w))
    canvas[ys, xs] = mm[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
    return canvas > 0.5


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", default="032")
    p.add_argument("--checkpoint", default="runs/v03/checkpoints/best.pt")
    p.add_argument("--n", type=int, default=8, help="layers to render (evenly spaced)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = DefectSegmenter(
        ModelConfig(backbone=cfg["backbone"], input_size=cfg["input_size"], logits_size=cfg["logits_size"])
    )
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    t = pq.read_table(
        INOVA_DIR / f"data/ticks/{args.build}.parquet",
        columns=["positions.position.z2", "frame_chamber", "frame_galvo"],
    )
    z2 = t["positions.position.z2"].to_numpy()
    fc = t["frame_chamber"].to_pylist()
    fg = t["frame_galvo"].to_pylist()
    bounds = np.where(np.diff(z2) > 0)[0]
    print(f"build {args.build}: {len(t)} ticks, {len(bounds)} layer boundaries")

    # --- per-layer extraction -------------------------------------------
    layers = []
    for li, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        g_idx = [i for i in range(a, b) if fg[i] is not None]
        c_idx = [i for i in range(a, b) if fc[i] is not None]
        if not g_idx or not c_idx:
            continue
        acc = None
        active = []
        for i in g_idx:
            m = galvo_mask(fg[i])
            if m.sum() > 20:
                active.append(i)
            acc = m if acc is None else (acc | m)
        if acc is None or acc.sum() < 100 or not active:
            continue
        pre = [i for i in c_idx if i < active[0]] or c_idx[:3]
        post = [i for i in c_idx if i > active[-1]] or c_idx[-3:]
        recoat_i = max(pre, key=lambda i: quality(decode_gray(fc[i])))
        scan_i = max(post, key=lambda i: quality(decode_gray(fc[i])))
        layers.append(dict(li=li, recoat=recoat_i, scan=scan_i, mask=acc))
    print(f"usable layers: {len(layers)}")

    # --- registration-lite: fit on the 5 largest-mask layers -------------
    fits = []
    for lay in sorted(layers, key=lambda d: -d["mask"].sum())[:5]:
        scan_g = decode_gray(fc[lay["scan"]])
        dark = -standardize(scan_g)
        fits.append(fit_similarity(dark, lay["mask"]))
    fits.sort(reverse=True)
    _, s, tx, ty = fits[0]
    print(f"registration-lite: scale {s:.2f}, tx {tx:.0f}, ty {ty:.0f} (corr {fits[0][0]:.4f})")

    # --- run the model on evenly spaced layers ---------------------------
    out_dir = FIGURES_DIR / f"inova_{args.build}"
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = layers[:: max(len(layers) // args.n, 1)][: args.n]
    for k, lay in enumerate(picks):
        if k == 0 or picks[k - 1]["scan"] is None:
            prev_g = decode_gray(fc[layers[max(0, layers.index(lay) - 1)]["scan"]])
        else:
            prev_g = decode_gray(fc[picks[k - 1]["scan"]])
        recoat_g = decode_gray(fc[lay["recoat"]])
        scan_g = decode_gray(fc[lay["scan"]])
        pmask = warp_mask(lay["mask"], scan_g.shape, s, tx, ty)

        size = 518
        def prep(g):
            return torch.tensor(standardize(cv2.resize(g, (size, size))))
        frames = torch.stack([prep(prev_g), prep(recoat_g), prep(scan_g)])[None]
        phase_ids = torch.tensor([[2, 0, 1]])
        bright = [float(g.mean()) / 255 for g in (prev_g, recoat_g, scan_g)]
        meta = torch.tensor([[[-1.0, bright[0], 1.0], [0.5, bright[1], 1.0], [0.0, bright[2], 1.0]]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        part = torch.tensor(cv2.resize(pmask.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool))[None]

        logits = model(frames.to(args.device), phase_ids.to(args.device), meta.to(args.device),
                       valid.to(args.device), part.to(args.device))
        probs = logits.float().sigmoid()[0].cpu().numpy()

        fig, axes = plt.subplots(1, 3 + len(SHOW_CLASSES), figsize=(6 * (3 + len(SHOW_CLASSES)), 6))
        axes[0].imshow(recoat_g, cmap="gray"); axes[0].set_title(f"post-recoat (t{lay['recoat']})")
        axes[1].imshow(scan_g, cmap="gray"); axes[1].set_title(f"post-scan (t{lay['scan']})")
        axes[2].imshow(scan_g, cmap="gray")
        axes[2].imshow(np.ma.masked_where(~pmask, pmask), cmap="spring", alpha=0.4)
        axes[2].set_title("galvo mask (registration-lite)")
        for j, c in enumerate(SHOW_CLASSES):
            m = probs[PEREGRINE_ALL_CLASSES.index(c)]
            im = axes[3 + j].imshow(m, cmap="inferno", vmin=0, vmax=1)
            axes[3 + j].set_title(f"P({c}) max={m.max():.2f}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Inova build {args.build} layer {lay['li']} — v03 zero-shot")
        fig.tight_layout()
        path = out_dir / f"layer{lay['li']:03d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()
