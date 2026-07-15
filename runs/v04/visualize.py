"""
Entry point: uv run python -m runs.v04.visualize --checkpoint <path>
    [--classes swelling super_elevation] [--n 6] [--builds 5]

For each requested class, picks the held-out layers with the LARGEST
ground-truth presence and renders a panel row per layer:

    after_powder | after_melt | GT mask (class) | predicted prob (class)

Figures land in runs/v01/figures/<class>/ (gitignored). Built to answer
"why does the model miss elevation-type classes" — look at whether the
defect is even visible in the imagery before blaming the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .constants import PEREGRINE_ALL_CLASSES, PEREGRINE_CLASSES
from .dataset import PEREGRINE_CACHE, PeregrineSequenceDataset, collate_layers
from .model import DefectSegmenter, ModelConfig

FIGURES_DIR = Path(__file__).parent / "figures"


def pick_layers(builds: list[int], cls: str, n: int) -> list[tuple[int, int, float]]:
    """(build, layer, frac) with the largest area-weighted GT fraction."""
    r = pd.read_parquet(PEREGRINE_CACHE / "regions.parquet")
    r = r[r["build"].isin(builds)]
    r["px"] = r[f"frac_{cls}"] * r["area_px"]
    by_layer = r.groupby(["build", "layer"])["px"].sum().sort_values(ascending=False)
    return [(b, l, v) for (b, l), v in by_layer.head(n).items()]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--classes", nargs="+", default=["swelling", "super_elevation", "recoater_streaking"])
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--builds", type=int, nargs="+", default=[3])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = DefectSegmenter(
        ModelConfig(
            backbone=cfg["backbone"],
            input_size=cfg["input_size"],
            logits_size=cfg["logits_size"],
        )
    )
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    ds = PeregrineSequenceDataset(
        builds=args.builds, k_frames=cfg.get("k_frames", 4),
        logits_size=cfg["logits_size"], augment=False,
    )
    key_to_idx = {k: i for i, k in enumerate(ds.keys)}

    for cls in args.classes:
        ci = PEREGRINE_ALL_CLASSES.index(cls)
        out_dir = FIGURES_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for build, layer, px in pick_layers(args.builds, cls, args.n):
            if (build, layer) not in key_to_idx:
                continue
            item = ds[key_to_idx[(build, layer)]]
            batch = collate_layers([item])
            logits = model(
                batch["frames"].to(args.device),
                batch["phase_ids"].to(args.device),
                batch["meta"].to(args.device),
                batch["frame_valid"].to(args.device),
                batch["part_mask"].to(args.device),
                batch["photometric"].to(args.device),
            )
            prob = logits.float().sigmoid()[0, ci].cpu().numpy()
            gt = item["target"][ci].numpy()

            with np.load(PEREGRINE_CACHE / f"build_{build}_layer_{layer:04d}.npz") as z:
                imgs = z["images"].astype(np.float32)

            fig, axes = plt.subplots(1, 4, figsize=(22, 6))
            for ax, im, title in [
                (axes[0], imgs[0], "after_powder"),
                (axes[1], imgs[1], "after_melt"),
            ]:
                ax.imshow(im, cmap="gray")
                ax.set_title(title)
            axes[2].imshow(imgs[1], cmap="gray")
            axes[2].imshow(np.ma.masked_where(gt < 0.5, gt), cmap="spring", alpha=0.6, vmin=0, vmax=1)
            axes[2].set_title(f"GT {cls} ({px:.0f} px @1842²)")
            im3 = axes[3].imshow(prob, cmap="inferno", vmin=0, vmax=1)
            axes[3].set_title(f"pred P({cls}) max={prob.max():.3f}")
            fig.colorbar(im3, ax=axes[3], fraction=0.046)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"build {build} layer {layer} — {cls}")
            fig.tight_layout()
            path = out_dir / f"build{build}_layer{layer:04d}.png"
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
