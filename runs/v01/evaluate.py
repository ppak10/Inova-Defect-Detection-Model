"""
Entry point: uv run python -m runs.v01.evaluate --checkpoint <path>

Evaluates a checkpoint on held-out builds (default: build 5, the
validation build): per-class region-level average precision, the same
metric train.py tracks, so checkpoints trained on different data are
comparable on identical ground.

Smoke test:
    uv run python -m runs.v01.evaluate --checkpoint runs/v01/checkpoints/best.pt --max-cases 8
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import LABEL_FRACTION_THRESHOLD, PEREGRINE_CLASSES
from .dataset import PeregrineSequenceDataset, collate_layers
from .model import DefectSegmenter, ModelConfig, region_scores
from .trainer import ANOMALY_IDX, average_precision


@torch.no_grad()
def evaluate(model, dl, device: str) -> dict[str, float]:
    model.eval()
    scores, labels = [], []
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(
                batch["frames"].to(device),
                batch["phase_ids"].to(device),
                batch["meta"].to(device),
                batch["frame_valid"].to(device),
                batch["part_mask"].to(device),
            )
        probs = logits.float().sigmoid()
        rs = region_scores(
            probs[:, ANOMALY_IDX], batch["region_map"].to(device), batch["labels"].shape[1]
        )
        rv = batch["region_valid"]
        scores.append(rs.cpu()[rv].numpy())
        labels.append(batch["labels"][rv].numpy())
    scores, labels = np.concatenate(scores), np.concatenate(labels)
    out = {}
    for j, c in enumerate(PEREGRINE_CLASSES):
        pos = labels[:, j] > LABEL_FRACTION_THRESHOLD
        out[c] = average_precision(scores[:, j], pos)
        out[f"{c}/n_pos"] = int(pos.sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--builds", type=int, nargs="+", default=[5])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-cases", type=int, default=None)
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
    model.to(args.device)

    ds = PeregrineSequenceDataset(
        builds=args.builds,
        k_frames=cfg.get("k_frames", 4),
        logits_size=cfg["logits_size"],
        augment=False,
    )
    if args.max_cases:
        ds.keys = ds.keys[: args.max_cases]
    dl = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.workers,
        collate_fn=collate_layers, pin_memory=True,
    )
    print(f"{args.checkpoint} (epoch {ckpt['epoch']}) on builds {args.builds}: {len(ds)} layers")

    results = evaluate(model, dl, args.device)
    aps = [v for k, v in results.items() if "/" not in k and not np.isnan(v)]
    for c in PEREGRINE_CLASSES:
        ap = results[c]
        print(f"  {c:22s} AP {'nan' if np.isnan(ap) else f'{ap:.4f}'}  (n_pos {results[f'{c}/n_pos']})")
    print(f"  mean AP (defined classes): {np.mean(aps):.4f}")


if __name__ == "__main__":
    main()
