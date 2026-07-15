"""
Entry point: uv run python -m runs.v05.train

v04 = v03 + photometric head channels (short-feed probe), region head
(swelling probe), debris copy-paste augmentation. Same build-3 holdout.
Checkpoint selection: region-head mean AP. wandb project
inova-defect-v05; checkpoints in runs/v05/checkpoints/.

Smoke test:
    uv run python -m runs.v05.train --max-cases 8 --epochs 1 --no-wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dataset import PeregrineSequenceDataset
from .model import DefectSegmenter, ModelConfig
from .trainer import Trainer

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
TRAIN_BUILDS = [1, 2, 4, 5]
VAL_BUILD = [3]  # debris/short-feed/swelling-rich — the classes that matter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone", default="base", choices=["small", "base", "large"])
    p.add_argument("--k-frames", type=int, default=5)
    p.add_argument("--frame-size", type=int, default=518)
    p.add_argument("--input-size", type=int, default=518)
    p.add_argument("--logits-size", type=int, default=256)
    p.add_argument("--region-loss-weight", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    train_ds = PeregrineSequenceDataset(
        builds=TRAIN_BUILDS, k_frames=args.k_frames,
        frame_size=args.frame_size, logits_size=args.logits_size,
    )
    val_ds = PeregrineSequenceDataset(
        builds=VAL_BUILD, k_frames=args.k_frames,
        frame_size=args.frame_size, logits_size=args.logits_size, augment=False,
    )
    if args.max_cases:
        train_ds.keys = train_ds.keys[: args.max_cases]
        val_ds.keys = val_ds.keys[: args.max_cases]
    print(f"train layers: {len(train_ds)}  val layers: {len(val_ds)}")

    model = DefectSegmenter(
        ModelConfig(
            backbone=args.backbone,
            input_size=args.input_size,
            logits_size=args.logits_size,
        )
    )
    n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"trainable params: {n_train / 1e6:.2f}M (backbone {args.backbone}, frozen)")

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(project="inova-defect-v05", config=vars(args))

    trainer = Trainer(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        region_loss_weight=args.region_loss_weight,
        workers=args.workers,
        device=args.device,
        wandb_run=run,
    )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        train_stats = trainer.train_epoch(epoch)
        metrics = trainer.validate(epoch)
        mean_ap = metrics["val/mean_ap"]
        print(
            f"epoch {epoch}: "
            + " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in {**train_stats, **metrics}.items())
        )
        torch.save(
            {"model": model.state_dict(), "config": vars(args), "epoch": epoch},
            CHECKPOINT_DIR / "last.pt",
        )
        if mean_ap == mean_ap and mean_ap > best:
            best = mean_ap
            torch.save(
                {"model": model.state_dict(), "config": vars(args), "epoch": epoch},
                CHECKPOINT_DIR / "best.pt",
            )
            print(f"  new best mean AP {best:.4f} -> checkpoints/best.pt")

    if run:
        run.finish()


if __name__ == "__main__":
    main()
