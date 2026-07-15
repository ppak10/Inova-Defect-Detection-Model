"""
Training loop for v04 = v03 + region-HEAD loss + photometric input.

Three loss terms on two outputs:
- pixel BCE on the dense map (with photometric channels in the head)
- dense-mean region loss (unchanged from v02/v03)
- region-head weighted BCE — the probe-validated path for part-state
  classes (0.336 linear-probe AP on swelling vs ~0.06 from the dense
  pipeline).

Validation reports BOTH region scores: ap_<c> = dense-mean (comparable
to v03), aph_<c> = region head. Checkpoint selection: mean over region-
head APs (the hypothesis under test).

v01 findings that drive this (runs/v01/README.md): validation
oscillated at constant LR (best at epoch 6/15), and the region term —
the one whose label semantics match part-level classes like
super_elevation — weighted all classes equally, so rare curl-family
classes contributed almost nothing to the gradient.

Region weighting: elementwise BCE scaled by (1 + (w_c - 1) * y) where
y is the soft fraction target — negatives keep weight 1, full
positives weight w_c = clamped inverse positive-rate measured from the
train split's region table.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .constants import (
    LABEL_FRACTION_THRESHOLD,
    PEREGRINE_ALL_CLASSES,
    PEREGRINE_CLASSES,
)
from .dataset import FRAC_COLS
from .model import DefectSegmenter, region_scores

ANOMALY_IDX = [PEREGRINE_ALL_CLASSES.index(c) for c in PEREGRINE_CLASSES]


def average_precision(scores: np.ndarray, positives: np.ndarray) -> float:
    if positives.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    pos = positives[order]
    cum = np.cumsum(pos)
    prec = cum / (np.arange(len(pos)) + 1)
    return float((prec * pos).sum() / pos.sum())


def measure_pos_weight(dataset, n_sample: int = 64, cap: float = 100.0) -> torch.Tensor:
    """Per-class positive-PIXEL rate over a sample of dense targets."""
    idx = np.linspace(0, len(dataset) - 1, min(n_sample, len(dataset))).astype(int)
    rates = torch.zeros(len(PEREGRINE_ALL_CLASSES))
    for i in idx:
        rates += dataset[int(i)]["target"].mean(dim=(1, 2))
    rates /= len(idx)
    return ((1 - rates) / rates.clamp(min=1e-6)).clamp(max=cap)


def measure_region_weight(dataset, cap: float = 100.0) -> torch.Tensor:
    """Per-class positive-REGION rate from the train region table."""
    fr = dataset.regions[FRAC_COLS].to_numpy()
    rates = (fr > LABEL_FRACTION_THRESHOLD).mean(axis=0)
    w = (1 - rates) / np.clip(rates, 1e-6, None)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


class Trainer:
    def __init__(
        self,
        model: DefectSegmenter,
        train_ds,
        val_ds,
        epochs: int,
        batch_size: int = 16,
        lr: float = 3e-4,
        region_loss_weight: float = 1.0,
        workers: int = 8,
        device: str = "cuda",
        wandb_run=None,
    ):
        from .dataset import collate_layers

        self.model = model.to(device)
        self.device = device
        self.wandb = wandb_run
        self.region_w = region_loss_weight
        self.pos_weight = measure_pos_weight(train_ds).to(device)
        self.region_pos_weight = measure_region_weight(train_ds).to(device)
        print("region pos_weight:", {
            c: round(float(w), 1)
            for c, w in zip(PEREGRINE_CLASSES, self.region_pos_weight)
        })
        self.opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=epochs, eta_min=lr * 0.02
        )
        self.train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            collate_fn=collate_layers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=collate_layers,
            pin_memory=True,
        )

    def _forward(self, batch) -> tuple[torch.Tensor, tuple, dict]:
        d = self.device
        n_regions = batch["labels"].shape[1]
        logits, region_logits = self.model(
            batch["frames"].to(d),
            batch["phase_ids"].to(d),
            batch["meta"].to(d),
            batch["frame_valid"].to(d),
            batch["part_mask"].to(d),
            batch["photometric"].to(d),
            region_map=batch["region_map"].to(d),
            n_regions=n_regions,
        )
        target = batch["target"].to(d)
        pixel_loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight[None, :, None, None]
        )
        with torch.autocast(d.split(":")[0] if d != "cpu" else "cpu", enabled=False):
            probs = logits.float().sigmoid()
            rs = region_scores(
                probs[:, ANOMALY_IDX], batch["region_map"].to(d), batch["labels"].shape[1]
            )
            rv = batch["region_valid"].to(d)
            y = batch["labels"].to(d)[rv]
            p = rs.clamp(1e-6, 1 - 1e-6)[rv]
            bce = -(y * p.log() + (1 - y) * (1 - p).log())
            w = 1 + (self.region_pos_weight[None, :] - 1) * y
            region_loss = (bce * w).sum() / w.sum().clamp(min=1.0)
            rl = region_logits.float()[:, :, ANOMALY_IDX][rv]
            bce_h = F.binary_cross_entropy_with_logits(rl, y, reduction="none")
            head_loss = (bce_h * w).sum() / w.sum().clamp(min=1.0)
        loss = pixel_loss + self.region_w * (region_loss + head_loss)
        return loss, (rs, rl.sigmoid()), {
            "pixel_loss": pixel_loss.item(),
            "region_loss": region_loss.item(),
            "head_loss": head_loss.item(),
        }

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        agg: dict[str, float] = {}
        for step, batch in enumerate(self.train_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                loss, _, parts = self._forward(batch)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            if self.wandb and step % 10 == 0:
                self.wandb.log(
                    {"epoch": epoch, "lr": self.sched.get_last_lr()[0],
                     **{f"train/{k}": v for k, v in parts.items()}}
                )
        self.sched.step()
        return {k: v / max(len(self.train_dl), 1) for k, v in agg.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        self.model.eval()
        scores, hscores, labels = [], [], []
        agg: dict[str, float] = {}
        for batch in self.val_dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                _, (rs, rh), parts = self._forward(batch)
            rv = batch["region_valid"]
            scores.append(rs.float().cpu()[rv].numpy())
            hscores.append(rh.float().cpu().numpy())
            labels.append(batch["labels"][rv].numpy())
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
        scores = np.concatenate(scores)
        hscores = np.concatenate(hscores)
        labels = np.concatenate(labels)
        metrics = {f"val/{k}": v / max(len(self.val_dl), 1) for k, v in agg.items()}
        aps = []
        for j, c in enumerate(PEREGRINE_CLASSES):
            pos = labels[:, j] > LABEL_FRACTION_THRESHOLD
            metrics[f"val/ap_{c}"] = average_precision(scores[:, j], pos)
            aph = average_precision(hscores[:, j], pos)
            metrics[f"val/aph_{c}"] = aph
            if not np.isnan(aph):
                aps.append(aph)
        metrics["val/mean_ap"] = float(np.mean(aps)) if aps else float("nan")
        metrics["val/mean_ap_dense"] = float(
            np.nanmean([metrics[f"val/ap_{c}"] for c in PEREGRINE_CLASSES])
        )
        if self.wandb:
            self.wandb.log({"epoch": epoch, **metrics})
        return metrics
