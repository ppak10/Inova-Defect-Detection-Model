# Run v04 — probe-driven fixes: photometric channels, region head, debris paste

Every change here traces to a v02/v03 diagnostic (see those READMEs):

## v03 → v04

| change | evidence |
|---|---|
| SegHead consumes a 4-channel PHOTOMETRIC stack (std recoat/scan + diffs vs prev_scan) at output resolution | short-feed probe: 6 pixel stats 0.578 AP vs frozen DINOv2 0.03-0.05 — self-supervised backbones are invariant to the brightness cue |
| dedicated RegionHead (MLP on region-pooled fused features), trained with the same per-class weights; checkpoint selection on region-head mean AP | swelling probe: linear on pooled frozen features 0.336 AP vs 0.061 from the dense pipeline |
| debris copy-paste augmentation (train-build crops only, p=0.5, pasted into current frames but NOT prev_scan) | debris optically obvious but 120 train positives; paste keeps build-3 val honest and creates a "new object appeared" temporal cue |

Unchanged: build-3 holdout, frozen DINOv2-base, K=5 sequences, cosine
LR, sensitivity-first deployment posture. Baselines to beat (build 3):
short-feed 0.535, swelling 0.061, debris 0.004, spatter 0.748,
streaking 0.112 (cross-build fragility — open problem, not addressed
by v04).

Eval reports BOTH ranking paths per class: `dense` (region-mean of the
dense map, v03-comparable) and `head` (region head).

## Results (2026-07-15, build-3 holdout; best ckpt = epoch 10 by head mean AP)

| class (n_pos) | v03 | v04 dense | v04 head | verdict |
|---|---|---|---|---|
| swelling (1,929) | 0.061 | 0.231 | **0.616** | region head VALIDATED — 10x v03, ~2x the probe ceiling |
| incomplete_spreading (378) | 0.535 | 0.497 (0.543 @ep20) | 0.199 | photometric parity-to-slight-gain; dense is the right path for it |
| debris (649) | 0.004 | 0.009 | 0.006 | paste FAILED to transfer to real build-3 debris |
| spatter (34,908) | 0.748 | 0.700 | 0.730 | parity |
| recoater_streaking (11,327) | 0.112 | 0.092 | 0.049 | cross-build fragility untouched (not a v04 target) |

Takeaways: (1) part-state classes belong to the region head — deploy
should read swelling from it; (2) checkpoint selection on head-mean-AP
sacrificed some short-feed (ep20 hit 0.543 dense) — v05: per-class-
family selection or a combined metric; (3) debris paste needs rethink:
8-100 source layers from 2 builds may lack appearance diversity, or
paste artifacts are too easy — consider harder blending (Poisson),
scale jitter, and pasting onto POWDER (Peregrine debris labels sit on
parts, but Inova debris lands anywhere); (4) streaking cross-build is
now the biggest open problem.

## Pipeline

```bash
uv run python -m runs.v04.train
uv run python -m runs.v04.evaluate --checkpoint runs/v04/checkpoints/best.pt
uv run python -m runs.v04.visualize --checkpoint runs/v04/checkpoints/best.pt --classes debris incomplete_spreading
```

NOTE: serve.py (the live endpoint) still runs v03 — its forward call
predates the photometric/region-head signature. Port serve to v04 after
results validate.
