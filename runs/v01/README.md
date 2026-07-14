# Run v01 — foundation-backbone dense defect segmentation, per-region output

Dense semantic segmentation of the 12 Peregrine classes from a DINOv2
encoder + light seg head. The served output is the dense per-class map
alone (bed/galvo coordinates); the agentic system attributes defects to
parts itself from prescribed geometry. Region aggregation (part regions
+ powder tiles) is internal — a training loss term and eval view. Trained on Peregrine (metal LPBF, dense expert masks);
transfers zero-shot to the Inova (polymer SLS, no labels) — human
confirm/reject of live detections later builds the Inova label set.

## Design

**Common input format** — a layer event is a SEQUENCE of K frames plus a
part mask, per-image standardized on a common 2D grid:
- frames: K grayscale views spanning the layer (Peregrine: after_powder
  + after_melt, expanded to K synthetic halogen states — black, dim,
  hazy, blown-out, glare gradients, laser-spot artifacts — calibrated
  against measured Inova stats | Inova: detail-scored chamber frames
  sampled across recoat+scan; many are genuinely uninformative, which
  the fusion module must handle)
- part mask (Peregrine `part_ids > 0` | Inova galvo trace ACCUMULATED
  over the layer's frames — frame_galvo is time-fading, not binary —
  then warped via `registration.py`: fisheye correction + homography,
  one-time calibration)

Shared frozen encoder per frame → attention pooling across frames
(padding mask for missing/blank) → fused feature map → seg head. The
phase identity of each frame (recoat vs scan) enters as an embedding,
not a fixed channel slot.

**Architecture** — DINOv2 ViT encoder (size configurable, start ViT-B;
frozen first, LoRA/unfreeze as ablations) → segmentation head over 12
classes → region aggregation (masked pooling over the region map) →
per-region multilabel scores. SAM/SAM2 encoders and SegFormer-B5 are
comparison ablations. Class imbalance is severe (spatter 26% of regions,
debris 0.1%): BCE with per-class pos_weight, report per-class AP.
Actuation-tier thresholds are deployment config, not baked into weights.

**Sim-to-real (metal→polymer) augmentation** — train-time degradation of
Peregrine toward measured Inova chamber conditions: residual fisheye
warp, JPEG artifacts, lighting gradients, blur/noise, contrast-polarity
jitter. Calibrate against real Inova frames before trusting it.

**Deployment** — stateless torch service on a reserved RTX 6000 Ada on
this machine; detections are advisory to the agent (no CPU fallback
required). GPU headroom enables a future ~10 fps streaming mode during
recoat.

## Data

`prepare.py` STREAMS Peregrine layer parquets from HF (local dev-subset
short-circuit) and caches only compact 1024² arrays + a region-label
parquet under `runs/v01/data/peregrine/` (~45 GB for all 17.5k layers;
1024 so thin streak hairlines survive). No full raw clones (422 GB
stays remote).

## Results (2026-07-14, head-to-head on all 3,575 build-5 layers)

| class (n_pos) | anomaly-1k ckpt | full-17.5k ckpt |
|---|---|---|
| recoater_streaking (54,389) | **0.771** | 0.675 |
| spatter (9,568) | 0.538 | **0.777** |
| incomplete_spreading (11) | 0.251 | **0.329** |
| swelling (91) | 0.009 | 0.006 |
| super_elevation (73) | 0.001 | 0.004 |
| mean (defined) | 0.314 | **0.358** |

Checkpoints: `best_anomaly1k.pt` (50 ep, builds 1-4 anomaly subset),
`best.pt` (15 ep @ constant LR — undertrained, oscillating val; add a
cosine schedule next run). KNOWN GAP: elevation-type classes (swelling,
super_elevation ≈ curl — the Inova-critical failure mode) are near zero
for BOTH models; hypotheses: subtle shading cues lost at 518 input /
frozen features, tiny positive counts. Investigate with visualize.py
before scaling anything else.

## Pipeline

```bash
uv run python -m runs.v01.prepare --source anomaly   # 1,750 curated layers
uv run python -m runs.v01.prepare --source build     # all 17,582 layers (streams ~422 GB once)
uv run python -m runs.v01.train
uv run python -m runs.v01.evaluate --checkpoint runs/v01/checkpoints/best.pt
uv run python -m runs.v01.visualize --checkpoint runs/v01/checkpoints/best.pt
uv run python -m runs.v01.export --checkpoint runs/v01/checkpoints/best.pt
```

Every stage supports `--max-cases 5` for smoke tests. wandb project:
`inova-defect-v01`.
