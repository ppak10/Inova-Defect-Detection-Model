# Run v02 — temporal context + rebalanced region supervision

Targets v01's diagnosed failure: elevation-type classes (swelling,
super_elevation ≈ curl — the Inova-critical failure mode) scored ~0 AP.
Visual inspection (runs/v01/figures/) showed swelling fires too broadly
and super_elevation is a persistent, PART-LEVEL state the model cannot
see from a single layer's imagery.

## v01 → v02

| change | why |
|---|---|
| prev_scan frame leads each sequence (K=5: prev_melt + 2 recoat + 2 scan) | elevation defects persist across layers; the cue is change/persistence, not single-frame appearance. Full-build cache has consecutive layers (anomaly subset didn't). |
| per-class weighting on the REGION loss (inverse positive-rate, cap 100) | region-level supervision matches part-level label semantics (super_elevation paints whole parts); v01 weighted all classes equally there, so rare classes barely contributed gradient. |
| cosine LR schedule (3e-4 → 6e-6) | v01 full-data run oscillated at constant LR; best epoch was 6/15. |
| dataloader frames at 518 px (was 1024) | model resizes to 518 anyway; augmenting at 1024 was ~4x wasted dataloader CPU. |
| more epochs (30) at full 17.5k layers | v01 full run was undertrained. |

Unchanged: model architecture (frozen DINOv2-base + fusion + seg head),
dense-map-only output contract, 12-class multi-label taxonomy,
1024² shared cache (READ FROM runs/v01/data/peregrine — do not
duplicate; prepare.py intentionally not copied into v02).

## Pipeline

```bash
uv run python -m runs.v02.train
uv run python -m runs.v02.evaluate --checkpoint runs/v02/checkpoints/best.pt
uv run python -m runs.v02.visualize --checkpoint runs/v02/checkpoints/best.pt
```

Baselines to beat (same eval: all 3,575 build-5 layers, region-level AP):
mean 0.358 / streaking 0.771 (v01 best_anomaly1k) / spatter 0.777,
short-feed 0.329 (v01 full ckpt). Success criterion: swelling and
super_elevation move off ~0.01 without giving back streaking.
