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

## Results (2026-07-14, all 3,575 build-5 layers, region-level AP)

| class (n_pos) | v01 anomaly-1k | v01 full | **v02** |
|---|---|---|---|
| recoater_streaking (54,389) | **0.771** | 0.675 | 0.663 |
| incomplete_spreading (11) | 0.251 | 0.329 | **0.482** |
| spatter (9,568) | 0.538 | **0.777** | 0.565 |
| swelling (91) | 0.009 | 0.006 | 0.003 |
| super_elevation (73) | 0.001 | 0.004 | 0.000 |
| mean | 0.314 | 0.358 | 0.343 |

**Verdict: the curl interventions FAILED.** Temporal prev_scan frame +
weighted region loss + cosine LR did not move swelling/super_elevation
off the floor (visuals unchanged: swelling fires too broadly,
super_elevation blind). Short-feed improved a lot (n=11 — noisy).

Post-mortem hypotheses for v03, in order of suspicion:
1. **iid halogen augmentation destroys the temporal signal** — each
   frame gets independent lighting + per-image standardization, so the
   prev-vs-current DIFFERENCE (the actual elevation cue) is noise at
   train time. Fix: correlated per-sequence lighting trajectories
   and/or an explicit photometric-normalized difference channel.
2. Region-mean scoring dilutes sparse-in-region classes (swelling GT is
   edge speckle inside large parts) — try top-k% pixel aggregation.
3. 518-px input may be too coarse for elevation shading — try 1036
   (74x74 patches; the 1024 cache exists for exactly this).
4. On the Inova, curl may be better seen THERMALLY (elevated part =
   less powder insulation = hotter in bedmatrix) — a transfer-side
   mitigation independent of Peregrine optics.
