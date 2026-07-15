# Run v05 — v04 issue fixes: paste v2, combined selection, richer photometrics

## v04 → v05

| change | v04 issue it fixes |
|---|---|
| debris paste v2: feathered-alpha blend + gain/scale jitter, paste anywhere; REGION labels updated for pasted pixels | paste v1 never reached the region head (label-update bug) and hard copy edges were shortcut-learnable; debris stayed at ~0.006 |
| checkpoint selection = mean over classes of max(dense AP, head AP) | head-only selection picked epoch 10 and gave back dense short-feed (0.543@ep20 -> 0.497) |
| photometric stack 4 -> 7 channels (+ box-std, box-|diff| maps) | photometrics only reached parity (0.497-0.543) with the 0.578 pixel-stat bar; hand the probe's winning statistics to the head directly |
| streaking cross-build probe (runs/v02/probe_curl.py --cls recoater_streaking) | streaking collapse on build 3 (0.09-0.11) is undiagnosed — diagnose before fixing |

Unchanged: build-3 holdout, frozen DINOv2-base, K=5, cosine LR, region
head + dense head with per-class output routing (part-state -> head,
localized events -> dense).

Baselines to beat (build 3): swelling head 0.616, short-feed dense
0.543 (ep20)/0.535 (v03), debris 0.009, spatter 0.748.

## Streaking probe verdict (runs/v02/probe_streaking.txt, 2026-07-15)

ALL feature types collapse on build-3 streaking (pixstats 0.065, cur
0.049, diff 0.092 vs 0.053 random — 12,909 train positives). Visual
check: build 3's after_powder frames are uniformly DARK and the
labeled streak lines are not visible in the optics at all. Combined
with v01 transferring streaking to held-out build 5 at 0.77:
**streaking generalizes across builds 1/2/4/5 — build 3 itself is the
anomaly** (unlabelable-from-optics annotations, likely different
exposure; it is also the scan-path-mismatch build). Decision: judge
streaking on build-5 holdout numbers, treat build-3 streak labels as
unusable, and do NOT spend model effort "fixing" it.

## Pipeline

```bash
uv run python -m runs.v05.train
uv run python -m runs.v05.evaluate --checkpoint runs/v05/checkpoints/best.pt
```
