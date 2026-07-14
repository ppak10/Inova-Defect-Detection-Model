# Run v03 — retargeted at the Inova's real failure modes

Priority reset (user, 2026-07-14): the failures actually seen on the
Inova are DEBRIS (compounds across layers, damages neighboring parts)
and SHORT FEED (over-rastering; visible post-recoat as the previous
layer's raster showing through, toward the overflow bin). Curl has not
been observed and super_elevation is untrainable cross-build.
Sensitivity > precision while detections are human-gated.

## v02 → v03

| change | why |
|---|---|
| val split: hold out BUILD 3, train 1,2,4,5 | debris has 649/769 positives in build 3 (build 5: ZERO); short-feed 378 (build 5: 11); swelling 1,929. v01/v02 measured the wrong classes on the wrong build. |
| region loss weight 1.0 → 3.0 | probe_curl.py: a linear probe on region-pooled frozen features hit 0.336 AP on swelling where the dense-head pipeline scored ~0 — region-level supervision is where part-state classes live. |
| headline metrics: debris AP, incomplete_spreading AP | matches the deployment priorities and remedies (extra recoat; skip/exclude part). |

Kept from v02: prev_scan temporal frame, per-class region weights,
cosine LR, 518-px frames, shared v01 cache.

**Short-feed probe verdict (runs/v02/probe_shortfeed.txt, build-3
test, 378 positives, prevalence 0.0095):** 6 raw pixel statistics
(region mean/std + pixel-level prev-diff stats) hit **AP 0.578** while
frozen DINOv2 features score 0.03-0.05 — self-supervised backbones are
trained to be INVARIANT to brightness/contrast, which is the entire
short-feed cue. Consequences: (1) a pixstat logistic detector is the
strongest short-feed alerter available — build it as a standalone
deployable; (2) v04: concatenate raw standardized image + prev-diff
channels into the seg head to restore the photometric information the
backbone discards. Also: debris has only 120 TRAIN positives under
this split (649/769 live in val build 3) — cross-build debris needs
k-fold or within-build splits, or Inova-side labels.

Deployment note: short-feed inference must run on the POST-RECOAT frame
in the recoat→scan gap (the remedy is another recoat pass, which must
precede the scan).

## Results (2026-07-14, build-3 holdout, 3,297 layers — the honest split)

| class (n_pos) | v03 AP | reference |
|---|---|---|
| incomplete_spreading (378) | **0.535** | pixstat probe 0.578 |
| spatter (34,908) | 0.748 | |
| recoater_streaking (11,327) | **0.112** | was 0.66-0.81 with build 3 IN train |
| swelling (1,929) | 0.061 | linear probe ceiling 0.336 |
| debris (649) | 0.004 | only 120 train positives |

Best epoch 7/30. (Do NOT quote the build-5 numbers for this run —
build 5 is in its training set.)

Findings:
1. **Short feed 0.535 with real statistics behind it** — first
   trustworthy number for the top-priority class; deep model ~parity
   with, not better than, 6 pixel statistics. v04: inject raw
   photometric channels into the head; ship the pixstat logistic as a
   standalone alerter regardless.
2. **Streaking does NOT generalize cross-build** (0.11 on build 3 vs
   0.66-0.81 when build 3 was trained on). Earlier streaking optimism
   was partly memorized build style. Needs investigation — build 3 is
   also the scan-path-mismatch build.
3. Swelling at 0.061 vs its 0.336 probe ceiling — the seg pipeline
   still underuses region-poolable signal (v04: dedicated region head
   on pooled features).
4. Debris is OPTICALLY OBVIOUS (dark objects in after_melt; see
   figures/debris/) but data-starved (120 train positives). Candidates:
   copy-paste debris augmentation, k-fold, Inova human-loop labels.

## Pipeline

```bash
uv run python -m runs.v03.train
uv run python -m runs.v03.evaluate --checkpoint runs/v03/checkpoints/best.pt
uv run python -m runs.v03.visualize --checkpoint runs/v03/checkpoints/best.pt --classes debris incomplete_spreading
```
