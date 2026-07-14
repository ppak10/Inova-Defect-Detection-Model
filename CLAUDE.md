# Inova Defect Detection Model

Layer-wise powder-bed defect detection for the SLS4All Inova MK1 (polymer
SLS, PA12). Trains here (GPU machine), deploys as a CPU background service
on the recorder host (`7810`), and feeds detections to the agentic system
in the parent repo so agents can react mid-print (recoater compensation,
part exclusion proposals).

## Where this fits

- Submodule of `Agentic-Additive-Manufacturing-Process-Optimization`
  (at `sls4all/Inova-Defect-Detection-Model`). Parent repo lives on `7810`
  at `/mnt/storage2/GitHub/Agentic-Additive-Manufacturing-Process-Optimization`
  and owns the recorder, Postgres, MCP tools, and agent harnesses.
- Event flow (planned): per-layer inference → `build_events` row in the
  parent's Postgres → notify agent plane (episodic run via
  `harness/run.ts` with the event as prompt).
- Actuation tiering (decided): short feed → agent applies recoater
  compensation (staged passes / dose); streaking → notify only (blade
  damage is physical); part defect (curl/damage) → agent PROPOSES
  exclusion, human ack gates it. Precision matters most where actuation
  is destructive — a false-positive exclusion kills a good part.

## Training data

### Inova capture (ours, polymer SLS)

Recorded per build by the parent repo's recorder. Raw frames live on
`7810` under `data/frames/<build_id>/` (NOT in this repo, NOT on the
training machine — pull via the HF dataset or rsync from 7810):

- `<ts>_chamber.jpg` — visible camera of bed, ~10 fps, ~650 px wide,
  fisheye at edges. Sintered part cross-sections read clearly darker
  than fresh powder. ~500k/build.
- `<ts>_galvo.png` — firmware's render of the CURRENT layer's laser scan
  geometry; clean binary part mask. This is the "where the laser
  actually traced" prior — better than CAD/G-code. ~10 fps.
- `<ts>_bedmatrix.json` — 32×24 spatial temperature grid (MLX90640-class),
  °C floats. Real low-res thermal image; use for cooling-gradient (curl
  precursor) and cold-spot-after-recoat (short feed) signals. ~25–50k/build.
- `<ts>_thermal.gif` — rendered thermal view; present on some builds
  (44 yes, 46 no — dropout unexplained, verify before relying on it).

Structured data (Postgres on 7810; exports in parent
`data/exports/`, mirrored to HF `ppak10/Inova-Mk1-Telemetry`):

- `telemetry/<build>.parquet` — long format `(ts, sensor_id, kind, value)`.
  Temps: `surface|surfaceAvg|surfaceMax|surfaceMin`, `quadrant1..4`,
  `printBed`, `powderBed`, chambers; plus `power/laser` on-off channel.
- `position_hf/<build>.parquet` — `(ts, x, y, z1, z2, r, has_homed)`;
  `z1` (print bed) steps down once per layer → layer boundaries for
  historical builds. Coarse (~4k rows/build).
- `plotter_commands` (Postgres table) — per-command `(build_id, ts,
  layer_idx, cmd_idx, op, x, y, laser, speed)`. Vector-level laser ground
  truth. PLUMBED BUT EMPTY as of 2026-07-13 — no print has run since the
  stream was added; first future build populates it.
- `frames.jsonl` export — `(id, build_id, ts, kind, path)` index of all
  frames.

**Known gap — layer alignment:** frames are timestamped, not
layer-indexed. Build a proper `frame → (layer, phase)` index where phase ∈
{recoat, scan}: from `plotter_commands.layer_idx` timestamps going
forward; from `z1` drops + galvo-image transitions for historical builds
(builds ~12–46 exist). This index is prerequisite plumbing for everything
else — build it first.

### Peregrine (pretraining / taxonomy reference)

`https://huggingface.co/datasets/ppak10/Peregrine-Dataset-v2023-11` —
ppak10's mirror of ORNL Peregrine v2023-11 (ORNL-Peregrine license, cite
Scime et al.). 5 builds, SS 316L, Concept Laser M2 (metal LPBF), 17,582
layers, 712 GB. Per layer: after-powder + after-melt float32 images
(1842²), pixel-wise masks for 12 anomaly classes, part-ID maps, laser
scan paths + exposure times, 19 sensor scalars. Use the
`anomaly_0250/0500/1000` benchmark configs for prototyping — don't pull
the full 712 GB.

Structural mapping to our capture is ~1:1: after-powder/after-melt ↔
post-recoat/post-scan chamber frames; part-ID maps ↔ galvo mask; scan
paths ↔ plotter_commands; sensor scalars ↔ telemetry (+ our bedmatrix is
spatially richer).

Taxonomy mapping (metal → polymer SLS): KEEP recoater streaking,
recoater hopping, incomplete spreading (= short feed), debris,
super-elevation/swelling (≈ curl — the killer SLS failure mode). DROP or
redefine spatter, over-melting, under-melting (melt-pool phenomena).
Target label set: ~6–7 classes.

## Model plan (decided in parent-repo session 2026-07-13)

- ~4M-param encoder-decoder: SegFormer-B0 or MobileNetV3-encoder U-Net.
  DSCNN (Scime et al. 2020, doi 10.1016/j.addma.2020.101453) is the
  reference architecture: parallel local/regional/global legs, multi-
  sensor fusion, machine-agnostic — mirror its fusion idea, not its code.
- Input = stacked channels per layer event: chamber frame, galvo mask
  warped into camera space, frame-diff vs previous layer, upsampled
  bedmatrix thermal channel.
- v1 head: per-REGION classification (galvo mask splits part vs powder
  regions; classify each) — literature (IJAMT 2026 CAD-guided lightweight
  CNN framework, doi 10.1007/s00170-026-17884-2) finds this beats blind
  dense segmentation for streaking/super-elevation, and it's far cheaper
  to label. v2: dense segmentation once pixel labels exist.
- One-time calibration needed: homography galvo/bed coords → chamber
  pixels (chamber cam has fisheye; may need distortion correction first).
- Label bootstrap: retro-label historical builds from 7810, prioritizing
  known failures. Pretrain on Peregrine configs, fine-tune on Inova.
- Consider classical-CV baselines first (streak lines along recoat axis,
  exposed-bed contrast for short feed) — they set the bar the CNN must
  beat and may suffice for v1 alerting.

## Deployment constraints — IMPORTANT

- Inference host `7810` GPUs: Quadro K2200 (4 GB, Maxwell sm_50) + 2×
  K620 (2 GB). Modern PyTorch wheels dropped sm_50 (2.8+cu126 needs CC
  ≥6.1; CUDA 13 drops Maxwell from toolkit). Do NOT plan GPU inference
  there. Decided: export ONNX, run `onnxruntime` on CPU — cadence is one
  inference per layer (~30–120 s), a few hundred ms on CPU is 100×
  headroom.
- Training happens on THIS machine (the GPU box) with a normal modern
  torch stack; only the ONNX artifact ships to 7810.

## Conventions (inherited from parent repo)

- Python via `uv` (`uv run …`); `uv run pytest` for tests.
- Precision > recall for anything that triggers actuation; recall
  matters for notify-only alerts. Tier accordingly.
- Reference data is fixed at its origin repo — don't hand-edit copies.
