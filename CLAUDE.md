# Inova Defect Detection Model

Layer-wise powder-bed defect detection for the SLS4All Inova MK1 (polymer
SLS, PA12). Trains here (GPU machine), deploys as a CPU background service
on the recorder host (`7810`), and feeds detections to the agentic system
in the parent repo so agents can react mid-print (recoater compensation,
part exclusion proposals).

## Repo layout

Experiments live in `runs/vXX/`, one directory per model iteration —
same convention as the AMT repo (`../AMT`): `constants.py`,
`prepare.py` (one-time data cache builder), `dataset.py`,
`registration.py` (galvo→chamber calibration), `model.py`,
`trainer.py`, `train.py`, `evaluate.py`, `visualize.py`, `export.py`
(ONNX), `README.md` (design rationale + v(X-1)→vX change table).
Run everything as `uv run python -m runs.vXX.<script>` (module form
required; scripts import `runs.*`). Every stage should support
`--max-cases 5` for smoke tests.

`runs/*/data|checkpoints|figures|exports` and `wandb/` are gitignored.
Experiment tracking: wandb, one project per run (`inova-defect-v01`).
Current run: **v05**. The 44 GB Peregrine cache is a shared artifact
under runs/v01/data/peregrine — later runs read it in place. State
after v01-v05 (details in each run README): swelling/curl 0.683 (v05
region head), short-feed 0.578 (pixstat logistic — beats all neural
attempts), streaking 0.771 (build-5 holdout; build-3 streak labels are
an exposure artifact, unusable), debris data-limited in Peregrine
(awaits Inova human-loop labels). Peregrine iteration is at
diminishing returns; the frontier is transfer-side (registration,
serve port to v04+ signature, human feedback loop). Inference service:
runs/v05/serve.py on :8100 (deploy/restart: runs/v05/deploy.sh) with
per-class routing — swelling/super_elevation from the region head,
incomplete_spreading from the pixstat logistic
(runs/v05/pixstat_shortfeed.npz), rest from the dense map. 7810 calls
POST /infer with the layer's raw frames (see serve.py contract;
runs/v03/replay.py is a reference client).

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
- PRIORITY CLASSES (user, 2026-07-14): the failure modes actually seen
  on the Inova are (1) DEBRIS that compounds across layers and damages
  other parts, and (2) SHORT FEED causing over-rastering of parts.
  Curl has NOT been observed lately — deprioritize swelling/
  super_elevation (super_elevation is also untrainable: 73/83 positives
  are in build 5). Planned remedies exist for both: extra recoat pass;
  skip/exclude part.
- Short feed detection WINDOW: visible post-recoat toward the overflow
  ("over powder") bin side as the previous layer's raster showing
  through. The remedy (another recoat) must run BEFORE the scan starts
  → inference must trigger on the post-recoat frame, in the
  recoat→scan gap, not at layer end.
- CURRENT operating point (human-gated phase): user prefers SENSITIVITY
  over precision — false positives are acceptable because a human makes
  the final call. The precision-first rule above kicks in only when
  actuation becomes autonomous.
- No agent light control exists, and halogens can't be pulsed freely
  (surface damage risk) — do not plan capture-side lighting fixes.
- Powder: all builds ran reused PA12-GF (glass-filled, ≥1 prior cycle);
  powder history/refresh ratio is untracked so appearance drift across
  builds is expected.

## Training data

### Inova capture (ours, polymer SLS)

ACCESS PATH (verified 2026-07-13): HF `ppak10/Inova-Mk1-Telemetry` is
self-sufficient — `data/ticks/<build>.parquet`, one row per 10 Hz tick
with frame_chamber/frame_galvo/frame_thermal EMBEDDED inline (HF Image
structs, null when no frame fell in the 100 ms window), ~64 sensor
columns, and 1 kHz `position_hf_burst` lists. 25 builds (001–046),
165 GB. Layer boundaries: **z2** advances once per deposited layer (NOT
z1 as previously noted; the dataset repo's timelapse scripts have
reference detection logic). Raw 32×24 bedmatrix floats are NOT in the
mirror — the thermal signal is the rendered frame_thermal image.
Stream ticks rather than downloading all 165 GB; small dev builds
(013/032/038) are local under /mnt/am/ppak/HuggingFace/Datasets/.

**Measured imaging reality (build 032/013/038 analysis, 2026-07-13 — do
not rediscover):**
- Chamber illumination is driven by the halogen BED HEATERS, not a
  dedicated light: frames swing from pitch black (halogens off, bed at
  temp) through hazy low-contrast gray to completely blown-out white
  (halogens on). Mean brightness is NOT a quality metric — both
  extremes are useless; score frames by gradient/detail energy with
  saturation+black penalties. In 032 every layer had ≥1 usable frame,
  but usable ≠ high-contrast: even good frames are hazy, sintered
  regions read as subtle darkening. 013 was 97% lit; lighting regime
  varies per build.
- The laser spot glows visibly in chamber frames during scanning (useful
  scan-progress signal, nuisance for single-frame semantics).
- `frame_galvo` is NOT a binary current-layer mask — it renders the
  laser trace with time-fade (white=recent → gray=older). The per-layer
  part mask must be ACCUMULATED (binarize + max) over the layer's galvo
  frames.
- `laser.power` at 10 Hz ticks is far undersampled (30 nonzero ticks
  across 91 layers in 032) — unusable for phase detection; use z2 steps
  + galvo-frame activity instead.
- Chamber frames must be rotated 90° CLOCKWISE to match the galvo image
  orientation (user-confirmed 2026-07-13). Apply before any
  registration/warping.
- Recoater travels LEFT → RIGHT in the rotated (galvo-aligned) view
  (user-confirmed). Streaks therefore run horizontally there; orient
  the common grid so this matches the Peregrine streak axis.
- Camera is never remounted/refocused between builds (user-confirmed) —
  registration is ONE-TIME. The lid opens/closes between builds, so
  allow a small per-build refinement, but the bed-plate alignment is
  central to the machine design.
- Bed geometry (sls4all.com): build chamber 177×177 mm, effective PA12
  build area 150×150 mm (×185 mm z). Whether the galvo image spans the
  full chamber or the effective area still needs verifying against a
  known part geometry.
- Halogen state IS partially in telemetry (measured on 032):
  `lights.lights.enabled` corr +0.56 with frame brightness,
  `powerman.power.current` (200–1200 W — the halogen draw) corr +0.45.
  Use as a causal lighting prior for frame quality/selection; imperfect
  because of thermal lag (halogens glow after power-off).
- Inova defect ground truth does NOT exist and is not reconstructable
  from memory (user-confirmed): failures were process-parameter-driven;
  recycled-powder builds (030/031) are prime suspects but unverified.
  Retrospective screening plan: self-supervised / embedding-outlier
  analysis (e.g. PCA over per-layer features) to surface anomalous
  layers for human review — not hand labeling.
- Consequence (decided): Inova inference input is a SEQUENCE of frames
  per layer (K frames spanning recoat+scan), fused by an
  attention-pooling module trained with synthetic halogen-state
  augmentation on Peregrine single images (black/dim/hazy/blown-out +
  glare gradients + laser-spot artifacts) — AMT-v03-style synthetic
  degradation, calibrated against these measured stats.

Origin capture (context; raw frames live on `7810` under
`data/frames/<build_id>/`):

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

## Model plan (revised 2026-07-13, this-repo session; supersedes the
## parent-repo ~4M-CPU plan)

- Foundation-backbone dense segmentation: DINOv2 ViT encoder (size
  configurable, start ViT-B; frozen first, LoRA/unfreeze as ablation) +
  segmentation head over the 12 Peregrine classes. GPU-primary inference
  (below) removed the small-model constraint; frozen foundation features
  are also the best bet against the metal→polymer domain gap. SAM/SAM2
  are class-agnostic — their role is backbone/ablation, not the model.
- Output contract (decided): the DENSE MAP ONLY — 12 independent
  per-class sigmoid channels (multi-label, NOT one-hot: anomalies
  overlay powder/printed; per-class thresholds enable actuation
  tiering) on the common bed grid, delivered in bed/galvo coordinates.
  The agentic system does its own map→part attribution from prescribed
  geometry. Region aggregation (part regions + 8x8 powder tiles) is
  INTERNAL: a training loss term (region fractions are exact full-res
  labels) and an eval view — not served.
- Common input format (3x512x512): post-recoat image, post-scan image,
  part mask — Peregrine (after_powder, after_melt, part_ids>0) ==
  Inova (chamber post-recoat, chamber post-scan, warped galvo mask).
  Per-image standardization; augment Peregrine toward measured Inova
  imaging conditions (AMT-v03 lesson). Thermal channel deferred to
  Inova fine-tuning (Peregrine has no analogue).
- Labels: train supervised on Peregrine only (Inova has NO defect
  labels). Transfer to Inova zero-shot; human confirm/reject of live
  detections builds the Inova label set for later fine-tuning.
- One-time calibration needed: homography galvo/bed coords → chamber
  pixels (chamber cam has fisheye; may need distortion correction first).
- Classical-CV baselines (streak lines along recoat axis, exposed-bed
  contrast for short feed) still set the bar the model must beat.
- Data handling (decided): NO full local clones — prepare.py streams
  layers from HF and caches only the compact 1024^2 common-format
  arrays (~45 GB for all 17.5k layers; 1024 so 1-2 px streak hairlines
  survive). Raw dev subsets live under
  /mnt/am/ppak/HuggingFace/Datasets/.

## Deployment (decided 2026-07-13, supersedes ONNX-on-7810-CPU plan)

- GPU-PRIMARY inference: a stateless torch service on THIS machine
  (3× RTX 6000 Ada; reserve ONE GPU for inference, train on the others).
  Contract: layer-event frames in → (region, class, score) rows out.
  The 7810 recorder calls it over the network; latency is irrelevant at
  one inference per layer (~30–120 s). An A4500 box is an alternative
  remote host.
- Availability model: detections are ADVISORY suggestions to the agentic
  system — if the GPU host is down, the agent proceeds without them.
  No CPU fallback required. (Optional ONNX-CPU export remains possible
  but is not a design constraint anymore.)
- 7810's own GPUs (K2200 4 GB + 2× K620, Maxwell sm_50) are ruled out:
  modern torch/CUDA dropped sm_50, and ~1.3 TFLOPS isn't worth the
  frozen-driver-stack tax.
- GPU headroom also enables the future streaming mode (score every
  recoat frame at ~10 fps to catch streaks mid-spread), which per-layer
  CPU inference never could.

## Conventions (inherited from parent repo)

- Python via `uv` (`uv run …`); `uv run pytest` for tests.
- Precision > recall for anything that triggers actuation; recall
  matters for notify-only alerts. Tier accordingly.
- Reference data is fixed at its origin repo — don't hand-edit copies.
