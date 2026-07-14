# Inova-Defect-Detection-Model

Layer-wise powder-bed defect detection for the [SLS4All Inova MK1](https://sls4all.com/)
polymer SLS printer. Detects spreading and part anomalies (recoater
streaking, short feed, curl, debris) from chamber-camera frames, the
firmware's galvo scan mask, and a 32×24 bed temperature matrix, so an
agentic process-optimization system can react mid-print.

Part of [Agentic-Additive-Manufacturing-Process-Optimization](https://github.com/ppak10/Agentic-Additive-Manufacturing-Process-Optimization)
(included there as a submodule). Pretraining draws on the
[Peregrine v2023-11 dataset](https://huggingface.co/datasets/ppak10/Peregrine-Dataset-v2023-11)
(ORNL, Scime et al.); fine-tuning uses the
[Inova-Mk1-Telemetry dataset](https://huggingface.co/datasets/ppak10/Inova-Mk1-Telemetry).

## Layout

- `inova_defect_detection/` — package: data loaders, galvo→chamber
  registration, model, training/export entry points.
- `tests/` — `uv run pytest`.

See `CLAUDE.md` for the full design context and data contracts.

## Setup

```sh
uv sync
uv run pytest
```
