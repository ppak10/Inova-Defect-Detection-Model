"""
Entry point: uv run python -m runs.v01.export --checkpoint ...

Exports the trained model to ONNX (runs/v01/exports/, gitignored) and
smoke-tests it under CPU onnxruntime — the artifact that ships to the
recorder host. Cadence there is one inference per layer (~30-120 s), so
a few hundred ms on CPU is 100x headroom.
"""
