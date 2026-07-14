"""Data loading: Inova frame/telemetry ingest and Peregrine pretraining sets.

Responsibilities (see CLAUDE.md "Training data"):
- frame -> (layer, phase) index construction: from plotter_commands
  layer_idx timestamps going forward; from position z1 drops + galvo
  transitions for historical builds.
- Layer-event sampling: post-recoat and post-scan chamber frames plus the
  nearest bedmatrix grid per layer.
- Peregrine anomaly_0250/0500/1000 config loaders for pretraining.
"""
