"""
Shared constants for v03 — v01 + temporal context & rebalanced region
supervision. See runs/v02/README.md for the v01→v02 change table and
runs/v01/constants.py for dataset verification notes.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Local dataset locations
# ---------------------------------------------------------------------------
DATASETS_ROOT = Path("/mnt/am/ppak/HuggingFace/Datasets")
PEREGRINE_DIR = DATASETS_ROOT / "Peregrine-Dataset-v2023-11"
INOVA_DIR = DATASETS_ROOT / "Inova-Mk1-Telemetry"

DATA_DIR = Path(__file__).parent / "data"  # v02's own outputs (gitignored)

# The 44 GB Peregrine layer cache is a shared artifact built by
# runs/v01/prepare.py — v02 reads it in place rather than duplicating.
PEREGRINE_CACHE = Path(__file__).parent.parent / "v01" / "data" / "peregrine"

# ---------------------------------------------------------------------------
# Common input format (unchanged from v01)
# ---------------------------------------------------------------------------
COMMON_SIZE = 1024
POWDER_GRID = 8
REGION_MIN_AREA_PX = 256
LABEL_FRACTION_THRESHOLD = 0.05

PEREGRINE_ALL_CLASSES = [
    "powder",
    "printed",
    "recoater_hopping",
    "recoater_streaking",
    "incomplete_spreading",
    "swelling",
    "debris",
    "super_elevation",
    "spatter",
    "misprint",
    "over_melting",
    "under_melting",
]

PEREGRINE_CLASSES = [
    "recoater_hopping",
    "recoater_streaking",
    "incomplete_spreading",
    "swelling",
    "debris",
    "super_elevation",
    "spatter",
    "misprint",
    "over_melting",
    "under_melting",
]

TRANSFER_CLASSES = [
    "recoater_hopping",
    "recoater_streaking",
    "incomplete_spreading",
    "swelling",
    "debris",
    "super_elevation",
]

# Inova facts (see repo CLAUDE.md "Measured imaging reality")
CHAMBER_ROTATE = "cw90"
PHASE_RECOAT = "recoat"
PHASE_SCAN = "scan"
