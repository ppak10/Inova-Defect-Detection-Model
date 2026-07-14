"""
Shared constants for v01 — layer-wise powder-bed defect detection.

Verified capture facts (see CLAUDE.md "Training data" — do not
rediscover):
- Inova ticks (HF ppak10/Inova-Mk1-Telemetry): one row per 10 Hz tick,
  frames embedded inline (frame_chamber jpg, frame_galvo png,
  frame_thermal gif), ~64 sensor columns, position_hf_burst lists.
  Layer boundaries: z2 advances once per deposited layer.
- Peregrine (HF ppak10/Peregrine-Dataset-v2023-11): one row per layer,
  1842x1842 float32 images in 0-255 range (after_powder, after_melt),
  uint32 part_ids map, 12 bool segmentation masks, scan_path (N,5).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Local dataset locations (downloaded via `hf download --local-dir`)
# ---------------------------------------------------------------------------
DATASETS_ROOT = Path("/mnt/am/ppak/HuggingFace/Datasets")
PEREGRINE_DIR = DATASETS_ROOT / "Peregrine-Dataset-v2023-11"
INOVA_DIR = DATASETS_ROOT / "Inova-Mk1-Telemetry"

DATA_DIR = Path(__file__).parent / "data"  # prepare.py cache (gitignored)

# ---------------------------------------------------------------------------
# Common input format — the intersection of both domains (AMT v03 lesson:
# define one format both domains deliver, augment toward the deployment
# domain). Registered top-down bed view, grayscale, per-image standardized.
# Peregrine 1842^2 is INTER_AREA-downsampled; the Inova chamber view
# (~650 px, fisheye) will be undistorted + warped onto the same grid by
# registration.py.
# ---------------------------------------------------------------------------
# 1024 (not 512): Peregrine streaks are 1-2 px hairlines at 1842^2 and
# must survive the downsample; GPU-primary inference removed the reason
# to go smaller. Models may still train at lower res by resizing at load.
COMMON_SIZE = 1024

# Input channels (both domains):
#   0: post-recoat image  (Peregrine after_powder | Inova post-recoat chamber)
#   1: post-scan image    (Peregrine after_melt   | Inova post-scan chamber)
#   2: part mask          (Peregrine part_ids > 0 | Inova warped galvo mask)
NUM_CHANNELS = 3

# Powder (background) regions: grid tiling of non-part pixels
POWDER_GRID = 8  # 8x8 tiles over the bed

# Region label = anomaly-class pixel fraction within the region; binarize
# with these at train time (tunable without re-running prepare).
REGION_MIN_AREA_PX = 256  # full-res px; skip slivers
LABEL_FRACTION_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Peregrine segmentation classes. ALL_CLASSES order = bit position in the
# cached dense-mask bitmask (uint16 per pixel, bit set = class present
# anywhere in the source-resolution block — "any coverage" rule so thin
# streaks survive 1842->COMMON_SIZE downsampling).
# ---------------------------------------------------------------------------
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

# Anomaly subset (label-vector positions in the region-fraction table).
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

# Metal LPBF -> polymer SLS transfer relevance (CLAUDE.md taxonomy
# mapping). Melt-pool classes are pretraining-only signal.
TRANSFER_CLASSES = [
    "recoater_hopping",
    "recoater_streaking",
    "incomplete_spreading",  # = short feed
    "swelling",  # ~ curl precursor
    "debris",
    "super_elevation",  # ~ curl
]

# Inova chamber frames must be rotated 90 deg CLOCKWISE to match the
# galvo image orientation (user-confirmed 2026-07-13); apply BEFORE any
# registration/warping. cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE).
CHAMBER_ROTATE = "cw90"

# Inova layer event phases
PHASE_RECOAT = "recoat"
PHASE_SCAN = "scan"
