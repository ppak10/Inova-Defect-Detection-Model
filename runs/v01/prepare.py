"""
One-time data cache builder for v01 (Peregrine side).

STREAMS Peregrine layer parquets from HF (no full local clone — decided
2026-07-13) and keeps only a compact training cache in the common input
format. Falls back to local files under PEREGRINE_DIR when present
(e.g. the anomaly_0250 dev subset).

- runs/v01/data/peregrine/build_<b>_layer_<l>.npz
    images     (2, 1024, 1024) float16 — after_powder, after_melt (0-255)
    part_mask  (1024, 1024)    bool    — part_ids > 0
    region_map (1024, 1024)    int16   — per-pixel region index (-1 = none)
    class_bits (1024, 1024)    uint16  — dense-mask bitmask; bit i =
        PEREGRINE_ALL_CLASSES[i] present anywhere in the source block
        ("any coverage", so 1-2 px streaks survive 1842->1024)
- runs/v01/data/peregrine/regions.parquet
    one row per region: (build, layer, region_idx, kind, part_id,
    area_px, frac_<class> x 10) — anomaly pixel fractions computed at
    FULL resolution. This is the per-region label table (agent output
    contract) used for the region-aggregation head + eval.

Regions: one per part id (galvo-mask analogue on Inova) plus an 8x8
grid tiling of the powder background.

Already-cached layers are skipped, so an interrupted stream resumes.
Delete a layer's npz (and keep regions.parquet) to force re-processing.

Usage:
    uv run python -m runs.v01.prepare [--source anomaly|build]
        [--max-cases 5] [--workers 8]
"""

from __future__ import annotations

import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .constants import (
    COMMON_SIZE,
    DATA_DIR,
    PEREGRINE_ALL_CLASSES,
    PEREGRINE_CLASSES,
    PEREGRINE_DIR,
    POWDER_GRID,
    REGION_MIN_AREA_PX,
)

HF_REPO = "datasets/ppak10/Peregrine-Dataset-v2023-11"
OUT_DIR = DATA_DIR / "peregrine"

KIND_PART = "part"
KIND_POWDER = "powder"


def _decode(buf: bytes) -> np.ndarray:
    return np.load(io.BytesIO(buf), allow_pickle=False)


def _read_table(spec: str):
    """spec is either a local path or an hf://-style repo path."""
    local = PEREGRINE_DIR / spec
    if local.exists():
        return pq.read_table(local)
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"{HF_REPO}/{spec}", "rb") as f:
        return pq.read_table(f)


def _powder_tile_map(shape: tuple[int, int]) -> np.ndarray:
    """Tile index (0..POWDER_GRID^2-1) for every pixel."""
    h, w = shape
    rows = np.minimum(np.arange(h) * POWDER_GRID // h, POWDER_GRID - 1)
    cols = np.minimum(np.arange(w) * POWDER_GRID // w, POWDER_GRID - 1)
    return rows[:, None] * POWDER_GRID + cols[None, :]


def process_layer(spec: str) -> list[dict] | None:
    """Build the npz cache + region label rows for one layer parquet."""
    row = _read_table(spec).to_pylist()[0]
    build, layer = int(row["build"]), int(row["layer"])
    out_npz = OUT_DIR / f"build_{build}_layer_{layer:04d}.npz"
    if out_npz.exists():
        return None

    powder_img = _decode(row["image_after_powder"])
    melt_img = _decode(row["image_after_melt"])
    part_ids = _decode(row["part_ids"])
    masks = {
        c: _decode(row[f"segmentation_{c}"]) for c in PEREGRINE_ALL_CLASSES
    }

    # --- regions + anomaly fractions at full resolution -------------------
    region_map_full = np.full(part_ids.shape, -1, dtype=np.int16)
    records: list[dict] = []

    pids = np.unique(part_ids)
    pids = pids[pids > 0]
    for idx, pid in enumerate(pids):
        m = part_ids == pid
        area = int(m.sum())
        if area < REGION_MIN_AREA_PX:
            continue
        region_map_full[m] = idx
        rec = {
            "build": build,
            "layer": layer,
            "region_idx": idx,
            "kind": KIND_PART,
            "part_id": int(pid),
            "area_px": area,
        }
        for c in PEREGRINE_CLASSES:
            rec[f"frac_{c}"] = float(masks[c][m].sum() / area)
        records.append(rec)

    n_parts = len(pids)
    tiles = _powder_tile_map(part_ids.shape)
    powder = part_ids == 0
    for t in range(POWDER_GRID * POWDER_GRID):
        m = powder & (tiles == t)
        area = int(m.sum())
        if area < REGION_MIN_AREA_PX:
            continue
        idx = n_parts + t
        region_map_full[m] = idx
        rec = {
            "build": build,
            "layer": layer,
            "region_idx": idx,
            "kind": KIND_POWDER,
            "part_id": 0,
            "area_px": area,
        }
        for c in PEREGRINE_CLASSES:
            rec[f"frac_{c}"] = float(masks[c][m].sum() / area)
        records.append(rec)

    # --- downsample to the common format ----------------------------------
    size = (COMMON_SIZE, COMMON_SIZE)
    images = np.stack(
        [
            cv2.resize(powder_img, size, interpolation=cv2.INTER_AREA),
            cv2.resize(melt_img, size, interpolation=cv2.INTER_AREA),
        ]
    ).astype(np.float16)
    part_mask = (
        cv2.resize(
            (part_ids > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    )
    region_map = cv2.resize(region_map_full, size, interpolation=cv2.INTER_NEAREST)

    class_bits = np.zeros(size, dtype=np.uint16)
    for i, c in enumerate(PEREGRINE_ALL_CLASSES):
        coverage = cv2.resize(
            masks[c].astype(np.float32), size, interpolation=cv2.INTER_AREA
        )
        class_bits |= (coverage > 0).astype(np.uint16) << i

    np.savez_compressed(
        out_npz,
        images=images,
        part_mask=part_mask,
        region_map=region_map,
        class_bits=class_bits,
    )
    return records


def list_layers(source: str) -> list[str]:
    """Layer parquet specs (paths relative to the repo/local root).

    The REMOTE listing is the source of truth — a partial local download
    must not hide layers. _read_table() short-circuits to local files
    per layer where they exist.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    specs = sorted(
        p.removeprefix(f"{HF_REPO}/")
        for p in fs.glob(f"{HF_REPO}/data/{source}/**/*.parquet")
    )
    if source != "anomaly":
        return specs
    # anomaly_0250 is a subset of 0500 is a subset of 1000 — dedupe by
    # filename (build_<b>_layer_<l>.parquet), keep first occurrence. Full
    # builds use per-build dirs with bare layer_<l>.parquet names, so
    # filename dedupe would wrongly collapse layers across builds there.
    seen: set[str] = set()
    return [s for s in specs if not (Path(s).name in seen or seen.add(Path(s).name))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="anomaly",
        choices=["anomaly", "build"],
        help="anomaly = benchmark configs; build = full 17.5k-layer builds",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    specs = list_layers(args.source)
    if args.max_cases:
        specs = specs[: args.max_cases]
    print(f"{len(specs)} layers from {args.source}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, recs in enumerate(pool.map(process_layer, specs, chunksize=1)):
            if recs:
                all_records.extend(recs)
            if (i + 1) % 25 == 0 or i + 1 == len(specs):
                print(f"  {i + 1}/{len(specs)} layers, {len(all_records)} new regions")

    regions = pd.DataFrame(all_records)
    out = OUT_DIR / "regions.parquet"
    if out.exists():  # merge with prior runs (resume / anomaly then build)
        prior = pd.read_parquet(out)
        regions = (
            pd.concat([prior, regions])
            .drop_duplicates(subset=["build", "layer", "region_idx"], keep="last")
            .reset_index(drop=True)
        )
    regions.to_parquet(out, index=False)
    print(f"wrote {out} ({len(regions)} regions)")


if __name__ == "__main__":
    main()
