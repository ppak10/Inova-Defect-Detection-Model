"""
Replay driver — plays the 7810 recorder's role against the inference
service for a deployment smoke test.

    uv run python -m runs.v03.replay [--build 032]
        [--url http://localhost:8100] [--limit 0]

Streams a historical build's ticks, assembles one payload per layer
with galvo activity (all chamber + galvo frames in the z2 window,
plus the previous layer's post-scan frame), POSTs to /infer, and
appends build_events-shaped rows to runs/v03/replay_events.jsonl:

    {"ts": ..., "build_id": ..., "kind": "defect_detection",
     "payload": {scores, alerts, quality, latency_ms}}

This is the same call pattern the recorder will use live (fires in the
recoat->scan gap for short feed; here we fire once per layer window).
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests

from .constants import INOVA_DIR

EVENTS_PATH = Path(__file__).parent / "replay_events.jsonl"


def b64(frame: dict) -> str:
    return base64.b64encode(frame["bytes"]).decode()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", default="032")
    p.add_argument("--url", default="http://localhost:8100")
    p.add_argument("--limit", type=int, default=0, help="max layers (0 = all)")
    args = p.parse_args()

    health = requests.get(f"{args.url}/health", timeout=10).json()
    print("service:", health["status"], "|", health["checkpoint"], "|", health["device"])

    t = pq.read_table(
        INOVA_DIR / f"data/ticks/{args.build}.parquet",
        columns=["ts", "positions.position.z2", "frame_chamber", "frame_galvo"],
    )
    z2 = t["positions.position.z2"].to_numpy()
    fc = t["frame_chamber"].to_pylist()
    fg = t["frame_galvo"].to_pylist()
    bounds = np.where(np.diff(z2) > 0)[0]

    prev_scan_b64 = None
    sent = 0
    with EVENTS_PATH.open("a") as out:
        for li, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            galvo = [fg[i] for i in range(a, b) if fg[i] is not None]
            chamber = [fc[i] for i in range(a, b) if fc[i] is not None]
            if not galvo or not chamber:
                continue
            payload = {
                "build_id": int(args.build),
                "layer": li,
                "chamber_frames": [b64(f) for f in chamber],
                "galvo_frames": [b64(f) for f in galvo],
                "prev_scan_frame": prev_scan_b64,
            }
            t0 = time.time()
            r = requests.post(f"{args.url}/infer", json=payload, timeout=120)
            r.raise_for_status()
            res = r.json()
            wall = (time.time() - t0) * 1000
            if res["quality"]["galvo_px"] == 0:
                continue  # heating/idle window — nothing scanned
            prev_scan_b64 = payload["chamber_frames"][-1]
            event = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "build_id": int(args.build),
                "kind": "defect_detection",
                "payload": {k: res[k] for k in ["layer", "scores", "alerts", "quality", "latency_ms"]},
            }
            out.write(json.dumps(event) + "\n")
            sent += 1
            top = max(res["scores"], key=res["scores"].get)
            print(
                f"layer {li:3d}: alerts={res['alerts'] or '-'} top={top}:{res['scores'][top]:.2f} "
                f"infer={res['latency_ms']:.0f}ms wall={wall:.0f}ms"
            )
            if args.limit and sent >= args.limit:
                break
    print(f"{sent} layer events -> {EVENTS_PATH}")


if __name__ == "__main__":
    main()
