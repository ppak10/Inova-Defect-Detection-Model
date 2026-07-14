"""
Inference service — the standing endpoint the 7810 recorder calls.

    uv run python -m runs.v03.serve [--port 8100] [--gpu 2]
        [--checkpoint runs/v03/checkpoints/best.pt]

Contract (see CLAUDE.md "Deployment"):
- GET  /health -> {status, checkpoint, device, classes}
- POST /infer  -> layer event in, detections out. Request JSON:
      {
        "build_id": int, "layer": int,
        "chamber_frames": [b64 jpg/png, ...],   # the layer's frames, raw
        "galvo_frames":   [b64 png, ...],       # same window
        "prev_scan_frame": b64 | null           # previous layer's post-scan
      }
  The service does frame quality selection, galvo accumulation,
  registration (config transform), inference. Response:
      {
        "scores": {class: max dense prob},
        "alerts": [classes over serve_config thresholds],
        "maps_png": {class: b64 grayscale PNG, for alert classes},
        "quality": {...frame selection diagnostics...},
        "latency_ms": float
      }

Thresholds live in runs/v03/serve_config.json — SENSITIVITY-FIRST while
detections are human-gated. Registration transform is config too;
refit when calibration improves (registration.py is the real fix).

Detections are ADVISORY: if this service is down, the agent proceeds
without it (decided 2026-07-13).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .constants import PEREGRINE_ALL_CLASSES, TRANSFER_CLASSES
from .dataset import standardize
from .inova_demo import decode_gray, galvo_mask, quality, warp_mask
from .model import DefectSegmenter, ModelConfig

CONFIG_PATH = Path(__file__).parent / "serve_config.json"
DEFAULT_CONFIG = {
    # sensitivity-first (human-gated phase)
    "thresholds": {c: 0.35 for c in TRANSFER_CLASSES},
    # registration-lite similarity galvo->rotated-chamber; refit per
    # calibration (registration.py will replace this)
    "registration": {"scale": 1.0, "tx": 0.0, "ty": 0.0},
    "input_size": 518,
}


def _b64_to_frame(b: str) -> dict:
    return {"bytes": base64.b64decode(b)}


def _map_to_b64png(m: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((m * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Engine:
    def __init__(self, checkpoint: str, device: str):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.model = DefectSegmenter(
            ModelConfig(
                backbone=cfg["backbone"],
                input_size=cfg["input_size"],
                logits_size=cfg["logits_size"],
            )
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self.device = device
        self.checkpoint = checkpoint
        self.config = DEFAULT_CONFIG | (
            json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        )

    @torch.no_grad()
    def infer(self, req: dict) -> dict:
        t0 = time.time()
        chamber = [_b64_to_frame(b) for b in req["chamber_frames"]]
        galvo = [_b64_to_frame(b) for b in req["galvo_frames"]]

        grays = [decode_gray(f) for f in chamber]
        quals = [quality(g) for g in grays]
        order = np.argsort(quals)
        recoat_g, scan_g = grays[order[-2]] if len(grays) > 1 else grays[order[-1]], grays[order[-1]]

        acc = None
        for f in galvo:
            m = galvo_mask(f)
            acc = m if acc is None else (acc | m)
        reg = self.config["registration"]
        pmask = (
            warp_mask(acc, scan_g.shape, reg["scale"], reg["tx"], reg["ty"])
            if acc is not None and acc.sum() > 0
            else np.zeros(scan_g.shape, dtype=bool)
        )

        prev_g = (
            decode_gray(_b64_to_frame(req["prev_scan_frame"]))
            if req.get("prev_scan_frame")
            else None
        )

        size = self.config["input_size"]

        def prep(g):
            return torch.tensor(standardize(cv2.resize(g, (size, size))))

        seq = [prev_g if prev_g is not None else np.zeros_like(recoat_g), recoat_g, scan_g]
        frames = torch.stack([prep(g) for g in seq])[None]
        phase_ids = torch.tensor([[2, 0, 1]])
        bright = [float(np.mean(g)) / 255 for g in seq]
        meta = torch.tensor(
            [[[-1.0, bright[0], 1.0], [0.5, bright[1], 1.0], [0.0, bright[2], 1.0]]],
            dtype=torch.float32,
        )
        valid = torch.tensor([[prev_g is not None, True, True]])
        part = torch.tensor(
            cv2.resize(pmask.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
        )[None]

        logits = self.model(
            frames.to(self.device),
            phase_ids.to(self.device),
            meta.to(self.device),
            valid.to(self.device),
            part.to(self.device),
        )
        probs = logits.float().sigmoid()[0].cpu().numpy()

        scores = {
            c: float(probs[PEREGRINE_ALL_CLASSES.index(c)].max()) for c in TRANSFER_CLASSES
        }
        thr = self.config["thresholds"]
        alerts = [c for c, s in scores.items() if s >= thr.get(c, 1.1)]
        maps = {
            c: _map_to_b64png(probs[PEREGRINE_ALL_CLASSES.index(c)]) for c in alerts
        }
        return {
            "build_id": req.get("build_id"),
            "layer": req.get("layer"),
            "scores": scores,
            "alerts": alerts,
            "maps_png": maps,
            "quality": {
                "n_chamber": len(grays),
                "best_quality": float(max(quals)) if quals else 0.0,
                "galvo_px": int(acc.sum()) if acc is not None else 0,
                "has_prev": prev_g is not None,
            },
            "latency_ms": (time.time() - t0) * 1000,
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--checkpoint", default="runs/v03/checkpoints/best.pt")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    engine = Engine(args.checkpoint, device)

    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="inova-defect-detection")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "checkpoint": engine.checkpoint,
            "device": engine.device,
            "classes": TRANSFER_CLASSES,
            "thresholds": engine.config["thresholds"],
        }

    @app.post("/infer")
    def infer(req: dict):
        return engine.infer(req)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
