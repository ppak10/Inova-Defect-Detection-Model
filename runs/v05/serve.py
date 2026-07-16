"""
Inference service (v05) — the standing endpoint the 7810 recorder calls.

    uv run python -m runs.v05.serve [--port 8100] [--gpu 2]

PER-CLASS ROUTING (see runs/v05/README.md results):
    swelling, super_elevation  <- v05 REGION HEAD (max over part regions)
    incomplete_spreading       <- PIXSTAT logistic (max over powder
                                  tiles; runs/v05/pixstat_shortfeed.npz)
    streaking, hopping, debris <- v05 DENSE map (max prob)

Contract (unchanged from v03 — 7810-side callers need no changes):
- GET  /health
- POST /infer {build_id, layer, chamber_frames: [b64...],
               galvo_frames: [b64...], prev_scan_frame: b64|null}
  -> {scores, sources, alerts, maps_png, quality, latency_ms}

Config: runs/v05/serve_config.json (thresholds — SENSITIVITY-FIRST
while human-gated — and the registration transform; replace when
registration.py delivers a real calibration).

Detections are ADVISORY: if this service is down the agent proceeds
without it. Deploy/restart via runs/v05/deploy.sh.
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

from .constants import PEREGRINE_ALL_CLASSES, POWDER_GRID, TRANSFER_CLASSES
from .dataset import standardize
from .fit_pixstat import region_stats
from .model import DefectSegmenter, ModelConfig

CONFIG_PATH = Path(__file__).parent / "serve_config.json"
PIXSTAT_PATH = Path(__file__).parent / "pixstat_shortfeed.npz"
HEAD_CLASSES = {"swelling", "super_elevation"}
PIXSTAT_CLASSES = {"incomplete_spreading"}
STAT_SIZE = 1024  # pixstat detector is fitted at cache-native res

DEFAULT_CONFIG = {
    "thresholds": {c: 0.35 for c in TRANSFER_CLASSES},
    "registration": {"scale": 1.0, "tx": 0.0, "ty": 0.0},
    "input_size": 518,
}


def rot_cw90(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def decode_gray(b64s: str) -> np.ndarray:
    g = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(b64s))).convert("L"), dtype=np.float32
    )
    return rot_cw90(g)


def quality(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (160, 120))
    c = small[20:100, 30:130]
    gx, gy = np.gradient(c)
    detail = float(np.sqrt(gx**2 + gy**2).mean())
    return detail * (1 - float((c > 245).mean())) * (1 - float((c < 15).mean()))


def galvo_accumulate(b64s: list[str]) -> np.ndarray | None:
    acc = None
    for b in b64s:
        a = np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("L"))
        m = a > 40
        acc = m if acc is None else (acc | m)
    return acc


def warp_mask(mask: np.ndarray, shape: tuple[int, int], s: float, tx: float, ty: float) -> np.ndarray:
    h, w = shape
    scale = s * (w / mask.shape[1])
    mm = cv2.resize(mask.astype(np.float32), None, fx=scale, fy=scale)
    canvas = np.zeros(shape, dtype=np.float32)
    y0, x0 = int((h - mm.shape[0]) // 2 + ty), int((w - mm.shape[1]) // 2 + tx)
    ys, xs = slice(max(y0, 0), min(y0 + mm.shape[0], h)), slice(max(x0, 0), min(x0 + mm.shape[1], w))
    canvas[ys, xs] = mm[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
    return canvas > 0.5


def build_region_map(part_mask: np.ndarray, min_area: int = 64) -> tuple[np.ndarray, int, np.ndarray]:
    """Parts (connected components) + POWDER_GRID^2 tiles -> region map.
    Returns (region_map, n_regions, is_part flags)."""
    n_cc, cc = cv2.connectedComponents(part_mask.astype(np.uint8))
    rmap = np.full(part_mask.shape, -1, dtype=np.int64)
    idx = 0
    flags = []
    for c in range(1, n_cc):
        m = cc == c
        if m.sum() < min_area:
            continue
        rmap[m] = idx
        flags.append(True)
        idx += 1
    h, w = part_mask.shape
    rows = np.minimum(np.arange(h) * POWDER_GRID // h, POWDER_GRID - 1)
    cols = np.minimum(np.arange(w) * POWDER_GRID // w, POWDER_GRID - 1)
    tiles = rows[:, None] * POWDER_GRID + cols[None, :]
    powder = ~part_mask
    for t in range(POWDER_GRID * POWDER_GRID):
        m = powder & (tiles == t)
        if m.sum() < min_area:
            continue
        rmap[m] = idx
        flags.append(False)
        idx += 1
    return rmap, idx, np.array(flags, dtype=bool)


def _map_to_b64png(m: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
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
        self.logits_size = cfg["logits_size"]
        px = np.load(PIXSTAT_PATH)
        self.px_w, self.px_b = px["w"], float(px["b"])
        self.px_mu, self.px_sd = px["mu"], px["sd"]
        self.config = DEFAULT_CONFIG | (
            json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        )

    @torch.no_grad()
    def infer(self, req: dict) -> dict:
        t0 = time.time()
        grays = [decode_gray(b) for b in req["chamber_frames"]]
        quals = [quality(g) for g in grays]
        top2 = sorted(np.argsort(quals)[-2:]) if len(grays) > 1 else [0, 0]
        recoat_g, scan_g = grays[top2[0]], grays[top2[1]]
        prev_g = decode_gray(req["prev_scan_frame"]) if req.get("prev_scan_frame") else None

        acc = galvo_accumulate(req["galvo_frames"])
        reg = self.config["registration"]
        pmask = (
            warp_mask(acc, scan_g.shape, reg["scale"], reg["tx"], reg["ty"])
            if acc is not None and acc.sum() > 0
            else np.zeros(scan_g.shape, dtype=bool)
        )

        size = self.config["input_size"]
        L = self.logits_size

        def prep(g):
            return torch.tensor(standardize(cv2.resize(g, (size, size))))

        seq = [prev_g if prev_g is not None else np.zeros_like(recoat_g), recoat_g, scan_g]
        pr, ps, pp = (standardize(cv2.resize(g, (size, size))) for g in (recoat_g, scan_g, seq[0]))

        def _box_std(a, k=15):
            m = cv2.blur(a, (k, k))
            return np.sqrt(np.clip(cv2.blur(a * a, (k, k)) - m * m, 0, None))

        photo = np.stack(
            [pr, ps, ps - pp, pr - pp, _box_std(pr),
             cv2.blur(np.abs(pr - pp), (15, 15)), cv2.blur(np.abs(ps - pp), (15, 15))]
        )
        photo_t = torch.nn.functional.interpolate(
            torch.tensor(photo)[None], size=(L, L), mode="bilinear"
        ).float()

        pmask_L = cv2.resize(pmask.astype(np.uint8), (L, L), interpolation=cv2.INTER_NEAREST).astype(bool)
        rmap_L, n_regions, is_part = build_region_map(pmask_L)
        n_regions = max(n_regions, 1)

        frames = torch.stack([prep(g) for g in seq])[None]
        phase_ids = torch.tensor([[2, 0, 1]])
        bright = [float(np.mean(g)) / 255 for g in seq]
        meta = torch.tensor(
            [[[-1.0, bright[0], 1.0], [0.5, bright[1], 1.0], [0.0, bright[2], 1.0]]],
            dtype=torch.float32,
        )
        valid = torch.tensor([[prev_g is not None, True, True]])
        part_1024 = torch.tensor(
            cv2.resize(pmask.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
        )[None]

        d = self.device
        dense, region_logits = self.model(
            frames.to(d), phase_ids.to(d), meta.to(d), valid.to(d),
            part_1024.to(d), photo_t.to(d),
            region_map=torch.tensor(rmap_L)[None].to(d), n_regions=n_regions,
        )
        probs = dense.float().sigmoid()[0].cpu().numpy()
        rprobs = region_logits.float().sigmoid()[0].cpu().numpy()  # (R, 12)

        # pixstat short-feed over powder tiles at fitted resolution
        px_score = 0.0
        if prev_g is not None:
            cur_s = standardize(cv2.resize(recoat_g, (STAT_SIZE, STAT_SIZE)))
            prev_s = standardize(cv2.resize(prev_g, (STAT_SIZE, STAT_SIZE)))
            pm_s = cv2.resize(pmask.astype(np.uint8), (STAT_SIZE, STAT_SIZE), cv2.INTER_NEAREST).astype(bool)
            rmap_s, n_s, is_part_s = build_region_map(pm_s, min_area=256)
            if n_s:
                stats = region_stats(cur_s, prev_s, rmap_s, n_s)
                okr = np.isfinite(stats).all(axis=1) & ~is_part_s[: len(stats)]
                if okr.any():
                    z = (stats[okr] - self.px_mu) / self.px_sd
                    px_score = float(
                        (1 / (1 + np.exp(-(z @ self.px_w + self.px_b)))).max()
                    )

        scores, sources = {}, {}
        for c in TRANSFER_CLASSES:
            ci = PEREGRINE_ALL_CLASSES.index(c)
            if c in PIXSTAT_CLASSES and prev_g is not None:
                scores[c], sources[c] = px_score, "pixstat"
            elif c in HEAD_CLASSES and is_part.any():
                scores[c] = float(rprobs[: len(is_part)][is_part, ci].max())
                sources[c] = "region_head"
            else:
                scores[c], sources[c] = float(probs[ci].max()), "dense"

        thr = self.config["thresholds"]
        alerts = [c for c, s in scores.items() if s >= thr.get(c, 1.1)]
        maps = {c: _map_to_b64png(probs[PEREGRINE_ALL_CLASSES.index(c)]) for c in alerts}
        return {
            "build_id": req.get("build_id"),
            "layer": req.get("layer"),
            "scores": scores,
            "sources": sources,
            "alerts": alerts,
            "maps_png": maps,
            "quality": {
                "n_chamber": len(grays),
                "best_quality": float(max(quals)) if quals else 0.0,
                "galvo_px": int(acc.sum()) if acc is not None else 0,
                "n_part_regions": int(is_part.sum()),
                "has_prev": prev_g is not None,
            },
            "latency_ms": (time.time() - t0) * 1000,
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--checkpoint", default="runs/v05/checkpoints/best.pt")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    engine = Engine(args.checkpoint, device)

    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="inova-defect-detection-v05")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "checkpoint": engine.checkpoint,
            "device": engine.device,
            "classes": TRANSFER_CLASSES,
            "sources": {
                c: ("pixstat" if c in PIXSTAT_CLASSES else "region_head" if c in HEAD_CLASSES else "dense")
                for c in TRANSFER_CLASSES
            },
            "thresholds": engine.config["thresholds"],
        }

    @app.post("/infer")
    def infer(req: dict):
        return engine.infer(req)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
