"""
─────────────────────────
Road hazard detection using YOLO11-nano (pothole_yolo11n, NCNN @ 416).

Classes detected (1 class — matches models/pothole_yolo11n.pt)
----------------
    0 → pothole

Class names come from road.classes in config.yaml and are then synced with
the names embedded in the loaded model, so swapping models needs no code edit.

Per-class confidence thresholds (runtime deployment only):
    pothole 0.35  (unknown classes fall back to confidence_threshold)
    (For paper mAP50 benchmarks, use a uniform threshold instead)

Severity is determined by bounding-box area and confidence score:
    Minor    < 3 000 px²  or conf < 0.70
    Moderate < 10 000 px²
    Severe   ≥ 10 000 px²
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pathlib import Path
from loguru import logger


# ─── Detection dataclass ─────────────────────────────────────────────────────

@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel)
    severity:   str = "minor"               # "minor" | "moderate" | "severe"
    area_px2:   int = 0
    timestamp:  float = field(default_factory=time.time)
    lat:        Optional[float] = None
    lon:        Optional[float] = None


# ─── RoadDamageDetector ───────────────────────────────────────────────────────

class RoadDamageDetector:
    """
    YOLO11-nano inference wrapper with severity classification.
    
    """

    # Defaults for models/pothole_yolo11n — overridden by road.classes in
    # config.yaml and then by the names embedded in the loaded model.
    CLASS_NAMES = {
        0: "pothole",
    }
    COLOURS = {
        "pothole":            (0, 0, 220),      # red
    }

    PER_CLASS_CONF = {
        0: 0.35,   # pothole — catch more, safety critical
    }
    SEVERITY_COLOURS = {
        "minor":    (0, 200, 80),
        "moderate": (0, 165, 255),
        "severe":   (0, 0, 220),
    }

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.model_path = cfg["model_path"]
        if cfg.get("classes"):
            self.CLASS_NAMES = {int(k): str(v) for k, v in cfg["classes"].items()}
        self.conf_thr   = cfg.get("confidence_threshold", 0.50)
        self.iou_thr    = cfg.get("iou_threshold", 0.35)   # tighter NMS — fewer duplicate boxes
        self.imgsz      = int(cfg.get("imgsz", 320))       # must match the NCNN export size
        # Runtime per-class filtering toggle (default on)
        self.use_per_class = cfg.get("per_class_thresholds", True)
        self._base_conf = min(self.PER_CLASS_CONF.values()) if self.use_per_class else self.conf_thr

        severity_cfg          = cfg.get("severity", {})
        self._minor_area_max  = severity_cfg.get("minor",    {}).get("area_px2_max", 3000)
        self._mod_area_max    = severity_cfg.get("moderate", {}).get("area_px2_max", 10000)

        # ── Temporal confirmation ──────────────────────────────────────────────
        # Require a hazard class to appear in `confirm_frames` of the last
        # `confirm_window` frames before it is treated as real. Filters out
        # single-frame false positives (shadows, tar seams, wet patches).
        self.temporal_enabled = cfg.get("temporal_confirmation", True)
        self.confirm_window   = max(1, int(cfg.get("confirm_window", 5)))
        self.confirm_frames   = max(1, int(cfg.get("confirm_frames", 3)))
        if self.confirm_frames > self.confirm_window:
            self.confirm_frames = self.confirm_window
            logger.warning("[Road] confirm_frames > confirm_window — clamped to window size")
        # Each entry is the set of class_ids detected in that frame
        self._frame_history: deque = deque(maxlen=self.confirm_window)

        self._model  = None
        self._device = "cpu"
        self._lock   = threading.Lock()
        self._latest: List[Detection] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        logger.info(f"[Road] Loading YOLO model: {self.model_path}")
        try:
            from ultralytics import YOLO
            # NCNN model (an "_ncnn_model" folder) runs CPU-only, ARM-optimised —
            # used on the Pi. A .pt model uses the GPU when one is available.
            is_ncnn = (str(self.model_path).rstrip("/").endswith("_ncnn_model")
                       or Path(self.model_path).is_dir())
            if is_ncnn:
                self._device = "cpu"
                self._model  = YOLO(self.model_path, task="detect")
                engine = "NCNN"
            else:
                self._device = 0 if self._has_cuda() else "cpu"
                self._model  = YOLO(self.model_path)
                self._model.to(self._device)
                engine = "PyTorch"
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False, device=self._device, imgsz=self.imgsz)
            model_names = getattr(self._model, "names", None)
            if model_names:
                model_names = {int(k): str(v) for k, v in dict(model_names).items()}
                if model_names != self.CLASS_NAMES:
                    logger.warning(f"[Road] road.classes {self.CLASS_NAMES} != model names "
                                   f"{model_names} — using the model's names")
                self.CLASS_NAMES = model_names
            logger.success(f"[Road] YOLO loaded ({engine}) on {str(self._device).upper()} ✓  "
                           f"classes={self.CLASS_NAMES}")
        except FileNotFoundError:
            logger.warning(f"[Road] Model file not found: {self.model_path}. "
                           "Using dummy detections.")
        except Exception as e:
            logger.error(f"[Road] Model load failed: {e}")

    @staticmethod
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray,
                lat: Optional[float] = None,
                lon: Optional[float] = None) -> List[Detection]:
        
        if self._model is None:
            return []
        results = self._model.predict(
            frame,
            conf=self._base_conf,
            iou=self.iou_thr,
            imgsz=self.imgsz,
            verbose=False,
            stream=False,
            device=self._device,
        )

        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])

                # ── Per-class confidence filter (runtime only) ──────────────
                if self.use_per_class:
                    threshold = self.PER_CLASS_CONF.get(cls_id, self.conf_thr)
                    if conf < threshold:
                        continue   # skip detections below this class's threshold

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area    = max(0, (x2 - x1)) * max(0, (y2 - y1))
                sev     = self._classify_severity(area, conf)
                cls_name = self.CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

                detections.append(Detection(
                    class_id   = cls_id,
                    class_name = cls_name,
                    confidence = round(conf, 3),
                    bbox       = (x1, y1, x2, y2),
                    severity   = sev,
                    area_px2   = area,
                    lat        = lat,
                    lon        = lon,
                ))

        detections = self._apply_temporal(detections)

        with self._lock:
            self._latest = detections
        return detections

    # ── Temporal confirmation ───────────────────────────────────────────────────

    def _apply_temporal(self, detections: List[Detection]) -> List[Detection]:
        """
        Keep only hazards confirmed across multiple frames.
        """
        if not self.temporal_enabled:
            return detections

        seen_now = {d.class_id for d in detections}
        self._frame_history.append(seen_now)

        confirmed: List[Detection] = []
        for d in detections:
            hits = sum(1 for frame_classes in self._frame_history
                       if d.class_id in frame_classes)
            if hits >= self.confirm_frames:
                confirmed.append(d)
        return confirmed

    # ── Severity ──────────────────────────────────────────────────────────────

    def _classify_severity(self, area: int, conf: float) -> str:
        if area >= self._mod_area_max:
            return "severe"
        if area >= self._minor_area_max:
            return "moderate"
        return "minor"

    # ── Annotation ────────────────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray,
                 detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame."""
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            colour = self.SEVERITY_COLOURS[d.severity]
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{d.class_name} {d.severity[0].upper()} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    @property
    def latest(self) -> List[Detection]:
        with self._lock:
            return list(self._latest)