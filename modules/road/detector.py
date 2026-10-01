"""
modules/road/detector.py
─────────────────────────
Road damage detection using YOLOv8-nano fine-tuned on RDD2022 +
1 200 custom-annotated Indian road frames.

Classes detected (5 classes — matches yolov8n_rdd_india.pt)
----------------
    0 → pothole
    1 → crack_longitudinal
    2 → crack_transverse
    3 → rutting
    4 → repair

Per-class confidence thresholds (runtime deployment only):
    pothole 0.35 / cracks 0.45 / rutting 0.50 / repair 0.55
    (For paper mAP50 benchmarks, use a uniform threshold instead)

Severity is determined by bounding-box area and confidence score:
    Minor    < 3 000 px²  or conf < 0.70
    Moderate < 10 000 px²
    Severe   ≥ 10 000 px²

Usage
-----
    detector = RoadDamageDetector(cfg["road"])
    detector.start()

    detections = detector.process(frame)
    for d in detections:
        print(d.class_name, d.severity, d.confidence)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
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
    YOLOv8-nano inference wrapper with severity classification.

    Parameters
    ----------
    cfg : dict  — the `road` section from config.yaml
    """

    CLASS_NAMES = {
        0: "pothole",
        1: "crack_longitudinal",
        2: "crack_transverse",
        3: "rutting",
        4: "repair",
    }
    COLOURS = {
        "pothole":            (0, 0, 220),      # red
        "crack_longitudinal": (0, 0, 0),        # black
        "crack_transverse":   (0, 165, 255),    # orange
        "rutting":            (42, 42, 165),    # brown
        "repair":             (180, 0, 220),    # purple
    }

    # ── Per-class confidence thresholds (RUNTIME ONLY) ─────────────────────
    # NOTE: These are for live deployment usability, NOT for mAP50 evaluation.
    # For paper benchmarks, use a single uniform threshold across all classes.
    # Rationale: potholes are safety-critical (catch more, lower threshold),
    # repairs are non-critical (only flag if very confident, higher threshold).
    PER_CLASS_CONF = {
        0: 0.35,   # pothole           — catch more, safety critical
        1: 0.45,   # crack_longitudinal
        2: 0.45,   # crack_transverse
        3: 0.50,   # rutting
        4: 0.55,   # repair            — only flag if very confident
    }
    SEVERITY_COLOURS = {
        "minor":    (0, 200, 80),
        "moderate": (0, 165, 255),
        "severe":   (0, 0, 220),
    }

    def __init__(self, cfg: dict):
        self.cfg        = cfg
        self.model_path = cfg["model_path"]
        # Base conf is the LOWEST of all per-class thresholds so the model
        # returns everything; we then filter per-class in process().
        self.conf_thr   = cfg.get("confidence_threshold", 0.50)
        self.iou_thr    = cfg.get("iou_threshold", 0.35)   # tighter NMS — fewer duplicate boxes
        # Runtime per-class filtering toggle (default on)
        self.use_per_class = cfg.get("per_class_thresholds", True)
        self._base_conf = min(self.PER_CLASS_CONF.values()) if self.use_per_class else self.conf_thr

        severity_cfg          = cfg.get("severity", {})
        self._minor_area_max  = severity_cfg.get("minor",    {}).get("area_px2_max", 3000)
        self._mod_area_max    = severity_cfg.get("moderate", {}).get("area_px2_max", 10000)

        self._model  = None
        self._lock   = threading.Lock()
        self._latest: List[Detection] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Load YOLOv8 model weights."""
        logger.info(f"[Road] Loading YOLOv8 model: {self.model_path}")
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            # Warm-up pass
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False)
            logger.success("[Road] YOLOv8-nano loaded ✓")
        except FileNotFoundError:
            logger.warning(f"[Road] Model file not found: {self.model_path}. "
                           "Using dummy detections.")
        except Exception as e:
            logger.error(f"[Road] Model load failed: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray,
                lat: Optional[float] = None,
                lon: Optional[float] = None) -> List[Detection]:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray  BGR image
        lat, lon : GPS coordinates to geo-tag each detection

        Returns
        -------
        List[Detection]
        """
        if self._model is None:
            return []

        # Run model at the LOWEST threshold so nothing is pre-filtered,
        # then apply per-class thresholds below.
        results = self._model.predict(
            frame,
            conf=self._base_conf,
            iou=self.iou_thr,
            verbose=False,
            stream=False,
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

        with self._lock:
            self._latest = detections
        return detections

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