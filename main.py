"""
main.py
────────
Driver Guard  |  Entry Point

Architecture
────────────
Two camera threads run in parallel:
  Thread-A: DMS pipeline  (IR cabin cam → face → EAR/MAR/pose → risk)
  Thread-B: Road pipeline (road cam → YOLOv8 → detections)
  Thread-C: GPS reader    (background serial / simulation)

All results flow into AlertManager which computes the composite risk
score and triggers GPIO / audio alerts.

DatabaseManager flushes events to SQLite in the background.

Flask + Socket.IO dashboard runs in Thread-D on port 5000.

Usage
─────
    python main.py                    # live mode (uses config.yaml)
    python main.py --simulate         # GPS simulation + no real camera
    python main.py --config my.yaml   # custom config file
"""

import argparse
import sys
import time
import threading
import signal
from pathlib import Path

import cv2
import yaml
from loguru import logger


# ── Internal modules ──────────────────────────────────────────────────────────
from modules.dms.driver_monitor     import DriverMonitor
from modules.road.detector          import RoadDamageDetector
from modules.gps.gps_reader         import GPSReader
from modules.database.db_manager    import DatabaseManager
from modules.alert.alert_manager    import AlertManager
from dashboard.app                  import create_app


# ─── Configuration ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded: {path}")
    return cfg


# ─── Camera thread helpers ────────────────────────────────────────────────────

class CameraCapture:
    """Thread-safe OpenCV camera wrapper with reconnect logic."""

    def __init__(self, source, width, height, fps, name="cam"):
        self.source = source
        self.width  = width
        self.height = height
        self.fps    = fps
        self.name   = name
        self._cap:  cv2.VideoCapture = None
        self._lock  = threading.Lock()
        self._frame = None
        self._stop  = threading.Event()

    def start(self):
        self._connect()
        t = threading.Thread(target=self._read_loop, daemon=True, name=f"Cam-{self.name}")
        t.start()

    def stop(self):
        self._stop.set()
        if self._cap and self._cap.isOpened():
            self._cap.release()

    def _connect(self):
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
        if not self._cap.isOpened():
            logger.warning(f"[{self.name}] Camera {self.source} not opened yet")

    def _read_loop(self):
        while not self._stop.is_set():
            if not self._cap.isOpened():
                logger.warning(f"[{self.name}] Reconnecting …")
                time.sleep(2)
                self._connect()
                continue
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


# ─── Pipeline threads ─────────────────────────────────────────────────────────

def dms_thread(dms: DriverMonitor, cam: CameraCapture,
               alert: AlertManager, db: DatabaseManager,
               show_preview: bool, stop: threading.Event):
    """Driver Monitoring System loop."""
    logger.info("[DMS] Pipeline thread started")
    prev_risk  = "low"
    frame_skip = 0

    while not stop.is_set():
        frame = cam.read()
        if frame is None:
            time.sleep(0.033)
            continue

        result = dms.process(frame)

        # Log on risk change
        if result.risk != prev_risk:
            db.log_driver_event(
                ear=result.ear, mar=result.mar,
                yaw=result.yaw, pitch=result.pitch,
                risk=result.risk, state=result.driver_state,
                eye_frames=result.eye_closed_frames,
                yawn_frames=result.yawn_frames,
                lat=alert.state.lat, lon=alert.state.lon,
            )
            prev_risk = result.risk
            logger.info(f"[DMS] Risk change → {result.risk.upper()} "
                        f"| State: {result.driver_state} "
                        f"| EAR: {result.ear:.2f} MAR: {result.mar:.2f}")

        # if show_preview:
        #     cv2.imshow("DMS — Driver Monitor", result.annotated_frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         stop.set()
        # if show_preview:
        #     display = result.annotated_frame if result.annotated_frame is not None else frame
        #     cv2.imshow("DMS — Driver Monitor", display)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         stop.set()
        if show_preview:
            display = result.annotated_frame if result.annotated_frame is not None else frame
            with _preview_lock:
                _preview_frames["dms"] = display

        time.sleep(1 / 30)


def road_thread(detector: RoadDamageDetector, cam: CameraCapture,
                alert: AlertManager, db: DatabaseManager,
                show_preview: bool, stop: threading.Event,
                writer: cv2.VideoWriter = None):
    """Road Damage Detection loop."""
    logger.info("[Road] Pipeline thread started")
    while not stop.is_set():
        frame = cam.read()
        if frame is None:
            time.sleep(0.033)
            continue

        # Save raw frame to recording if enabled
        if writer is not None:
            writer.write(frame)

        gps = alert.state
        dets = detector.process(frame, lat=gps.lat, lon=gps.lon)

        for d in dets:
            db.log_road_event(
                class_name=d.class_name,
                confidence=d.confidence,
                severity=d.severity,
                area_px2=d.area_px2,
                lat=d.lat, lon=d.lon,
            )
            logger.info(f"[Road] {d.class_name} | {d.severity.upper()} "
                        f"| conf={d.confidence:.2f} "
                        f"| area={d.area_px2}px²")

        # if show_preview:
        #     annotated = detector.annotate(frame, dets)
        #     cv2.imshow("Road Damage Detector", annotated)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         stop.set()
        if show_preview:
            annotated = detector.annotate(frame, dets)
            with _preview_lock:
                _preview_frames["road"] = annotated

        time.sleep(1 / 30)


def alert_sync_thread(alert: AlertManager, dms: DriverMonitor,
                      road: RoadDamageDetector, gps: GPSReader,
                      stop: threading.Event):
    """Syncs DMS + road results into AlertManager at 10 Hz."""
    while not stop.is_set():
        alert.update(
            dms_result      = dms.latest,
            road_detections = road.latest,
            gps_fix         = gps.latest,
        )
        time.sleep(0.1)

# ─── Shared preview frames ────────────────────────────────────────────────────
_preview_frames = {}
_preview_lock   = threading.Lock()
# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Driver Guard")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--simulate", action="store_true",
                        help="Enable GPS simulation and camera fallback")
    parser.add_argument("--preview",  action="store_true",
                        help="Show OpenCV preview windows")
    parser.add_argument("--no-web",   action="store_true",
                        help="Skip launching the web dashboard")
    parser.add_argument("--record",   action="store_true",
                        help="Save road camera footage to data/recordings/ for training data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.simulate:
        cfg["gps"]["simulation"] = True
        logger.info("Simulation mode enabled")

    # ── Video recording setup ─────────────────────────────────────────────────
    road_writer = None
    if args.record:
        from datetime import datetime
        import os
        rec_dir = Path("data/recordings")
        rec_dir.mkdir(parents=True, exist_ok=True)
        timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rec_path   = str(rec_dir / f"road_{timestamp}.mp4")
        road_cfg   = cfg["cameras"]["road"]
        road_writer = cv2.VideoWriter(
            rec_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            road_cfg.get("fps", 30),
            (road_cfg.get("width", 640), road_cfg.get("height", 480)),
        )
        logger.info(f"[Record] Saving road footage → {rec_path}")

    # ── Setup logger ──────────────────────────────────────────────────────────
    logger.remove()
    logger.add(sys.stderr, level=cfg["system"].get("log_level", "INFO"),
               format="<green>{time:HH:mm:ss}</green> | {level} | {message}")
    logger.add("data/logs/driver_guard_{time:YYYY-MM-DD}.log",
               rotation="1 day", retention="7 days", level="DEBUG")

    logger.info("=" * 50)
    logger.info("  Driver Guard  v2.1")
    logger.info("=" * 50)

    # ── Instantiate modules ───────────────────────────────────────────────────
    db      = DatabaseManager(cfg["database"])
    gps     = GPSReader(cfg["gps"])
    dms     = DriverMonitor(cfg["dms"])
    road    = RoadDamageDetector(cfg["road"])
    alert   = AlertManager(cfg)

    # ── Cameras ───────────────────────────────────────────────────────────────
    dms_cam_cfg  = cfg["cameras"]["dms"]
    road_cam_cfg = cfg["cameras"]["road"]

    # dms_cam  = CameraCapture(
    #     dms_cam_cfg["source"],  dms_cam_cfg["width"],
    #     dms_cam_cfg["height"],  dms_cam_cfg["fps"],  "DMS")
    # road_cam = CameraCapture(
    #     road_cam_cfg["source"], road_cam_cfg["width"],
    #     road_cam_cfg["height"], road_cam_cfg["fps"], "Road")

    dms_cam  = CameraCapture(
        dms_cam_cfg["source"],  dms_cam_cfg["width"],
        dms_cam_cfg["height"],  dms_cam_cfg["fps"],  "DMS")

    # If both cameras use the same source (single webcam), share the instance
    if dms_cam_cfg["source"] == road_cam_cfg["source"]:
        road_cam = dms_cam
        logger.info("Single camera mode — DMS and Road sharing camera 0")
    else:
        road_cam = CameraCapture(
            road_cam_cfg["source"], road_cam_cfg["width"],
            road_cam_cfg["height"], road_cam_cfg["fps"], "Road")

    # ── Start all modules ─────────────────────────────────────────────────────
    stop = threading.Event()

    db.start()
    gps.start()
    dms.start()
    road.start()
    alert.start()
    dms_cam.start()
    time.sleep(3)
    road_cam.start()

    db.log_system("INFO", "Driver Guard system started")
    logger.success("All modules online ✓")

    # ── Launch threads ────────────────────────────────────────────────────────
    threads = [
        threading.Thread(target=dms_thread,
            args=(dms, dms_cam, alert, db, args.preview, stop),
            daemon=True, name="DMSPipeline"),
        threading.Thread(target=road_thread,
            args=(road, road_cam, alert, db, args.preview, stop, road_writer),
            daemon=True, name="RoadPipeline"),
        threading.Thread(target=alert_sync_thread,
            args=(alert, dms, road, gps, stop),
            daemon=True, name="AlertSync"),
    ]
    for t in threads:
        t.start()

    # ── Web dashboard ─────────────────────────────────────────────────────────
    if not args.no_web:
        dash_cfg = cfg.get("alert", {})
        host = dash_cfg.get("dashboard_host", "0.0.0.0")
        port = dash_cfg.get("dashboard_port", 5000)
        app, socketio = create_app(alert, db, cfg)

        def _run_dash():
            logger.info(f"[Dashboard] http://{host}:{port}")
            socketio.run(app, host=host, port=port,
                         use_reloader=False, log_output=False)

        threading.Thread(target=_run_dash, daemon=True, name="Dashboard").start()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def _shutdown(sig, _):
        logger.info(f"Signal {sig} received — shutting down …")
        stop.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Press Ctrl+C to stop")
    # try:
    #     while not stop.is_set():
    #         time.sleep(0.5)
    # except KeyboardInterrupt:
    #     stop.set()
    try:
        while not stop.is_set():
            if args.preview:
                with _preview_lock:
                    frames = dict(_preview_frames)
                for title, frm in frames.items():
                    if frm is not None:
                        cv2.imshow("DMS — Driver Monitor" if title == "dms" else "Road Damage Detector", frm)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop.set()
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        stop.set()

    logger.info("Stopping modules …")
    if road_writer is not None:
        road_writer.release()
        logger.info("[Record] Road footage saved ✓")
    dms_cam.stop()
    road_cam.stop()
    gps.stop()
    alert.stop()
    db.stop()
    cv2.destroyAllWindows()
    logger.success("Driver Guard stopped cleanly.")


if __name__ == "__main__":
    main()