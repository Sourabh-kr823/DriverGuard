"""
evaluate.py
────────────
DriverGuard Performance & Accuracy Evaluation

Benchmarks every component of the pipeline on this device and prints
a formatted report.  Run after any change to see real numbers.

Usage
─────
    python evaluate.py                         # full evaluation (auto-detect model)
    python evaluate.py --quick                 # skip long video runs
    python evaluate.py --no-cpu                # skip PyTorch comparison (NCNN only)
    python evaluate.py --model models/yolov8n_rdd_india_ncnn_320
    python evaluate.py --imgsz 640             # override inference size
    python evaluate.py --compare 320 640       # compare two NCNN export sizes
"""

import argparse
import os
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import yaml
from loguru import logger

# ── Silence noisy loggers during benchmark ────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="WARNING")

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"

def fmt_fps(seconds: float) -> str:
    return f"{1/seconds:.1f}" if seconds > 0 else "∞"

def percentile(data, pct):
    k = (len(data) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[f]
    return data[f] + (k - f) * (data[c] - data[f])

def get_system_info() -> dict:
    info = {}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("Model"):
                    info["board"] = line.split(":")[-1].strip()
                    break
    except Exception:
        pass
    try:
        import subprocess
        temp = subprocess.check_output(["vcgencmd", "measure_temp"],
                                       stderr=subprocess.DEVNULL, text=True)
        info["temp"] = temp.strip().replace("temp=", "")
    except Exception:
        info["temp"] = "N/A"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram"] = f"{kb / 1024 / 1024:.1f} GB"
                    break
    except Exception:
        pass
    import platform
    info["arch"] = platform.machine()
    info["python"] = platform.python_version()
    return info


def load_video_frames(path: str, max_frames: int = 200) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)
    cap.release()
    return frames


def find_ncnn_model(stem: str = "yolov8n_rdd_india", imgsz: int | None = None) -> str | None:
    """Find the best NCNN model in models/ directory.

    Naming convention (Ultralytics requires _ncnn_model suffix):
        320 (default): models/<stem>_ncnn_model
        640:           models/<stem>_640_ncnn_model
    """
    models_dir = BASE_DIR / "models"
    if imgsz and imgsz != 320:
        # Sized variant: <stem>_<size>_ncnn_model
        path = models_dir / f"{stem}_{imgsz}_ncnn_model"
        if path.exists():
            return str(path)
    if imgsz == 320 or imgsz is None:
        # Default 320 export or auto-detect
        path = models_dir / f"{stem}_ncnn_model"
        if path.exists():
            return str(path)
    if imgsz is None:
        # Try sized variants as fallback
        for sz in [640, 416]:
            path = models_dir / f"{stem}_{sz}_ncnn_model"
            if path.exists():
                return str(path)
    return None


def detect_imgsz(model_path: str) -> int:
    """Infer the imgsz from the model folder name or its metadata."""
    # Check metadata file first
    meta_path = Path(model_path) / "export_meta.json"
    if meta_path.exists():
        import json
        try:
            return json.loads(meta_path.read_text())["imgsz"]
        except Exception:
            pass
    name = os.path.basename(model_path)
    # Pattern: stem_640_ncnn_model → 640
    for sz in [640, 416, 320]:
        if f"_{sz}_ncnn_model" in name:
            return sz
    return 320


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark: YOLO Road Detection
# ═══════════════════════════════════════════════════════════════════════════════

def bench_yolo(model_path: str, frames: list, label: str, imgsz: int = 320) -> dict:
    from ultralytics import YOLO
    model = YOLO(model_path)

    # Warmup (3 frames)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(dummy, verbose=False, imgsz=imgsz)

    latencies = []
    total_dets = 0
    class_counts = {}
    confidences = []
    for frame in frames:
        t0 = time.perf_counter()
        results = model.predict(frame, verbose=False, imgsz=imgsz, conf=0.35)
        latencies.append(time.perf_counter() - t0)
        for r in results:
            if r.boxes is not None:
                total_dets += len(r.boxes)
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_names = {0: "pothole", 1: "crack_long", 2: "crack_trans",
                                 3: "rutting", 4: "repair"}
                    name = cls_names.get(cls_id, f"cls_{cls_id}")
                    class_counts[name] = class_counts.get(name, 0) + 1
                    confidences.append(conf)

    latencies.sort()
    return {
        "label":      label,
        "frames":     len(frames),
        "imgsz":      imgsz,
        "mean":       statistics.mean(latencies),
        "median":     statistics.median(latencies),
        "p95":        percentile(latencies, 95),
        "p99":        percentile(latencies, 99),
        "min":        min(latencies),
        "max":        max(latencies),
        "fps":        1.0 / statistics.mean(latencies),
        "total_dets": total_dets,
        "class_counts": class_counts,
        "avg_conf":   statistics.mean(confidences) if confidences else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark: DMS (MediaPipe)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_dms(cfg: dict, frames: list) -> dict:
    from backend.dms.driver_monitor import DriverMonitor

    dms = DriverMonitor(cfg, draw_overlays=False)
    dms.start()

    # Warmup
    for f in frames[:3]:
        dms.process(f)

    latencies = []
    states = {"ALERT": 0, "DROWSY": 0, "DISTRACTED": 0, "FATIGUED": 0, "LOW VISIBILITY": 0}
    for frame in frames:
        t0 = time.perf_counter()
        result = dms.process(frame)
        latencies.append(time.perf_counter() - t0)
        states[result.driver_state] = states.get(result.driver_state, 0) + 1

    dms.stop()
    latencies.sort()
    return {
        "frames": len(frames),
        "mean":   statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95":    percentile(latencies, 95),
        "min":    min(latencies),
        "max":    max(latencies),
        "fps":    1.0 / statistics.mean(latencies),
        "states": states,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark: Full pipeline (DMS + Road on same frame set)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_pipeline(cfg: dict, frames: list) -> dict:
    from backend.dms.driver_monitor import DriverMonitor
    from backend.road.detector import RoadDamageDetector

    dms = DriverMonitor(cfg["dms"], draw_overlays=False)
    dms.start()
    road = RoadDamageDetector(cfg["road"])
    road.start()

    # Warmup
    for f in frames[:3]:
        dms.process(f)
        road.process(f)

    latencies = []
    for frame in frames:
        t0 = time.perf_counter()
        dms.process(frame)
        road.process(frame)
        latencies.append(time.perf_counter() - t0)

    dms.stop()
    latencies.sort()
    return {
        "frames": len(frames),
        "mean":   statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95":    percentile(latencies, 95),
        "fps":    1.0 / statistics.mean(latencies),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark: YOLO mAP (if validation data exists)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_accuracy(model_path: str, imgsz: int = 320) -> dict | None:
    data_yaml = BASE_DIR / "data" / "finetune_india.yaml"
    if not data_yaml.exists():
        return None
    with open(data_yaml) as f:
        dcfg = yaml.safe_load(f)
    val_path = Path(dcfg.get("path", "")) / dcfg.get("val", "valid/images")
    if not val_path.exists():
        return None

    from ultralytics import YOLO
    model = YOLO(model_path)
    try:
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False)
        return {
            "mAP50":    round(metrics.box.map50, 4),
            "mAP50_95": round(metrics.box.map, 4),
            "precision": round(metrics.box.mp, 4),
            "recall":    round(metrics.box.mr, 4),
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Benchmark: Video detection test
# ═══════════════════════════════════════════════════════════════════════════════

def bench_video_detections(model_path: str, video_path: str,
                           max_frames: int = 300, imgsz: int = 320) -> dict:
    from ultralytics import YOLO
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    class_counts = {}
    severity_counts = {"minor": 0, "moderate": 0, "severe": 0}
    frame_count = 0
    det_frames = 0
    latencies = []
    confidences = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frame_count += 1

        t0 = time.perf_counter()
        results = model.predict(frame, verbose=False, imgsz=imgsz, conf=0.35)
        latencies.append(time.perf_counter() - t0)

        has_det = False
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                has_det = True
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_names = {0: "pothole", 1: "crack_long", 2: "crack_trans",
                             3: "rutting", 4: "repair"}
                name = cls_names.get(cls_id, f"cls_{cls_id}")
                class_counts[name] = class_counts.get(name, 0) + 1
                confidences.append(conf)

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                sev = "severe" if area >= 10000 else ("moderate" if area >= 3000 else "minor")
                severity_counts[sev] += 1
        if has_det:
            det_frames += 1

    cap.release()
    return {
        "video":           os.path.basename(video_path),
        "frames_analyzed": frame_count,
        "frames_with_det": det_frames,
        "det_rate":        f"{det_frames/max(frame_count,1)*100:.1f}%",
        "class_counts":    class_counts,
        "severity_counts": severity_counts,
        "total_dets":      sum(class_counts.values()),
        "avg_conf":        statistics.mean(confidences) if confidences else 0,
        "mean_latency":    statistics.mean(latencies) if latencies else 0,
        "fps":             1.0 / statistics.mean(latencies) if latencies else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Report printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(results: dict):
    W = 62
    line = "═" * W
    thin = "─" * W

    print()
    print(f"╔{line}╗")
    print(f"║{'DRIVERGUARD — EVALUATION REPORT':^{W}}║")
    print(f"║{results['timestamp']:^{W}}║")
    print(f"╚{line}╝")

    # System info
    si = results.get("system", {})
    print(f"\n  SYSTEM")
    print(f"  {thin}")
    print(f"  Board      : {si.get('board', 'N/A')}")
    print(f"  Arch       : {si.get('arch', 'N/A')}")
    print(f"  RAM        : {si.get('ram', 'N/A')}")
    print(f"  Temperature: {si.get('temp', 'N/A')}")
    print(f"  Python     : {si.get('python', 'N/A')}")

    # YOLO benchmarks
    for yolo in results.get("yolo_benchmarks", []):
        print(f"\n  YOLO — {yolo['label']}")
        print(f"  {thin}")
        print(f"  Frames     : {yolo['frames']}")
        print(f"  Mean       : {fmt_ms(yolo['mean']):>10}  ({fmt_fps(yolo['mean']):>6} FPS)")
        print(f"  Median     : {fmt_ms(yolo['median']):>10}")
        print(f"  P95        : {fmt_ms(yolo['p95']):>10}")
        print(f"  P99        : {fmt_ms(yolo['p99']):>10}")
        print(f"  Min        : {fmt_ms(yolo['min']):>10}")
        print(f"  Max        : {fmt_ms(yolo['max']):>10}")
        print(f"  Detections : {yolo['total_dets']}")
        if yolo.get("class_counts"):
            print(f"  Classes    : ", end="")
            for cls, cnt in sorted(yolo["class_counts"].items(), key=lambda x: -x[1]):
                print(f"{cls}={cnt}  ", end="")
            print()
        if yolo.get("avg_conf"):
            print(f"  Avg conf   : {yolo['avg_conf']:.3f}")

    # DMS benchmark
    dms = results.get("dms_benchmark")
    if dms:
        print(f"\n  DMS (MediaPipe)")
        print(f"  {thin}")
        print(f"  Frames     : {dms['frames']}")
        print(f"  Mean       : {fmt_ms(dms['mean']):>10}  ({fmt_fps(dms['mean']):>6} FPS)")
        print(f"  Median     : {fmt_ms(dms['median']):>10}")
        print(f"  P95        : {fmt_ms(dms['p95']):>10}")
        print(f"  Min        : {fmt_ms(dms['min']):>10}")
        print(f"  Max        : {fmt_ms(dms['max']):>10}")
        print(f"  States     : ", end="")
        for s, c in dms["states"].items():
            if c > 0:
                print(f"{s}={c}  ", end="")
        print()

    # Pipeline benchmark
    pipe = results.get("pipeline_benchmark")
    if pipe:
        print(f"\n  FULL PIPELINE (DMS + Road combined)")
        print(f"  {thin}")
        print(f"  Frames     : {pipe['frames']}")
        print(f"  Mean       : {fmt_ms(pipe['mean']):>10}  ({fmt_fps(pipe['mean']):>6} FPS)")
        print(f"  Median     : {fmt_ms(pipe['median']):>10}")
        print(f"  P95        : {fmt_ms(pipe['p95']):>10}")
        achievable = min(20, pipe['fps'])
        print(f"  Max target : {achievable:.0f} FPS (at current load)")

    # Accuracy
    acc = results.get("accuracy")
    if acc and "error" not in acc:
        print(f"\n  MODEL ACCURACY (validation set)")
        print(f"  {thin}")
        print(f"  mAP@50     : {acc['mAP50']:.4f}")
        print(f"  mAP@50-95  : {acc['mAP50_95']:.4f}")
        print(f"  Precision  : {acc['precision']:.4f}")
        print(f"  Recall     : {acc['recall']:.4f}")

    # Video detection tests
    for vd in results.get("video_detections", []):
        print(f"\n  VIDEO — {vd['video']}")
        print(f"  {thin}")
        print(f"  Frames     : {vd['frames_analyzed']}")
        print(f"  With dets  : {vd['frames_with_det']} ({vd['det_rate']})")
        print(f"  Total dets : {vd.get('total_dets', '?')}")
        print(f"  Avg conf   : {vd.get('avg_conf', 0):.3f}")
        print(f"  Avg FPS    : {vd['fps']:.1f}")
        if vd["class_counts"]:
            print(f"  Classes    : ", end="")
            for cls, cnt in sorted(vd["class_counts"].items(), key=lambda x: -x[1]):
                print(f"{cls}={cnt}  ", end="")
            print()
        print(f"  Severity   : ", end="")
        for s, c in vd["severity_counts"].items():
            print(f"{s}={c}  ", end="")
        print()

    # ── Comparison tables ─────────────────────────────────────────────────────
    yolo_list = results.get("yolo_benchmarks", [])

    # NCNN vs PyTorch speedup
    pt = next((y for y in yolo_list if "PyTorch" in y["label"]), None)
    nc = next((y for y in yolo_list if "NCNN" in y["label"]), None)
    if pt and nc:
        speedup = pt["mean"] / nc["mean"]
        print(f"\n  NCNN SPEEDUP")
        print(f"  {thin}")
        print(f"  PyTorch    : {fmt_ms(pt['mean'])} / frame  ({pt['fps']:.1f} FPS)")
        print(f"  NCNN       : {fmt_ms(nc['mean'])} / frame  ({nc['fps']:.1f} FPS)")
        print(f"  Speedup    : {speedup:.2f}x faster")

    # NCNN 320 vs 640 comparison
    ncnn_models = [y for y in yolo_list if "NCNN" in y["label"]]
    if len(ncnn_models) >= 2:
        print(f"\n  NCNN MODEL COMPARISON (320 vs 640)")
        print(f"  {thin}")
        print(f"  {'Model':<25} {'Latency':>10} {'FPS':>8} {'Dets':>6} {'Avg Conf':>10}")
        print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*6} {'─'*10}")
        for m in ncnn_models:
            print(f"  {m['label']:<25} {fmt_ms(m['mean']):>10} {m['fps']:>7.1f} "
                  f"{m['total_dets']:>6} {m['avg_conf']:>10.3f}")

        # Video comparison if available
        vid_groups = {}
        for vd in results.get("video_detections", []):
            key = vd["video"]
            vid_groups.setdefault(key, []).append(vd)
        for video_name, vds in vid_groups.items():
            if len(vds) >= 2:
                print(f"\n  VIDEO COMPARISON — {video_name}")
                print(f"  {'─'*58}")
                print(f"  {'Model':<20} {'Det Rate':>10} {'Dets':>6} {'Conf':>8} {'FPS':>8}")
                print(f"  {'─'*20} {'─'*10} {'─'*6} {'─'*8} {'─'*8}")
                for vd in vds:
                    lbl = vd.get("model_label", "?")
                    print(f"  {lbl:<20} {vd['det_rate']:>10} {vd.get('total_dets',0):>6} "
                          f"{vd.get('avg_conf',0):>8.3f} {vd['fps']:>7.1f}")

    # ── Save report ──────────────────────────────────────────────────────────
    ts_tag = results['timestamp'][:10]
    report_path = BASE_DIR / "data" / "test_results" / f"eval_{ts_tag}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write(f"DriverGuard Evaluation — {results['timestamp']}\n")
        f.write(f"{'=' * 50}\n\n")
        for yolo in results.get("yolo_benchmarks", []):
            f.write(f"YOLO {yolo['label']}: {fmt_ms(yolo['mean'])} mean, "
                    f"{yolo['fps']:.1f} FPS, {yolo['total_dets']} dets, "
                    f"avg_conf={yolo.get('avg_conf',0):.3f}\n")
        if dms:
            f.write(f"DMS: {fmt_ms(dms['mean'])} mean, {dms['fps']:.1f} FPS\n")
        if pipe:
            f.write(f"Pipeline: {fmt_ms(pipe['mean'])} mean, {pipe['fps']:.1f} FPS\n")
        if acc and "error" not in acc:
            f.write(f"mAP@50: {acc['mAP50']:.4f} | Precision: {acc['precision']:.4f} "
                    f"| Recall: {acc['recall']:.4f}\n")
        for vd in results.get("video_detections", []):
            lbl = vd.get("model_label", "")
            f.write(f"\nVideo {vd['video']} [{lbl}]: {vd['det_rate']} detection rate, "
                    f"{vd['fps']:.1f} FPS, {vd.get('total_dets',0)} dets, "
                    f"avg_conf={vd.get('avg_conf',0):.3f}\n")
            for cls, cnt in sorted(vd["class_counts"].items(), key=lambda x: -x[1]):
                f.write(f"  {cls}: {cnt}\n")
        if pt and nc:
            f.write(f"\nNCNN speedup: {pt['mean']/nc['mean']:.2f}x\n")

    print(f"\n  Report saved → {report_path}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DriverGuard Evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode — fewer frames, skip video tests")
    parser.add_argument("--video", type=str, default=None,
                        help="Specific video file to test")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames per benchmark (default 100)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model path to benchmark")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override inference image size")
    parser.add_argument("--no-cpu", action="store_true",
                        help="Skip PyTorch/CPU comparison (only benchmark NCNN)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Alias for --no-cpu")
    parser.add_argument("--compare", nargs="+", type=int, metavar="SIZE",
                        help="Compare NCNN models at these sizes (e.g. --compare 320 640)")
    args = parser.parse_args()

    skip_pytorch = args.no_cpu or args.no_compare
    n_frames = 30 if args.quick else args.frames

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {"timestamp": timestamp}

    print("\n  DriverGuard Evaluation starting...")
    print(f"  Frames per test: {n_frames}")

    # ── System info ───────────────────────────────────────────────────────────
    print("  Collecting system info...")
    results["system"] = get_system_info()

    # ── Load test frames ──────────────────────────────────────────────────────
    video_path = args.video
    if not video_path:
        for v in ["Pune_Road.mp4", "Pothole.mp4", "test_video.mp4"]:
            if (BASE_DIR / v).exists():
                video_path = str(BASE_DIR / v)
                break

    if not video_path:
        print("  ERROR: No test video found. Place a .mp4 file in the project root.")
        return

    print(f"  Loading frames from {os.path.basename(video_path)}...")
    frames = load_video_frames(video_path, max_frames=n_frames)
    print(f"  Loaded {len(frames)} frames (640x480)")

    # ── Resolve models to benchmark ───────────────────────────────────────────
    pt_path = "models/yolov8n_rdd_india.pt"
    results["yolo_benchmarks"] = []

    if args.compare:
        # Compare mode: benchmark NCNN at each specified size
        for sz in args.compare:
            ncnn_path = find_ncnn_model(imgsz=sz)
            if ncnn_path:
                label = f"NCNN @ imgsz={sz}"
                print(f"  Benchmarking YOLO — {label}...")
                results["yolo_benchmarks"].append(
                    bench_yolo(ncnn_path, frames, label, imgsz=sz))
            else:
                print(f"  WARNING: No NCNN model found for imgsz={sz}")
                print(f"  Export one with: python scripts/export_ncnn.py --imgsz {sz}")

    elif args.model:
        # Explicit model
        imgsz = args.imgsz or detect_imgsz(args.model)
        label = f"{os.path.basename(args.model)} @ imgsz={imgsz}"
        print(f"  Benchmarking YOLO — {label}...")
        results["yolo_benchmarks"].append(
            bench_yolo(args.model, frames, label, imgsz=imgsz))

    else:
        # Auto-detect: benchmark whatever NCNN model is available
        ncnn_path = find_ncnn_model()
        if ncnn_path:
            imgsz = args.imgsz or detect_imgsz(ncnn_path)
            label = f"NCNN @ imgsz={imgsz}"
            print(f"  Benchmarking YOLO — {label}...")
            results["yolo_benchmarks"].append(
                bench_yolo(ncnn_path, frames, label, imgsz=imgsz))

        # PyTorch comparison (unless --no-cpu)
        if os.path.exists(pt_path) and not skip_pytorch:
            imgsz_pt = args.imgsz or 320
            print(f"  Benchmarking YOLO — PyTorch @ imgsz={imgsz_pt} (comparison)...")
            results["yolo_benchmarks"].append(
                bench_yolo(pt_path, frames, f"PyTorch @ imgsz={imgsz_pt}", imgsz=imgsz_pt))

    # ── DMS benchmark ─────────────────────────────────────────────────────────
    print("  Benchmarking DMS (MediaPipe)...")
    try:
        results["dms_benchmark"] = bench_dms(cfg["dms"], frames)
    except Exception as e:
        print(f"  DMS benchmark failed: {e}")
        results["dms_benchmark"] = None

    # ── Full pipeline benchmark ───────────────────────────────────────────────
    if not args.quick:
        print("  Benchmarking full pipeline (DMS + Road)...")
        try:
            results["pipeline_benchmark"] = bench_pipeline(cfg, frames[:50])
        except Exception as e:
            print(f"  Pipeline benchmark failed: {e}")
            results["pipeline_benchmark"] = None
    else:
        results["pipeline_benchmark"] = None

    # ── Accuracy (if validation data available) ───────────────────────────────
    results["accuracy"] = None
    if not args.quick:
        active_model = (args.model or find_ncnn_model() or pt_path)
        if os.path.exists(active_model):
            imgsz_val = args.imgsz or detect_imgsz(active_model)
            print(f"  Checking model accuracy ({os.path.basename(active_model)} @ {imgsz_val})...")
            results["accuracy"] = bench_accuracy(active_model, imgsz=imgsz_val)
            if results["accuracy"] is None:
                print("  Skipped — no local validation dataset found")

    # ── Video detection tests ─────────────────────────────────────────────────
    results["video_detections"] = []
    if not args.quick:
        videos = [args.video] if args.video else []
        if not videos:
            for v in ["Pune_Road.mp4", "Pothole.mp4", "test_video.mp4"]:
                if (BASE_DIR / v).exists():
                    videos.append(str(BASE_DIR / v))

        if args.compare:
            # Run each video with each NCNN size for side-by-side comparison
            for sz in args.compare:
                ncnn_path = find_ncnn_model(imgsz=sz)
                if not ncnn_path:
                    continue
                for vp in videos:
                    label = f"NCNN-{sz}"
                    print(f"  Testing {os.path.basename(vp)} with {label}...")
                    vd = bench_video_detections(ncnn_path, vp, max_frames=200, imgsz=sz)
                    vd["model_label"] = label
                    results["video_detections"].append(vd)
        else:
            active_model = args.model or find_ncnn_model() or pt_path
            imgsz_vid = args.imgsz or detect_imgsz(active_model)
            label = os.path.basename(active_model)
            for vp in videos:
                print(f"  Testing detections on {os.path.basename(vp)}...")
                vd = bench_video_detections(active_model, vp, max_frames=200, imgsz=imgsz_vid)
                vd["model_label"] = label
                results["video_detections"].append(vd)

    # ── Print report ──────────────────────────────────────────────────────────
    print_report(results)


if __name__ == "__main__":
    main()
