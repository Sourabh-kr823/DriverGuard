"""
DriverGuard evaluation — only measured, defensible results:
  1. Road-model validation  — mAP50, mAP50-95, per-class AP50/precision/recall
  2. Inference benchmark     — ms/frame + FPS (GPU+CPU for .pt, CPU for NCNN)
  3. Test suite              — pytest pass/fail (correctness evidence)

Run from the repo root:
    python scripts/evaluate.py
    python scripts/evaluate.py --no-val       # skip dataset validation
    python scripts/evaluate.py --no-cpu       # skip the (slow) CPU benchmark (.pt only)
    python scripts/evaluate.py --no-tests
    python scripts/evaluate.py --device cpu

Works with both the .pt model (GPU when available) and an exported NCNN model
(a "_ncnn_model" folder, CPU-only). Writes evaluation_report.md and PNG charts
when matplotlib is available. DMS drowsiness precision/recall needs labeled
ground-truth video, which is not part of the logged data — this script does
NOT fabricate a DMS accuracy figure.
"""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LINE = "─" * 70


def load_config():
    p = ROOT / "config.yaml"
    return (yaml.safe_load(open(p)) or {}) if p.exists() else {}


def is_ncnn(model_path):
    return (str(model_path).rstrip("/").endswith("_ncnn_model")
            or Path(model_path).is_dir())


def model_size_mb(model_path):
    mp = Path(model_path)
    try:
        if mp.is_dir():                      # NCNN folder — sum the .bin weights
            size = sum(f.stat().st_size for f in mp.glob("*.bin"))
        else:
            size = mp.stat().st_size
        return round(size / 1e6, 1)
    except Exception:
        return None


def load_model(model_path):
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        return None, None, f"ultralytics/torch not available: {e}"
    if not Path(model_path).exists():
        return None, None, f"model not found: {model_path}"
    model = YOLO(model_path, task="detect") if is_ncnn(model_path) else YOLO(model_path)
    return model, torch, None


# ── 1. Inference benchmark ──────────────────────────────────────────────────

def _time_device(model, device, runs, ncnn=False):
    import numpy as np
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    if not ncnn:
        model.to(device)                      # NCNN has no .to() — CPU only
    model(blank, verbose=False, device=device)          # warm-up (not timed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(blank, verbose=False, device=device)
        times.append((time.perf_counter() - t0) * 1000)
    avg = statistics.mean(times)
    return {"ms": round(avg, 1), "fps": round(1000 / avg, 1) if avg else 0}


def benchmark(model, torch, model_path, device_arg, do_cpu):
    out = {}
    if is_ncnn(model_path):
        out["engine"] = "NCNN"
        out["cpu"] = _time_device(model, "cpu", 10, ncnn=True)
    else:
        out["engine"] = "PyTorch"
        has_cuda = bool(torch.cuda.is_available())
        primary = device_arg if device_arg else (0 if has_cuda else "cpu")
        if str(primary) == "0":
            out["gpu"] = _time_device(model, 0, 30)
        else:
            out["cpu"] = _time_device(model, "cpu", 5)
        if do_cpu and "cpu" not in out:
            out["cpu"] = _time_device(model, "cpu", 5)   # for the comparison

    out["size_mb"] = model_size_mb(model_path)
    try:
        out["params_m"] = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    except Exception:
        out["params_m"] = None      # NCNN model isn't a torch module
    return out


# ── 2. Road-model validation ────────────────────────────────────────────────

def validate_road(model, device_arg, data_yaml, ncnn=False, imgsz=640):
    if not Path(data_yaml).exists():
        return {"error": f"dataset yaml not found: {data_yaml}"}
    try:
        import torch
        dev = "cpu" if ncnn else (device_arg if device_arg
                                  else (0 if torch.cuda.is_available() else "cpu"))
        m = model.val(data=str(data_yaml), verbose=False, device=dev, imgsz=imgsz)
    except Exception as e:
        return {"error": f"validation failed: {e}"}
    res = {"map50": round(float(m.box.map50) * 100, 1),
           "map5095": round(float(m.box.map) * 100, 1),
           "per_class": []}
    try:
        names = model.names
        for k, ci in enumerate(m.box.ap_class_index):
            res["per_class"].append({
                "name": names[int(ci)],
                "ap50": round(float(m.box.ap50[k]) * 100, 1),
                "precision": round(float(m.box.p[k]) * 100, 1),
                "recall": round(float(m.box.r[k]) * 100, 1)})
    except Exception:
        pass
    return res


# ── 3. Test suite ───────────────────────────────────────────────────────────

def run_tests():
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=no"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    except Exception as e:
        return {"error": f"could not run pytest: {e}"}
    import re
    summary = ((p.stdout or p.stderr).strip().splitlines() or [""])[-1]
    mp = re.search(r"(\d+)\s+passed", summary)
    mf = re.search(r"(\d+)\s+failed", summary)
    return {"passed": int(mp.group(1)) if mp else 0,
            "failed": int(mf.group(1)) if mf else 0,
            "summary": summary}


# ── Charts ──────────────────────────────────────────────────────────────────

def make_charts(road, bench, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    if road and not road.get("error") and road.get("per_class"):
        names = [c["name"] for c in road["per_class"]]
        ap = [c["ap50"] for c in road["per_class"]]
        plt.figure(figsize=(7, 4))
        plt.bar(names, ap, color="#B91C3C")
        plt.ylabel("AP50 (%)"); plt.title("Per-class AP50 — Road Model (RDD2022 val)")
        plt.ylim(0, 100); plt.xticks(rotation=20, ha="right"); plt.tight_layout()
        p = outdir / "per_class_ap50.png"; plt.savefig(p, dpi=130); plt.close(); made.append(p)

    if bench and "gpu" in bench and "cpu" in bench:          # only for .pt (GPU vs CPU)
        vals = [bench["cpu"]["ms"], bench["gpu"]["ms"]]
        plt.figure(figsize=(5, 4))
        bars = plt.bar(["CPU", "GPU"], vals, color=["#9CA3AF", "#3F6B2A"])
        plt.ylabel("ms / frame (lower = faster)")
        plt.title("Road Inference Speed — CPU vs GPU")
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width()/2, v, f"{v} ms", ha="center", va="bottom")
        plt.tight_layout()
        p = outdir / "gpu_vs_cpu_speed.png"; plt.savefig(p, dpi=130); plt.close(); made.append(p)

    return made


# ── Report ──────────────────────────────────────────────────────────────────

def build_report(bench, road, tests, charts):
    L = ["# DriverGuard — Evaluation Report", ""]

    L.append("## 1. Road-Model Validation (RDD2022 val split)")
    if road.get("error"):
        L.append(f"_Skipped — {road['error']}_")
    else:
        L += [f"- **mAP50: {road['map50']}%**   ·   mAP50-95: {road['map5095']}%", ""]
        if road["per_class"]:
            L += ["| Class | AP50 % | Precision % | Recall % |", "|---|---|---|---|"]
            for c in road["per_class"]:
                L.append(f"| {c['name']} | {c['ap50']} | {c['precision']} | {c['recall']} |")
    L.append("")

    L.append("## 2. Inference Benchmark")
    if bench.get("error"):
        L.append(f"_Skipped — {bench['error']}_")
    else:
        eng = bench.get("engine", "")
        if "gpu" in bench:
            L.append(f"- **GPU ({eng}): {bench['gpu']['ms']} ms/frame (~{bench['gpu']['fps']} FPS)**")
        if "cpu" in bench:
            lead = "**" if "gpu" not in bench else ""
            end = "**" if "gpu" not in bench else ""
            L.append(f"- {lead}CPU ({eng}): {bench['cpu']['ms']} ms/frame (~{bench['cpu']['fps']} FPS){end}")
        if "gpu" in bench and "cpu" in bench and bench["gpu"]["ms"]:
            L.append(f"- Speedup: **{round(bench['cpu']['ms']/bench['gpu']['ms'], 1)}× faster on GPU**")
        params = f"{bench['params_m']} M" if bench.get("params_m") else "n/a (NCNN)"
        L.append(f"- Model size: {bench['size_mb']} MB   ·   Parameters: {params}")
    L.append("")

    L.append("## 3. Test Suite (pytest)")
    if tests.get("error"):
        L.append(f"_Skipped — {tests['error']}_")
    else:
        L.append(f"- **{tests['passed']} passed, {tests['failed']} failed**  ({tests['summary']})")
    L.append("")

    L.append("## Note on DMS accuracy")
    L.append("Drowsiness precision/recall requires labeled ground-truth video and is not "
             "computed here, so no DMS accuracy figure is fabricated. DMS correctness is "
             "evidenced by the test suite above and the manual test-scenario results.")
    if charts:
        L += ["", "## Charts", *[f"- {p.name}" for p in charts]]
    return "\n".join(L)


# ── Entry ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DriverGuard evaluation")
    ap.add_argument("--device", default=None, help="'cpu' or a CUDA index (.pt only)")
    ap.add_argument("--data", default=str(ROOT / "data" / "finetune_india.yaml"))
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--no-cpu", action="store_true", help="skip the slow CPU benchmark (.pt only)")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="validation input size (e.g. 320 vs 640 to compare accuracy)")
    args = ap.parse_args()

    cfg = load_config()
    model_path = cfg.get("road", {}).get("model_path", "models/yolov8n_rdd_india.pt")

    print("Running evaluation …\n")
    model, torch, err = load_model(model_path)
    if err:
        bench = {"error": err}; road = {"error": err}
    else:
        ncnn = is_ncnn(model_path)
        bench = benchmark(model, torch, model_path, args.device, not args.no_cpu)
        road = {"error": "skipped (--no-val)"} if args.no_val \
            else validate_road(model, args.device, args.data, ncnn=ncnn, imgsz=args.imgsz)

    tests = {"error": "skipped (--no-tests)"} if args.no_tests else run_tests()
    charts = [] if args.no_charts else make_charts(road, bench, ROOT / "evaluation_charts")

    report = build_report(bench, road, tests, charts)
    print(LINE); print(report); print(LINE)
    (ROOT / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(f"\nSaved: {ROOT / 'evaluation_report.md'}")
    if charts:
        print("Charts: " + ", ".join(str(p) for p in charts))


if __name__ == "__main__":
    main()