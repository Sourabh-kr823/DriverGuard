# Models

This folder contains the AI model files used by DriverGuard.

## Files

### ✅ Included in repository
| File | Size | Purpose |
|------|------|---------|
| `yolov8n_rdd_india.pt` | ~6.2MB | YOLOv8-nano road damage model trained on RDD2022 (58.8% mAP50) |

### ❌ Download separately (too large for git)
| File | Size | Download |
|------|------|----------|
| `shape_predictor_68_face_landmarks.dat` | ~99MB | [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |

## Setup after cloning

```bash
# After git clone, download the dlib model:
# 1. Go to http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# 2. Extract the .dat file
# 3. Place it in this models/ folder

# The YOLOv8 model is already included — no extra download needed
```

## Model Details

### yolov8n_rdd_india.pt
- **Architecture**: YOLOv8-nano (lightweight, edge-optimised)
- **Dataset**: RDD2022 — Road Damage Dataset 2022 (22,700+ images, 6 countries)
- **mAP50**: 58.8%
- **Classes**: 5
  - `0` — pothole
  - `1` — crack_longitudinal
  - `2` — crack_transverse
  - `3` — rutting
  - `4` — repair
- **Input size**: 640×640
- **Inference**: ~20-30ms on GPU, ~100ms on CPU

### shape_predictor_68_face_landmarks.dat
- **Source**: dlib library (Davis King, 2014)
- **Purpose**: 68-point facial landmark detection for DMS pipeline
- **Used by**: `modules/dms/driver_monitor.py` when backend = "dlib" or "both"
- **Note**: System works without this file when using `backend: mediapipe` in config.yaml
