
# 🚗 DriverGuard v2.1

> **Real-time Vehicle Safety Monitor** — Drowsiness Detection + Road Damage Detection on a unified live dashboard.

![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)

---

## 📌 Overview

DriverGuard is a dual-pipeline vehicle safety system that simultaneously monitors:

- **Driver (DMS)** — Eye closure (EAR), yawning (MAR), and head pose to detect drowsiness and distraction in real time
- **Road** — YOLOv8-nano detects potholes, cracks, rutting, and road repairs from a forward-facing camera

All detections are fused into a risk score, logged to SQLite, and streamed to a live Flask + Socket.IO dashboard with GPS mapping, EAR trend charts, and session statistics.

---

## 🖥️ Dashboard Preview

> *Screenshots will be added after road test*

| Live Feed | Session Stats | GPS Map |
|-----------|--------------|---------|
| DMS + Road windows | Risk gauge + counters | Hazard location pins |

---

## ✅ Implemented Features (Phase I)

| Module | Details |
|--------|---------|
| **Pipeline A — DMS** | EAR/MAR drowsiness detection via MediaPipe 468-pt + dlib 68-pt |
| **Auto-calibration** | Personal EAR/MAR baseline per driver, top-75% / bottom-50% filters |
| **Head pose** | Yaw/pitch sustained counter — prevents DISTRACTED flicker false triggers |
| **Multi-driver** | NEW DRIVER button on dashboard triggers fresh recalibration |
| **Pipeline B — Road** | YOLOv8n trained on RDD2022, 58.8% mAP50, 5-class detection |
| **GPS Handler** | NEO-6M serial read + simulation mode (`--simulate` flag) |
| **SQLite Logger** | WAL mode, 3-table schema (driver events, road events, sessions) |
| **Risk Fusion** | Weighted scoring + exponential smoothing |
| **Live Dashboard** | Flask + Socket.IO at localhost:5000 with circular gauge, EAR chart, GPS map |
| **Session Scoping** | All stats, history, and exports filtered to current session only |

---

## 🔭 Planned Features (Phase II — Raspberry Pi 5)

- IR camera on MIPI CSI port for night-time DMS
- Active buzzer alert via GPIO BCM-17
- OBD-II vehicle speed + RPM data fusion
- YOLOv8n fine-tuning on India-specific road footage
- Head-pose distraction detection improvement
- Emergency SOS via GSM/4G module
- Cloud sync — AWS IoT / Firebase road hazard database
- City-wide road damage heat map export
- Mobile companion app for real-time family alerts
- Fleet management portal with driver scoring

---

## 🛠️ Hardware

### Current (Development)
| Component | Details |
|-----------|---------|
| Laptop | Driver-facing camera (`source: 1`) |
| USB Webcam | Road-facing camera (`source: 0`) |
| GPS | NEO-6M USB dongle (simulation mode available) |

### Target (Phase II Deployment)
| Component | Details |
|-----------|---------|
| SBC | Raspberry Pi 5 8GB |
| DMS Camera | IR Camera Module (MIPI CSI) |
| Road Camera | USB Webcam |
| GPS | NEO-6M UART dongle |
| Power | 27W USB-C supply |
| Storage | 32GB microSD |
| Cooling | Case with fan |
| Alert | Active buzzer (GPIO BCM-17) |

---

## 📁 Project Structure

```
DriverGuard/
├── main.py                    # Entry point
├── config.yaml                # All thresholds and settings
├── finetune_india.py          # Road model fine-tuning script
│
├── modules/
│   ├── dms/
│   │   ├── driver_monitor.py  # DMS pipeline orchestrator
│   │   ├── ear_mar.py         # EAR + MAR computation (6-pt formula)
│   │   └── head_pose.py       # Yaw/pitch estimation
│   ├── road/
│   │   └── road_detector.py   # YOLOv8 road damage pipeline
│   ├── gps/
│   │   └── gps_handler.py     # NEO-6M serial + simulation
│   ├── risk/
│   │   └── risk_fusion.py     # Weighted risk scoring
│   └── db/
│       └── db_manager.py      # SQLite WAL logger
│
├── dashboard/
│   └── app.py                 # Flask + Socket.IO dashboard
│
├── models/
│   └── yolov8n_rdd_india.pt   # Road damage model (not in git — too large)
│
└── data/
    └── logs/                  # SQLite DB + session logs
```

---

## ⚙️ Setup

### Prerequisites
- Windows 10/11 or Raspberry Pi OS
- Anaconda / Miniconda
- CUDA-capable GPU (optional but recommended for road model)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/K4s7/DriverGuard.git
cd DriverGuard

# 2. Create conda environment
conda create -n driverguard python=3.10
conda activate driverguard

# 3. Install dependencies
pip install -r requirements.txt
```

### Download large files separately
These are too large for git — get them from the team:

| File | Source |
|------|--------|
| `models/shape_predictor_68_face_landmarks.dat` | [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |
| `models/yolov8n_rdd_india.pt` | Get from team (trained weights) |

---

## 🚀 Usage

```bash
conda activate driverguard
cd DriverGuard

# Full run with GPS simulation + preview windows
python main.py --simulate --preview

# Without preview windows (headless / Pi deployment)
python main.py --simulate
```

Open `http://localhost:5000` in your browser for the live dashboard.

### Camera configuration (`config.yaml`)
```yaml
cameras:
  dms:  1    # Driver-facing (laptop built-in)
  road: 0    # Road-facing  (USB webcam)
```

---

## 🤖 Road Model

| Detail | Value |
|--------|-------|
| Architecture | YOLOv8-nano |
| Dataset | RDD2022 (multi-country via Roboflow) |
| Current mAP50 | 58.8% |
| Classes | pothole, crack_longitudinal, crack_transverse, rutting, repair |
| Input size | 640×640 |

### Fine-tuning on India road footage
```bash
# After collecting road test footage and labelling on Roboflow:
python finetune_india.py

# Optional args
python finetune_india.py --epochs 75 --batch 4 --device 0
```

---

## 📊 DMS Thresholds

| Parameter | Value | Notes |
|-----------|-------|-------|
| EAR threshold | Auto-calibrated | Baseline − 0.08, min 0.15 |
| EAR consec frames | 20 fr (~667ms) | Triggers DROWSY |
| Drowsy watch | 15 fr (~500ms) | Triggers FATIGUED warning |
| MAR threshold | Auto-calibrated | Baseline + 0.38 |
| MAR consec frames | 12 fr (~400ms) | Triggers FATIGUED (yawn) |
| Head yaw limit | 30° sustained 8 fr | Triggers DISTRACTED |
| Head pitch limit | 20° sustained 8 fr | Triggers DISTRACTED |

---

## 👥 Team

| Name | GitHub | Role |
|------|--------|------|
| Sourabh Kumar | [@Sourabh-kr823] (https://github.com/Sourabh-kr823) | Lead - Road detection pipeline, YOLOv8 model training |
| Vaibhav Gupta | [@vaibhavgupta4621] (https://github.com/vaibhavgupta4621) | DMS pipeline, EAR/MAR computation |
| Saumya Raj | [@saumyaraj1925] (https://github.com/saumyaraj1925) | Live web dashboard , Frontend Handling |
| Sumit Kumar | [@sumitkumar93041] (https://github.com/sumitkumar93041) | Backend and Risk fusion Engine development |

---

## 📅 Roadmap

```
Phase I  ✅ Complete    → Laptop prototype, dual-camera, live dashboard
Phase II 🔄 In Progress → Raspberry Pi 5 deployment, hardware integration  
Phase III 📋 Planned    → Cloud sync, fleet management, mobile app
```

---

## 📄 License

This project is developed as part of academic coursework.  
© 2026 DriverGuard Team. All rights reserved.
=======
# DriverGuard
Vehicle Safety Monitor - Driver Monitoring System + Road Damage Detection
>>>>>>> 7e69633f207fa4210e9bb22afe6f0236d92b10a4
