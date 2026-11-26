# 🚗 DriverGuard

> **Edge AI system for real-time fusion of driver drowsiness and road hazard detection on a Raspberry Pi 5** — one system that watches the driver and the road at the same time, combines both into a single speed-aware risk level, and runs entirely on the Pi's CPU with no AI accelerator.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![YOLO11](https://img.shields.io/badge/YOLO11-nano-purple)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarker-green?logo=google)
![NCNN](https://img.shields.io/badge/Inference-NCNN%20%28CPU%29-lightgrey)
![Flask](https://img.shields.io/badge/Flask-Socket.IO-black?logo=flask)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-c51a4a?logo=raspberrypi)

---

## 📌 Overview

DriverGuard runs two pipelines in parallel on a Raspberry Pi 5:

- **Driver monitoring** — a NoIR camera facing the driver measures the Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR) and head pose with MediaPipe, after a short per-driver calibration, to detect drowsiness, yawning and distraction.
- **Road hazard detection** — a USB camera facing the road runs a pothole-only YOLO11-nano model, exported to NCNN, with temporal confirmation and size-based severity.

A Risk Fusion Engine combines the driver state, road hazards and vehicle speed from GPS into one risk level (LOW / MODERATE / HIGH). It drives a buzzer and voice alerts, logs every event with its GPS position to SQLite, and streams everything to a live web dashboard.

---

## ⭐ Engineering Highlights

- **Evaluation-first approach:** every accuracy claim is measured on a held-out test set (precision, recall, mAP), speed is measured on the target device rather than a GPU, and what has *not* been measured yet is stated openly — see [Evaluation Methodology](#-evaluation-methodology).
- **End-to-end computer vision pipeline:** merged three public datasets (about 11,600 training images, 25,359 potholes), trained a YOLO11-nano detector, evaluated it on a held-out Indian test set, and deployed it to an edge device.
- **Edge AI optimisation:** NCNN export and a 416 px input size chosen by experiment (512 px gave no accuracy gain for about 1.5× the compute), so two neural networks run in real time on a Raspberry Pi 5 CPU with no GPU or accelerator.
- **Multi-model fusion:** combines face-landmark analysis (MediaPipe), object detection (YOLO11) and GPS speed into one explainable risk score.
- **Adaptive per-user calibration:** thresholds adjust to each driver's face in the first seconds, with outlier filtering that excludes blinks and speech.
- **Structured, queryable data:** every driver and road event is stored as a typed record with time, measurements, risk state and GPS position, ready for SQL queries, CSV export and PDF session reports.
- **Clean, tested Python:** modular packages per pipeline, a pytest suite covering the core logic and edge cases (blank, corrupted and very dark frames), a watchdog that restarts stalled threads, Docker and a systemd service.

---

## 🧰 Tech Stack

| Area | Tools |
|------|-------|
| AI / ML | YOLO11 (Ultralytics), MediaPipe Face Landmarker, NCNN, OpenCV, NumPy, SciPy |
| Backend | Python 3.11, multi-threading, Flask, Flask-Socket.IO, SQLite |
| Frontend | HTML, CSS, JavaScript, Chart.js, Leaflet (maps) |
| Hardware / IoT | Raspberry Pi 5, Pi Camera Module 3 NoIR, USB camera, NEO-7M GPS (NMEA over UART), GPIO buzzer |
| Tooling | pytest, Docker, systemd, Git |

---

## 📊 Results

| Measure | Result |
|---------|--------|
| Pothole detection, held-out Indian test set (402 images, 756 potholes) | Precision 65.3% · Recall 55.0% · **mAP@0.5 59.5%** · mAP@0.5:0.95 27.9% |
| Pothole detection, validation set (886 images, 1,290 potholes) | Precision 66.9% · Recall 52.0% · mAP@0.5 57.1% · mAP@0.5:0.95 26.7% |
| Road model speed on Pi 5 CPU (NCNN, 416 px) | ≈ 51 ms per frame (≈ 19.6 FPS) |
| Driver monitoring speed on Pi 5 CPU | ≈ 9.8 ms per frame |
| Complete system | Keeps pace with both 20 FPS cameras (road model runs on every second frame) |
| Prototype test scenarios (eyes open / closed, yawning, head turned, pothole) | All passed |

Drowsiness detection was tested in controlled sessions with team members. Precision and recall for drowsiness are not reported because that needs labelled driving video, which has not been collected yet.

---

## 🧪 Evaluation Methodology

**How do we know the system is any good?** These are the rules followed when measuring it.

**Road model**
- **Separate test set.** 402 Indian road images (756 potholes) were kept out of training and used only for the final numbers; a separate validation set (886 images) was used during training.
- **Metrics chosen for the use case.**
  - *Precision* — how many alerts are real potholes (false alarms annoy drivers).
  - *Recall* — how many real potholes are caught (missed hazards are the safety risk).
  - *mAP@0.5* — overall detection quality.
  - *mAP@0.5:0.95* — how tightly the boxes fit, which matters because severity is graded from box size.
- **Design decisions backed by measurement.** 416 px and 512 px inputs were compared on the same data: 512 px scored 58.9% mAP@0.5 against 59.5% for 416 px, so the cheaper 416 px model was kept.
- **Speed measured where it runs.** Timings are from the Raspberry Pi 5 CPU with the NCNN model, not from a training GPU.
- **Numbers in context.** Indian roads are a hard benchmark: in the CRDDC'2022 road-damage challenge the best team's F1 score on Indian images was 0.583. Accuracy figures from other papers use different datasets, so they are not directly comparable to this test set.
- **Reducing false alarms at run time.** A detection must appear in at least 2 of the last 4 processed frames before it triggers an alert, which filters out one-frame flickers.

**Driver monitoring**
- **Why calibration was needed.** Phase I logs showed one driver's normal open-eye EAR at 0.217 while others sat at 0.277–0.287, so one fixed threshold would either miss eye closures or fire constantly for some drivers. Thresholds are now set per driver at start-up, with outliers (blinks, speech) filtered out.
- **Scenario tests.** Eyes open, eyes closed, yawning and head turned were each checked against the expected state and risk level.
- **Unit tests.** The pytest suite checks EAR / MAR computation, severity grading, temporal confirmation, risk scoring, speed scaling, GPS parsing and degraded inputs.

**Not measured yet (stated honestly)**
- Drowsiness precision and recall on labelled video (planned with the YawDD dataset).
- Accuracy under live driving with real GPS speed; bench tests used the GPS simulation mode.

---

## 🗂️ Data Model

All events are written to SQLite (WAL mode) as structured records:

| Table | One row per | Key fields |
|-------|-------------|------------|
| `driver_events` | Driver-state reading | time, EAR, MAR, yaw, pitch, state, risk level, eye-closure / yawn frame counts, latitude, longitude |
| `road_events` | Confirmed pothole | time, class, confidence, severity, box area, latitude, longitude |
| `system_log` | System message | time, level, message |

Because each hazard carries a GPS position and time, the data can answer questions such as "which stretches of road have repeated severe potholes" or "how often did this driver become drowsy per hour". The dashboard uses it for the hazard map and heatmap, the 80 m hazard-ahead warning, CSV export and the PDF session report.

---

## ✅ Features

| Module | Details |
|--------|---------|
| **Driver monitoring** | EAR, MAR and head pose (yaw / pitch) from MediaPipe Face Landmarker; optional dlib 68-point backend |
| **Per-driver calibration** | EAR baseline from the upper 75% of samples (excludes blinks), threshold = baseline − 0.08 (min 0.15); MAR baseline from the lower 50% of samples, threshold = baseline + 0.38 |
| **New driver** | NEW DRIVER button on the dashboard starts a fresh calibration without reloading models |
| **Low-light handling** | Dark frames are enhanced with CLAHE (Contrast-Limited Adaptive Histogram Equalization); a dark frame with no face is reported as LOW VISIBILITY instead of ALERT |
| **Road hazard detection** | YOLO11-nano, pothole only, 416 px input, NCNN on the Pi CPU; confidence 0.35, NMS IoU 0.35 |
| **Temporal confirmation** | A pothole counts only if it appears in at least 2 of the last 4 processed frames |
| **Severity** | By box area on 640×480 frames: minor < 3,000 px², moderate < 10,000 px², otherwise severe |
| **Risk Fusion Engine** | Driver risk = 0.45 × eye closure + 0.25 × yawning + 0.30 × head deviation, scaled by speed (×1.0 up to 40 km/h, rising to ×1.35 at 80 km/h); a severe pothole raises LOW to MODERATE |
| **GPS and proximity alerts** | NEO-7M GPS on the Pi UART; each pothole is geo-tagged, and a warning is given when a logged hazard lies within 80 m ahead; optional OpenStreetMap road-surface check; simulation mode for bench tests |
| **Alerts** | KY-012 active buzzer on GPIO 17; two voice channels with espeak-ng (female voice for driver alerts, male voice for road alerts) |
| **Event logging** | SQLite in WAL (write-ahead logging) mode: driver events, road events and system log |
| **Live dashboard** | Flask + Socket.IO on port 5000: risk gauge, driver measurements, road counters by severity, EAR trend chart, hazard map with heatmap view, alert history, driver scorecard, FPS and latency, weather (optional), dark / light theme, keyboard shortcuts |
| **Exports** | Driver and road events as CSV; PDF session report |
| **Reliability** | Watchdog restarts any pipeline thread silent for more than 8 s; systemd service starts DriverGuard at boot |

---

## 🛠️ Hardware

| Component | Details |
|-----------|---------|
| Board | Raspberry Pi 5 (8 GB) |
| Driver camera | Raspberry Pi Camera Module 3 NoIR (CSI), 640×480 at 20 FPS |
| Road camera | USB webcam, 640×480 at 20 FPS |
| GPS | u-blox NEO-7M on the Pi UART (`/dev/serial0`, 9600 baud) |
| Buzzer | KY-012 active buzzer (GPIO 17) |
| Audio | Speaker for voice alerts |
| Power | Raspberry Pi 27 W USB-C power supply |

No GPU or AI accelerator is used; all inference runs on the Pi 5's ARM CPU.

---

## 📁 Project Structure

```
DriverGuard/
├── main.py                       # Entry point (starts all pipelines + dashboard)
├── config.yaml                   # Cameras, thresholds, risk weights, GPS, alerts
├── requirements.txt
├── .env.template                 # Copy to .env for the optional weather API key
├── driver_guard.service          # systemd service for auto-start on the Pi
├── Dockerfile · docker-compose.yml
├── evaluate.py                   # Speed / accuracy benchmark of the road model
│
├── backend/
│   ├── dms/
│   │   ├── driver_monitor.py     # Driver pipeline: calibration, states, low light
│   │   ├── ear_mar.py            # EAR + MAR computation
│   │   ├── head_pose.py          # Yaw / pitch estimation
│   │   └── mp_face_mesh.py       # MediaPipe Face Landmarker wrapper
│   ├── road/
│   │   ├── detector.py           # YOLO11 pothole detection, confirmation, severity
│   │   └── proximity_alert.py    # Hazard-ahead warnings (GPS + OpenStreetMap)
│   ├── gps/
│   │   └── gps_reader.py         # NEO-7M serial reader + simulation mode
│   ├── alert/
│   │   ├── alert_manager.py      # Risk Fusion Engine + buzzer
│   │   └── voice_alert.py        # Two-voice spoken alerts
│   └── database/
│       └── db_manager.py         # SQLite (WAL) event logger
│
├── dashboard/
│   └── app.py                    # Flask + Socket.IO live dashboard
│
├── utils/
│   ├── camera.py                 # Opens Pi CSI camera, USB camera or video file
│   ├── risk_score.py             # Risk-score math
│   └── watchdog.py               # Thread watchdog / auto-restart
│
├── scripts/
│   ├── export_ncnn.py            # Export the .pt model to NCNN for the Pi
│   ├── test_images.py            # Run the road model on images or a video
│   ├── evaluate.py               # Validation, benchmark, tests and charts
│   ├── check_system.py           # Pre-flight check of cameras, models, GPS
│   ├── check_cameras.py          # List available cameras
│   ├── inject_test.py            # Insert test hazards into the database
│   ├── clear_db.py               # Clear the event database
│   └── ...                       # Dataset conversion, merging and training helpers
│
├── tests/                        # pytest suite (8 files)
│
├── models/
│   ├── pothole_yolo11n.pt        # Trained pothole model (YOLO11-nano)
│   └── README.md
│
└── data/logs/                    # SQLite database (not tracked)
```

---

## ⚙️ Setup (Raspberry Pi 5)

For Raspberry Pi OS (64-bit).

```bash
# 1. System packages (Pi camera library and voice engine)
sudo apt install python3-picamera2 espeak-ng

# 2. Clone the repository
git clone https://github.com/Sourabh-kr823/DriverGuard.git
cd DriverGuard

# 3. Virtual environment that can see picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. MediaPipe face model
wget -O models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# 5. Export the pothole model to NCNN (creates models/pothole_yolo11n_416_ncnn_model)
python scripts/export_ncnn.py

# 6. Optional: weather widget
cp .env.template .env      # then add your OPENWEATHER_API_KEY
```

Check everything is in place:

```bash
python scripts/check_system.py
```

> **Laptop / development use:** set the camera `source` values in `config.yaml` to OpenCV indices (for example `0` and `1`) or to a video file. The dlib backend is optional and only needed if `dms.backend` is set to `dlib` or `both`.

---

## 🚀 Usage

```bash
source .venv/bin/activate

# Normal run
python main.py

# Bench test with simulated GPS and OpenCV preview windows
python main.py --simulate --preview

# Save road footage to data/recordings/ while running
python main.py --record

# Run without the web dashboard
python main.py --no-web

# Test the road model on an image or video
python scripts/test_images.py --source path/to/image_or_video

# Run the test suite (pip install pytest first)
pytest
```

Open `http://<pi-address>:5000` (or `http://localhost:5000` on the Pi) for the live dashboard.

### Start automatically at boot

```bash
sudo cp driver_guard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now driver_guard
```

Edit the `User` and paths in `driver_guard.service` to match your Pi first.

---

## 🤖 AI / ML Pipeline (Road Model)

**Data → training → evaluation → edge deployment**

1. **Data:** pothole labels taken from three datasets and merged into one class, plus road images without potholes to cut false alarms.
2. **Training:** YOLO11-nano trained on a cloud GPU.
3. **Evaluation:** precision, recall and mAP measured on Indian road images never seen in training.
4. **Optimisation:** input sizes compared (416 px vs 512 px) and the model exported to NCNN for the ARM CPU.
5. **Deployment:** runs live on the Pi 5 with temporal confirmation and severity grading.


| Detail | Value |
|--------|-------|
| Architecture | YOLO11-nano (about 2.6 million parameters) |
| Classes | 1 — pothole |
| Input size | 416 × 416 |
| Runtime format | NCNN (Pi 5 CPU) |
| Training data | Pothole labels from RDD2022, the Indian Roads Dataset and a second Indian pothole dataset; about 11,600 images with 25,359 potholes, plus about 800 road images without potholes to reduce false alarms |
| Test data | 402 held-out Indian road images with 756 potholes |
| Training | Cloud GPU |

A 512 px version reached 58.9% mAP@0.5 on the same data, no better than 416 px while needing about 1.5 times as many pixels per frame, so the 416 px model is deployed.

---

## 📊 Driver Monitoring Thresholds

Camera runs at 20 FPS, so 20 frames = 1 second.

| State | Condition | Risk level |
|-------|-----------|------------|
| DROWSY | Eyes closed for 13 frames (650 ms) | HIGH |
| FATIGUED (drowsy watch) | Eyes closed for 10 frames (500 ms) | MODERATE |
| FATIGUED (yawning) | Mouth open beyond the MAR threshold for 8 frames (400 ms) | MODERATE |
| DISTRACTED | Yaw beyond 30° or pitch beyond 20° for 5 frames (250 ms) | MODERATE |
| LOW VISIBILITY | Frame too dark and no face found | — |

EAR and MAR thresholds are set by the per-driver calibration at start-up; all values can be changed in `config.yaml`.

---

## 🔭 Future Work

- Measure drowsiness precision and recall on labelled video (for example the YawDD dataset)
- Road trials with live GPS input and real vehicle speed
- Larger road model on a low-cost accelerator such as the Raspberry Pi AI HAT+
- More labelled Indian road images, including night and rain
- Infrared illumination for night-time driver monitoring
- Phone-use detection
- Tune the fusion weights with an ablation study

---

## 👤 Author

**Sourabh Kumar** — [@Sourabh-kr823](https://github.com/Sourabh-kr823)<br>

---

## 📅 Roadmap

```
Phase I   ✅ Complete → Laptop prototype, dual camera, live dashboard
Phase II  ✅ Complete → Raspberry Pi 5 deployment, YOLO11 pothole model, GPS, buzzer, voice alerts
Phase III 📋 In Progress  → Road trials, labelled drowsiness evaluation, accelerator-based model
```

---

## 📄 License

This project is developed as part of academic coursework.
© 2026 DriverGuard. All rights reserved.
