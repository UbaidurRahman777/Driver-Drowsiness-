# Driver Monitoring System

> Real-time detection of driver drowsiness, yawning, and head-pose inattention
> using MediaPipe FaceMesh and OpenCV.


## Overview

This system analyses a live webcam feed and raises alerts when:

| Event | Detection Method | Alert |
|-------|-----------------|-------|
| **Drowsiness** | Eye Aspect Ratio (EAR) drops below threshold for ≥ N frames | Visual banner + audio beep |
| **Yawning** | Mouth Aspect Ratio (MAR) rises above threshold for ≥ N frames | Visual banner + audio beep |
| **Inattention** | Head pitch / yaw / roll exceeds threshold for ≥ N frames | Visual banner + audio beep |

All three detectors use a **debounce frame counter** to suppress single-frame noise and false positives.


## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Face mesh & landmarks | `mediapipe` | ≥ 0.10.9 |
| Image capture & processing | `opencv-python` | ≥ 4.9.0 |
| Numerical computation | `numpy` | ≥ 1.26.0 |
| Audio alerts | `pygame` | ≥ 2.5.2 |
| Testing | `pytest` | ≥ 8.1.0 |

> **No model files required** — MediaPipe bundles its own FaceMesh model.
> **No `dlib`, `imutils`, or `scipy`** — all replaced by modern equivalents.


## Project Structure

```
driver_monitoring_system/
├── main.py                        # CLI entry point
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── settings.py                # All thresholds & app config (dataclasses)
├── src/
│   ├── detectors/
│   │   ├── eye_detector.py        # EAR-based drowsiness detection
│   │   ├── mouth_detector.py      # MAR-based yawn detection
│   │   └── head_pose_detector.py  # 3-DOF head pose via solvePnP
│   ├── alerts/
│   │   └── alert_manager.py       # Audio beep + cooldown management
│   ├── utils/
│   │   ├── camera.py              # cv2.VideoCapture abstraction
│   │   └── drawing.py             # All HUD / overlay rendering
│   └── monitoring/
│       └── driver_monitor.py      # Main orchestrator + session stats
├── logs/                          # Auto-generated rotating log files
└── tests/
    └── test_detectors.py          # Pytest unit tests for EAR / MAR
```


## Installation

### 1. Clone the repository

```bash
git clone https://github.com/UbaidurRahman777/Driver-Drowsiness-.git
cd driver_monitoring_system
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run with default settings

```bash
python main.py
```

### Common options

```bash
# Use a different camera (e.g. USB camera at index 1)
python main.py --source 1

# Change resolution
python main.py --width 1920 --height 1080

# Adjust detection sensitivity
python main.py --ear-thresh 0.22 --mar-thresh 0.55

# Disable audio alerts
python main.py --no-audio

# Increase log verbosity
python main.py --log-level DEBUG
```

Press **`q`** to quit. A session summary is printed to the console on exit.


## Configuration

All default values live in [`config/settings.py`](config/settings.py).
Edit this file to permanently change thresholds without touching detection logic.

| Setting | Default | Description |
|---------|---------|-------------|
| `ear_threshold` | `0.25` | EAR below this → eye closed |
| `ear_consecutive_frames` | `15` | Frames of closure before alert (~0.5 s at 30 fps) |
| `mar_threshold` | `0.60` | MAR above this → mouth open |
| `mar_consecutive_frames` | `20` | Frames of opening before yawn alert |
| `pitch_threshold` | `20.0°` | Head nod limit |
| `yaw_threshold` | `30.0°` | Head turn limit |
| `roll_threshold` | `20.0°` | Head tilt limit |
| `cooldown_seconds` | `2.0` | Min gap between same-type audio alerts |


## Running the Tests

```bash
python -m pytest tests/ -v
```

Expected output:

```
tests/test_detectors.py::TestEyeAspectRatio::test_open_eye_returns_high_ear  PASSED
tests/test_detectors.py::TestEyeAspectRatio::test_closed_eye_returns_low_ear PASSED
...
```

## How It Works

### Eye Aspect Ratio (EAR)

```
       p2    p3
p1  •——•——•——•  • p4
       p6    p5

EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
```

An EAR close to **0** means the eye is fully closed.
Values typically range from **0.20 (closed) to 0.40 (open)**.

### Mouth Aspect Ratio (MAR)

```
MAR = ‖top−bottom‖ / ‖left−right‖
```

A MAR above **0.60** indicates a wide-open mouth consistent with yawning.

### Head Pose Estimation

Six stable facial landmarks are matched against a canonical 3-D face model
using `cv2.solvePnP` (Perspective-n-Point).  The resulting rotation vector
is decomposed into **pitch** (nod), **yaw** (turn), and **roll** (tilt) in degrees.


## Dissertation Notes

This system was developed as part of a dissertation on real-time driver
monitoring.  The implementation replaces the legacy `dlib` + `imutils`
pipeline (circa 2017) with a modern MediaPipe-based architecture, yielding:

- **No external model files** (no `.dat` download required)
- **Faster inference** on CPU via MediaPipe's GPU-optimised graph
- **478 landmarks** vs the original 68 — higher spatial resolution
- **Modular, testable codebase** with full type annotations
- **Debounced yawn detection** (the original had none — a known false-positive source)


## Author

*Syed Muhammad Ubaid Ur Rahman* — Dissertation Project, 2026
`The University of Roehampton`

`In partial fulfilment of the requirements for the degree of`

`Master of Science in Computing`
