# Driver Monitoring System

> Real-time detection of driver drowsiness, yawning and head-pose inattention using MediaPipe FaceMesh and OpenCV.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-orange)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

![Demo](demo.png)

---

## Overview

A CPU-only driver monitoring system that analyses a live camera feed and raises an alert when the driver shows signs of fatigue or inattention. Three independent detectors run in parallel, each debounced over consecutive frames so that a single blink, a spoken word or a brief glance away does not trigger a false alarm.

| Event | Detection method | Alert |
|-------|-----------------|-------|
| **Drowsiness** | Eye Aspect Ratio falls below threshold for ≥ 15 frames | Visual banner + 880 Hz beep |
| **Yawning** | Mouth Aspect Ratio rises above threshold for ≥ 20 frames | Visual banner + 660 Hz beep |
| **Inattention** | Head pitch, yaw or roll exceeds threshold for ≥ 15 frames | Visual banner + 440 Hz beep |

## Performance

Measured on a standard laptop CPU with no GPU acceleration and no dedicated inference hardware.

| Metric | Result |
|--------|--------|
| Sustained throughput | **23.3 fps** |
| Face detection rate | **88.8%** |
| Facial landmarks tracked | 478 |
| Detection latency (15-frame debounce) | ~0.5 s at 30 fps |
| Model files to download | None |

Running above 20 fps on commodity CPU hardware is the design constraint that matters here: it means the system is deployable on the low-cost embedded hardware realistically found in a vehicle, rather than requiring a discrete GPU.

## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Face mesh and landmarks | `mediapipe` | 0.10.14 |
| Image capture and processing | `opencv-python` | ≥ 4.9.0 |
| Numerical computation | `numpy` | ≥ 1.26.0 |
| Audio alerts | `pygame-ce` | ≥ 2.4.0 |
| Testing | `pytest` | ≥ 8.1.0 |

MediaPipe bundles its own FaceMesh model, so there are no `.dat` weights to download. `dlib`, `imutils` and `scipy` are not required.

## Quick Start

```bash
git clone https://github.com/UbaidurRahman777/Driver-Drowsiness-.git
cd Driver-Drowsiness-

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

Press `q` to quit. A session summary is printed to the console on exit.

### Common options

```bash
python main.py --source 1                      # different camera index
python main.py --width 1920 --height 1080      # change resolution
python main.py --ear-thresh 0.22 --mar-thresh 0.55   # adjust sensitivity
python main.py --no-audio                      # visual alerts only
python main.py --log-level DEBUG               # verbose logging
```

## How It Works

### Eye Aspect Ratio (EAR)

```
       p2    p3
p1  •——•——•——•  • p4
       p6    p5

EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
```

EAR approaches 0 as the eye closes, ranging roughly from 0.20 (closed) to 0.40 (open). The 0.25 threshold sits below the resting open-eye value for most subjects while staying above full closure, and the 15-frame counter distinguishes a microsleep from an ordinary blink, which lasts 3–4 frames at 30 fps.

### Mouth Aspect Ratio (MAR)

```
MAR = ‖top−bottom‖ / ‖left−right‖
```

A MAR above 0.60 indicates a mouth opening consistent with a yawn. The 20-frame debounce is deliberately longer than the eye counter because speech produces frequent brief openings above the threshold; a yawn sustains it.

### Head Pose Estimation

Six stable facial landmarks are matched against a canonical 3-D face model using `cv2.solvePnP` (Perspective-n-Point). The resulting rotation vector is decomposed into pitch (nod), yaw (turn) and roll (tilt) in degrees. The yaw threshold is set wider than pitch and roll at ±30° because normal mirror and shoulder checks involve lateral head movement that should not be flagged.

## Configuration

All defaults live in [`config/settings.py`](config/settings.py) as typed dataclasses. Edit that file to change behaviour permanently without touching detection logic.

| Setting | Default | Description |
|---------|---------|-------------|
| `ear_threshold` | `0.25` | EAR below this counts as eye closed |
| `ear_consecutive_frames` | `15` | Frames of closure before alert |
| `mar_threshold` | `0.60` | MAR above this counts as mouth open |
| `mar_consecutive_frames` | `20` | Frames before a yawn alert |
| `pitch_threshold` | `20.0°` | Head nod limit |
| `yaw_threshold` | `30.0°` | Head turn limit |
| `roll_threshold` | `20.0°` | Head tilt limit |
| `cooldown_seconds` | `2.0` | Minimum gap between same-type audio alerts |

## Project Structure

```
├── main.py                        # CLI entry point
├── requirements.txt
├── config/
│   └── settings.py                # All thresholds and app config (dataclasses)
├── src/
│   ├── detectors/
│   │   ├── eye_detector.py        # EAR-based drowsiness detection
│   │   ├── mouth_detector.py      # MAR-based yawn detection
│   │   └── head_pose_detector.py  # 3-DOF head pose via solvePnP
│   ├── alerts/
│   │   └── alert_manager.py       # Audio beep and cooldown management
│   ├── utils/
│   │   ├── camera.py              # cv2.VideoCapture abstraction
│   │   └── drawing.py             # HUD and overlay rendering
│   └── monitoring/
│       └── driver_monitor.py      # Orchestrator and session statistics
├── logs/                          # Auto-generated rotating log files
└── tests/
    └── test_detectors.py          # Unit tests for EAR and MAR geometry
```

## Tests

```bash
python -m pytest tests/ -v
```

Ten unit tests cover the EAR and MAR geometry against synthetic landmark sets representing open, closed and partially closed states.

## Design Notes

The system was developed as an MSc dissertation project. It replaces the widely-copied `dlib` + `imutils` pipeline from around 2017 with a MediaPipe-based architecture:

- **478 landmarks instead of 68**, giving finer eye and mouth contours and better behaviour under head rotation and partial occlusion.
- **No external model files**, removing the `shape_predictor_68_face_landmarks.dat` download and the `dlib` build step that makes the original difficult to install.
- **Debounced yawn detection**, absent from the original implementation and a documented source of false positives during speech.
- **Modular, type-annotated codebase** with detection logic separated from rendering, alerting and capture, so thresholds and detectors can be unit-tested without a camera.

### Limitations and further work

Honest scope, since this is a prototype rather than a production system:

- Thresholds are fixed rather than calibrated per driver; EAR baselines vary meaningfully between individuals and with eyewear.
- Evaluation was carried out under indoor lighting on a limited set of subjects. Performance under low light, direct sunlight and infrared illumination is untested.
- The 88.8% face detection rate means roughly one frame in nine yields no landmarks; a Kalman filter or similar temporal smoothing would help bridge those gaps.
- No integration with vehicle CAN data, which in a real deployment would provide corroborating signals such as steering variance and lane position.

Assessed informally against the driver monitoring expectations set out in ISO 21448 (SOTIF) and the Euro NCAP assessment protocols.

## Author

**Syed Muhammad Ubaid ur Rahman**
MSc Computing, University of Roehampton, 2026
[github.com/UbaidurRahman777](https://github.com/UbaidurRahman777)

## Licence

MIT — see [LICENSE](LICENSE).
