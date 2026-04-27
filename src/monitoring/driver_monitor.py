"""
src/monitoring/driver_monitor.py
==================================
Main orchestrator for the Driver Monitoring System.

Ties together the camera, MediaPipe FaceMesh, all detectors,
the alert manager, and the HUD drawing utilities into a single
``DriverMonitor`` class with a clean ``run()`` loop.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any

import cv2
import mediapipe as mp
import numpy as np

from config.settings import AppConfig
from src.alerts.alert_manager import AlertManager, AlertType
from src.detectors.eye_detector import EyeDetector
from src.detectors.head_pose_detector import HeadPoseDetector
from src.detectors.mouth_detector import MouthDetector
from src.utils.camera import CameraManager
from src.utils import drawing

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    """Accumulated statistics for a monitoring session."""
    start_time: float = field(default_factory=time.monotonic)
    total_frames: int = 0
    faces_detected: int = 0
    drowsy_events: int = 0
    yawn_events: int = 0
    distraction_events: int = 0

    # State trackers to count transitions (not per-frame)
    _prev_drowsy: bool = False
    _prev_yawning: bool = False
    _prev_distracted: bool = False

    def update(
        self,
        face_found: bool,
        eye_result: Dict[str, Any],
        mouth_result: Dict[str, Any],
        pose_result: Dict[str, Any],
    ) -> None:
        self.total_frames += 1
        if face_found:
            self.faces_detected += 1

        # Count rising edges only (event onset)
        drowsy = eye_result.get("is_drowsy", False)
        yawning = mouth_result.get("is_yawning", False)
        distracted = pose_result.get("is_distracted", False)

        if drowsy and not self._prev_drowsy:
            self.drowsy_events += 1
        if yawning and not self._prev_yawning:
            self.yawn_events += 1
        if distracted and not self._prev_distracted:
            self.distraction_events += 1

        self._prev_drowsy = drowsy
        self._prev_yawning = yawning
        self._prev_distracted = distracted

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def summary(self) -> str:
        elapsed = self.elapsed_seconds
        fps = self.total_frames / elapsed if elapsed > 0 else 0
        return (
            f"\n{'='*50}\n"
            f"  SESSION SUMMARY\n"
            f"{'='*50}\n"
            f"  Duration         : {elapsed:.1f}s\n"
            f"  Frames processed : {self.total_frames} ({fps:.1f} fps)\n"
            f"  Frames with face : {self.faces_detected}\n"
            f"  Drowsy events    : {self.drowsy_events}\n"
            f"  Yawn events      : {self.yawn_events}\n"
            f"  Distraction evts : {self.distraction_events}\n"
            f"{'='*50}\n"
        )


class DriverMonitor:
    """
    Real-time driver monitoring system.

    Detects:
        - Drowsiness via Eye Aspect Ratio (EAR)
        - Yawning via Mouth Aspect Ratio (MAR)
        - Inattention via 3-DOF head pose estimation

    Usage::

        monitor = DriverMonitor(AppConfig())
        monitor.run()
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        t = config.thresholds
        a = config.alerts

        self._camera = CameraManager(
            source=config.camera.source,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
        )
        self._eye_det = EyeDetector(t.ear_threshold, t.ear_consecutive_frames)
        self._mouth_det = MouthDetector(t.mar_threshold, t.mar_consecutive_frames)
        self._head_det = HeadPoseDetector(
            t.pitch_threshold, t.yaw_threshold,
            t.roll_threshold, t.head_pose_consecutive_frames,
        )
        self._alert_mgr = AlertManager(
            enable_audio=a.enable_audio,
            cooldown_seconds=a.cooldown_seconds,
            alert_duration_ms=a.alert_duration_ms,
        )
        self._stats = SessionStats()

        # MediaPipe FaceMesh
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    # Main loop

    def run(self) -> None:
        """Start the monitoring loop. Press 'q' to quit."""
        logger.info("Starting %s v%s", self.config.app_name, self.config.version)

        if not self._camera.initialize():
            logger.error("Cannot open camera — aborting.")
            return

        try:
            self._loop()
        finally:
            self._shutdown()

    def _loop(self) -> None:
        empty_eye = {"is_drowsy": False, "avg_ear": 0.0,
                     "left_ear": 0.0, "right_ear": 0.0, "closed_frames": 0}
        empty_mouth = {"is_yawning": False, "mar": 0.0, "yawn_frames": 0}
        empty_pose = {"success": False, "is_distracted": False,
                      "pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        while True:
            ret, frame = self._camera.read_frame()
            if not ret or frame is None:
                logger.warning("Frame read failed — skipping.")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

            eye_res, mouth_res, pose_res = empty_eye, empty_mouth, empty_pose
            face_found = False

            if results.multi_face_landmarks:
                face_found = True
                landmarks = results.multi_face_landmarks[0]
                shape = frame.shape

                eye_res = self._eye_det.process(landmarks, shape)
                mouth_res = self._mouth_det.process(landmarks, shape)
                pose_res = self._head_det.process(landmarks, shape)

                # Fire alerts
                if eye_res["is_drowsy"]:
                    self._alert_mgr.trigger(AlertType.DROWSINESS)
                if mouth_res["is_yawning"]:
                    self._alert_mgr.trigger(AlertType.YAWNING)
                if pose_res["is_distracted"]:
                    self._alert_mgr.trigger(AlertType.HEAD_POSE)

                # Draw overlays
                if self.config.display.show_landmarks:
                    drawing.draw_eye_landmarks(frame, eye_res)
                    drawing.draw_mouth_landmarks(frame, mouth_res)
                if self.config.display.show_head_axes:
                    drawing.draw_head_pose_axis(frame, pose_res)

            face_count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
            drawing.draw_face_count(frame, face_count)

            if self.config.display.show_stats_panel:
                drawing.draw_stats_panel(frame, eye_res, mouth_res, pose_res)

            drawing.draw_alert_banners(frame, eye_res, mouth_res, pose_res)

            self._stats.update(face_found, eye_res, mouth_res, pose_res)

            cv2.imshow(self.config.display.window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit requested by user.")
                break
    # Shutdown

    def _shutdown(self) -> None:
        self._camera.release()
        self._face_mesh.close()
        cv2.destroyAllWindows()
        print(self._stats.summary())
        logger.info("Session ended.%s", self._stats.summary())
