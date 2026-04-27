"""
src/detectors/head_pose_detector.py
=====================================
3-DOF head pose estimation (pitch, yaw, roll) using the
Perspective-n-Point (PnP) algorithm with six MediaPipe
facial anchor points and a canonical 3-D face model.

Outputs pitch / yaw / roll in degrees and a distraction flag
when any angle exceeds its configured threshold.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Six stable MediaPipe FaceMesh landmark indices for PnP
HEAD_POSE_IDX = [
    1,    # Nose tip
    152,  # Chin
    226,  # Left eye — outer corner
    446,  # Right eye — outer corner
    57,   # Left mouth corner
    287,  # Right mouth corner
]

# Corresponding canonical 3-D face model points (millimetres, origin = nose tip)
FACE_3D_MODEL = np.array(
    [
        (0.0,    0.0,    0.0),       # Nose tip
        (0.0,  -330.0,  -65.0),      # Chin
        (-225.0, 170.0, -135.0),     # Left eye outer corner
        (225.0,  170.0, -135.0),     # Right eye outer corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0,  -150.0, -125.0),    # Right mouth corner
    ],
    dtype=np.float64,
)


class HeadPoseDetector:
    """
    Estimates head pose (pitch, yaw, roll) from MediaPipe face landmarks.

    Uses ``cv2.solvePnP`` (ITERATIVE) with an approximate camera matrix
    derived from the frame dimensions.  While not as accurate as a
    calibrated camera, this approach is hardware-agnostic and sufficient
    for driver monitoring purposes.

    Args:
        pitch_threshold: Alert if |pitch| exceeds this value (degrees).
        yaw_threshold:   Alert if |yaw|   exceeds this value (degrees).
        roll_threshold:  Alert if |roll|  exceeds this value (degrees).
        consecutive_frames: Frames above threshold before alert fires.
    """

    def __init__(
        self,
        pitch_threshold: float = 20.0,
        yaw_threshold: float = 30.0,
        roll_threshold: float = 20.0,
        consecutive_frames: int = 15,
    ) -> None:
        self.pitch_threshold = pitch_threshold
        self.yaw_threshold = yaw_threshold
        self.roll_threshold = roll_threshold
        self.consecutive_frames = consecutive_frames

        self._distracted_frames: int = 0
        self._is_distracted: bool = False

    # Private helpers

    @staticmethod
    def _camera_matrix(frame_shape: Tuple[int, ...]) -> np.ndarray:
        """Build an approximate pinhole camera matrix from frame dimensions."""
        h, w = frame_shape[:2]
        f = w  # Approximate focal length = image width
        cx, cy = w / 2.0, h / 2.0
        return np.array(
            [[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64
        )

    @staticmethod
    def _rotation_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a 3×3 rotation matrix to Euler angles (pitch, yaw, roll)
        in degrees.

        Returns:
            (pitch, yaw, roll) in degrees.
        """
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0.0

        return math.degrees(x), math.degrees(y), math.degrees(z)

    # Main processing

    def process(
        self,
        face_landmarks: Any,
        frame_shape: Tuple[int, ...],
    ) -> Dict[str, Any]:
        """
        Estimate head pose for one frame.

        Args:
            face_landmarks: A single MediaPipe NormalizedLandmarkList.
            frame_shape: (H, W, C) shape tuple.

        Returns:
            Dictionary with pitch/yaw/roll, solvePnP outputs, and flags.
            ``success`` key is False if solvePnP fails.
        """
        h, w = frame_shape[:2]

        image_pts = np.array(
            [
                (face_landmarks.landmark[i].x * w,
                 face_landmarks.landmark[i].y * h)
                for i in HEAD_POSE_IDX
            ],
            dtype=np.float64,
        )

        cam_matrix = self._camera_matrix(frame_shape)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL,
            image_pts,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return {
                "success": False,
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "is_distracted": self._is_distracted,
            }

        rot_matrix, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_to_euler(rot_matrix)

        # Project nose direction arrow
        nose_end, _ = cv2.projectPoints(
            np.array([(0.0, 0.0, 500.0)]),
            rvec, tvec, cam_matrix, dist_coeffs,
        )

        # Distraction check
        above_threshold = (
            abs(pitch) > self.pitch_threshold
            or abs(yaw) > self.yaw_threshold
            or abs(roll) > self.roll_threshold
        )
        if above_threshold:
            self._distracted_frames += 1
            if self._distracted_frames >= self.consecutive_frames:
                self._is_distracted = True
        else:
            self._distracted_frames = max(0, self._distracted_frames - 1)
            if self._distracted_frames == 0:
                self._is_distracted = False

        return {
            "success": True,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": cam_matrix,
            "dist_coeffs": dist_coeffs,
            "nose_end_2d": nose_end,
            "image_points": image_pts,
            "distracted_frames": self._distracted_frames,
            "is_distracted": self._is_distracted,
        }

    def reset(self) -> None:
        """Reset internal state."""
        self._distracted_frames = 0
        self._is_distracted = False
