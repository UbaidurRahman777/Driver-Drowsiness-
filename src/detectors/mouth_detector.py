"""
src/detectors/mouth_detector.py
================================
Mouth Aspect Ratio (MAR) based yawn detection using
MediaPipe FaceMesh landmarks.

The MAR mirrors the EAR formula applied to the inner lip
landmarks — a high MAR indicates a wide-open mouth (yawn).
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe inner-lip landmark indices used for MAR
# left=78, right=308, top=13, bottom=14

MOUTH_LEFT: int = 78
MOUTH_RIGHT: int = 308
MOUTH_TOP: int = 13
MOUTH_BOTTOM: int = 14

# Outer hull indices for visualisation
MOUTH_HULL_IDX: List[int] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]


class MouthDetector:
    """
    Detects yawning via Mouth Aspect Ratio (MAR).

    A sustained rise above ``mar_threshold`` for at least
    ``consecutive_frames`` frames triggers the yawning flag.

    Args:
        mar_threshold: MAR value above which the mouth is considered open.
        consecutive_frames: Minimum open frames before alert fires.
    """

    def __init__(
        self,
        mar_threshold: float = 0.60,
        consecutive_frames: int = 20,
    ) -> None:
        self.mar_threshold = mar_threshold
        self.consecutive_frames = consecutive_frames

        self._yawn_frames: int = 0
        self._is_yawning: bool = False

    # Static geometry helpers

    @staticmethod
    def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def compute_mar(
        left: Tuple[float, float],
        right: Tuple[float, float],
        top: Tuple[float, float],
        bottom: Tuple[float, float],
    ) -> float:
        """
        Compute the Mouth Aspect Ratio.

        Args:
            left:   Left mouth corner (x, y).
            right:  Right mouth corner (x, y).
            top:    Inner upper lip centre (x, y).
            bottom: Inner lower lip centre (x, y).

        Returns:
            MAR scalar; 0.0 if horizontal distance is zero.
        """
        vertical = MouthDetector._euclidean(top, bottom)
        horizontal = MouthDetector._euclidean(left, right)
        return vertical / horizontal if horizontal > 0 else 0.0

    # Main processing

    def process(
        self,
        face_landmarks: Any,
        frame_shape: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """
        Process one frame's face landmarks and update yawning state.

        Args:
            face_landmarks: A single MediaPipe NormalizedLandmarkList.
            frame_shape: (H, W, C) shape of the current frame.

        Returns:
            Dictionary containing MAR value, pixel coords, and flags.
        """
        h, w = frame_shape[:2]

        def px(idx: int) -> Tuple[float, float]:
            lm = face_landmarks.landmark[idx]
            return lm.x * w, lm.y * h

        left = px(MOUTH_LEFT)
        right = px(MOUTH_RIGHT)
        top = px(MOUTH_TOP)
        bottom = px(MOUTH_BOTTOM)
        hull_coords = [px(i) for i in MOUTH_HULL_IDX]

        mar = self.compute_mar(left, right, top, bottom)

        # Debounced counter
        if mar > self.mar_threshold:
            self._yawn_frames += 1
            if self._yawn_frames >= self.consecutive_frames:
                self._is_yawning = True
        else:
            self._yawn_frames = max(0, self._yawn_frames - 1)
            if self._yawn_frames == 0:
                self._is_yawning = False

        return {
            "mar": mar,
            "yawn_frames": self._yawn_frames,
            "is_yawning": self._is_yawning,
            "mouth_corners": (left, right, top, bottom),
            "hull_coords": hull_coords,
        }

    def reset(self) -> None:
        """Reset internal state."""
        self._yawn_frames = 0
        self._is_yawning = False
