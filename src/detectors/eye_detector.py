"""
src/detectors/eye_detector.py
==============================
Eye Aspect Ratio (EAR) based drowsiness detection using
MediaPipe FaceMesh 478-landmark model.

Reference:
    Soukupová & Čech, "Real-Time Eye Blink Detection using
    Facial Landmarks", CVWW 2016.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices — eyes
# Each list follows the EAR convention: [p1, p2, p3, p4, p5, p6]
# where p1/p4 are the horizontal corners and p2,p3,p5,p6 are vertical pairs.
# ---------------------------------------------------------------------------
LEFT_EYE_IDX: List[int] = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX: List[int] = [33, 160, 158, 133, 153, 144]


class EyeDetector:
    """
    Detects driver drowsiness via Eye Aspect Ratio (EAR).

    A sustained drop below ``ear_threshold`` for at least
    ``consecutive_frames`` frames triggers the drowsy flag.
    A decay counter prevents false resets from single open-eye frames.

    Args:
        ear_threshold: EAR value below which an eye is considered closed.
        consecutive_frames: Minimum closed frames before alert fires.
    """

    def __init__(
        self,
        ear_threshold: float = 0.25,
        consecutive_frames: int = 15,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames

        self._closed_frames: int = 0
        self._is_drowsy: bool = False

    # ------------------------------------------------------------------
    # Static geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def compute_ear(coords: List[Tuple[float, float]]) -> float:
        """
        Compute the Eye Aspect Ratio for one eye.

        Args:
            coords: Six (x, y) landmark points in EAR order.

        Returns:
            EAR scalar (float); 0.0 if the horizontal distance is zero.
        """
        A = EyeDetector._euclidean(coords[1], coords[5])
        B = EyeDetector._euclidean(coords[2], coords[4])
        C = EyeDetector._euclidean(coords[0], coords[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def process(
        self,
        face_landmarks: Any,
        frame_shape: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """
        Process one frame's face landmarks and update drowsiness state.

        Args:
            face_landmarks: A single MediaPipe NormalizedLandmarkList.
            frame_shape: (H, W, C) shape of the current frame.

        Returns:
            Dictionary containing EAR values, pixel coords, and flags.
        """
        h, w = frame_shape[:2]

        def px(idx: int) -> Tuple[float, float]:
            lm = face_landmarks.landmark[idx]
            return lm.x * w, lm.y * h

        left_coords = [px(i) for i in LEFT_EYE_IDX]
        right_coords = [px(i) for i in RIGHT_EYE_IDX]

        left_ear = self.compute_ear(left_coords)
        right_ear = self.compute_ear(right_coords)
        avg_ear = (left_ear + right_ear) / 2.0

        # Debounced counter
        if avg_ear < self.ear_threshold:
            self._closed_frames += 1
            if self._closed_frames >= self.consecutive_frames:
                self._is_drowsy = True
        else:
            self._closed_frames = max(0, self._closed_frames - 1)
            if self._closed_frames == 0:
                self._is_drowsy = False

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "closed_frames": self._closed_frames,
            "is_drowsy": self._is_drowsy,
            "left_eye_coords": left_coords,
            "right_eye_coords": right_coords,
        }

    def reset(self) -> None:
        """Reset internal state (call between sessions)."""
        self._closed_frames = 0
        self._is_drowsy = False
