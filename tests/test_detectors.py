"""
tests/test_detectors.py
========================
Unit tests for the EAR and MAR detector geometry helpers.

Run with:
    python -m pytest tests/ -v
"""

import pytest
import numpy as np

from src.detectors.eye_detector import EyeDetector
from src.detectors.mouth_detector import MouthDetector


# ---------------------------------------------------------------------------
# EyeDetector — compute_ear
# ---------------------------------------------------------------------------

class TestEyeAspectRatio:

    def test_open_eye_returns_high_ear(self):
        """A wide-open eye should produce an EAR well above the typical threshold."""
        coords = [
            (0.0, 0.0),    # p1 left
            (1.0, -2.0),   # p2 top-left
            (3.0, -2.0),   # p3 top-right
            (4.0, 0.0),    # p4 right
            (3.0, 2.0),    # p5 bottom-right
            (1.0, 2.0),    # p6 bottom-left
        ]
        ear = EyeDetector.compute_ear(coords)
        assert ear > 0.25, f"Expected EAR > 0.25 for open eye, got {ear:.4f}"

    def test_closed_eye_returns_low_ear(self):
        """Nearly-closed eye (top and bottom landmarks almost touching)."""
        coords = [
            (0.0, 0.0),
            (1.0, -0.05),
            (3.0, -0.05),
            (4.0, 0.0),
            (3.0, 0.05),
            (1.0, 0.05),
        ]
        ear = EyeDetector.compute_ear(coords)
        assert ear < 0.25, f"Expected EAR < 0.25 for closed eye, got {ear:.4f}"

    def test_zero_horizontal_distance_returns_zero(self):
        """If all points overlap horizontally, EAR must return 0 (no div-by-zero)."""
        coords = [(0.0, 0.0)] * 6
        ear = EyeDetector.compute_ear(coords)
        assert ear == 0.0

    def test_ear_non_negative(self):
        """EAR should never be negative."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            pts = [(float(x), float(y)) for x, y in rng.uniform(-5, 5, (6, 2))]
            ear = EyeDetector.compute_ear(pts)
            assert ear >= 0.0


# ---------------------------------------------------------------------------
# EyeDetector — drowsiness state machine
# ---------------------------------------------------------------------------

class TestEyeDetectorStateMachine:

    def _make_closed_landmarks(self, frame_h=100, frame_w=100):
        """Return a mock face_landmarks object with closed-eye coords."""
        class FakeLM:
            def __init__(self, x, y):
                self.x = x / frame_w
                self.y = y / frame_h

        class FakeLandmarks:
            def __init__(self):
                self.landmark = {i: FakeLM(50, 50) for i in range(478)}

        return FakeLandmarks()

    def test_not_drowsy_below_consecutive_threshold(self):
        det = EyeDetector(ear_threshold=0.99, consecutive_frames=10)
        lm = self._make_closed_landmarks()
        result = None
        for _ in range(5):
            result = det.process(lm, (100, 100, 3))
        assert not result["is_drowsy"]

    def test_drowsy_after_consecutive_threshold(self):
        det = EyeDetector(ear_threshold=0.99, consecutive_frames=5)
        lm = self._make_closed_landmarks()
        result = None
        for _ in range(6):
            result = det.process(lm, (100, 100, 3))
        assert result["is_drowsy"]

    def test_reset_clears_state(self):
        det = EyeDetector(ear_threshold=0.99, consecutive_frames=3)
        lm = self._make_closed_landmarks()
        for _ in range(10):
            det.process(lm, (100, 100, 3))
        det.reset()
        assert not det._is_drowsy
        assert det._closed_frames == 0


# ---------------------------------------------------------------------------
# MouthDetector — compute_mar
# ---------------------------------------------------------------------------

class TestMouthAspectRatio:

    def test_open_mouth_high_mar(self):
        mar = MouthDetector.compute_mar(
            left=(0.0, 0.0),
            right=(10.0, 0.0),
            top=(5.0, -4.0),
            bottom=(5.0, 4.0),
        )
        assert mar > 0.6, f"Expected MAR > 0.6 for open mouth, got {mar:.4f}"

    def test_closed_mouth_low_mar(self):
        mar = MouthDetector.compute_mar(
            left=(0.0, 0.0),
            right=(10.0, 0.0),
            top=(5.0, -0.1),
            bottom=(5.0, 0.1),
        )
        assert mar < 0.1, f"Expected MAR < 0.1 for closed mouth, got {mar:.4f}"

    def test_zero_horizontal_returns_zero(self):
        mar = MouthDetector.compute_mar(
            left=(5.0, 0.0),
            right=(5.0, 0.0),
            top=(5.0, -2.0),
            bottom=(5.0, 2.0),
        )
        assert mar == 0.0
