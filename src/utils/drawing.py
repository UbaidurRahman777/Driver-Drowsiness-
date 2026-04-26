"""
src/utils/drawing.py
=====================
All OpenCV overlay / HUD drawing helpers.

Keeping drawing logic in one place makes it easy to restyle the
HUD without touching detector or monitoring logic.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette  (BGR)
# ---------------------------------------------------------------------------
CLR_GREEN = (0, 220, 100)
CLR_RED = (0, 60, 220)
CLR_AMBER = (0, 165, 255)
CLR_WHITE = (255, 255, 255)
CLR_BLACK = (0, 0, 0)
CLR_CYAN = (255, 220, 0)
CLR_PANEL_BG = (20, 20, 20)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = 0.45
FONT_MD = 0.60
FONT_LG = 0.80


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _text_with_shadow(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float = FONT_MD,
    colour: Tuple[int, int, int] = CLR_WHITE,
    thickness: int = 1,
) -> None:
    """Draw text with a dark shadow for readability on any background."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1), FONT, scale, CLR_BLACK, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, colour, thickness, cv2.LINE_AA)


def _draw_eye_hull(
    frame: np.ndarray,
    coords: List[Tuple[float, float]],
    colour: Tuple[int, int, int],
) -> None:
    pts = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
    hull = cv2.convexHull(pts)
    cv2.drawContours(frame, [hull], -1, colour, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Stats panel
# ---------------------------------------------------------------------------

def draw_stats_panel(
    frame: np.ndarray,
    eye_result: Dict[str, Any],
    mouth_result: Dict[str, Any],
    pose_result: Dict[str, Any],
) -> None:
    """
    Render a semi-transparent stats panel in the top-left corner.

    Displays EAR, MAR and head-pose angles with colour-coded values.
    """
    panel_w, panel_h = 280, 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), CLR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (8, 8), (8 + panel_w, 8 + panel_h), CLR_CYAN, 1, cv2.LINE_AA)

    ear = eye_result.get("avg_ear", 0.0)
    mar = mouth_result.get("mar", 0.0)
    pitch = pose_result.get("pitch", 0.0)
    yaw = pose_result.get("yaw", 0.0)
    roll = pose_result.get("roll", 0.0)

    ear_clr = CLR_RED if eye_result.get("is_drowsy") else CLR_GREEN
    mar_clr = CLR_RED if mouth_result.get("is_yawning") else CLR_GREEN
    pose_clr = CLR_RED if pose_result.get("is_distracted") else CLR_GREEN

    lines = [
        (f"EAR: {ear:.3f}", ear_clr),
        (f"MAR: {mar:.3f}", mar_clr),
        (f"Pitch: {pitch:+.1f}°  Yaw: {yaw:+.1f}°  Roll: {roll:+.1f}°", pose_clr),
    ]

    for i, (txt, clr) in enumerate(lines):
        _text_with_shadow(frame, txt, (16, 34 + i * 32), FONT_SM, clr)


# ---------------------------------------------------------------------------
# Alert banners
# ---------------------------------------------------------------------------

def draw_alert_banners(
    frame: np.ndarray,
    eye_result: Dict[str, Any],
    mouth_result: Dict[str, Any],
    pose_result: Dict[str, Any],
) -> None:
    """
    Draw bold alert banners along the top of the frame when an event is active.
    """
    h, w = frame.shape[:2]
    banners = []

    if eye_result.get("is_drowsy"):
        banners.append(("⚠  DROWSINESS DETECTED", CLR_RED))
    if mouth_result.get("is_yawning"):
        banners.append(("⚠  YAWNING DETECTED", CLR_AMBER))
    if pose_result.get("is_distracted"):
        banners.append(("⚠  HEAD POSE ALERT", CLR_AMBER))

    for idx, (msg, clr) in enumerate(banners):
        y_pos = h - 30 - idx * 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_pos - 24), (w, y_pos + 8), CLR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        (tw, _), _ = cv2.getTextSize(msg, FONT, FONT_LG, 2)
        _text_with_shadow(frame, msg, ((w - tw) // 2, y_pos), FONT_LG, clr, 2)


# ---------------------------------------------------------------------------
# Landmark overlays
# ---------------------------------------------------------------------------

def draw_eye_landmarks(
    frame: np.ndarray,
    eye_result: Dict[str, Any],
) -> None:
    """Draw convex hull outlines around each eye."""
    colour = CLR_RED if eye_result.get("is_drowsy") else CLR_GREEN
    for key in ("left_eye_coords", "right_eye_coords"):
        coords = eye_result.get(key)
        if coords:
            _draw_eye_hull(frame, coords, colour)


def draw_mouth_landmarks(
    frame: np.ndarray,
    mouth_result: Dict[str, Any],
) -> None:
    """Draw convex hull around the mouth region."""
    colour = CLR_RED if mouth_result.get("is_yawning") else CLR_GREEN
    hull_coords = mouth_result.get("hull_coords")
    if hull_coords:
        _draw_eye_hull(frame, hull_coords, colour)


def draw_head_pose_axis(
    frame: np.ndarray,
    pose_result: Dict[str, Any],
) -> None:
    """Draw a nose-direction arrow to visualise head pose."""
    if not pose_result.get("success"):
        return

    image_pts = pose_result.get("image_points")
    nose_end = pose_result.get("nose_end_2d")

    if image_pts is None or nose_end is None:
        return

    p1 = (int(image_pts[0][0]), int(image_pts[0][1]))
    p2 = (int(nose_end[0][0][0]), int(nose_end[0][0][1]))

    colour = CLR_RED if pose_result.get("is_distracted") else CLR_CYAN
    cv2.arrowedLine(frame, p1, p2, colour, 2, cv2.LINE_AA, tipLength=0.3)


# ---------------------------------------------------------------------------
# Face-count badge
# ---------------------------------------------------------------------------

def draw_face_count(frame: np.ndarray, count: int) -> None:
    """Small badge showing how many faces are detected."""
    label = f"Faces: {count}"
    _text_with_shadow(frame, label, (frame.shape[1] - 130, 28), FONT_SM, CLR_CYAN)
