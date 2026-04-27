"""
src/utils/camera.py
===================
Camera abstraction — wraps cv2.VideoCapture with clean
initialisation, frame reading, and resource release.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages a single camera capture session.

    Usage (context manager)::

        with CameraManager(source=0, width=1280, height=720) as cam:
            ret, frame = cam.read_frame()
    """

    def __init__(
        self,
        source: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count: int = 0

    def initialize(self) -> bool:
        """Open the camera and configure resolution / FPS."""
        logger.info("Initialising camera (source=%s)", self.source)
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error("Failed to open camera source: %s", self.source)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Flush a few frames so the sensor stabilises
        for _ in range(5):
            self._cap.read()

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera ready: %dx%d @ %.1f fps", actual_w, actual_h, actual_fps
        )
        return True

    def release(self) -> None:
        """Release the capture handle."""
        if self._cap is not None:
            self._cap.release()
            logger.info("Camera released after %d frames", self._frame_count)

    # Frame I/O

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read one frame from the camera.

        Returns:
            (success, frame) — frame is None when success is False.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    # Properties

    @property
    def frame_count(self) -> int:
        """Total frames successfully read in this session."""
        return self._frame_count

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # Context manager support

    def __enter__(self) -> "CameraManager":
        self.initialize()
        return self

    def __exit__(self, *_) -> None:
        self.release()
