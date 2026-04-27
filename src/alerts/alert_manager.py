"""
src/alerts/alert_manager.py
============================
Manages audio and visual alerts with per-type cooldown timers
so the driver is not bombarded with repeated beeps.

Audio uses pygame.mixer with a programmatically generated sine-wave
tone — no external sound files required.
"""

import logging
import threading
import time
from enum import Enum, auto
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AlertType(Enum):
    DROWSINESS = auto()
    YAWNING = auto()
    HEAD_POSE = auto()


# Tone frequency (Hz) per alert type
_FREQUENCIES: Dict[AlertType, int] = {
    AlertType.DROWSINESS: 880,
    AlertType.YAWNING: 660,
    AlertType.HEAD_POSE: 440,
}


class AlertManager:
    """
    Fires audio beeps and tracks visual alert state.

    Each alert type has an independent cooldown so that a drowsiness
    alert does not suppress a head-pose alert.

    Args:
        enable_audio: Whether to play beep tones.
        cooldown_seconds: Minimum gap between alerts of the same type.
        alert_duration_ms: Length of each beep in milliseconds.
    """

    def __init__(
        self,
        enable_audio: bool = True,
        cooldown_seconds: float = 2.0,
        alert_duration_ms: int = 500,
    ) -> None:
        self.enable_audio = enable_audio
        self.cooldown_seconds = cooldown_seconds
        self.alert_duration_ms = alert_duration_ms

        self._last_alert: Dict[AlertType, float] = {}
        self._audio_ok: bool = False

        if enable_audio:
            self._init_audio()

    # Audio initialisation

    def _init_audio(self) -> None:
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._audio_ok = True
            logger.info("Audio alerts initialised (pygame mixer)")
        except ImportError:
            logger.warning("pygame not installed — audio alerts disabled")
        except Exception as exc:
            logger.warning("Audio init failed: %s", exc)

    # Internal beep generator

    def _play_beep(self, frequency: int) -> None:
        """Generate and play a pure-tone beep in a background thread."""
        if not self._audio_ok:
            return

        def _beep() -> None:
            try:
                import pygame
                import numpy as np

                sample_rate = 44100
                n = int(sample_rate * self.alert_duration_ms / 1000)
                t = np.linspace(0, self.alert_duration_ms / 1000, n, endpoint=False)
                wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(wave)
                sound.play()
                pygame.time.wait(self.alert_duration_ms)
            except Exception as exc:
                logger.debug("Beep playback error: %s", exc)

        threading.Thread(target=_beep, daemon=True).start()

    # Public API
    
    def trigger(self, alert_type: AlertType) -> bool:
        """
        Attempt to fire an alert of the given type.

        Respects the cooldown period — returns True if the alert actually
        fired, False if suppressed by the cooldown.

        Args:
            alert_type: Which kind of alert to fire.

        Returns:
            True if the alert fired; False if on cooldown.
        """
        now = time.monotonic()
        last = self._last_alert.get(alert_type, 0.0)

        if now - last < self.cooldown_seconds:
            return False  # Still in cooldown

        self._last_alert[alert_type] = now
        freq = _FREQUENCIES.get(alert_type, 440)

        logger.info("Alert triggered: %s", alert_type.name)
        self._play_beep(freq)
        return True

    def is_on_cooldown(self, alert_type: AlertType) -> bool:
        """Return True if this alert type is still in its cooldown window."""
        elapsed = time.monotonic() - self._last_alert.get(alert_type, 0.0)
        return elapsed < self.cooldown_seconds

    def reset(self) -> None:
        """Clear all cooldown timers."""
        self._last_alert.clear()
