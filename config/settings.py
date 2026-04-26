"""
config/settings.py
==================
Centralised configuration for the Driver Monitoring System.
All thresholds, camera settings, and display options live here.
"""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """Camera capture settings."""
    source: int = 0          # 0 = default webcam
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class DetectionThresholds:
    """
    Thresholds that govern alert triggering.

    EAR (Eye Aspect Ratio):  lower value = more closed.
    MAR (Mouth Aspect Ratio): higher value = more open.
    Head pose angles are in degrees.
    Consecutive-frame counters debounce instantaneous noise.
    """
    # -- Eye / Drowsiness --
    ear_threshold: float = 0.25
    ear_consecutive_frames: int = 15        # ~0.5 s at 30 fps

    # -- Mouth / Yawning --
    mar_threshold: float = 0.60
    mar_consecutive_frames: int = 20        # ~0.67 s at 30 fps

    # -- Head Pose --
    pitch_threshold: float = 20.0           # nodding down / up
    yaw_threshold: float = 30.0             # turning left / right
    roll_threshold: float = 20.0            # tilting sideways
    head_pose_consecutive_frames: int = 15


@dataclass
class AlertConfig:
    """Audio and visual alert settings."""
    enable_audio: bool = True
    enable_visual: bool = True
    cooldown_seconds: float = 2.0           # min time between same-type alerts

    # Beep frequencies (Hz) — different tone per event type
    drowsiness_freq: int = 880
    yawn_freq: int = 660
    head_pose_freq: int = 440
    alert_duration_ms: int = 500


@dataclass
class DisplayConfig:
    """HUD / overlay display settings."""
    show_landmarks: bool = True
    show_face_mesh: bool = False
    show_head_axes: bool = True
    show_stats_panel: bool = True
    window_title: str = "Driver Monitoring System"


@dataclass
class LoggingConfig:
    """Logging behaviour."""
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_file: bool = True


@dataclass
class AppConfig:
    """Root application configuration — compose all sub-configs here."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    app_name: str = "Driver Monitoring System"
    version: str = "2.0.0"
