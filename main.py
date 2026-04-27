"""
main.py
=======
Entry point for the Driver Monitoring System.

Run with defaults:
    python main.py

Override camera source:
    python main.py --source 1

Disable audio alerts:
    python main.py --no-audio

Show all options:
    python main.py --help
"""

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path
from config.settings import AppConfig, CameraConfig, AlertConfig
from src.monitoring.driver_monitor import DriverMonitor

def _configure_logging(log_dir: str, level: str, to_file: bool) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]

    if to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            Path(log_dir) / "session.log",
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        handlers.append(fh)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="driver-monitor",
        description="Real-time Driver Monitoring System — drowsiness, yawn & head-pose detection",
    )
    parser.add_argument(
        "--source", type=int, default=0,
        metavar="INT",
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Capture width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Capture height in pixels (default: 720)",
    )
    parser.add_argument(
        "--ear-thresh", type=float, default=0.25,
        metavar="FLOAT",
        help="EAR threshold for eye closure (default: 0.25)",
    )
    parser.add_argument(
        "--mar-thresh", type=float, default=0.60,
        metavar="FLOAT",
        help="MAR threshold for yawning (default: 0.60)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio beep alerts",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    # Build config, overriding defaults with CLI flags
    config = AppConfig()
    config.camera = CameraConfig(
        source=args.source,
        width=args.width,
        height=args.height,
    )
    config.thresholds.ear_threshold = args.ear_thresh
    config.thresholds.mar_threshold = args.mar_thresh
    config.alerts = AlertConfig(enable_audio=not args.no_audio)
    config.logging.log_level = args.log_level

    _configure_logging(
        config.logging.log_dir,
        config.logging.log_level,
        config.logging.log_to_file,
    )

    logger = logging.getLogger(__name__)
    logger.info("Driver Monitoring System v%s — press 'q' to quit", config.version)

    monitor = DriverMonitor(config)
    monitor.run()


if __name__ == "__main__":
    main()
