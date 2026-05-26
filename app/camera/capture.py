from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    device_id: int = 0
    warmup_frames: int = 4


class Webcam:
    """OpenCV webcam wrapper (default driver resolution)."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0

    @property
    def fps(self) -> float:
        return self._fps

    def open(self) -> None:
        if sys.platform == "win32":
            cap = cv2.VideoCapture(self._config.device_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self._config.device_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._config.device_id}")
        for _ in range(max(0, self._config.warmup_frames)):
            cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if 1.0 < fps <= 240.0 else 30.0
        self._cap = cap

    def setup_window(self, window_name: str) -> None:
        from app.camera.window import setup_window
        if self._cap is None:
            raise RuntimeError("Webcam not open")
        setup_window(self._cap, window_name)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
