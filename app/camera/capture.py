from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    device_id: int = 0
    width: int = 0
    height: int = 0
    warmup_frames: int = 6


def _configure_resolution(cap: cv2.VideoCapture, width: int, height: int) -> None:
    """width/height <= 0: both unset → request camera maximum; one set → that axis only."""
    if width <= 0 and height <= 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000.0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000.0)
        return
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))


class Webcam:
    """Camera capture with platform-specific backend and nominal FPS."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        if sys.platform == "win32":
            cap = cv2.VideoCapture(self._config.device_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self._config.device_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._config.device_id}")

        _configure_resolution(cap, self._config.width, self._config.height)

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
        ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> Webcam:
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
