from __future__ import annotations

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, path: str, fps: float, frame_size: tuple[int, int]) -> None:
        w, h = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
