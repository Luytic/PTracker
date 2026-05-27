from __future__ import annotations

import numpy as np


def bbox_from_center(
    frame: np.ndarray,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    fh, fw = frame.shape[:2]
    x1 = int(round(max(0.0, cx - w * 0.5)))
    y1 = int(round(max(0.0, cy - h * 0.5)))
    x2 = int(min(fw - 1, x1 + w))
    y2 = int(min(fh - 1, y1 + h))
    return x1, y1, x2, y2


def clamp_bbox_xyxy(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = int(frame_shape[0]), int(frame_shape[1])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    return x1, y1, x2, y2


def roi_xywh_to_xyxy(roi_xywh: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = roi_xywh
    return x, y, x + w, y + h
