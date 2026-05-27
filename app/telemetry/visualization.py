from __future__ import annotations

import cv2
import numpy as np


def draw_tracking_overlay(
    frame: np.ndarray,
    *,
    bbox: tuple[int, int, int, int] | None,
    point: tuple[float, float] | None,
    tracking_state: str,
    debug: bool = False,
    **kwargs: object,
) -> np.ndarray:
    """User overlay: bbox + tip marker (debug HUD added later)."""
    out = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0) if tracking_state == "visible" else (0, 190, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    if point is not None:
        x, y = int(point[0]), int(point[1])
        cv2.drawMarker(out, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
    return out


def draw_depth_trajectory(
    frame: np.ndarray, points_xyz: list[tuple[float, float, float]]
) -> np.ndarray:
    if len(points_xyz) < 2:
        return frame
    out = frame
    for i in range(1, len(points_xyz)):
        x1, y1, s1 = points_xyz[i - 1]
        x2, y2, s2 = points_xyz[i]
        scale = float((s1 + s2) * 0.5)
        # 2× sensitivity vs z_relative (0.5 + 0.5·(scale−1)): full scale−1 drives color/thickness.
        z = float(np.clip(0.5 + (scale - 1.0), 0.0, 1.0))
        b = int(np.clip((1.0 - z) * 255, 20, 255))
        g = int(np.clip(80 + z * 140, 40, 255))
        r = int(np.clip(z * 255, 20, 255))
        thickness = int(np.clip(1 + z * 5, 1, 6))
        cv2.line(
            out,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (b, g, r),
            thickness,
            cv2.LINE_AA,
        )
    return out
