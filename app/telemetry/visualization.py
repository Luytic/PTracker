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
    search_roi: tuple[int, int, int, int] | None = None,
    flow_roi: tuple[int, int, int, int] | None = None,
    flow_dx: float = 0.0,
    flow_dy: float = 0.0,
    flow_valid: bool = False,
    flow_n_points: int = 0,
    flow_feature_tracks: tuple[tuple[float, float, float, float], ...] = (),
    flow_rejected_tracks: tuple[tuple[float, float, float, float], ...] = (),
    confidence: float = 0.0,
    method_used: str = "",
    object_scale_ratio: float = 1.0,
    bbox_scale_step: float = 1.0,
    bbox_scale_updated: bool = False,
    latency_total_ms: float = 0.0,
    latency_cv_ms: float = 0.0,
    latency_nn_ms: float = 0.0,
    latency_kf_ms: float = 0.0,
) -> np.ndarray:
    """User overlay: bbox + tip marker. Debug adds search ROI, timings, labels."""
    out = frame.copy()

    if debug and search_roi is not None:
        sx1, sy1, sx2, sy2 = search_roi
        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (0, 140, 255), 1, cv2.LINE_AA)

    if debug and flow_roi is not None:
        fx1, fy1, fx2, fy2 = flow_roi
        color = (80, 220, 80) if flow_valid else (80, 80, 220)
        cv2.rectangle(out, (fx1, fy1), (fx2, fy2), color, 1, cv2.LINE_AA)
        if flow_valid and point is not None:
            px, py = int(point[0]), int(point[1])
            tx, ty = int(round(px + flow_dx)), int(round(py + flow_dy))
            cv2.arrowedLine(out, (px, py), (tx, ty), (100, 255, 100), 1, tipLength=0.35)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if tracking_state == "visible":
            color = (0, 255, 0)
        elif tracking_state == "uncertain":
            color = (0, 190, 255)
        else:
            color = (0, 80, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    if point is not None:
        x, y = int(point[0]), int(point[1])
        cv2.drawMarker(out, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)

    if debug:
        cv2.putText(
            out,
            f"mode={tracking_state} conf={confidence:.2f} method={method_used}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"lat={latency_total_ms:.1f}ms cv={latency_cv_ms:.1f} "
            f"nn={latency_nn_ms:.1f} kf={latency_kf_ms:.1f}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (220, 220, 120),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"scale={object_scale_ratio:.2f}x bbox"
            + (f" d={bbox_scale_step:.3f}" if bbox_scale_updated else ""),
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        flow_label = (
            f"flow=({flow_dx:+.1f},{flow_dy:+.1f}) n={flow_n_points}"
            if flow_valid
            else "flow=off"
        )
        cv2.putText(
            out,
            flow_label,
            (12, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (140, 255, 180) if flow_valid else (140, 140, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"flow feat: used={len(flow_feature_tracks)} "
            f"rej={len(flow_rejected_tracks)}",
            (12, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 220, 180),
            1,
            cv2.LINE_AA,
        )
    return out


def _draw_feature_track(
    out: np.ndarray,
    track: tuple[float, float, float, float],
    color: tuple[int, int, int],
    *,
    dot_only: bool = False,
) -> None:
    x1, y1, x2, y2 = track
    ix1, iy1 = int(round(x1)), int(round(y1))
    ix2, iy2 = int(round(x2)), int(round(y2))
    if dot_only or abs(ix2 - ix1) + abs(iy2 - iy1) < 1:
        cv2.circle(out, (ix1, iy1), 3, color, -1, lineType=cv2.LINE_AA)
    else:
        cv2.line(out, (ix1, iy1), (ix2, iy2), color, 2, cv2.LINE_AA)
        cv2.circle(out, (ix2, iy2), 3, color, -1, lineType=cv2.LINE_AA)


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
