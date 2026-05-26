from __future__ import annotations

import cv2
import numpy as np

from app.camera import Webcam

QUICK_ROI_MIN_SIDE_FRACTION = 0.05
_CLICK_MAX_MOVE_PX = 5
_MIN_ROI_PX = 4


def quick_roi_side_px(frame_shape: tuple[int, ...]) -> int:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    return max(_MIN_ROI_PX, int(round(min(w, h) * QUICK_ROI_MIN_SIDE_FRACTION)))


def roi_xywh_at_point(
    frame_shape: tuple[int, ...],
    cx: int,
    cy: int,
    *,
    side_px: int | None = None,
) -> tuple[int, int, int, int]:
    """Square ROI centered on (cx, cy), clamped to frame bounds."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    size = side_px if side_px is not None else quick_roi_side_px(frame_shape)
    size = max(_MIN_ROI_PX, min(size, w, h))
    half = size // 2
    x = int(np.clip(cx - half, 0, max(0, w - size)))
    y = int(np.clip(cy - half, 0, max(0, h - size)))
    return x, y, size, size


class RoiSelector:
    """Interactive ROI: drag rectangle or single-click quick box (5% of min frame side)."""

    def __init__(self, window_name: str) -> None:
        self._window_name = window_name

    def select(
        self,
        webcam: Webcam,
        *,
        setup: bool = False,
        prompt: str = (
            "Drag LMB ROI | click LMB = 5% quick box | q/ESC quit"
        ),
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        if setup:
            webcam.setup_window(self._window_name)

        state: dict[str, object] = {
            "dragging": False,
            "start": (0, 0),
            "current": (0, 0),
            "done": False,
            "roi": None,
            "frame_shape": None,
        }

        def _mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            _ = flags, param
            shape = state["frame_shape"]
            if shape is None:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                state["dragging"] = True
                state["start"] = (x, y)
                state["current"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and bool(state["dragging"]):
                state["current"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and bool(state["dragging"]):
                state["dragging"] = False
                state["current"] = (x, y)
                x0, y0 = state["start"]  # type: ignore[assignment]
                x1, y1 = state["current"]  # type: ignore[assignment]
                if max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0))) <= _CLICK_MAX_MOVE_PX:
                    state["roi"] = roi_xywh_at_point(shape, int(x1), int(y1))
                    state["done"] = True
                    return
                rx, ry = min(int(x0), int(x1)), min(int(y0), int(y1))
                rw, rh = abs(int(x1) - int(x0)), abs(int(y1) - int(y0))
                if rw >= _MIN_ROI_PX and rh >= _MIN_ROI_PX:
                    state["roi"] = (rx, ry, rw, rh)
                    state["done"] = True

        cv2.setMouseCallback(self._window_name, _mouse)
        selected_frame = None
        try:
            while True:
                ok, frame = webcam.read()
                if not ok or frame is None:
                    raise RuntimeError("Cannot read frame during ROI selection")
                selected_frame = frame
                state["frame_shape"] = frame.shape
                vis = frame.copy()
                if bool(state["dragging"]):
                    x0, y0 = state["start"]  # type: ignore[assignment]
                    x1, y1 = state["current"]  # type: ignore[assignment]
                    moved = max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0)))
                    if moved <= _CLICK_MAX_MOVE_PX:
                        qx, qy, qw, qh = roi_xywh_at_point(frame.shape, int(x1), int(y1))
                        cv2.rectangle(
                            vis,
                            (qx, qy),
                            (qx + qw, qy + qh),
                            (0, 255, 120),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.rectangle(
                            vis,
                            (int(x0), int(y0)),
                            (int(x1), int(y1)),
                            (0, 220, 255),
                            2,
                            cv2.LINE_AA,
                        )
                cv2.putText(
                    vis,
                    prompt,
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )
                side = quick_roi_side_px(frame.shape)
                cv2.putText(
                    vis,
                    f"quick box: {side}x{side}px ({QUICK_ROI_MIN_SIDE_FRACTION * 100:.0f}% min side)",
                    (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 220, 180),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(self._window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return None
                if bool(state["done"]):
                    roi = state["roi"]  # type: ignore[assignment]
                    assert selected_frame is not None
                    return selected_frame, roi  # type: ignore[return-value]
        finally:
            cv2.setMouseCallback(self._window_name, lambda *_args: None)
