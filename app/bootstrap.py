from __future__ import annotations

import threading

import cv2

from app.camera import Webcam, WINDOW_NAME
from tracker.tracking import TrackingPipeline, create_tracking_pipeline


def preload_pipeline(
    webcam: Webcam,
    *,
    fps: float,
    pentiptrack_version: str,
    nn_interval: int,
    window_name: str = WINDOW_NAME,
) -> TrackingPipeline:
    """Load PyTorch + PenTipTrack while showing live camera preview."""
    webcam.setup_window(window_name)
    load_state: dict[str, object] = {"pipeline": None, "error": None}

    def _load() -> None:
        try:
            load_state["pipeline"] = create_tracking_pipeline(
                fps=fps,
                pentiptrack_version=pentiptrack_version,
                nn_interval=nn_interval,
            )
        except BaseException as exc:
            load_state["error"] = exc

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()

    while thread.is_alive():
        ok, frame = webcam.read()
        if not ok or frame is None:
            raise RuntimeError("Cannot read frame while loading model")
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            raise RuntimeError("Model load cancelled by user")
        thread.join(timeout=0.02)

    thread.join()
    err = load_state["error"]
    if err is not None:
        raise RuntimeError(f"Failed to load PenTipTrack: {err}") from err
    pipeline = load_state["pipeline"]
    assert isinstance(pipeline, TrackingPipeline)
    return pipeline
