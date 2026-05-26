from __future__ import annotations

import cv2

WINDOW_NAME = "Pen Tracker"


def setup_window(cap: cv2.VideoCapture, window_name: str = WINDOW_NAME) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fw > 0 and fh > 0:
            cv2.resizeWindow(window_name, fw, fh)
    except Exception:
        pass


def destroy_window(window_name: str = WINDOW_NAME) -> None:
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
