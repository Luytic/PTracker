from __future__ import annotations
import cv2
from app.camera import CameraConfig, Webcam, WINDOW_NAME, destroy_window
from app.cli import parse_args

def main(argv=None) -> None:
    config = parse_args(argv)
    cam = Webcam(CameraConfig(device_id=config.camera))
    cam.open()
    cam.setup_window(WINDOW_NAME)
    print("Camera preview — q/ESC to quit")
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                break
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    finally:
        cam.release()
        destroy_window()

if __name__ == "__main__":
    main()
