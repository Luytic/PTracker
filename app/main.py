from __future__ import annotations

from app.bootstrap import preload_pipeline
from app.camera import CameraConfig, Webcam
from app.cli import parse_args
from app.config import AppConfig
from app.session import WebcamTrackingSession


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    webcam = Webcam(
        CameraConfig(
            device_id=config.camera,
            width=config.width,
            height=config.height,
        )
    )
    webcam.open()
    w, h = webcam.frame_size
    res_note = "max" if config.width <= 0 and config.height <= 0 else "requested"
    print(f"Camera capture: {w}x{h} ({res_note})")
    try:
        pipeline = preload_pipeline(
            webcam,
            fps=webcam.fps,
            pentiptrack_version=config.pentiptrack_version,
            nn_interval=config.nn_interval,
        )
        print(
            f"Model loaded: {config.tracker_label} | NN every {config.nn_interval} frame(s)"
        )
        WebcamTrackingSession(config, webcam, pipeline).run()
    finally:
        webcam.release()


if __name__ == "__main__":
    main()
