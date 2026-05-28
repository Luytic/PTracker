from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera: int = 0
    output: str = "demo_output.mp4"
    log: str = "tracking_log.jsonl"
    max_frames: int = 0
    width: int = 0
    height: int = 0
    pentiptrack_version: str = "v2"
    nn_interval: int = 3
    debug: bool = False

    @property
    def tracker_label(self) -> str:
        return f"PenTipTrack {self.pentiptrack_version.upper()}"
