from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    camera: int = 0
    output: str = "demo_output.mp4"
    log: str = "tracking_log.jsonl"
    max_frames: int = 0
    width: int = 1280
    height: int = 720

    @property
    def tracker_label(self) -> str:
        return "PenTipTrack"
