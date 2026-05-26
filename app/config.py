from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    camera: int = 0
    output: str = "demo_output.mp4"
