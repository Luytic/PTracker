from __future__ import annotations
import argparse
from app.config import AppConfig

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Webcam pen-tip tracker")
    p.add_argument("--camera", type=int, default=0)
    return p

def parse_args(argv=None) -> AppConfig:
    args = build_parser().parse_args(argv)
    return AppConfig(camera=args.camera)
