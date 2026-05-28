from __future__ import annotations
import argparse
from app.config import AppConfig

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-time webcam pen-tip tracker")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--output", type=str, default="demo_output.mp4")
    p.add_argument("--log", type=str, default="tracking_log.jsonl")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--pentiptrack-version", choices=("v2", "v3"), default="v3")
    p.add_argument("--nn-interval", type=int, default=3, metavar="N")
    return p

def parse_args(argv=None) -> AppConfig:
    args = build_parser().parse_args(argv)
    if args.nn_interval < 1:
        raise SystemExit("--nn-interval must be >= 1")
    return AppConfig(
        camera=args.camera,
        output=args.output,
        log=args.log,
        max_frames=args.max_frames,
        width=args.width,
        height=args.height,
        pentiptrack_version=args.pentiptrack_version,
        nn_interval=args.nn_interval,
    )
