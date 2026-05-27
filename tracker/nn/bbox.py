"""Backward-compatible re-export; prefer tracker.nn.coords."""

from tracker.nn.coords import Center, Corner, corner2center

__all__ = ["Center", "Corner", "corner2center"]
