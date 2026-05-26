"""Pen-tip tracking package.

Canonical entry points:
  from tracker.tracking import TrackingPipeline, create_tracking_pipeline
  from tracker.types import PipelineOutput
"""

from tracker.tracking import TrackingPipeline, create_tracking_pipeline
from tracker.types import PipelineOutput

__all__ = ["TrackingPipeline", "PipelineOutput", "create_tracking_pipeline"]
