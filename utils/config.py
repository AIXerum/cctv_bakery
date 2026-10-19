from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    window_title: str = "Bakery Throughput Tracker"
    default_source: str = "1.mp4"
    model_path: str = "weights.pt"
    min_confidence: float = 0.25
    # Run the detector every N frames for better performance.
    infer_every_n_frames: int = 10
    refresh_interval_ms: int = 33

