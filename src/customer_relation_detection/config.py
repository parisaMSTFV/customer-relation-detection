"""Configuration loading for the relation-detection case study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "analysis.json"


@dataclass(frozen=True)
class AnalysisConfig:
    seed: int
    buildings: int
    units_per_building: int
    validation_share: float
    test_share: float
    threshold_min: float
    threshold_max: float
    threshold_step: float
    max_component_size: int


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AnalysisConfig:
    config = AnalysisConfig(**json.loads(path.read_text(encoding="utf-8")))
    if config.buildings < 30 or config.units_per_building < 2:
        raise ValueError("Synthetic location dimensions are too small")
    if config.validation_share <= 0 or config.test_share <= 0:
        raise ValueError("Validation and test shares must be positive")
    if config.validation_share + config.test_share >= 0.6:
        raise ValueError("Training buildings must remain the majority")
    if not 0 < config.threshold_min < config.threshold_max <= 1:
        raise ValueError("Threshold range is invalid")
    if config.threshold_step <= 0:
        raise ValueError("threshold_step must be positive")
    if config.max_component_size < 2:
        raise ValueError("max_component_size must be at least two")
    return config
