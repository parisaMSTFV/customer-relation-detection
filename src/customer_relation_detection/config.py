"""Configuration loading for the relation-detection case study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any


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
    component_size_candidates: tuple[int, ...] = (4, 5, 6, 8, 10)
    max_block_size: int = 250
    address_history_days: int = 365
    precision_floor: float = 0.90
    false_merge_cost: float = 5.0
    false_split_cost: float = 1.0
    policy_version: str = "relation-policy-v2"
    policy_threshold: float = 0.82
    policy_max_component_size: int = 6
    stability_seeds: tuple[int, ...] = (1, 7, 42, 99, 123)
    stability_buildings: int = 60


def _default_config_values() -> dict[str, Any]:
    resource = files("customer_relation_detection").joinpath("resources/analysis.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_config(path: Path | None = None) -> AnalysisConfig:
    """Load a packaged default config or an explicit local policy file."""
    values = (
        _default_config_values()
        if path is None
        else json.loads(Path(path).read_text(encoding="utf-8"))
    )
    if "component_size_candidates" in values:
        values["component_size_candidates"] = tuple(values["component_size_candidates"])
    if "stability_seeds" in values:
        values["stability_seeds"] = tuple(values["stability_seeds"])
    config = AnalysisConfig(**values)
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
    if not config.component_size_candidates or min(config.component_size_candidates) < 2:
        raise ValueError("component_size_candidates must contain values of at least two")
    if max(config.component_size_candidates) > config.max_component_size:
        raise ValueError("component_size_candidates cannot exceed max_component_size")
    if config.max_block_size < 2:
        raise ValueError("max_block_size must be at least two")
    if config.address_history_days < 1:
        raise ValueError("address_history_days must be positive")
    if not 0 < config.precision_floor <= 1:
        raise ValueError("precision_floor must be in (0, 1]")
    if config.false_merge_cost <= 0 or config.false_split_cost <= 0:
        raise ValueError("Error costs must be positive")
    if not config.policy_version.strip():
        raise ValueError("policy_version must not be blank")
    if not 0 < config.policy_threshold <= 1:
        raise ValueError("policy_threshold must be in (0, 1]")
    if not 2 <= config.policy_max_component_size <= config.max_component_size:
        raise ValueError("policy_max_component_size is outside the allowed range")
    if len(set(config.stability_seeds)) != len(config.stability_seeds):
        raise ValueError("stability_seeds must be unique")
    if config.stability_seeds and config.stability_buildings < 30:
        raise ValueError("stability_buildings must be at least 30")
    return config
