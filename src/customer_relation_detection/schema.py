"""Schema and separation checks for observations and ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class DataValidationError(ValueError):
    """Raised when synthetic inputs violate the public pipeline contract."""


@dataclass(frozen=True)
class ValidationSummary:
    orders: int
    accounts: int
    true_groups: int
    buildings: int
    checks_passed: int


def validate_inputs(orders: pd.DataFrame, truth: pd.DataFrame) -> ValidationSummary:
    order_columns = {
        "order_id",
        "account_id",
        "city",
        "street",
        "building",
        "unit",
        "postcode",
        "family_token",
        "order_date",
    }
    truth_columns = {
        "account_id",
        "true_group_id",
        "true_relation_type",
        "building_id",
        "unit_id",
        "noise_level",
        "split",
    }
    missing = order_columns.difference(orders.columns)
    if missing:
        raise DataValidationError(f"Missing order columns: {sorted(missing)}")
    missing_truth = truth_columns.difference(truth.columns)
    if missing_truth:
        raise DataValidationError(f"Missing truth columns: {sorted(missing_truth)}")
    forbidden = {"true_group_id", "true_relation_type", "split", "noise_level"}
    if forbidden.intersection(orders.columns):
        raise DataValidationError("Ground-truth labels must not appear in observations")
    if orders.empty or truth.empty:
        raise DataValidationError("Inputs must not be empty")
    if orders["order_id"].duplicated().any():
        raise DataValidationError("order_id must be unique")
    if truth["account_id"].duplicated().any():
        raise DataValidationError("Ground truth must have one row per account")
    if set(orders["account_id"]) != set(truth["account_id"]):
        raise DataValidationError("Observation and truth account sets must match")
    if not truth["split"].isin(["train", "validation", "test"]).all():
        raise DataValidationError("Unsupported split label")
    if truth.groupby("building_id")["split"].nunique().max() != 1:
        raise DataValidationError("A building must not cross data splits")
    return ValidationSummary(
        orders=len(orders),
        accounts=truth["account_id"].nunique(),
        true_groups=truth["true_group_id"].nunique(),
        buildings=truth["building_id"].nunique(),
        checks_passed=9,
    )
