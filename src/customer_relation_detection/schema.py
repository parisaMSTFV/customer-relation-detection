"""Fail-closed schemas for operational observations and benchmark labels."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ORDER_COLUMNS = {
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
TRUTH_COLUMNS = {
    "account_id",
    "true_group_id",
    "true_relation_type",
    "building_id",
    "unit_id",
    "noise_level",
    "split",
}
FORBIDDEN_OBSERVATION_COLUMNS = {
    "true_group_id",
    "true_relation_type",
    "building_id",
    "unit_id",
    "split",
    "noise_level",
    "is_related",
}
SUPPORTED_SPLITS = {"train", "validation", "test"}


class DataValidationError(ValueError):
    """Raised when inputs violate the public pipeline contract."""


@dataclass(frozen=True)
class ValidationSummary:
    orders: int
    accounts: int
    true_groups: int
    buildings: int
    checks_passed: int


@dataclass(frozen=True)
class ObservationSummary:
    orders: int
    accounts: int
    first_order_date: str
    last_order_date: str
    checks_passed: int


def _missing_or_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _contains_non_scalar(series: pd.Series) -> bool:
    return bool(series.map(lambda value: isinstance(value, (dict, list, set, tuple))).any())


def validate_observations(orders: pd.DataFrame) -> ObservationSummary:
    """Validate unlabeled observations before normalization or scoring."""
    missing = ORDER_COLUMNS.difference(orders.columns)
    if missing:
        raise DataValidationError(f"Missing order columns: {sorted(missing)}")
    leaked = FORBIDDEN_OBSERVATION_COLUMNS.intersection(orders.columns)
    if leaked:
        raise DataValidationError(
            f"Ground-truth labels must not appear in observations: {sorted(leaked)}"
        )
    if orders.empty:
        raise DataValidationError("Observations must not be empty")
    for identifier in ("order_id", "account_id"):
        if _missing_or_blank(orders[identifier]).any():
            raise DataValidationError(f"{identifier} must not be null or blank")
    if orders["order_id"].duplicated().any():
        raise DataValidationError("order_id must be unique")
    text_columns = ["city", "street", "building", "unit", "postcode", "family_token"]
    if any(_contains_non_scalar(orders[column]) for column in text_columns):
        raise DataValidationError("Address fields must contain scalar values")
    nonempty_location = pd.DataFrame(
        {
            column: ~_missing_or_blank(orders[column])
            for column in ["city", "street", "building", "postcode"]
        }
    ).any(axis=1)
    if not nonempty_location.all():
        raise DataValidationError("Every order must contain at least one location field")
    lengths = orders[text_columns].astype("string").apply(lambda column: column.str.len())
    if lengths.gt(512).any().any():
        raise DataValidationError("Address fields must not exceed 512 characters")
    dates = pd.to_datetime(orders["order_date"], errors="coerce", utc=True, format="mixed")
    if dates.isna().any():
        raise DataValidationError("order_date must contain valid, non-null dates")
    return ObservationSummary(
        orders=len(orders),
        accounts=orders["account_id"].nunique(),
        first_order_date=dates.min().date().isoformat(),
        last_order_date=dates.max().date().isoformat(),
        checks_passed=10,
    )


def validate_inputs(orders: pd.DataFrame, truth: pd.DataFrame) -> ValidationSummary:
    """Validate the synthetic evaluation boundary and split separation."""
    validate_observations(orders)
    missing_truth = TRUTH_COLUMNS.difference(truth.columns)
    if missing_truth:
        raise DataValidationError(f"Missing truth columns: {sorted(missing_truth)}")
    if truth.empty:
        raise DataValidationError("Ground truth must not be empty")
    for identifier in ("account_id", "true_group_id", "building_id", "split"):
        if _missing_or_blank(truth[identifier]).any():
            raise DataValidationError(f"Truth {identifier} must not be null or blank")
    if truth["account_id"].duplicated().any():
        raise DataValidationError("Ground truth must have one row per account")
    if set(orders["account_id"].astype(str)) != set(truth["account_id"].astype(str)):
        raise DataValidationError("Observation and truth account sets must match")
    if not truth["split"].isin(SUPPORTED_SPLITS).all():
        raise DataValidationError("Unsupported split label")
    missing_splits = SUPPORTED_SPLITS.difference(set(truth["split"]))
    if missing_splits:
        raise DataValidationError(
            f"Ground truth is missing required splits: {sorted(missing_splits)}"
        )
    if truth.groupby("building_id")["split"].nunique().max() != 1:
        raise DataValidationError("A building must not cross data splits")
    if truth.groupby("true_group_id")["building_id"].nunique().max() != 1:
        raise DataValidationError("A true group must not cross buildings")
    if not truth["noise_level"].isin(["clean", "moderate", "heavy"]).all():
        raise DataValidationError("Unsupported noise level")
    return ValidationSummary(
        orders=len(orders),
        accounts=truth["account_id"].nunique(),
        true_groups=truth["true_group_id"].nunique(),
        buildings=truth["building_id"].nunique(),
        checks_passed=18,
    )
