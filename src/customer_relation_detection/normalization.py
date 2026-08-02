"""Deterministic normalization and account-level address signatures."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import pandas as pd

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def normalize_text(value: object) -> str:
    """Normalize case, Unicode, digits, punctuation, and common address terms."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).translate(DIGITS).lower().strip()
    replacements = {
        r"\bst\.?\b": "street",
        r"\bbldg\.?\b": "building",
        r"\bapt\.?\b": "unit",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_number_field(value: object) -> str:
    normalized = normalize_text(value)
    numbers = re.findall(r"\d+", normalized)
    return str(int(numbers[-1])) if numbers else normalized


def normalize_postcode(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(value))


def _mode(values: Iterable[str]) -> str:
    series = pd.Series(list(values), dtype="string")
    if series.empty:
        return ""
    counts = series.value_counts(dropna=False)
    return str(sorted(counts[counts == counts.max()].index.astype(str))[0])


def build_account_signatures(orders: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated orders into one normalized record per synthetic account."""
    frame = orders.copy()
    frame["city_norm"] = frame["city"].map(normalize_text)
    frame["street_norm"] = frame["street"].map(normalize_text)
    frame["building_norm"] = frame["building"].map(normalize_number_field)
    frame["unit_norm"] = frame["unit"].map(normalize_number_field)
    frame["postcode_norm"] = frame["postcode"].map(normalize_postcode)
    frame["family_token_norm"] = frame["family_token"].map(normalize_text)
    signatures = (
        frame.groupby("account_id", as_index=False)
        .agg(
            city_norm=("city_norm", _mode),
            street_norm=("street_norm", _mode),
            building_norm=("building_norm", _mode),
            unit_norm=("unit_norm", _mode),
            postcode_norm=("postcode_norm", _mode),
            family_token_norm=("family_token_norm", _mode),
            observed_orders=("order_id", "nunique"),
        )
        .sort_values("account_id", ignore_index=True)
    )
    signatures["exact_address_key"] = (
        signatures["city_norm"]
        + "|"
        + signatures["street_norm"]
        + "|"
        + signatures["building_norm"]
        + "|"
        + signatures["unit_norm"]
        + "|"
        + signatures["postcode_norm"]
    )
    return signatures
