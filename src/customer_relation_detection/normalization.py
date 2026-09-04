"""Unicode-safe normalization and time-aware account address signatures."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import pandas as pd

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
PERSIAN_CHARACTERS = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
        "ـ": "",
        "\u200c": " ",
        "\u200d": " ",
        "\ufeff": " ",
    }
)
TOKEN_REPLACEMENTS = {"st": "street", "bldg": "building", "apt": "unit"}
NUMBER_KEYWORDS = {
    "building": {"building", "bldg", "ساختمان", "پلاک"},
    "unit": {"unit", "apt", "واحد"},
}
ADDRESS_COLUMNS = (
    "city_norm",
    "street_norm",
    "building_norm",
    "unit_norm",
    "postcode_norm",
)


def _present(value: object) -> bool:
    try:
        return value is not None and not bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_text(value: object) -> str:
    """Normalize scripts, digits, punctuation, spacing, and address abbreviations."""
    if not _present(value):
        return ""
    text = (
        unicodedata.normalize("NFKC", str(value))
        .translate(DIGITS)
        .translate(PERSIAN_CHARACTERS)
        .casefold()
        .strip()
    )
    characters: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category.startswith(("L", "N")):
            characters.append(character)
        elif category.startswith("M"):
            continue
        else:
            characters.append(" ")
    tokens = [TOKEN_REPLACEMENTS.get(token, token) for token in "".join(characters).split()]
    return " ".join(tokens)


def normalize_number_field(value: object, field: str | None = None) -> str:
    """Extract an unambiguous number, preferring a field-specific keyword."""
    normalized = normalize_text(value)
    tokens = normalized.split()
    if field is not None:
        if field not in NUMBER_KEYWORDS:
            raise ValueError(f"Unsupported number field: {field}")
        keywords = NUMBER_KEYWORDS[field]
        for position, token in enumerate(tokens[:-1]):
            if token in keywords and tokens[position + 1].isdigit():
                return str(int(tokens[position + 1]))
    numbers = re.findall(r"\d+", normalized)
    if len(numbers) == 1:
        return str(int(numbers[0]))
    return ""


def normalize_postcode(value: object) -> str:
    return "".join(character for character in normalize_text(value) if character.isalnum())


def _mode(values: Iterable[str]) -> str:
    series = pd.Series(list(values), dtype="string")
    if series.empty:
        return ""
    counts = series.value_counts(dropna=False)
    return str(sorted(counts[counts == counts.max()].index.astype(str))[0])


def _select_address_tuple(group: pd.DataFrame) -> pd.Series:
    """Choose one observed tuple by frequency, recency, then a stable lexical tie-break."""
    summary = (
        group.groupby(list(ADDRESS_COLUMNS), dropna=False, as_index=False)
        .agg(address_observations=("order_id", "nunique"), last_observed_at=("_observed_at", "max"))
        .sort_values(
            ["address_observations", "last_observed_at", *ADDRESS_COLUMNS],
            ascending=[False, False, *([True] * len(ADDRESS_COLUMNS))],
            kind="mergesort",
        )
    )
    return summary.iloc[0]


def build_account_signatures(
    orders: pd.DataFrame,
    *,
    snapshot_date: object | None = None,
    history_days: int = 365,
) -> pd.DataFrame:
    """Build one recent, actually observed address tuple per account.

    Each account uses only observations at or before ``snapshot_date`` and inside a
    rolling history window relative to that account's latest eligible observation.
    Address fields are selected as a tuple, so the result cannot combine fields from
    different historical addresses.
    """
    if history_days < 1:
        raise ValueError("history_days must be positive")
    frame = orders.copy()
    frame["_observed_at"] = pd.to_datetime(frame["order_date"], errors="raise", utc=True)
    if snapshot_date is not None:
        snapshot = pd.to_datetime(snapshot_date, errors="raise", utc=True)
        frame = frame[frame["_observed_at"] <= snapshot].copy()
    if frame.empty:
        raise ValueError("No observations are available at the requested snapshot date")
    latest_by_account = frame.groupby("account_id")["_observed_at"].transform("max")
    cutoff = latest_by_account - pd.to_timedelta(history_days, unit="D")
    frame = frame[frame["_observed_at"] >= cutoff].copy()

    frame["city_norm"] = frame["city"].map(normalize_text)
    frame["street_norm"] = frame["street"].map(normalize_text)
    frame["building_norm"] = frame["building"].map(
        lambda value: normalize_number_field(value, "building")
    )
    frame["unit_norm"] = frame["unit"].map(lambda value: normalize_number_field(value, "unit"))
    frame["postcode_norm"] = frame["postcode"].map(normalize_postcode)
    frame["family_token_norm"] = frame["family_token"].map(normalize_text)

    records: list[dict[str, object]] = []
    for account_id, group in frame.groupby("account_id", sort=True):
        selected = _select_address_tuple(group)
        chosen = group
        for column in ADDRESS_COLUMNS:
            chosen = chosen[chosen[column] == selected[column]]
        record: dict[str, object] = {
            "account_id": str(account_id),
            **{column: str(selected[column]) for column in ADDRESS_COLUMNS},
            "family_token_norm": _mode(chosen["family_token_norm"]),
            "observed_orders": int(group["order_id"].nunique()),
            "address_observations": int(selected["address_observations"]),
            "address_history_count": int(group[list(ADDRESS_COLUMNS)].drop_duplicates().shape[0]),
            "address_as_of_date": selected["last_observed_at"].date().isoformat(),
        }
        records.append(record)
    signatures = pd.DataFrame.from_records(records).sort_values("account_id", ignore_index=True)
    complete = signatures[list(ADDRESS_COLUMNS)].ne("").all(axis=1)
    signatures["exact_address_key"] = ""
    signatures.loc[complete, "exact_address_key"] = signatures.loc[
        complete, list(ADDRESS_COLUMNS)
    ].agg("|".join, axis=1)
    return signatures
