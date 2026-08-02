"""Generate synthetic account, order, address, and relation observations."""

from __future__ import annotations

import string

import numpy as np
import pandas as pd

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _typo(text: str, rng: np.random.Generator) -> str:
    positions = [index for index, value in enumerate(text) if value in string.ascii_letters]
    if not positions:
        return text
    position = int(rng.choice(positions))
    return text[:position] + text[position + 1 :]


def _account_address_variant(
    city: str,
    street: str,
    building: int,
    unit: int,
    postcode: str,
    noise_level: str,
    rng: np.random.Generator,
) -> dict[str, str]:
    if noise_level == "clean":
        return {
            "city": city,
            "street": street,
            "building": f"Building {building}",
            "unit": f"Unit {unit}",
            "postcode": postcode,
        }
    if noise_level == "moderate":
        return {
            "city": city.upper(),
            "street": street.replace("Street", "St."),
            "building": f"Bldg-{building}",
            "unit": f"Apt {str(unit).translate(PERSIAN_DIGITS)}",
            "postcode": postcode.replace("-", " "),
        }
    unit_draw = rng.random()
    if unit_draw < 0.18:
        unit_value = ""
    elif unit_draw < 0.36:
        unit_value = str(unit + 1).translate(PERSIAN_DIGITS)
    else:
        unit_value = str(unit).translate(PERSIAN_DIGITS)
    return {
        "city": city,
        "street": _typo(street.replace("Street", "St"), rng),
        "building": f"B {str(building).translate(PERSIAN_DIGITS)}",
        "unit": unit_value,
        "postcode": "" if rng.random() < 0.65 else postcode.replace("-", ""),
    }


def generate_orders(
    seed: int = 42,
    buildings: int = 180,
    units_per_building: int = 4,
    validation_share: float = 0.2,
    test_share: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create noisy address observations and separate account-level ground truth."""
    rng = np.random.default_rng(seed)
    building_ids = np.arange(1, buildings + 1)
    shuffled = rng.permutation(building_ids)
    validation_count = int(round(buildings * validation_share))
    test_count = int(round(buildings * test_share))
    split_map = {value: "train" for value in building_ids}
    for value in shuffled[:validation_count]:
        split_map[int(value)] = "validation"
    for value in shuffled[validation_count : validation_count + test_count]:
        split_map[int(value)] = "test"

    order_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    account_position = 0
    order_position = 0
    group_position = 0
    cities = ["North City", "Central City", "South City"]
    street_names = ["Cedar", "Maple", "Willow", "Oak", "Pine", "Elm"]

    for building_id in building_ids:
        city = cities[(building_id - 1) % len(cities)]
        street_number = (building_id - 1) // 6 + 1
        street = f"{street_names[(building_id - 1) % 6]} Street {street_number}"
        postcode = f"PC-{(building_id - 1) % 3 + 1}-{building_id:04d}"
        for unit in range(1, units_per_building + 1):
            group_position += 1
            group_id = f"REL-{group_position:05d}"
            size = int(rng.choice([1, 2, 3, 4, 6], p=[0.44, 0.26, 0.16, 0.10, 0.04]))
            if size == 1:
                relation_type = "singleton"
            elif size >= 6:
                relation_type = "shared_location"
            else:
                relation_type = "household_signal" if rng.random() < 0.65 else "shared_residence"
            shared_family_token = f"FAM-{group_position:05d}"
            for _member in range(size):
                account_position += 1
                account_id = f"ACC-{account_position:06d}"
                noise_level = str(rng.choice(["clean", "moderate", "heavy"], p=[0.30, 0.48, 0.22]))
                variant = _account_address_variant(
                    city, street, int(building_id), unit, postcode, noise_level, rng
                )
                if relation_type == "household_signal" and rng.random() < 0.82:
                    family_token = shared_family_token
                else:
                    family_token = f"FAM-X-{account_position:06d}"
                truth_rows.append(
                    {
                        "account_id": account_id,
                        "true_group_id": group_id,
                        "true_relation_type": relation_type,
                        "building_id": int(building_id),
                        "unit_id": unit,
                        "noise_level": noise_level,
                        "split": split_map[int(building_id)],
                    }
                )
                order_count = int(rng.integers(2, 6))
                for _ in range(order_count):
                    order_position += 1
                    order_rows.append(
                        {
                            "order_id": f"ORD-{order_position:07d}",
                            "account_id": account_id,
                            **variant,
                            "family_token": family_token,
                            "order_date": (
                                pd.Timestamp("2025-01-01")
                                + pd.Timedelta(days=int(rng.integers(0, 365)))
                            ),
                        }
                    )
    orders = pd.DataFrame.from_records(order_rows).sort_values("order_id", ignore_index=True)
    truth = pd.DataFrame.from_records(truth_rows).sort_values("account_id", ignore_index=True)
    return orders, truth
