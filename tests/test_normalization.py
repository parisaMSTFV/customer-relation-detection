import pandas as pd

from customer_relation_detection.normalization import (
    build_account_signatures,
    normalize_number_field,
    normalize_text,
)
from customer_relation_detection.synthetic import generate_orders


def test_address_abbreviations_and_unicode_digits_normalize() -> None:
    assert normalize_text(" Cedar St. 7 ") == "cedar street 7"
    assert normalize_number_field("Apt ۰۳") == "3"
    assert normalize_number_field("Bldg-۱۲") == "12"


def test_persian_text_and_field_specific_numbers_are_preserved() -> None:
    assert normalize_text("تهران، خیابان ولیعصر") == "تهران خیابان ولیعصر"
    assert normalize_text("خيابان كاشاني") == "خیابان کاشانی"
    assert normalize_number_field("ساختمان ۱۲ طبقه ۳", "building") == "12"
    assert normalize_number_field("ساختمان ۱۲ طبقه ۳") == ""


def test_account_signatures_have_one_row_per_account() -> None:
    orders, truth = generate_orders(buildings=30)
    signatures = build_account_signatures(orders)
    assert len(signatures) == len(truth)
    assert signatures["account_id"].is_unique


def test_signature_is_one_observed_tuple_and_respects_snapshot() -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "account_id": "A1",
                "city": "Tehran",
                "street": "Valiasr St.",
                "building": "Building 12",
                "unit": "Unit 3",
                "postcode": "111",
                "family_token": "F1",
                "order_date": "2025-01-01",
            },
            {
                "order_id": "O2",
                "account_id": "A1",
                "city": "Tehran",
                "street": "Zand St.",
                "building": "Building 9",
                "unit": "Unit 2",
                "postcode": "222",
                "family_token": "F1",
                "order_date": "2025-02-01",
            },
            {
                "order_id": "O3",
                "account_id": "A1",
                "city": "Shiraz",
                "street": "Valiasr St.",
                "building": "Building 9",
                "unit": "Unit 2",
                "postcode": "222",
                "family_token": "F1",
                "order_date": "2025-03-01",
            },
        ]
    )
    latest = build_account_signatures(orders).iloc[0]
    assert (latest["city_norm"], latest["street_norm"], latest["building_norm"]) == (
        "shiraz",
        "valiasr street",
        "9",
    )
    historical = build_account_signatures(orders, snapshot_date="2025-01-15").iloc[0]
    assert historical["building_norm"] == "12"
