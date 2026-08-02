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


def test_account_signatures_have_one_row_per_account() -> None:
    orders, truth = generate_orders(buildings=30)
    signatures = build_account_signatures(orders)
    assert len(signatures) == len(truth)
    assert signatures["account_id"].is_unique
