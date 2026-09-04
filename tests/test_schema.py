import pytest

from customer_relation_detection.schema import (
    DataValidationError,
    validate_inputs,
    validate_observations,
)
from customer_relation_detection.synthetic import generate_orders


def test_valid_synthetic_inputs_pass() -> None:
    orders, truth = generate_orders(buildings=30)
    summary = validate_inputs(orders, truth)
    assert summary.buildings == 30
    assert summary.checks_passed == 18


def test_truth_label_in_observations_is_rejected() -> None:
    orders, truth = generate_orders(buildings=30)
    orders["true_group_id"] = "LEAK"
    with pytest.raises(DataValidationError, match="must not appear"):
        validate_inputs(orders, truth)


def test_duplicate_order_id_is_rejected() -> None:
    orders, truth = generate_orders(buildings=30)
    orders.loc[1, "order_id"] = orders.loc[0, "order_id"]
    with pytest.raises(DataValidationError, match="order_id must be unique"):
        validate_inputs(orders, truth)


def test_blank_account_and_invalid_date_are_rejected() -> None:
    orders, _ = generate_orders(buildings=30)
    orders.loc[0, "account_id"] = " "
    with pytest.raises(DataValidationError, match="account_id must not"):
        validate_observations(orders)
    orders, _ = generate_orders(buildings=30)
    orders["order_date"] = orders["order_date"].astype("object")
    orders.loc[0, "order_date"] = "not-a-date"
    with pytest.raises(DataValidationError, match="valid"):
        validate_observations(orders)


def test_unlabeled_observations_have_an_independent_contract() -> None:
    orders, _ = generate_orders(buildings=30)
    summary = validate_observations(orders)
    assert summary.accounts == orders["account_id"].nunique()
