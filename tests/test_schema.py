import pytest

from customer_relation_detection.schema import DataValidationError, validate_inputs
from customer_relation_detection.synthetic import generate_orders


def test_valid_synthetic_inputs_pass() -> None:
    orders, truth = generate_orders(buildings=30)
    summary = validate_inputs(orders, truth)
    assert summary.buildings == 30
    assert summary.checks_passed == 9


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
