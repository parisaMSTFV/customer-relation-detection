import pandas as pd

from customer_relation_detection.synthetic import generate_orders


def test_generation_is_deterministic() -> None:
    first_orders, first_truth = generate_orders(seed=42, buildings=30)
    second_orders, second_truth = generate_orders(seed=42, buildings=30)
    pd.testing.assert_frame_equal(first_orders, second_orders)
    pd.testing.assert_frame_equal(first_truth, second_truth)


def test_ground_truth_is_not_in_order_observations() -> None:
    orders, truth = generate_orders(buildings=30)
    assert "true_group_id" not in orders.columns
    assert "split" not in orders.columns
    assert set(orders["account_id"]) == set(truth["account_id"])


def test_buildings_do_not_cross_splits() -> None:
    _, truth = generate_orders(buildings=30)
    assert truth.groupby("building_id")["split"].nunique().max() == 1
