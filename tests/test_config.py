from customer_relation_detection.config import load_config


def test_default_config_is_loaded_from_package_resources() -> None:
    config = load_config()
    assert config.policy_version == "relation-policy-v2"
    assert config.policy_max_component_size in config.component_size_candidates
