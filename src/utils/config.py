"""Configuration loading and management utilities."""

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve environment variables
    config = _resolve_env_vars(config)

    return config


def _resolve_env_vars(config: Dict) -> Dict:
    """Recursively resolve environment variables in config."""
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Replace ${VAR_NAME} with environment variable
        if config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.environ.get(var_name, config)
        return config
    else:
        return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configurations, with override_config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Configuration to override base

    Returns:
        Merged configuration
    """
    result = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: Dict) -> bool:
    """
    Validate configuration has required fields.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ["model", "training", "data"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate model config
    model_config = config["model"]
    required_model_fields = ["vision_encoder", "text_encoder", "fusion", "heads"]
    for field in required_model_fields:
        if field not in model_config:
            raise ValueError(f"Missing required model config field: {field}")

    # Validate training config
    training_config = config["training"]
    required_training_fields = ["max_epochs", "inner_lr"]
    for field in required_training_fields:
        if field not in training_config:
            raise ValueError(f"Missing required training config field: {field}")

    return True


class ConfigNamespace:
    """Convert dictionary config to namespace for easier access."""

    def __init__(self, config: Dict):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict:
        """Convert namespace back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        return str(self.to_dict())
