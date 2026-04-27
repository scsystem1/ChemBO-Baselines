"""
Centralized configuration loader for sensitive strings (SMILES, AWS region, etc.).

Loads from smiles_config.json (gitignored) or falls back to environment variables.
Copy smiles_config.example.json -> smiles_config.json and fill in real values.
"""

import os
import json

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smiles_config.json")
_config_cache = None


def _load_config():
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if os.path.exists(_CONFIG_FILE):
        with open(_CONFIG_FILE, "r") as f:
            _config_cache = json.load(f)
    else:
        _config_cache = {}

    return _config_cache


def get(key, default=None):
    """Return a config value from smiles_config.json or the environment."""
    cfg = _load_config()
    value = cfg.get(key) or os.environ.get(key) or default
    if value is None:
        raise ValueError(
            f"Missing config key '{key}'. "
            f"Set it in smiles_config.json or as an environment variable. "
            f"See smiles_config.example.json for the expected keys."
        )
    return value
