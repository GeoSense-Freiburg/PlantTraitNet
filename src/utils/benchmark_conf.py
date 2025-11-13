"""
Parses a params.yaml file (located at src/benchmarking/params.yaml)
to return a config object that can be used in other scripts.
"""

import yaml
from box import ConfigBox
from pathlib import Path


# Resolve path relative to this file’s location
PARAMS_PATH = Path(__file__).resolve().parents[1] / "params.yaml"


def parse_params() -> dict:
    """
    Parse a params.yaml file from a relative path and return it as a dictionary.
    """
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"params.yaml not found at: {PARAMS_PATH}")
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config(subset: str = None) -> ConfigBox:
    """
    Get a ConfigBox object (attribute-accessible dict) from params.yaml.
    If a subset is specified, returns only that section.
    """
    config = ConfigBox(parse_params())
    return config[subset] if subset is not None else config

if __name__ == "__main__":
    print(get_config())
