"""Config loading utility."""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
