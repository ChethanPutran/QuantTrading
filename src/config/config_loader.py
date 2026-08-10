import yaml
from pathlib import Path


class ConfigLoader:
    """Loads YAML configuration files."""

    def __init__(self, config_path: str, constraints_path: str):
        self.config = self._load_yaml(config_path)
        self.constraints = self._load_yaml(constraints_path)

    def _load_yaml(self, path: str) -> dict:
        with open(Path(path), "r") as f:
            return yaml.safe_load(f)

    def get_config(self) -> dict:
        return self.config

    def get_constraints(self) -> dict:
        return self.constraints