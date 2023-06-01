from dataclasses import dataclass
import json
from pathlib import Path
from typing import Union


@dataclass
class Config:
    context_distribution_path: Path
    results_path: Path
    width: int
    height: int

    def __post_init__(self):
        self.context_distribution_path = Path(self.context_distribution_path)
        self.results_path = Path(self.results_path)

    @staticmethod
    def from_json(path: Union[str, Path]):
        path = Path(path)
        with path.open() as f:
            data = json.load(f)

        return Config(**data)
