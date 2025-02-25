from dataclasses import dataclass
import json
from pathlib import Path
from typing import (
    List,
    Union,
)


@dataclass
class Config:
    f: str
    results_path: str
    width: int
    height: int
    batchsize: int
    lr: float
    weight_decay: float
    epochs: int
    linear_num: int
    stride: int
    droprate: float
    warm_epoch: int
    patience: int
    gpu_ids: List[int]
    model_name: str
    model_dir: str
    ms: List[float]
    test_dataloader: str
    data_dir: str = ""
    nclasses: int = 0
    step: int = 1
    verbose: bool = False
    debug_test: bool = False
    no_cache: bool = False
    st_reid: bool = False

    @staticmethod
    def from_json(path: Union[str, Path]):
        path = Path(path)

        with path.open() as f:
            data = json.load(f)
        data["f"] = str(path)

        return Config(**data)

    def to_json(self, path: Union[str, Path]):
        path = Path(path)
        with path.open("w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
