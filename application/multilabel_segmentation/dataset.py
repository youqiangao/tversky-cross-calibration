from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from tversky_cross_calibration.config import dataset_config, paper_config

from .semantic_dataset import build_semantic_dataset


DATASET_CHOICES = ("voc2012", "cityscapes")

DEFAULT_DATA_ROOTS = {
    name: str(dataset_config(name)["data_root"]) for name in DATASET_CHOICES
}
DEFAULT_TRAINING_CONFIGS: Dict[str, Dict[str, float | int]] = {
    name: {
        "image_size": int(dataset_config(name)["image_size"]),
        "batch_size": int(dataset_config(name)["batch_size"]),
        "epochs": int(paper_config()["training"]["max_epochs"]),
        "lr": float(paper_config()["training"]["learning_rate"]),
        "weight_decay": float(paper_config()["training"]["weight_decay"]),
    }
    for name in DATASET_CHOICES
}


def normalize_dataset_name(dataset_name: str | None) -> str:
    name = "voc2012" if dataset_name is None else dataset_name.lower()
    if name not in DATASET_CHOICES:
        raise ValueError(f"Unsupported paper dataset {name!r}; expected one of {DATASET_CHOICES}.")
    return name


def default_data_root_for(dataset_name: str | None) -> str:
    name = normalize_dataset_name(dataset_name)
    if name not in DATASET_CHOICES:
        raise ValueError(f"Unsupported paper dataset {name!r}; expected one of {DATASET_CHOICES}.")
    return DEFAULT_DATA_ROOTS[name]


def default_training_config_for(dataset_name: str | None) -> Dict[str, float | int]:
    return dict(DEFAULT_TRAINING_CONFIGS[normalize_dataset_name(dataset_name)])


def validate_integer_mask(target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid_mask = target != int(ignore_index)
    if bool(valid_mask.any().item()):
        valid_values = target[valid_mask]
        if int(valid_values.min().item()) < 0 or int(valid_values.max().item()) >= int(num_classes):
            raise ValueError("Target contains a valid class index outside the configured class range.")
    return valid_mask


class MultilabelSemanticDataset(Dataset):
    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset
        self.num_classes = int(base_dataset.num_classes)
        self.ignore_index = int(base_dataset.ignore_index)
        self.samples = base_dataset.samples
        self.class_names = tuple(f"class_{index}" for index in range(self.num_classes))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        item = dict(self.base_dataset[index])
        item["valid_mask"] = validate_integer_mask(item["mask"], self.num_classes, self.ignore_index)
        return item


def build_dataset(
    dataset_name: str,
    root: str | Path,
    split: str,
    image_size: int,
    augment: bool = False,
    normalize: bool = True,
    max_samples: int | None = None,
) -> MultilabelSemanticDataset:
    base_dataset = build_semantic_dataset(
        dataset_name=normalize_dataset_name(dataset_name),
        root=root,
        split=split,
        image_size=image_size,
        augment=augment,
        normalize=normalize,
        max_samples=max_samples,
    )
    return MultilabelSemanticDataset(base_dataset)
