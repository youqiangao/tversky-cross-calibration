from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


HF_REPO_ID = "youqiangao/tversky-cross-calibration-checkpoints"

PAPER_CHECKPOINTS = (
    {"task": "binary", "dataset": "oxford_pet", "model": "unet", "num_classes": 2, "source": "outputs/application/binary/oxford_pet/unet/best.pt"},
    {"task": "binary", "dataset": "oxford_pet", "model": "fcn8", "num_classes": 2, "source": "outputs/application/binary/oxford_pet/fcn8/best.pt"},
    {"task": "binary", "dataset": "isic2017", "model": "unet", "num_classes": 2, "source": "outputs/application/binary/isic2017/unet/best.pt"},
    {"task": "binary", "dataset": "isic2017", "model": "fcn8", "num_classes": 2, "source": "outputs/application/binary/isic2017/fcn8/best.pt"},
    {"task": "binary", "dataset": "kvasir_seg", "model": "unet", "num_classes": 2, "source": "outputs/application/binary/kvasir_seg/unet/best.pt"},
    {"task": "binary", "dataset": "kvasir_seg", "model": "fcn8", "num_classes": 2, "source": "outputs/application/binary/kvasir_seg/fcn8/best.pt"},
    {"task": "multilabel_segmentation", "dataset": "voc2012", "model": "unet", "num_classes": 21, "source": "outputs/application/multilabel_segmentation/voc2012/unet/best.pt"},
    {"task": "multilabel_segmentation", "dataset": "voc2012", "model": "fcn8", "num_classes": 21, "source": "outputs/application/multilabel_segmentation/voc2012/fcn8/best.pt"},
    {"task": "multilabel_segmentation", "dataset": "cityscapes", "model": "unet", "num_classes": 19, "source": "outputs/application/multilabel_segmentation/cityscapes/unet/best.pt"},
    {"task": "multilabel_segmentation", "dataset": "cityscapes", "model": "fcn8", "num_classes": 19, "source": "outputs/application/multilabel_segmentation/cityscapes/fcn8/best.pt"},
)


def checkpoint_paths(item: Mapping[str, Any]) -> tuple[str, str]:
    relative = f"{item['task']}/{item['dataset']}/{item['model']}/best.pt"
    local = f"outputs/application/{relative}"
    return relative, local


def hf_download_command(item: Mapping[str, Any]) -> str:
    remote, _ = checkpoint_paths(item)
    return f"hf download {HF_REPO_ID} {remote} --local-dir outputs/application"


def is_model_state_dict(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and bool(payload)
        and all(isinstance(key, str) for key in payload)
        and all(isinstance(value, torch.Tensor) for value in payload.values())
    )


def extract_model_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if is_model_state_dict(payload):
        return payload
    if isinstance(payload, Mapping) and is_model_state_dict(payload.get("model_state_dict")):
        return payload["model_state_dict"]
    raise ValueError("Checkpoint does not contain a valid model state_dict.")


def infer_dataset_and_model(checkpoint_path: str | Path) -> tuple[str, str]:
    path = Path(checkpoint_path)
    if path.name != "best.pt" or len(path.parents) < 2:
        raise ValueError(f"Cannot infer dataset/model from checkpoint path: {path}")
    return path.parent.parent.name, path.parent.name
