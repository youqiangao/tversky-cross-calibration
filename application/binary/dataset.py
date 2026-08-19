from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from tversky_cross_calibration.config import dataset_config, paper_config


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASET_CHOICES = ("isic2017", "oxford_pet", "kvasir_seg")
DEFAULT_DATA_ROOTS: Dict[str, str] = {
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


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    image_path: Path
    mask_path: Path | None


class BinarySegmentationDataset(Dataset):
    num_classes = 2
    ignore_index = 255

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        normalize: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        self.image_size = int(image_size)
        self.augment = augment
        self.normalize = normalize
        self.max_samples = None if max_samples is None else int(max_samples)
        self.samples = self._limit_samples(self._build_samples())

    def _build_samples(self) -> List[SampleRecord]:
        raise NotImplementedError

    def _limit_samples(self, samples: Sequence[SampleRecord]) -> List[SampleRecord]:
        items = list(samples)
        if not items:
            raise RuntimeError(f"No samples found for split '{self.split}' under {self.root}.")
        if self.max_samples is not None:
            if self.max_samples <= 0:
                raise ValueError(f"max_samples must be positive when provided, got {self.max_samples}.")
            items = items[: self.max_samples]
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        return image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

    def _resize_mask(self, mask: Image.Image) -> Image.Image:
        return mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

    def _random_flip(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def _random_vertical_flip(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def _random_rotate(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        angle = random.choice((0, 90, 180, 270))
        if angle == 0:
            return image, mask
        return image.rotate(angle), mask.rotate(angle)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.tensor(array.tolist(), dtype=torch.float32)
        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor

    def _mask_to_tensor(self, mask: Image.Image) -> torch.Tensor:
        array = np.asarray(mask, dtype=np.uint8)
        if array.ndim == 3:
            array = array[..., 0]
        binary = self._mask_array_to_binary(array)
        return torch.tensor(binary.tolist(), dtype=torch.int64)

    def _mask_array_to_binary(self, array: np.ndarray) -> np.ndarray:
        return (array > 0).astype(np.int64)

    def load_binary_mask_from_path(self, mask_path: str | Path) -> np.ndarray:
        array = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if array.ndim == 3:
            array = array[..., 0]
        return self._mask_array_to_binary(array).astype(np.uint8)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        original_size = (image.height, image.width)
        mask = Image.open(sample.mask_path) if sample.mask_path is not None else Image.fromarray(np.zeros(original_size, dtype=np.uint8))

        if self.augment:
            image, mask = self._random_flip(image, mask)
            if self.split in {"train"} and isinstance(self, (ISIC2017Dataset, KvasirSEGDataset)):
                image, mask = self._random_vertical_flip(image, mask)
                image, mask = self._random_rotate(image, mask)

        return {
            "image": self._image_to_tensor(self._resize_image(image)),
            "mask": self._mask_to_tensor(self._resize_mask(mask)),
            "image_id": sample.image_id,
            "original_size": original_size,
        }


class ISIC2017Dataset(BinarySegmentationDataset):
    SPLIT_CONFIG: Dict[str, Tuple[str, str | None]] = {
        "train": ("ISIC-2017_Training_Data", "ISIC-2017_Training_Part1_GroundTruth"),
        "val": ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth"),
        "test": ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth"),
    }

    def _build_samples(self) -> List[SampleRecord]:
        if self.split not in self.SPLIT_CONFIG:
            raise ValueError(f"Unsupported split '{self.split}'. Expected one of {sorted(self.SPLIT_CONFIG)}.")
        image_dir_name, mask_dir_name = self.SPLIT_CONFIG[self.split]
        image_dir = self.root / image_dir_name
        mask_dir = self.root / mask_dir_name if mask_dir_name is not None else None
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if mask_dir is not None and not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        samples: List[SampleRecord] = []
        for image_path in sorted(image_dir.glob("ISIC_*.jpg")):
            image_id = image_path.stem
            mask_path = None if mask_dir is None else mask_dir / f"{image_id}_segmentation.png"
            if mask_path is not None and not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_id}: {mask_path}")
            samples.append(SampleRecord(image_id=image_id, image_path=image_path, mask_path=mask_path))
        return samples


class OxfordIIITPetDataset(BinarySegmentationDataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        normalize: bool = True,
        max_samples: int | None = None,
        val_split: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        self.val_split = float(val_split)
        self.split_seed = int(split_seed)
        super().__init__(root, split, image_size, augment, normalize, max_samples)

    def _annotation_ids(self, split_file: Path) -> List[str]:
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        image_ids: List[str] = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                image_ids.append(stripped.split()[0])
        if not image_ids:
            raise RuntimeError(f"No image ids found in split file: {split_file}")
        return image_ids

    def _split_trainval_ids(self, image_ids: Sequence[str]) -> List[str]:
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(f"val_split must be between 0 and 1, got {self.val_split}.")
        shuffled = list(image_ids)
        rng = random.Random(self.split_seed)
        rng.shuffle(shuffled)
        val_count = max(1, int(math.ceil(len(shuffled) * self.val_split)))
        return sorted(shuffled[:val_count] if self.split == "val" else shuffled[val_count:])

    def _build_samples(self) -> List[SampleRecord]:
        images_dir = self.root / "images"
        trimaps_dir = self.root / "annotations" / "trimaps"
        annotations_dir = self.root / "annotations"
        if not images_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")
        if not trimaps_dir.exists():
            raise FileNotFoundError(f"Trimap directory not found: {trimaps_dir}")

        if self.split in {"train", "val"}:
            image_ids = self._split_trainval_ids(self._annotation_ids(annotations_dir / "trainval.txt"))
        elif self.split == "test":
            image_ids = sorted(self._annotation_ids(annotations_dir / "test.txt"))
        else:
            raise ValueError("Oxford-IIIT Pet split must be one of 'train', 'val', or 'test'.")

        samples: List[SampleRecord] = []
        for image_id in image_ids:
            image_path = images_dir / f"{image_id}.jpg"
            mask_path = trimaps_dir / f"{image_id}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing Oxford-IIIT Pet image: {image_path}")
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing Oxford-IIIT Pet trimap: {mask_path}")
            samples.append(SampleRecord(image_id=image_id, image_path=image_path, mask_path=mask_path))
        return samples

    def _mask_array_to_binary(self, array: np.ndarray) -> np.ndarray:
        if array.ndim == 3:
            array = array[..., 0]
        return np.isin(array, (1, 3)).astype(np.int64)


class KvasirSEGDataset(BinarySegmentationDataset):
    def _build_samples(self) -> List[SampleRecord]:
        if self.split not in {"train", "val", "test"}:
            raise ValueError("Kvasir-SEG split must be one of 'train', 'val', or 'test'.")
        image_dir = self.root / "paper_split" / self.split / "images"
        mask_dir = self.root / "paper_split" / self.split / "masks"
        if not image_dir.exists():
            raise FileNotFoundError(f"Kvasir image directory not found: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Kvasir mask directory not found: {mask_dir}")

        samples: List[SampleRecord] = []
        for image_path in sorted(image_dir.glob("*.jpg")):
            image_id = image_path.stem
            mask_path = mask_dir / f"{image_id}.jpg"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing Kvasir mask for {image_id}: {mask_path}")
            samples.append(SampleRecord(image_id=image_id, image_path=image_path, mask_path=mask_path))
        return samples


def normalize_dataset_name(dataset_name: str | None) -> str:
    name = "isic2017" if dataset_name is None else dataset_name.lower()
    if name not in DATASET_CHOICES:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of {DATASET_CHOICES}.")
    return name


def default_data_root_for(dataset_name: str | None) -> str:
    return DEFAULT_DATA_ROOTS[normalize_dataset_name(dataset_name)]


def default_training_config_for(dataset_name: str | None) -> Dict[str, float | int]:
    return dict(DEFAULT_TRAINING_CONFIGS[normalize_dataset_name(dataset_name)])


def build_dataset(
    dataset_name: str,
    root: str | Path,
    split: str,
    image_size: int = 256,
    augment: bool = False,
    normalize: bool = True,
    max_samples: int | None = None,
    val_split: float = 0.1,
    split_seed: int = 42,
) -> Dataset:
    name = normalize_dataset_name(dataset_name)
    if name == "isic2017":
        return ISIC2017Dataset(root=root, split=split, image_size=image_size, augment=augment, normalize=normalize, max_samples=max_samples)
    if name == "oxford_pet":
        return OxfordIIITPetDataset(
            root=root,
            split=split,
            image_size=image_size,
            augment=augment,
            normalize=normalize,
            max_samples=max_samples,
            val_split=val_split,
            split_seed=split_seed,
        )
    return KvasirSEGDataset(root=root, split=split, image_size=image_size, augment=augment, normalize=normalize, max_samples=max_samples)
