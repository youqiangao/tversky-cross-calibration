"""Paper-only VOC 2012 and Cityscapes dataset readers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CITYSCAPES_ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
    32: 17, 33: 18,
}


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    image_path: Path
    mask_path: Path


class PaperSemanticDataset(Dataset):
    ignore_index = 255

    def __init__(self, root, split, image_size, augment=False, normalize=True, max_samples=None):
        self.root = Path(root)
        self.split = split.lower()
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        self.max_samples = max_samples
        self.samples = self._limit(self._build_samples())

    def _build_samples(self):
        raise NotImplementedError

    def _limit(self, samples):
        samples = list(samples)
        if not samples:
            raise RuntimeError(f"No samples found for {self.split!r} under {self.root}.")
        return samples if self.max_samples is None else samples[: int(self.max_samples)]

    def _mask_array_to_target(self, array):
        if array.ndim == 3:
            array = array[..., 0]
        return array.astype(np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path)
        original_size = (image.height, image.width)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        image_tensor = torch.from_numpy(
            (np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0).copy()
        )
        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        target = torch.from_numpy(self._mask_array_to_target(np.asarray(mask)).copy()).long()
        return {"image": image_tensor, "mask": target, "image_id": sample.image_id, "original_size": original_size}

    def _partition_train(self, samples):
        shuffled = list(samples)
        random.Random(42).shuffle(shuffled)
        validation_count = int(math.ceil(len(shuffled) * 0.1))
        chosen = shuffled[validation_count:] if self.split == "train" else shuffled[:validation_count]
        return sorted(chosen, key=lambda item: item.image_id)


class VOC2012Dataset(PaperSemanticDataset):
    num_classes = 21

    def _build_samples(self):
        if self.split not in {"train", "val", "test"}:
            raise ValueError("VOC split must be train, val, or test.")
        root = self.root / "VOCdevkit" / "VOC2012"
        official_split = "train" if self.split in {"train", "val"} else "val"
        ids_path = root / "ImageSets" / "Segmentation" / f"{official_split}.txt"
        ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
        samples = [SampleRecord(i, root / "JPEGImages" / f"{i}.jpg", root / "SegmentationClass" / f"{i}.png") for i in ids]
        return self._partition_train(samples) if self.split in {"train", "val"} else samples


class CityscapesDataset(PaperSemanticDataset):
    num_classes = 19

    def _build_samples(self):
        if self.split not in {"train", "val", "test"}:
            raise ValueError("Cityscapes split must be train, val, or test.")
        official_split = "train" if self.split in {"train", "val"} else "val"
        images_root = self.root / "leftImg8bit" / official_split
        masks_root = self.root / "gtFine" / official_split
        samples = []
        for image_path in sorted(images_root.glob("*/*_leftImg8bit.png")):
            image_id = image_path.stem.replace("_leftImg8bit", "")
            mask = masks_root / image_path.parent.name / f"{image_id}_gtFine_labelIds.png"
            samples.append(SampleRecord(image_id, image_path, mask))
        return self._partition_train(samples) if self.split in {"train", "val"} else samples

    def _mask_array_to_target(self, array):
        if array.ndim == 3:
            array = array[..., 0]
        target = np.full(array.shape, self.ignore_index, dtype=np.int64)
        for source_id, train_id in CITYSCAPES_ID_TO_TRAIN_ID.items():
            target[array == source_id] = train_id
        return target


def build_semantic_dataset(dataset_name, **kwargs):
    if dataset_name == "voc2012":
        return VOC2012Dataset(**kwargs)
    if dataset_name == "cityscapes":
        return CityscapesDataset(**kwargs)
    raise ValueError(f"Unsupported paper dataset: {dataset_name!r}.")
