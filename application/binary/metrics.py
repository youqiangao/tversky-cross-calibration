from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch


@dataclass
class SegmentationMetrics:
    pixel_accuracy: float
    dice: float
    iou: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class RunningSegmentationMetrics:
    def __init__(self, ignore_index: int | None = 255) -> None:
        self.ignore_index = ignore_index
        self.total_correct = 0
        self.total_pixels = 0
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.image_count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = torch.argmax(logits, dim=1)
        valid_mask = torch.ones_like(targets, dtype=torch.bool) if self.ignore_index is None else targets != self.ignore_index
        if not torch.any(valid_mask):
            return

        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        self.total_correct += (valid_predictions == valid_targets).sum().item()
        self.total_pixels += valid_targets.numel()

        for prediction, target, mask in zip(predictions, targets, valid_mask):
            if not torch.any(mask):
                continue
            pred_fg = (prediction[mask] == 1).float()
            target_fg = (target[mask] == 1).float()
            intersection = (pred_fg * target_fg).sum()
            pred_sum = pred_fg.sum()
            target_sum = target_fg.sum()
            union = pred_sum + target_sum - intersection
            dice_denominator = pred_sum + target_sum
            dice_value = torch.where(dice_denominator > 0, 2.0 * intersection / dice_denominator, torch.zeros_like(intersection))
            iou_value = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
            self.dice_sum += float(dice_value.item())
            self.iou_sum += float(iou_value.item())
            self.image_count += 1

    def compute(self) -> SegmentationMetrics:
        pixel_accuracy = self.total_correct / max(self.total_pixels, 1)
        dice = self.dice_sum / max(self.image_count, 1)
        iou = self.iou_sum / max(self.image_count, 1)
        return SegmentationMetrics(pixel_accuracy=float(pixel_accuracy), dice=float(dice), iou=float(iou))

    def compute_rounded(self, digits: int = 6) -> Dict[str, float]:
        return {key: round(value, digits) for key, value in self.compute().to_dict().items()}
