from __future__ import annotations

from typing import Dict

import torch


def masked_index_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if targets.ndim != 3 or logits.shape[0] != targets.shape[0] or logits.shape[-2:] != targets.shape[-2:]:
        raise ValueError("Expected logits [B,C,H,W] and integer targets [B,H,W] with matching dimensions.")
    valid_mask = valid_mask.bool()
    safe_targets = targets.masked_fill(~valid_mask, 0).to(torch.int64)
    logits_float = logits.float()
    positive_logits = logits_float.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    per_pixel = torch.nn.functional.softplus(logits_float).sum(dim=1) - positive_logits
    denominator = (valid_mask.sum() * logits.shape[1]).clamp_min(1)
    return per_pixel.masked_select(valid_mask).sum() / denominator


class RunningMultilabelMetrics:
    def __init__(self, num_classes: int, threshold: float = 0.5) -> None:
        self.num_classes = int(num_classes)
        self.threshold = float(threshold)
        self.sums = {"macro_dice": 0.0, "macro_iou": 0.0, "micro_dice": 0.0, "micro_iou": 0.0}
        self.image_count = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> None:
        predictions = torch.sigmoid(logits.detach()) >= self.threshold
        targets = targets.detach().to(torch.int64)
        valid_mask = valid_mask.detach().bool()
        predictions = torch.logical_and(predictions, valid_mask.unsqueeze(1))

        for sample_index in range(predictions.shape[0]):
            pred_flat = predictions[sample_index].reshape(self.num_classes, -1)
            target_flat = targets[sample_index].reshape(-1)
            valid_flat = valid_mask[sample_index].reshape(-1)
            valid_targets = target_flat[valid_flat]
            if valid_targets.numel() == 0:
                raise ValueError("Cannot compute multilabel metrics for an image with no valid pixels.")
            truth_count = torch.bincount(valid_targets, minlength=self.num_classes).to(torch.float64)
            present_classes = truth_count > 0
            if not bool(present_classes.any().item()):
                raise ValueError("Cannot compute macro metrics for an image with no present ground-truth class.")
            predicted_count = pred_flat.sum(dim=1).to(torch.float64)
            selected_true_predictions = pred_flat[:, valid_flat].gather(0, valid_targets.unsqueeze(0)).squeeze(0)
            tp = torch.bincount(
                valid_targets,
                weights=selected_true_predictions.to(torch.float64),
                minlength=self.num_classes,
            )
            fp = predicted_count - tp
            fn = truth_count - tp
            # The paper's empirical convention averages only over classes that
            # are present in the ground-truth mask of this image.
            dice_by_class = (2 * tp) / (2 * tp + fp + fn).clamp_min(1)
            iou_by_class = tp / (tp + fp + fn).clamp_min(1)
            self.sums["macro_dice"] += float(
                dice_by_class[present_classes].mean().item()
            )
            self.sums["macro_iou"] += float(
                iou_by_class[present_classes].mean().item()
            )
            tp_micro, fp_micro, fn_micro = tp.sum(), fp.sum(), fn.sum()
            self.sums["micro_dice"] += float(
                ((2 * tp_micro) / (2 * tp_micro + fp_micro + fn_micro).clamp_min(1)).item()
            )
            self.sums["micro_iou"] += float(
                (tp_micro / (tp_micro + fp_micro + fn_micro).clamp_min(1)).item()
            )
            self.image_count += 1

    def compute_rounded(self, digits: int = 6) -> Dict[str, float]:
        return {key: round(value / max(self.image_count, 1), digits) for key, value in self.sums.items()}
