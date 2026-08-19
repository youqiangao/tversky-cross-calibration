from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm

from tversky_cross_calibration import predict_rank

from .dataset import DATASET_CHOICES, default_data_root_for
from .engine import create_dataloader, load_trained_model
from .utils import (
    ensure_dir,
    logits_to_probabilities,
    resize_logits_to_size,
    resolve_device,
    save_csv,
    save_json,
)


DEFAULT_MODEL_NAMES = ("unet", "fcn8")
OPTIMIZERS = {"RankDice": "macro-Dice", "RankIoU": "macro-IoU"}


class BinaryCounts:
    def __init__(self) -> None:
        self.correct = 0
        self.pixels = 0
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.image_count = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().bool().reshape(-1)
        target = target.detach().bool().reshape(-1)
        self.correct += int((prediction == target).sum().item())
        self.pixels += int(target.numel())

        intersection = torch.logical_and(prediction, target).sum().to(torch.float64)
        pred_sum = prediction.sum().to(torch.float64)
        target_sum = target.sum().to(torch.float64)
        union = pred_sum + target_sum - intersection
        self.dice_sum += float((2.0 * intersection / (pred_sum + target_sum).clamp_min(1.0)).item())
        self.iou_sum += float((intersection / union.clamp_min(1.0)).item())
        self.image_count += 1

    def compute(self) -> Dict[str, float]:
        return {
            "pixel_accuracy": round(self.correct / max(self.pixels, 1), 10),
            "dice": round(self.dice_sum / max(self.image_count, 1), 10),
            "iou": round(self.iou_sum / max(self.image_count, 1), 10),
        }


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_path(dataset_name: str, model_name: str) -> Path | None:
    current = Path("outputs/application/binary") / dataset_name / model_name / "best.pt"
    if current.exists():
        return current
    if dataset_name == "oxford_pet":
        legacy = Path("outputs/oxford_pet") / model_name / "best.pt"
        if legacy.exists():
            return legacy
    return None


def resolved_model_paths(dataset_name: str, requested_models: Sequence[str]) -> List[tuple[str, Path]]:
    model_names = tuple(requested_models) if requested_models else DEFAULT_MODEL_NAMES
    resolved: List[tuple[str, Path]] = []
    for model_name in model_names:
        checkpoint = _checkpoint_path(dataset_name, model_name)
        if checkpoint is not None:
            resolved.append((model_name, checkpoint))
    return resolved


@torch.no_grad()
def evaluate_checkpoint(
    *,
    dataset_name: str,
    model_label: str,
    checkpoint_path: Path,
    split: str,
    num_workers: int,
    device: torch.device,
    output_dir: Path,
    max_samples: int | None,
    refit: bool,
) -> List[Dict[str, Any]]:
    sample_suffix = "all" if max_samples is None else f"max{int(max_samples)}"
    result_dir = ensure_dir(output_dir / dataset_name / model_label)
    csv_path = result_dir / f"{split}_{sample_suffix}_full_rank_metrics.csv"
    if csv_path.exists() and not refit:
        return _read_rows(csv_path)

    model, checkpoint = load_trained_model(checkpoint_path, device)
    loader = create_dataloader(
        dataset_name=dataset_name,
        data_root=default_data_root_for(dataset_name),
        split=split,
        image_size=int(checkpoint["image_size"]),
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        augment=False,
        val_split=float(checkpoint.get("val_split", 0.1)),
        split_seed=int(checkpoint.get("split_seed", 42)),
        max_samples=max_samples,
    )
    sample_lookup = {sample.image_id: sample for sample in loader.dataset.samples}
    counters = {name: BinaryCounts() for name in OPTIMIZERS}
    image_count = 0

    progress = tqdm(loader, desc=f"binary-rank:{dataset_name}:{model_label}:{split}", leave=False)
    for batch in progress:
        image_id = batch["image_id"][0]
        sample = sample_lookup[image_id]
        if sample.mask_path is None:
            raise RuntimeError(f"Original mask is unavailable for {dataset_name}:{split}:{image_id}.")

        target = torch.as_tensor(
            loader.dataset.load_binary_mask_from_path(sample.mask_path),
            dtype=torch.bool,
            device=device,
        )
        logits = model(batch["image"].to(device, non_blocking=True))
        full_logits = resize_logits_to_size(logits, tuple(target.shape))
        foreground_probability = logits_to_probabilities(full_logits)[:, 1:2]

        for optimizer_name, rank_name in OPTIMIZERS.items():
            prediction = predict_rank(
                foreground_probability,
                metric="dice" if rank_name.endswith("Dice") else "iou",
                aggregation="macro",
            )[0, 0]
            counters[optimizer_name].update(prediction, target)
        image_count += 1

    rows: List[Dict[str, Any]] = []
    for optimizer_name, metrics in counters.items():
        rows.append(
            {
                "dataset_name": dataset_name,
                "model": model_label,
                "checkpoint": str(checkpoint_path),
                "split": split,
                "num_images": image_count,
                "training_image_size": int(checkpoint["image_size"]),
                "evaluation_resolution": "original_mask",
                "optimizer": optimizer_name,
                **metrics.compute(),
            }
        )

    save_csv(rows, csv_path)
    save_json({"rows": rows}, csv_path.with_suffix(".json"))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full-resolution RankDice/RankIoU evaluation for binary segmentation.")
    parser.add_argument("--dataset", action="append", choices=DATASET_CHOICES, default=[])
    parser.add_argument("--model", action="append", choices=DEFAULT_MODEL_NAMES, default=[])
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/application/binary_rank_eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    all_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dataset_name in args.dataset or DATASET_CHOICES:
        paths = resolved_model_paths(dataset_name, args.model)
        if not paths:
            missing.append(dataset_name)
        for model_label, checkpoint_path in paths:
            all_rows.extend(
                evaluate_checkpoint(
                    dataset_name=dataset_name,
                    model_label=model_label,
                    checkpoint_path=checkpoint_path,
                    split=args.split,
                    num_workers=args.num_workers,
                    device=device,
                    output_dir=output_dir,
                    max_samples=args.max_samples,
                    refit=args.refit,
                )
            )

    summary_path = ensure_dir(output_dir) / f"{args.split}_full_rank_metrics_summary.csv"
    save_csv(all_rows, summary_path)
    save_json({"rows": all_rows, "datasets_without_checkpoints": missing}, summary_path.with_suffix(".json"))
    print({"summary_path": str(summary_path), "rows": all_rows, "datasets_without_checkpoints": missing})


if __name__ == "__main__":
    main()
