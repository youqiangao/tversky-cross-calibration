from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from application.binary.utils import resize_logits_to_size
from tversky_cross_calibration import predict_rank

from .conventions import MACRO_CLASS_CONVENTION
from .dataset import DATASET_CHOICES, default_data_root_for
from .engine import create_dataloader, load_trained_model
from .metrics import RunningMultilabelMetrics, masked_index_bce_with_logits
from .utils import ensure_dir, resolve_device, save_csv, save_json


DEFAULT_MODELS = ("unet", "fcn8")
OPTIMIZERS = ("macro-Dice", "macro-IoU", "micro-Dice", "micro-IoU")


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_original_target(loader, sample) -> torch.Tensor:
    with Image.open(sample.mask_path) as mask:
        target = loader.dataset.base_dataset._mask_array_to_target(np.asarray(mask))
    return torch.as_tensor(target, dtype=torch.int64)


def _rank_metrics_path(output_dir, dataset, model_label, split, max_samples) -> Path:
    sample_suffix = "all" if max_samples is None else f"max{max_samples}"
    filename = f"{split}_{sample_suffix}_original_{MACRO_CLASS_CONVENTION}_macro_rank_metrics.csv"
    return Path(output_dir) / dataset / model_label / filename


def _rows_match_protocol(rows, expected_resolution: str) -> bool:
    return bool(rows) and all(
        row.get("evaluation_resolution") == expected_resolution
        and row.get("macro_class_convention") == MACRO_CLASS_CONVENTION
        for row in rows
    )


@torch.no_grad()
def evaluate_rank_checkpoint(
    dataset, model_label, checkpoint_path, split, batch_size, num_workers, device,
    output_dir, max_samples=None, refit=False,
):
    if batch_size != 1:
        raise ValueError("Original-resolution rank evaluation requires batch_size=1.")
    expected_resolution = "original"
    csv_path = _rank_metrics_path(output_dir, dataset, model_label, split, max_samples)
    ensure_dir(csv_path.parent)
    if csv_path.exists() and not refit:
        rows = _read_rows(csv_path)
        if _rows_match_protocol(rows, expected_resolution):
            return rows
    model, checkpoint = load_trained_model(checkpoint_path, device)
    loader = create_dataloader(dataset, default_data_root_for(dataset), split, checkpoint["image_size"], batch_size, num_workers, False, False, max_samples)
    sample_lookup = {sample.image_id: sample for sample in loader.dataset.samples}
    counters = {name: RunningMultilabelMetrics(checkpoint["num_classes"], threshold=0.5) for name in OPTIMIZERS}
    loss_sum, image_count = 0.0, 0
    for batch in tqdm(loader, desc=f"rank:{dataset}:{model_label}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        logits = model(images)
        sample = sample_lookup[batch["image_id"][0]]
        targets = _load_original_target(loader, sample).unsqueeze(0).to(device, non_blocking=True)
        valid_mask = targets != int(checkpoint["ignore_index"])
        logits = resize_logits_to_size(logits, targets.shape[-2:])
        probabilities = torch.sigmoid(logits)
        loss_sum += float(masked_index_bce_with_logits(logits, targets, valid_mask).item()) * images.shape[0]
        image_count += images.shape[0]
        for name in OPTIMIZERS:
            aggregation, metric = name.lower().split("-")
            prediction = predict_rank(
                probabilities,
                metric=metric,
                aggregation=aggregation,
                valid_mask=valid_mask,
            )
            synthetic_logits = torch.where(prediction, 20.0, -20.0)
            counters[name].update(synthetic_logits, targets, valid_mask)
    rows = []
    for name in OPTIMIZERS:
        rows.append({
            "dataset_name": dataset, "model": model_label, "checkpoint": str(checkpoint_path),
            "split": split, "num_images": image_count, "optimizer": name,
            "evaluation_resolution": expected_resolution,
            "macro_class_convention": MACRO_CLASS_CONVENTION,
            "bce_loss": round(loss_sum / max(image_count, 1), 10), **counters[name].compute_rounded(10),
        })
    save_csv(rows, csv_path)
    save_json({"records": rows}, csv_path.with_suffix(".json"))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank-based evaluation for independent-channel semantic segmentation.")
    parser.add_argument("--dataset", action="append", choices=DATASET_CHOICES, default=[])
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS, default=[])
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/application/multilabel_segmentation_rank_eval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    all_rows = []
    for dataset in args.dataset or DATASET_CHOICES:
        for model in args.model or DEFAULT_MODELS:
            checkpoint = Path("outputs/application/multilabel_segmentation") / dataset / model / "best.pt"
            if checkpoint.exists():
                all_rows.extend(evaluate_rank_checkpoint(
                    dataset, model, checkpoint, args.split, args.batch_size, args.num_workers,
                    device, args.output_dir, args.max_samples, args.refit,
                ))
    print({"records": all_rows})


if __name__ == "__main__":
    main()
