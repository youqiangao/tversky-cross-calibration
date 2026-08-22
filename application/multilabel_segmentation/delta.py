from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from application.binary.utils import resize_logits_to_size

from .dataset import DATASET_CHOICES, build_dataset, default_data_root_for, normalize_dataset_name
from .engine import create_dataloader, load_trained_model
from .utils import ensure_dir, resolve_device, save_csv, save_json


DEFAULT_MODELS = ("unet", "fcn8")
DELTA_VARIANT = "present_macro_mean_class_scaled_micro"


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_original_target(dataset, mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as mask:
        return dataset.base_dataset._mask_array_to_target(np.asarray(mask))


def _summarize(rows, model, checkpoint, dataset, split, eta, num_classes, cached):
    return {
        "model": model,
        "checkpoint": str(checkpoint),
        "dataset_name": dataset,
        "split": split,
        "num_images": len(rows),
        "num_classes": num_classes,
        "avg_present_classes": round(float(np.mean([float(row["num_present_classes"]) for row in rows])), 6),
        "avg_dimension_d": round(float(np.mean([float(row["dimension_d"]) for row in rows])), 6),
        "eta": float(eta),
        "delta_macro_hat": round(float(np.mean([float(row["delta_macro"]) for row in rows])), 10),
        "delta_micro_hat": round(float(np.mean([float(row["delta_micro"]) for row in rows])), 10),
        "cached": cached,
    }


@torch.no_grad()
def compute_checkpoint(checkpoint_path, model_label, dataset_name, data_root, split, batch_size, num_workers, device, output_dir, eta=12.0, max_samples=None, refit=False):
    dataset_name = normalize_dataset_name(dataset_name)
    model, checkpoint = load_trained_model(checkpoint_path, device)
    num_classes = int(checkpoint["num_classes"])
    suffix = "all" if max_samples is None else f"max{max_samples}"
    details_path = ensure_dir(output_dir) / f"{model_label}_{split}_{suffix}_{DELTA_VARIANT}_details.csv"
    if details_path.exists() and not refit:
        rows = _read_rows(details_path)
        if rows and all(abs(float(row["eta"]) - eta) < 1e-12 for row in rows):
            return _summarize(rows, model_label, checkpoint_path, dataset_name, split, eta, num_classes, True)
    loader = create_dataloader(dataset_name, data_root, split, checkpoint["image_size"], batch_size, num_workers, False, False, max_samples)
    lookup = {sample.image_id: sample for sample in loader.dataset.samples}
    rows: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"delta:{dataset_name}:{model_label}", leave=False):
        images = batch["image"].to(device)
        logits = model(images)
        for index, image_id in enumerate(batch["image_id"]):
            height = int(batch["original_size"][0][index])
            width = int(batch["original_size"][1][index])
            sample = lookup[image_id]
            target = _load_original_target(loader.dataset, sample.mask_path)
            valid_np = target != loader.dataset.ignore_index
            valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
            probabilities = torch.sigmoid(resize_logits_to_size(logits[index : index + 1], (height, width)))[0]
            class_sums = probabilities[:, valid].sum(dim=1)
            present = [int(value) for value in np.unique(target[valid_np]) if 0 <= int(value) < num_classes]
            present_sums = class_sums[torch.as_tensor(present, device=device)]
            macro = float((1 / torch.maximum(present_sums, present_sums.new_tensor(eta))).mean().item())
            total = class_sums.sum()
            micro = float((num_classes / torch.maximum(total, total.new_tensor(num_classes * eta))).item())
            rows.append({
                "family": "multilabel_segmentation", "dataset_name": dataset_name, "model": model_label,
                "checkpoint": str(checkpoint_path), "split": split, "image_id": image_id,
                "height": height, "width": width, "num_classes": num_classes,
                "num_present_classes": len(present), "dimension_d": int(valid.sum().item()),
                "class_prob_sum_total": round(float(total.item()), 6),
                "macro_class_sum_min": round(float(present_sums.min().item()), 6),
                "macro_class_sum_max": round(float(present_sums.max().item()), 6),
                "eta": eta, "delta_macro": round(macro, 10), "delta_micro": round(micro, 10),
            })
    save_csv(rows, details_path)
    save_json({"records": rows}, details_path.with_suffix(".json"))
    return _summarize(rows, model_label, checkpoint_path, dataset_name, split, eta, num_classes, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Delta for independent-channel semantic segmentation.")
    parser.add_argument("--dataset", action="append", choices=DATASET_CHOICES, default=[])
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS, default=[])
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eta", type=float, default=12.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    datasets = args.dataset or list(DATASET_CHOICES)
    models = args.model or list(DEFAULT_MODELS)
    device = resolve_device(args.device)
    all_summaries = []
    for dataset in datasets:
        output_dir = Path("outputs/application/multilabel_segmentation") / dataset / "delta"
        summaries = []
        for model in models:
            checkpoint = Path("outputs/application/multilabel_segmentation") / dataset / model / "best.pt"
            if checkpoint.exists():
                summaries.append(compute_checkpoint(checkpoint, model, dataset, default_data_root_for(dataset), args.split, args.batch_size, args.num_workers, device, output_dir, args.eta, args.max_samples, args.refit))
        save_csv(summaries, output_dir / "delta_summary.csv")
        save_json({"summaries": summaries}, output_dir / "delta_summary.json")
        all_summaries.extend(summaries)
    print({"summaries": all_summaries})


if __name__ == "__main__":
    main()
