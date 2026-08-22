from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from application.binary.dataset import DATASET_CHOICES, build_dataset, default_data_root_for, normalize_dataset_name
from application.checkpoints import infer_dataset_and_model
from application.binary.engine import create_dataloader, load_trained_model
from application.binary.utils import ensure_dir, load_checkpoint, logits_to_probabilities, resize_logits_to_size, resolve_device, save_csv, save_json


DEFAULT_MODEL_NAMES = ("unet", "fcn8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute binary real-application evaluation Delta.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--model-label", action="append", default=[])
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--eta", type=float, default=12.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--refit", action="store_true")
    return parser


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: Dict[str, Any], key: str) -> float:
    return float(row[key])


def _rows_match_eta(rows: Sequence[Dict[str, Any]], eta: float) -> bool:
    return bool(rows) and all(abs(_as_float(row, "eta") - float(eta)) < 1e-12 for row in rows)


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "model"


def _details_path(output_dir: Path, model_label: str, split: str, max_samples: int | None) -> Path:
    sample_suffix = "_all" if max_samples is None else f"_max{int(max_samples)}"
    return output_dir / f"{_safe_label(model_label)}_{split}{sample_suffix}_details.csv"


def _summary_from_details(
    rows: Sequence[Dict[str, Any]],
    *,
    model_label: str,
    checkpoint_path: str,
    dataset_name: str,
    split: str,
    eta: float,
    cached: bool,
) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError(f"No Delta rows available for {model_label} on {dataset_name}:{split}.")
    return {
        "model": model_label,
        "checkpoint": checkpoint_path,
        "dataset_name": dataset_name,
        "split": split,
        "num_images": len(rows),
        "avg_dimension_d": round(float(np.mean([_as_float(row, "dimension_d") for row in rows])), 6),
        "eta": float(eta),
        "delta_hat": round(float(np.mean([_as_float(row, "delta") for row in rows])), 10),
        "cached": bool(cached),
    }


def _resolve_requested_models(args: argparse.Namespace, dataset_name: str) -> Sequence[tuple[str, str]]:
    if args.checkpoint:
        labels = list(args.model_label)
        if labels and len(labels) != len(args.checkpoint):
            raise ValueError("When --model-label is provided, its count must match the number of --checkpoint values.")
        resolved: List[tuple[str, str]] = []
        for index, checkpoint in enumerate(args.checkpoint):
            label = labels[index] if labels else f"{Path(checkpoint).parent.name}_{Path(checkpoint).stem}"
            resolved.append((label, checkpoint))
        return resolved

    resolved = []
    for model_name in DEFAULT_MODEL_NAMES:
        checkpoint = Path("outputs/application/binary") / dataset_name / model_name / "best.pt"
        if checkpoint.exists():
            resolved.append((model_name, str(checkpoint)))
    return resolved


@torch.no_grad()
def compute_delta_for_checkpoint(
    checkpoint_path: str | Path,
    model_label: str,
    dataset_name: str,
    data_root: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dir: str | Path,
    val_split: float | None = None,
    split_seed: int | None = None,
    eta: float = 12.0,
    max_samples: int | None = None,
    refit: bool = False,
) -> Dict[str, Any]:
    dataset_name = normalize_dataset_name(dataset_name)
    output_dir = ensure_dir(output_dir)
    details_path = _details_path(output_dir, model_label, split, max_samples)
    checkpoint_str = str(checkpoint_path)

    if details_path.exists() and not refit:
        rows = read_csv_rows(details_path)
        if _rows_match_eta(rows, eta):
            return _summary_from_details(
                rows,
                model_label=model_label,
                checkpoint_path=checkpoint_str,
                dataset_name=dataset_name,
                split=split,
                eta=eta,
                cached=True,
            )

    model, checkpoint = load_trained_model(checkpoint_path, device)
    resolved_val_split = float(checkpoint.get("val_split", 0.1) if val_split is None else val_split)
    resolved_split_seed = int(checkpoint.get("split_seed", 42) if split_seed is None else split_seed)
    loader = create_dataloader(
        dataset_name=dataset_name,
        data_root=data_root,
        split=split,
        image_size=int(checkpoint["image_size"]),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        augment=False,
        val_split=resolved_val_split,
        split_seed=resolved_split_seed,
        max_samples=max_samples,
    )
    sample_lookup = {sample.image_id: sample for sample in loader.dataset.samples}
    details: List[Dict[str, Any]] = []

    progress = tqdm(loader, desc=f"delta:{dataset_name}:{model_label}:{split}", leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        logits = model(images)
        image_ids = batch["image_id"]
        original_sizes = batch["original_size"]

        for index in range(images.size(0)):
            image_id = image_ids[index]
            height = int(original_sizes[0][index])
            width = int(original_sizes[1][index])
            sample_logits = resize_logits_to_size(logits[index : index + 1], (height, width))
            foreground_probabilities = logits_to_probabilities(sample_logits)[0, 1]
            sample = sample_lookup[image_id]
            if sample.mask_path is None:
                valid_mask = torch.ones((height, width), dtype=torch.bool, device=foreground_probabilities.device)
            else:
                mask_array = loader.dataset.load_binary_mask_from_path(sample.mask_path)
                valid_mask = torch.tensor(mask_array != loader.dataset.ignore_index, dtype=torch.bool, device=foreground_probabilities.device)
            dimension_d = int(valid_mask.sum().item())
            prob_sum = float(foreground_probabilities[valid_mask].sum().item())
            s_value = max(prob_sum, float(eta))
            details.append(
                {
                    "family": "binary",
                    "dataset_name": dataset_name,
                    "model": model_label,
                    "checkpoint": checkpoint_str,
                    "split": split,
                    "image_id": image_id,
                    "height": height,
                    "width": width,
                    "dimension_d": dimension_d,
                    "prob_sum": round(prob_sum, 6),
                    "s_value": round(s_value, 6),
                    "eta": float(eta),
                    "delta": round(1.0 / s_value, 10),
                }
            )

    save_csv(details, details_path)
    save_json({"records": details}, details_path.with_suffix(".json"))
    return _summary_from_details(
        details,
        model_label=model_label,
        checkpoint_path=checkpoint_str,
        dataset_name=dataset_name,
        split=split,
        eta=eta,
        cached=False,
    )


def compute_mask_count_baseline(
    dataset_name: str,
    data_root: str | Path,
    split: str,
    output_dir: str | Path,
    val_split: float = 0.1,
    split_seed: int = 42,
    eta: float = 12.0,
    max_samples: int | None = None,
    refit: bool = False,
) -> Dict[str, Any]:
    dataset_name = normalize_dataset_name(dataset_name)
    output_dir = ensure_dir(output_dir)
    model_label = "count_ones_baseline"
    details_path = _details_path(output_dir, model_label, split, max_samples)
    if details_path.exists() and not refit:
        rows = read_csv_rows(details_path)
        if _rows_match_eta(rows, eta):
            return _summary_from_details(
                rows,
                model_label=model_label,
                checkpoint_path="",
                dataset_name=dataset_name,
                split=split,
                eta=eta,
                cached=True,
            )

    dataset = build_dataset(
        dataset_name=dataset_name,
        root=data_root,
        split=split,
        image_size=256,
        augment=False,
        normalize=False,
        max_samples=max_samples,
        val_split=val_split,
        split_seed=split_seed,
    )
    details: List[Dict[str, Any]] = []
    for sample in tqdm(dataset.samples, desc=f"delta:{dataset_name}:count:{split}", leave=False):
        with Image.open(sample.image_path) as image:
            height, width = image.height, image.width
        if sample.mask_path is None:
            raise RuntimeError(f"Mask-count baseline requires masks, but none were found for {sample.image_id}.")
        mask_array = dataset.load_binary_mask_from_path(sample.mask_path)
        dimension_d = int((mask_array != dataset.ignore_index).sum())
        ones_count = int((mask_array == 1).sum())
        s_value = max(float(ones_count), float(eta))
        details.append(
            {
                "family": "binary",
                "dataset_name": dataset_name,
                "model": model_label,
                "checkpoint": "",
                "split": split,
                "image_id": sample.image_id,
                "height": height,
                "width": width,
                "dimension_d": dimension_d,
                "prob_sum": ones_count,
                "s_value": round(s_value, 6),
                "eta": float(eta),
                "delta": round(1.0 / s_value, 10),
            }
        )

    save_csv(details, details_path)
    save_json({"records": details}, details_path.with_suffix(".json"))
    return _summary_from_details(
        details,
        model_label=model_label,
        checkpoint_path="",
        dataset_name=dataset_name,
        split=split,
        eta=eta,
        cached=False,
    )


def run_for_dataset(args: argparse.Namespace, dataset_name: str, device: torch.device) -> List[Dict[str, Any]]:
    data_root = args.data_root or default_data_root_for(dataset_name)
    output_dir = ensure_dir(args.output_dir or Path("outputs/application/binary") / dataset_name / "delta")
    val_split = 0.1 if args.val_split is None else float(args.val_split)
    split_seed = 42 if args.split_seed is None else int(args.split_seed)
    summaries: List[Dict[str, Any]] = []
    model_items = _resolve_requested_models(args, dataset_name)
    if args.checkpoint and not model_items:
        raise ValueError("No checkpoints were resolved.")

    for model_label, checkpoint in model_items:
        summaries.append(
            compute_delta_for_checkpoint(
                checkpoint_path=checkpoint,
                model_label=model_label,
                dataset_name=dataset_name,
                data_root=data_root,
                split=args.split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                output_dir=output_dir,
                val_split=val_split,
                split_seed=split_seed,
                eta=args.eta,
                max_samples=args.max_samples,
                refit=args.refit,
            )
        )

    summaries.append(
        compute_mask_count_baseline(
            dataset_name=dataset_name,
            data_root=data_root,
            split=args.split,
            output_dir=output_dir,
            val_split=val_split,
            split_seed=split_seed,
            eta=args.eta,
            max_samples=args.max_samples,
            refit=args.refit,
        )
    )
    save_csv(summaries, output_dir / "delta_summary.csv")
    save_json({"summaries": summaries}, output_dir / "delta_summary.json")
    return summaries


def infer_dataset_from_args(args: argparse.Namespace) -> Sequence[str]:
    if args.dataset is not None:
        return [normalize_dataset_name(args.dataset)]
    if args.checkpoint:
        payload = load_checkpoint(args.checkpoint[0], map_location="cpu")
        dataset_name = payload.get("dataset_name") or infer_dataset_and_model(args.checkpoint[0])[0]
        return [normalize_dataset_name(dataset_name)]
    return DATASET_CHOICES


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    all_summaries: List[Dict[str, Any]] = []
    for dataset_name in infer_dataset_from_args(args):
        all_summaries.extend(run_for_dataset(args, dataset_name, device))
    print({"summaries": all_summaries})


if __name__ == "__main__":
    main()
