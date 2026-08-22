from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from application.checkpoints import infer_dataset_and_model

from application.binary.utils import resize_logits_to_size

from .dataset import DATASET_CHOICES, default_data_root_for
from .engine import create_dataloader, load_trained_model
from .metrics import RunningMultilabelMetrics, masked_index_bce_with_logits
from .utils import ensure_dir, load_checkpoint, resolve_device, save_json, save_probability_map


@torch.no_grad()
def export_predictions(checkpoint_path, data_root, split, batch_size, num_workers, device, output_dir, dataset_name=None, max_samples=None):
    model, checkpoint = load_trained_model(checkpoint_path, device)
    dataset_name = dataset_name or checkpoint.get("dataset_name") or infer_dataset_and_model(checkpoint_path)[0]
    loader = create_dataloader(dataset_name, data_root, split, checkpoint["image_size"], batch_size, num_workers, False, False, max_samples)
    threshold = float(checkpoint.get("threshold", 0.5))
    metrics = RunningMultilabelMetrics(checkpoint["num_classes"], threshold)
    output_dir = ensure_dir(output_dir)
    probabilities_dir = ensure_dir(output_dir / "probabilities")
    masks_dir = ensure_dir(output_dir / "pred_masks")
    running_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = masked_index_bce_with_logits(logits, targets, valid_mask)
        running_loss += float(loss.item()) * images.shape[0]
        metrics.update(logits, targets, valid_mask)
        for index, image_id in enumerate(batch["image_id"]):
            size = (int(batch["original_size"][0][index]), int(batch["original_size"][1][index]))
            probabilities = torch.sigmoid(resize_logits_to_size(logits[index : index + 1], size))[0]
            probabilities_np = probabilities.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            predictions_np = (probabilities_np >= threshold).astype(np.uint8)
            save_probability_map(probabilities_np, probabilities_dir / f"{image_id}_probs.npy")
            save_probability_map(predictions_np, masks_dir / f"{image_id}_pred.npy")
    result = metrics.compute_rounded()
    result["loss"] = round(running_loss / max(len(loader.dataset), 1), 6)
    save_json({"checkpoint": str(checkpoint_path), "split": split, "metrics": result}, output_dir / "metrics.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export independent-channel semantic segmentation predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    dataset = args.dataset or checkpoint.get("dataset_name") or infer_dataset_and_model(args.checkpoint)[0]
    output = args.output_dir or Path(args.checkpoint).resolve().parent / f"{args.split}_predictions"
    print(export_predictions(args.checkpoint, args.data_root or default_data_root_for(dataset), args.split, args.batch_size, args.num_workers, resolve_device(args.device), output, dataset, args.max_samples))


if __name__ == "__main__":
    main()
