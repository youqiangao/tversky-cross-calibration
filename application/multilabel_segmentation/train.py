from __future__ import annotations

import argparse

from .dataset import DATASET_CHOICES, default_data_root_for, default_training_config_for
from .engine import train_model
from .utils import resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train semantic segmentation datasets with independent class channels.")
    parser.add_argument("--model", choices=("unet", "fcn8"), required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="outputs/application/multilabel_segmentation")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    defaults = default_training_config_for(args.dataset)
    seed_everything(args.seed)
    checkpoint = train_model(
        args.model, args.dataset, args.data_root or default_data_root_for(args.dataset), args.output_dir,
        int(args.image_size or defaults["image_size"]), int(args.batch_size or defaults["batch_size"]),
        int(args.epochs or defaults["epochs"]), float(args.lr or defaults["lr"]),
        float(args.weight_decay or defaults["weight_decay"]), args.num_workers, resolve_device(args.device),
        args.max_train_samples, args.max_val_samples, args.resume_checkpoint, False,
    )
    print(f"Best checkpoint saved to: {checkpoint}")


if __name__ == "__main__":
    main()
