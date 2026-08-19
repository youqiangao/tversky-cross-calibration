from __future__ import annotations

import argparse

from .dataset import DATASET_CHOICES, default_data_root_for, default_training_config_for
from .engine import train_model
from .utils import resolve_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train binary segmentation models.")
    parser.add_argument("--model", choices=("unet", "fcn8"), required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="outputs/application/binary")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
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
    checkpoint_path = train_model(
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=args.data_root or default_data_root_for(args.dataset),
        output_dir=args.output_dir,
        image_size=int(args.image_size or defaults["image_size"]),
        batch_size=int(args.batch_size or defaults["batch_size"]),
        epochs=int(args.epochs or defaults["epochs"]),
        lr=float(args.lr or defaults["lr"]),
        weight_decay=float(args.weight_decay or defaults["weight_decay"]),
        num_workers=args.num_workers,
        device=resolve_device(args.device),
        val_split=args.val_split,
        split_seed=args.split_seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(f"Best checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
