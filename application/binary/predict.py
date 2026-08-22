from __future__ import annotations

import argparse
from pathlib import Path

from application.checkpoints import infer_dataset_and_model

from .dataset import DATASET_CHOICES, default_data_root_for
from .engine import export_predictions
from .utils import load_checkpoint, resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export binary segmentation predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    dataset_name = args.dataset or checkpoint.get("dataset_name") or infer_dataset_and_model(args.checkpoint)[0]
    output_dir = args.output_dir or str(Path(args.checkpoint).resolve().parent / f"{args.split}_predictions")
    metrics = export_predictions(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root or default_data_root_for(dataset_name),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=resolve_device(args.device),
        output_dir=Path(output_dir),
        dataset_name=dataset_name,
        max_samples=args.max_samples,
    )
    print(metrics)


if __name__ == "__main__":
    main()
