from __future__ import annotations

import argparse

from application.checkpoints import infer_dataset_and_model

from .dataset import DATASET_CHOICES, default_data_root_for
from .engine import evaluate_checkpoint
from .utils import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate independent-channel semantic segmentation checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    dataset = args.dataset or checkpoint.get("dataset_name") or infer_dataset_and_model(args.checkpoint)[0]
    print(evaluate_checkpoint(args.checkpoint, args.data_root or default_data_root_for(dataset), args.split, args.batch_size, args.num_workers, resolve_device(args.device), args.output_dir, dataset, args.max_samples))


if __name__ == "__main__":
    main()
