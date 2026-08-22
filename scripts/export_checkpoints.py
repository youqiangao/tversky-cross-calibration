#!/usr/bin/env python3
"""Export the ten paper checkpoints as inference-only state dictionaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.checkpoints import PAPER_CHECKPOINTS, checkpoint_paths, extract_model_state_dict


def tensors_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def export_one(source: Path, destination: Path) -> int:
    payload = torch.load(source, map_location="cpu")
    state = dict(extract_model_state_dict(payload))
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    exported = torch.load(destination, map_location="cpu")
    if not tensors_equal(state, exported):
        raise RuntimeError(f"Exported tensors differ from source: {source}")
    return destination.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/checkpoint_release"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in PAPER_CHECKPOINTS:
        remote_path, _ = checkpoint_paths(item)
        source = Path(str(item["source"]))
        if not source.exists():
            raise FileNotFoundError(source)
        destination = args.output_dir / remote_path
        size = export_one(source, destination)
        print(f"exported {destination} ({size} bytes)")


if __name__ == "__main__":
    main()
