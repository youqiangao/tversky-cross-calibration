from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from tversky_cross_calibration.reproducibility import seed_everything


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_csv(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def resize_logits_to_size(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)
