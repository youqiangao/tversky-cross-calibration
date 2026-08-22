"""Deterministic execution and cache provenance helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import scipy
import torch

CANONICAL_SEED = 42
CACHE_SCHEMA = 3


def seed_everything(seed: int = CANONICAL_SEED, deterministic: bool = True) -> None:
    if int(seed) != CANONICAL_SEED:
        raise ValueError(f"Canonical paper experiments require seed={CANONICAL_SEED}.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataloader_generator() -> torch.Generator:
    return torch.Generator().manual_seed(CANONICAL_SEED)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cache_metadata(script: str, parameters: dict[str, Any]) -> dict[str, Any]:
    try:
        import rankseg
        rankseg_version = getattr(rankseg, "__version__", "0.0.5")
    except ImportError:
        rankseg_version = "unavailable"
    return {
        "schema": CACHE_SCHEMA,
        "script": script,
        "seed": CANONICAL_SEED,
        "parameters": parameters,
        "git_commit": _git_commit(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "rankseg": rankseg_version,
        },
    }


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".meta.json")


def cache_matches(csv_path: Path, expected: dict[str, Any]) -> bool:
    path = metadata_path(csv_path)
    if not csv_path.exists() or not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    # A cache remains reusable across documentation-only commits.
    actual.pop("git_commit", None)
    comparable = dict(expected)
    comparable.pop("git_commit", None)
    return actual == comparable


def write_cache_metadata(csv_path: Path, metadata: dict[str, Any]) -> None:
    path = metadata_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "CANONICAL_SEED",
    "CACHE_SCHEMA",
    "cache_matches",
    "cache_metadata",
    "dataloader_generator",
    "seed_everything",
    "seed_worker",
    "write_cache_metadata",
]
