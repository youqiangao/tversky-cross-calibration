"""Authoritative paper configuration access."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG_PATH = PROJECT_ROOT / "configs" / "paper.yaml"


@lru_cache(maxsize=1)
def paper_config() -> dict[str, Any]:
    with PAPER_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or int(config.get("seed", -1)) != 42:
        raise ValueError("configs/paper.yaml must define the canonical seed as 42.")
    return config


def dataset_config(name: str) -> dict[str, Any]:
    return dict(paper_config()["training"]["datasets"][name])


__all__ = ["PAPER_CONFIG_PATH", "PROJECT_ROOT", "dataset_config", "paper_config"]
