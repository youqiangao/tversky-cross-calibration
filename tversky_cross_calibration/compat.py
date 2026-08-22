"""Temporary call-shape compatibility for the original simulation scripts."""

from __future__ import annotations

import torch

from .rank import predict_rank


def _legacy_result(output: torch.Tensor, metric: str):
    prediction = predict_rank(output, metric)
    flattened = prediction.flatten(start_dim=2)
    tau = flattened.sum(dim=-1).to(dtype=torch.float32)
    cutpoint = torch.zeros_like(tau)
    return prediction, tau, cutpoint


def rank_dice(output: torch.Tensor, **_ignored):
    return _legacy_result(output, "dice")


def rank_iou(output: torch.Tensor, **_ignored):
    return _legacy_result(output, "iou")


__all__ = ["rank_dice", "rank_iou"]
