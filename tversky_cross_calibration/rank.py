"""Unified exact/official RankSEG inference used by every experiment.

The exact solver is retained for the small synthetic problems in the paper.
Large problems are delegated to the public ``rankseg`` package (version 0.0.5
in the reproducibility environment), avoiding the historical local
approximation that used to live in this repository.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from poibin import PoiBin
from .config import paper_config

Metric = Literal["dice", "iou"]
Aggregation = Literal["macro", "micro"]
RANK_CONFIG = paper_config()["rank_inference"]
DEFAULT_EXACT_THRESHOLD = int(RANK_CONFIG["exact_threshold"])


def _validate_probabilities(probabilities: torch.Tensor) -> None:
    if probabilities.ndim < 3:
        raise ValueError("probabilities must have shape [batch, classes, ...].")
    if not torch.is_floating_point(probabilities):
        raise TypeError("probabilities must be a floating-point tensor.")
    if torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1].")


def _expand_valid_mask(probabilities: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(probabilities, dtype=torch.bool)
    mask = valid_mask.to(device=probabilities.device, dtype=torch.bool)
    if mask.ndim == probabilities.ndim - 1:
        mask = mask.unsqueeze(1)
    try:
        return torch.broadcast_to(mask, probabilities.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} is not compatible with "
            f"probabilities shape {tuple(probabilities.shape)}."
        ) from exc


def _exact_rank_vector(probabilities: torch.Tensor, metric: Metric) -> torch.Tensor:
    """Return the exact Bayes rank decision for one one-dimensional vector."""
    if probabilities.ndim != 1:
        raise ValueError("The exact solver accepts one-dimensional vectors only.")
    dimension = probabilities.numel()
    prediction = torch.zeros(dimension, dtype=torch.bool, device=probabilities.device)
    if dimension == 0:
        return prediction

    sorted_prob, sorted_index = torch.sort(probabilities, descending=True)
    best_score = torch.tensor(0.0, dtype=torch.float64, device=probabilities.device)
    best_tau = 0

    if metric == "dice":
        accumulated = torch.zeros(dimension, dtype=torch.float64, device=probabilities.device)
        denominators = torch.arange(1, dimension + 1, dtype=torch.float64, device=probabilities.device)
        for tau in range(1, dimension + 1):
            excluded = torch.cat((sorted_prob[: tau - 1], sorted_prob[tau:]))
            pmf = PoiBin(excluded.detach().cpu().double().numpy()).pmf(np.arange(dimension))
            accumulated += sorted_prob[tau - 1].double() * torch.as_tensor(
                pmf, dtype=torch.float64, device=probabilities.device
            )
            score = 2.0 * torch.sum(accumulated / (tau + denominators))
            if score > best_score:
                best_score, best_tau = score, tau
    else:
        for tau in range(1, dimension + 1):
            included = sorted_prob[:tau].detach().cpu().double().numpy()
            excluded = sorted_prob[tau:].detach().cpu().double().numpy()
            pmf_in = torch.as_tensor(
                PoiBin(included).pmf(np.arange(tau + 1)),
                dtype=torch.float64,
                device=probabilities.device,
            )
            pmf_out = torch.as_tensor(
                PoiBin(excluded).pmf(np.arange(dimension - tau + 1)),
                dtype=torch.float64,
                device=probabilities.device,
            )
            true_positive = torch.arange(tau + 1, dtype=torch.float64, device=probabilities.device)[:, None]
            false_negative = torch.arange(
                dimension - tau + 1, dtype=torch.float64, device=probabilities.device
            )[None, :]
            score = torch.sum((true_positive / (tau + false_negative)) * pmf_in[:, None] * pmf_out[None, :])
            if score > best_score:
                best_score, best_tau = score, tau

    prediction[sorted_index[:best_tau]] = True
    return prediction


def _official_rank_vector(probabilities: torch.Tensor, metric: Metric) -> torch.Tensor:
    """Delegate a large vector to the official RankSEG 0.0.5 API."""
    try:
        # RankSEG 0.0.5 accesses scipy.stats during import on some SciPy builds.
        import scipy.stats  # noqa: F401
        from rankseg.functional import rankseg as official_rankseg
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Large rank inference requires RankSEG==0.0.5. Install the pinned "
            "reproducibility dependencies before running this experiment."
        ) from exc

    shaped = probabilities.reshape(1, 1, 1, -1)
    if metric == "dice":
        result = official_rankseg(
            shaped,
            metric="dice",
            smooth=float(RANK_CONFIG["smooth"]),
            output_mode="multilabel",
            solver=str(RANK_CONFIG["dice"]["solver"]),
            pruning_prob=float(RANK_CONFIG["pruning_prob"]),
            eps=float(RANK_CONFIG["dice"]["eps"]),
        )
    else:
        result = official_rankseg(
            shaped,
            metric="iou",
            smooth=float(RANK_CONFIG["smooth"]),
            output_mode="multilabel",
            solver=str(RANK_CONFIG["iou"]["solver"]),
            pruning_prob=float(RANK_CONFIG["pruning_prob"]),
        )
    return result.reshape(-1).to(device=probabilities.device, dtype=torch.bool)


def _solve_vector(probabilities: torch.Tensor, metric: Metric, exact_threshold: int) -> torch.Tensor:
    if probabilities.numel() <= exact_threshold:
        return _exact_rank_vector(probabilities, metric)
    return _official_rank_vector(probabilities, metric)


def predict_rank(
    probabilities: torch.Tensor,
    metric: Metric,
    aggregation: Aggregation = "macro",
    valid_mask: torch.Tensor | None = None,
    exact_threshold: int = DEFAULT_EXACT_THRESHOLD,
) -> torch.Tensor:
    """Compute metric-specific rank predictions.

    ``macro`` solves one vector per image and class. ``micro`` pools all valid
    class/pixel entries within each image. Invalid pixels are excluded from the
    optimization and are always returned as ``False``.
    """
    _validate_probabilities(probabilities)
    metric = metric.lower()
    aggregation = aggregation.lower()
    if metric not in {"dice", "iou"}:
        raise ValueError("metric must be 'dice' or 'iou'.")
    if aggregation not in {"macro", "micro"}:
        raise ValueError("aggregation must be 'macro' or 'micro'.")
    if exact_threshold < 0:
        raise ValueError("exact_threshold must be non-negative.")

    valid = _expand_valid_mask(probabilities, valid_mask)
    flat_prob = probabilities.reshape(probabilities.shape[0], probabilities.shape[1], -1)
    flat_valid = valid.reshape_as(flat_prob)
    flat_prediction = torch.zeros_like(flat_valid)

    for batch_index in range(flat_prob.shape[0]):
        if aggregation == "micro":
            selected = flat_valid[batch_index].reshape(-1)
            solved = _solve_vector(flat_prob[batch_index].reshape(-1)[selected], metric, exact_threshold)
            flat_prediction[batch_index].reshape(-1)[selected] = solved
        else:
            for class_index in range(flat_prob.shape[1]):
                selected = flat_valid[batch_index, class_index]
                solved = _solve_vector(
                    flat_prob[batch_index, class_index][selected], metric, exact_threshold
                )
                flat_prediction[batch_index, class_index][selected] = solved

    return flat_prediction.reshape_as(probabilities)


__all__ = ["predict_rank"]
