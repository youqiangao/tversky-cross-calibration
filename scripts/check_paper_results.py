"""Compare full release summaries with the values displayed in the paper."""

from __future__ import annotations

from typing import Any, Iterable


BINARY_DELTA = {
    ("oxford_pet", "unet"): 2.79e-5,
    ("oxford_pet", "fcn8"): 2.30e-5,
    ("isic2017", "unet"): 2.62e-6,
    ("isic2017", "fcn8"): 2.27e-6,
    ("kvasir_seg", "unet"): 4.25e-5,
    ("kvasir_seg", "fcn8"): 4.93e-5,
}
BINARY_PERFORMANCE = {
    ("oxford_pet", "unet"): (0.9339, 0.8839),
    ("oxford_pet", "fcn8"): (0.9194, 0.8592),
    ("isic2017", "unet"): (0.8402, 0.7562),
    ("isic2017", "fcn8"): (0.8464, 0.7556),
    ("kvasir_seg", "unet"): (0.8495, 0.7696),
    ("kvasir_seg", "fcn8"): (0.8515, 0.7685),
}
MULTILABEL_DELTA = {
    ("voc2012", "unet"): (1.37e-4, 1.30e-4),
    ("voc2012", "fcn8"): (1.74e-4, 1.30e-4),
    ("cityscapes", "unet"): (1.02e-4, 1.04e-5),
    ("cityscapes", "fcn8"): (1.44e-4, 1.03e-5),
}
MULTILABEL_PERFORMANCE = {
    ("voc2012", "unet", "micro"): (0.7635, 0.6485),
    ("voc2012", "fcn8", "micro"): (0.7553, 0.6399),
    ("cityscapes", "unet", "micro"): (0.9312, 0.8738),
    ("cityscapes", "fcn8", "micro"): (0.8829, 0.7939),
    ("voc2012", "unet", "macro"): (0.6364, 0.5402),
    ("voc2012", "fcn8", "macro"): (0.5844, 0.4889),
    ("cityscapes", "unet", "macro"): (0.6881, 0.6031),
    ("cityscapes", "fcn8", "macro"): (0.5164, 0.4337),
}


def _rows_by(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    return {tuple(str(row[field]) for field in fields): row for row in rows}


def check_paper_results(summaries: dict[str, list[dict[str, Any]]]) -> list[str]:
    issues: list[str] = []
    binary_delta = _rows_by(summaries["binary_delta"], ("dataset_name", "model"))
    for key, expected in BINARY_DELTA.items():
        actual = float(binary_delta[key]["delta_hat"])
        if f"{actual:.2e}" != f"{expected:.2e}":
            issues.append(f"binary delta {key}: got {actual:.10g}, paper displays {expected:.2e}")

    binary_performance = _rows_by(
        summaries["binary_performance"], ("dataset_name", "model", "optimizer")
    )
    for (dataset, model), expected in BINARY_PERFORMANCE.items():
        for optimizer in ("RankDice", "RankIoU"):
            row = binary_performance[(dataset, model, optimizer)]
            actual = (float(row["dice"]), float(row["iou"]))
            if tuple(round(value, 4) for value in actual) != expected:
                issues.append(f"binary performance {(dataset, model, optimizer)}: got {actual}, paper displays {expected}")

    multilabel_delta = _rows_by(summaries["multilabel_delta"], ("dataset_name", "model"))
    for key, expected in MULTILABEL_DELTA.items():
        row = multilabel_delta[key]
        actual = (float(row["delta_macro_hat"]), float(row["delta_micro_hat"]))
        if tuple(f"{value:.2e}" for value in actual) != tuple(f"{value:.2e}" for value in expected):
            issues.append(f"multilabel delta {key}: got {actual}, paper displays {expected}")

    multilabel_performance = _rows_by(
        summaries["multilabel_performance"], ("dataset_name", "model", "optimizer")
    )
    for (dataset, model, scope), expected in MULTILABEL_PERFORMANCE.items():
        fields = (f"{scope}_dice", f"{scope}_iou")
        for optimizer in (f"{scope}-Dice", f"{scope}-IoU"):
            row = multilabel_performance[(dataset, model, optimizer)]
            actual = tuple(float(row[field]) for field in fields)
            if tuple(round(value, 4) for value in actual) != expected:
                issues.append(f"multilabel performance {(dataset, model, optimizer)}: got {actual}, paper displays {expected}")
    return issues
