import argparse
import csv
import os
import sys
from itertools import combinations
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
MPLCONFIG_DIR = (SCRIPT_DIR / "../tmp/matplotlib").resolve()
FONTCONFIG_DIR = (SCRIPT_DIR / "../tmp/fontconfig").resolve()
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
FONTCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
os.environ["XDG_CACHE_HOME"] = str(FONTCONFIG_DIR.parent)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter

from metrics import dice_coeff, iou
from tversky_cross_calibration.config import PROJECT_ROOT, paper_config
from tversky_cross_calibration.reproducibility import cache_matches, cache_metadata, seed_everything, write_cache_metadata

DEVICE = "cpu"
METRIC_SPECS = [
    ("dice", "micro-Dice"),
    ("iou", "micro-IoU"),
    ("tversky_1_0_5", r"micro-TI$_{1,0.5}$"),
    ("tversky_1_2", r"micro-TI$_{1,2}$"),
]
METRIC_DISPLAY_NAMES = {metric_key: metric_label for metric_key, metric_label in METRIC_SPECS}
PAIR_SPECS = list(combinations([metric_key for metric_key, _ in METRIC_SPECS], 2))
METRIC_WEIGHTS = {
    "dice": 1.0,
    "tversky_1_0_5": 1.5,
    "iou": 2.0,
    "tversky_1_2": 3.0,
}
BOUND_LINE_STYLES = {
    "upper": {"color": "#d62728", "linestyle": "--", "linewidth": 1.5},
    "lower": {"color": "#2ca02c", "linestyle": "-.", "linewidth": 1.5},
}


def build_probability_map_sim_5_1(width, height):
    prob = torch.zeros((1, 3, width, height), device=DEVICE)
    prob[:, 0, :, :] = 0.8
    prob[:, 1, :, :] = 0.1
    prob[:, 2, :, :] = 0.1
    return prob


def build_probability_map_sim_5_2(width, height):
    prob = torch.zeros((1, 3, width, height), device=DEVICE)
    prob[:, 0, :, :] = 0.4
    prob[:, 1, :, :] = 0.3
    prob[:, 2, :, :] = 0.3
    return prob


def sample_targets(prob):
    sample_size, num_classes, width, height = prob.shape
    categorical_probs = prob.permute(0, 2, 3, 1).reshape(-1, num_classes)
    sampled_labels = torch.multinomial(categorical_probs, num_samples=1).reshape(sample_size, width, height)
    target = torch.zeros_like(prob, dtype=torch.bool)
    target.scatter_(1, sampled_labels.unsqueeze(1), True)
    return target


def build_argmax_prediction(prob):
    predicted_labels = prob.argmax(dim=1, keepdim=True)
    prediction = torch.zeros_like(prob, dtype=torch.bool)
    prediction.scatter_(1, predicted_labels, True)
    return prediction


def micro_tversky_index(prediction, target, alpha, beta):
    sample_size = prediction.shape[0]
    pred_flat = prediction.reshape(sample_size, -1).float()
    target_flat = target.reshape(sample_size, -1).float()
    true_positive = (pred_flat * target_flat).sum(dim=1)
    pred_positive = pred_flat.sum(dim=1)
    target_positive = target_flat.sum(dim=1)
    false_negative = target_positive - true_positive
    false_positive = pred_positive - true_positive
    denominator = true_positive + alpha * false_positive + beta * false_negative
    return torch.where(denominator > 0, true_positive / denominator, torch.zeros_like(denominator))


def compute_metric_values(prediction, target):
    return {
        "dice": dice_coeff(prediction, target),
        "iou": iou(prediction, target),
        "tversky_1_0_5": micro_tversky_index(prediction, target, alpha=1.0, beta=0.5),
        "tversky_1_2": micro_tversky_index(prediction, target, alpha=1.0, beta=2.0),
    }


def flip_prediction_multiclass(prediction, flip_prob):
    if flip_prob <= 0.0:
        return prediction.clone()

    num_classes = prediction.shape[1]
    labels = prediction.to(torch.int64).argmax(dim=1)
    flip_mask = torch.rand(labels.shape, device=prediction.device) < flip_prob
    if flip_mask.any():
        random_offset = torch.randint(
            1,
            num_classes,
            size=labels.shape,
            device=prediction.device,
        )
        flipped_labels = (labels + random_offset) % num_classes
        labels = torch.where(flip_mask, flipped_labels, labels)

    flipped_prediction = torch.zeros_like(prediction, dtype=torch.bool)
    flipped_prediction.scatter_(1, labels.unsqueeze(1), True)
    return flipped_prediction


def expand_predictions_by_mask(prediction_51, prediction_52, sim51_mask, output_shape):
    prediction = torch.empty(output_shape, dtype=torch.bool, device=DEVICE)
    sim52_mask = ~sim51_mask
    if sim51_mask.any():
        prediction[sim51_mask] = prediction_51.expand(int(sim51_mask.sum().item()), -1, -1, -1)
    if sim52_mask.any():
        prediction[sim52_mask] = prediction_52.expand(int(sim52_mask.sum().item()), -1, -1, -1)
    return prediction


def simulate_argmax(sample_size=100, width=10, height=1):
    sim51_mask = torch.rand(sample_size, device=DEVICE) < 0.5
    prob_map_51 = build_probability_map_sim_5_1(width=width, height=height)
    prob_map_52 = build_probability_map_sim_5_2(width=width, height=height)
    prob = torch.where(
        sim51_mask.view(sample_size, 1, 1, 1),
        prob_map_51.expand(sample_size, -1, -1, -1),
        prob_map_52.expand(sample_size, -1, -1, -1),
    )
    target = sample_targets(prob)

    prediction_51 = build_argmax_prediction(prob_map_51)
    prediction_52 = build_argmax_prediction(prob_map_52)
    base_prediction = expand_predictions_by_mask(
        prediction_51,
        prediction_52,
        sim51_mask,
        prob.shape,
    )
    base_scores = compute_metric_values(base_prediction, target)
    return {
        "target": target,
        "sim51_mask": sim51_mask,
        "scores": base_scores,
        "prediction": base_prediction,
    }


def summarize_metric_values(values):
    values = values.double()
    count = int(values.numel())
    if count == 0:
        return {"count": 0, "sum": 0.0, "sum_sq": 0.0, "mean": 0.0, "std_error": 0.0}

    value_sum = float(values.sum().item())
    value_sum_sq = float((values * values).sum().item())
    mean = value_sum / count
    if count > 1:
        variance = (value_sum_sq - value_sum * value_sum / count) / (count - 1)
        variance = max(variance, 0.0)
        std_error = float(np.sqrt(variance) / np.sqrt(count))
    else:
        std_error = 0.0
    return {
        "count": count,
        "sum": value_sum,
        "sum_sq": value_sum_sq,
        "mean": mean,
        "std_error": std_error,
    }


def combine_metric_summaries(*summaries):
    count = sum(summary["count"] for summary in summaries)
    value_sum = sum(summary["sum"] for summary in summaries)
    value_sum_sq = sum(summary["sum_sq"] for summary in summaries)
    if count == 0:
        return {"mean": 0.0, "std_error": 0.0}

    mean = value_sum / count
    if count > 1:
        variance = (value_sum_sq - value_sum * value_sum / count) / (count - 1)
        variance = max(variance, 0.0)
        std_error = float(np.sqrt(variance) / np.sqrt(count))
    else:
        std_error = 0.0
    return {"mean": mean, "std_error": std_error}


def summarize_metric(metric_values):
    values = np.asarray(metric_values, dtype=float)
    return float(values.mean()), float(values.std() / np.sqrt(len(values)))


def summarize_prediction_scores(prediction, target):
    metric_values = compute_metric_values(prediction, target)
    return {
        metric_key: summarize_metric_values(values)
        for metric_key, values in metric_values.items()
    }


def build_group_flip_summaries(base_prediction, target, flip_probs):
    summaries = {}
    for flip_prob in flip_probs:
        flipped_prediction = flip_prediction_multiclass(base_prediction, float(flip_prob))
        summaries[float(flip_prob)] = summarize_prediction_scores(flipped_prediction, target)
    return summaries


def print_summary(width, height, scores):
    print("#" * 20)
    print(f"width: {width}; height: {height}")
    summary_parts = []
    for metric_key, metric_label in METRIC_SPECS:
        metric_mean, metric_std = summarize_metric(scores[metric_key].cpu().numpy())
        summary_parts.append(f"{metric_label}: {metric_mean:.3f}({metric_std:.3f})")
    print("shared argmax baseline: " + "; ".join(summary_parts))


def run_flip_sweep(sample_size=1000, width=10000, height=1):
    flip_rows = []
    simulation = simulate_argmax(sample_size=sample_size, width=width, height=height)
    target = simulation["target"]
    sim51_mask = simulation["sim51_mask"]
    sim52_mask = ~sim51_mask
    scores = simulation["scores"]
    prediction = simulation["prediction"]
    flip_probs = np.arange(0.0, 1.0001, 0.02)
    baselines = {
        metric_key: float(scores[metric_key].mean().item())
        for metric_key, _ in METRIC_SPECS
    }
    num_sim_51 = int(sim51_mask.sum().item())
    num_sim_52 = int(sim52_mask.sum().item())

    print("#" * 20)
    print(f"shared argmax prediction; width: {width}; height: {height}")
    print(f"Precomputing sim-5-1 flip sweep for {num_sim_51} samples")
    sim51_summaries = build_group_flip_summaries(
        prediction[sim51_mask],
        target[sim51_mask],
        flip_probs,
    )
    print(f"Precomputing sim-5-2 flip sweep for {num_sim_52} samples")
    sim52_summaries = build_group_flip_summaries(
        prediction[sim52_mask],
        target[sim52_mask],
        flip_probs,
    )

    for flip_prob_51 in flip_probs:
        for flip_prob_52 in flip_probs:
            sim51_score = sim51_summaries[float(flip_prob_51)]
            sim52_score = sim52_summaries[float(flip_prob_52)]
            row = {
                "flip_prob_51": float(flip_prob_51),
                "flip_prob_52": float(flip_prob_52),
                "width": width,
                "height": height,
                "num_sim_51": num_sim_51,
                "num_sim_52": num_sim_52,
            }
            for metric_key, _ in METRIC_SPECS:
                metric_summary = combine_metric_summaries(
                    sim51_score[metric_key],
                    sim52_score[metric_key],
                )
                row[f"{metric_key}_mean"] = metric_summary["mean"]
                row[f"{metric_key}_std"] = metric_summary["std_error"]
            flip_rows.append(row)

    return flip_rows, baselines, scores


def build_excess_risk_rows(flip_rows, baselines):
    excess_rows = []
    for index, row in enumerate(flip_rows):
        excess_row = {
            "index": index,
            "flip_prob_51": row["flip_prob_51"],
            "flip_prob_52": row["flip_prob_52"],
            "width": row["width"],
            "height": row["height"],
        }
        for metric_key, _ in METRIC_SPECS:
            excess_row[f"excess_{metric_key}"] = baselines[metric_key] - row[f"{metric_key}_mean"]
        excess_rows.append(excess_row)
    return excess_rows


def infer_baselines_from_flip_rows(flip_rows):
    for row in flip_rows:
        if np.isclose(row["flip_prob_51"], 0.0) and np.isclose(row["flip_prob_52"], 0.0):
            return {
                metric_key: row[f"{metric_key}_mean"]
                for metric_key, _ in METRIC_SPECS
            }
    raise ValueError("Cannot infer baselines from flip CSV; missing zero-flip row.")


def save_csv(rows, output_path, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(input_path):
    with input_path.open(newline="") as csv_file:
        return [coerce_csv_row(row) for row in csv.DictReader(csv_file)]


def coerce_csv_row(row):
    int_fields = {"index", "width", "height", "num_sim_51", "num_sim_52"}
    coerced = {}
    for key, value in row.items():
        if key in int_fields:
            coerced[key] = int(float(value))
        else:
            coerced[key] = float(value)
    return coerced


def infer_width_height(rows, default_width, default_height):
    if not rows:
        return default_width, default_height
    return int(rows[0].get("width", default_width)), int(rows[0].get("height", default_height))


def csv_has_columns(input_path, required_fields):
    if not input_path.exists():
        return False
    with input_path.open(newline="") as csv_file:
        fieldnames = csv.DictReader(csv_file).fieldnames or []
    return set(required_fields).issubset(fieldnames)


def get_axis_limits(values, padding_ratio=0.12, min_padding=0.002):
    values = np.asarray(values, dtype=float)
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    padding = max(span * padding_ratio, min_padding)
    return lower - padding, upper + padding


def build_two_decimal_ticks(lower, upper, step=0.01):
    start = np.floor(lower / step) * step
    end = np.ceil(upper / step) * step
    ticks = np.arange(start, end + step * 0.5, step)
    rounded_ticks = np.round(ticks, 2)
    _, unique_indices = np.unique(rounded_ticks, return_index=True)
    return rounded_ticks[np.sort(unique_indices)]


def get_pair_slopes(x_metric, y_metric):
    weight_x = METRIC_WEIGHTS[x_metric]
    weight_y = METRIC_WEIGHTS[y_metric]
    ratio_xy = weight_x / weight_y
    ratio_yx = weight_y / weight_x
    return min(ratio_xy, ratio_yx), max(ratio_xy, ratio_yx)


def bound_diagnostics(x_values, y_values, lower_slope, upper_slope, tolerance=1e-6):
    """Report violations without changing or filtering any simulated point."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    lower_violation = np.maximum(lower_slope * x_values - y_values, 0.0)
    upper_violation = np.maximum(y_values - upper_slope * x_values, 0.0)
    violation = np.maximum(lower_violation, upper_violation)
    return {
        "num_points": int(x_values.size),
        "num_violations": int(np.count_nonzero(violation > tolerance)),
        "num_lower_violations": int(np.count_nonzero(lower_violation > tolerance)),
        "num_upper_violations": int(np.count_nonzero(upper_violation > tolerance)),
        "max_violation": float(violation.max(initial=0.0)),
        "tolerance": float(tolerance),
    }


def plot_excess_risks(excess_rows, output_path):
    metric_keys = [metric_key for metric_key, _ in METRIC_SPECS]
    pair_layout = [
        ("dice", "iou"),
        ("dice", "tversky_1_0_5"),
        ("dice", "tversky_1_2"),
        ("iou", "tversky_1_0_5"),
        ("iou", "tversky_1_2"),
        ("tversky_1_0_5", "tversky_1_2"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    diagnostics = []

    for pair_index, (x_metric, y_metric) in enumerate(pair_layout):
        row_index = pair_index // 3
        col_index = pair_index % 3
        ax = axes[row_index, col_index]
        x_values = np.array([row[f"excess_{x_metric}"] for row in excess_rows], dtype=float)
        y_values = np.array([row[f"excess_{y_metric}"] for row in excess_rows], dtype=float)
        lower_slope, upper_slope = get_pair_slopes(x_metric, y_metric)
        diagnostic = bound_diagnostics(x_values, y_values, lower_slope, upper_slope)
        diagnostic.update({"x_metric": x_metric, "y_metric": y_metric})
        diagnostics.append(diagnostic)
        ax.scatter(x_values, y_values, s=10, color="#1f77b4", alpha=0.35, linewidths=0)
        x_min = min(0.0, float(x_values.min(initial=0.0)))
        x_max = max(0.1, float(x_values.max(initial=0.0)))
        x_line = np.linspace(x_min, x_max, 200)
        ax.plot(
            x_line,
            lower_slope * x_line,
            label="lower bound",
            **BOUND_LINE_STYLES["lower"],
        )
        ax.plot(
            x_line,
            upper_slope * x_line,
            label="upper bound",
            **BOUND_LINE_STYLES["upper"],
        )
        ax.set_xlabel(f"Excess {METRIC_DISPLAY_NAMES[x_metric]} risk", fontsize=20)
        ax.set_ylabel(f"Excess {METRIC_DISPLAY_NAMES[y_metric]} risk", fontsize=20)
        ax.grid(False)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.minorticks_off()
        ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return diagnostics


def main(refit=False, sample_size=10000, width=50000, height=1):
    seed_everything(42)
    print(f"Running sim-5 on {DEVICE}")

    flip_csv = (SCRIPT_DIR / "../tmp/data/sim-5-flip.csv").resolve()
    excess_csv = (SCRIPT_DIR / "../tmp/data/sim-5-excess.csv").resolve()
    figure_path = PROJECT_ROOT / "figures" / "sim-5.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata("sim-5", {"sample_size": sample_size, "width": width, "height": height, "flip_grid": [0.0, 1.0, 0.02]})

    flip_fieldnames = [
        "flip_prob_51",
        "flip_prob_52",
        "width",
        "height",
        "num_sim_51",
        "num_sim_52",
    ]
    for metric_key, _ in METRIC_SPECS:
        flip_fieldnames.extend([f"{metric_key}_mean", f"{metric_key}_std"])

    excess_fieldnames = [
        "index",
        "flip_prob_51",
        "flip_prob_52",
        "width",
        "height",
    ]
    for metric_key, _ in METRIC_SPECS:
        excess_fieldnames.append(f"excess_{metric_key}")

    excess_cache_compatible = csv_has_columns(excess_csv, excess_fieldnames)
    flip_cache_compatible = csv_has_columns(flip_csv, flip_fieldnames)

    if not refit and excess_cache_compatible and cache_matches(excess_csv, metadata):
        print(f"Loading existing excess-risk CSV from {excess_csv}")
        excess_rows = read_csv_rows(excess_csv)
        width, height = infer_width_height(excess_rows, default_width=width, default_height=height)
    elif not refit and flip_cache_compatible and cache_matches(flip_csv, metadata):
        print(f"Loading existing flip CSV from {flip_csv}")
        flip_rows = read_csv_rows(flip_csv)
        baselines = infer_baselines_from_flip_rows(flip_rows)
        excess_rows = build_excess_risk_rows(flip_rows, baselines)
        width, height = infer_width_height(excess_rows, default_width=width, default_height=height)
        save_csv(excess_rows, excess_csv, excess_fieldnames)
        write_cache_metadata(flip_csv, metadata)
        write_cache_metadata(excess_csv, metadata)
    else:
        if refit:
            print("refit=True, rerunning simulation even if CSV files already exist")
        elif excess_csv.exists() or flip_csv.exists():
            print("Existing sim-5 cache is incompatible with the new schema, rerunning simulation")
        else:
            print("No existing CSV found, running simulation")
        flip_rows, baselines, scores = run_flip_sweep(
            sample_size=sample_size,
            width=width,
            height=height,
        )
        print_summary(width, height, scores)
        excess_rows = build_excess_risk_rows(flip_rows, baselines)
        save_csv(flip_rows, flip_csv, flip_fieldnames)
        save_csv(excess_rows, excess_csv, excess_fieldnames)
        write_cache_metadata(flip_csv, metadata)
        write_cache_metadata(excess_csv, metadata)

    diagnostics = plot_excess_risks(excess_rows, figure_path)
    diagnostics_csv = (SCRIPT_DIR / "../tmp/data/sim-5-bound-diagnostics.csv").resolve()
    save_csv(diagnostics, diagnostics_csv, ["x_metric", "y_metric", "num_points", "num_violations", "num_lower_violations", "num_upper_violations", "max_violation", "tolerance"])
    for row in diagnostics:
        print(f"{row['x_metric']}->{row['y_metric']} bound violations: {row['num_violations']}/{row['num_points']}; max={row['max_violation']:.3e}; tolerance={row['tolerance']:.1e}")

    print(f"Flip CSV: {flip_csv}")
    print(f"Excess-risk CSV: {excess_csv}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    simulation_config = paper_config()["simulation"]
    parser = argparse.ArgumentParser(
        description=(
            "Mix sim-5-1 and sim-5-2 with equal probability, apply a double-flip "
            "perturbation to the shared argmax predictor, and compare pairwise "
            "micro-metric excess risks."
        )
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Rerun the simulation and overwrite the cached CSV files.",
    )
    parser.add_argument("--sample-size", type=int, default=int(simulation_config["repetitions"]))
    parser.add_argument("--width", type=int, default=int(simulation_config["excess_risk"]["dimension"]))
    parser.add_argument("--height", type=int, default=1)
    args = parser.parse_args()
    main(refit=args.refit, sample_size=args.sample_size, width=args.width, height=args.height)
