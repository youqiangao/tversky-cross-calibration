import argparse
import csv
import os
import sys
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

from metrics import dice_coeff, iou, macro_dice_coeff, macro_iou
from tversky_cross_calibration.compat import rank_dice, rank_iou
from tversky_cross_calibration.config import PROJECT_ROOT, paper_config
from tversky_cross_calibration.reproducibility import cache_matches, cache_metadata, seed_everything, write_cache_metadata

DEVICE = "cpu"
C1 = 72.0
NUM_CLASSES = 3
ETA = 12.0


def build_probability_map_sim_3_1(width, height):
    num_class = 3
    prob = torch.zeros((1, num_class, width, height), device=DEVICE)
    block_width = int(width / 10)

    prob[:, 0, 0:block_width, :] = 0.9
    prob[:, 1, 0:block_width, :] = 0.5
    prob[:, 2, 0:block_width, :] = 0.4

    prob[:, 0, block_width : (2 * block_width), :] = 0.8
    prob[:, 1, block_width : (2 * block_width), :] = 0.29
    prob[:, 2, block_width : (2 * block_width), :] = 0.24

    prob[:, 0, 2 * block_width :, :] = 0.5
    return prob


def build_probability_map_sim_3_2(width, height):
    num_class = 3
    prob = torch.zeros((1, num_class, width, height), device=DEVICE)
    block_width = int(width / 10)

    prob[:, 0, 0:block_width, :] = 0.9
    prob[:, 1, 0:block_width, :] = 0.8
    prob[:, 2, 0:block_width, :] = 0.7

    prob[:, 0, block_width : (2 * block_width), :] = 0.8
    prob[:, 1, block_width : (2 * block_width), :] = 0.7
    prob[:, 2, block_width : (2 * block_width), :] = 0.6

    prob[:, :, 2 * block_width :, :] = 0.6
    return prob


def build_probability_map(sample_size, width, height):
    prob = build_probability_map_sim_3_2(width=width, height=height)
    return prob.expand(sample_size, -1, -1, -1)


def use_exact_rankseg(width):
    return width < 200


def reshape_micro_prediction(prediction_flat, reference_shape):
    return prediction_flat.reshape(reference_shape)


def run_rankseg_once(prob_single, exact):
    prediction_iou_single, _, _ = rank_iou(prob_single, device=DEVICE, verbose=0, exact=exact)
    prediction_dice_single, _, _ = rank_dice(prob_single, device=DEVICE, verbose=0, exact=exact)
    return prediction_iou_single, prediction_dice_single


def expand_predictions_by_mask(prediction_31, prediction_32, sim31_mask, output_shape):
    prediction = torch.empty(output_shape, dtype=torch.bool, device=DEVICE)
    sim32_mask = ~sim31_mask
    if sim31_mask.any():
        prediction[sim31_mask] = prediction_31.expand(int(sim31_mask.sum().item()), -1, -1, -1)
    if sim32_mask.any():
        prediction[sim32_mask] = prediction_32.expand(int(sim32_mask.sum().item()), -1, -1, -1)
    return prediction


def run_mixed_rankseg_predictions(prob_map_31, prob_map_32, sim31_mask, exact):
    macro_iou_31, macro_dice_31 = run_rankseg_once(prob_map_31, exact)
    macro_iou_32, macro_dice_32 = run_rankseg_once(prob_map_32, exact)
    macro_iou = expand_predictions_by_mask(macro_iou_31, macro_iou_32, sim31_mask, prob_map_31.expand(len(sim31_mask), -1, -1, -1).shape)
    macro_dice = expand_predictions_by_mask(macro_dice_31, macro_dice_32, sim31_mask, prob_map_31.expand(len(sim31_mask), -1, -1, -1).shape)

    prob_micro_31 = prob_map_31.reshape(1, 1, -1, 1)
    prob_micro_32 = prob_map_32.reshape(1, 1, -1, 1)
    micro_iou_flat_31, micro_dice_flat_31 = run_rankseg_once(prob_micro_31, exact)
    micro_iou_flat_32, micro_dice_flat_32 = run_rankseg_once(prob_micro_32, exact)
    micro_iou_31 = reshape_micro_prediction(micro_iou_flat_31, prob_map_31.shape)
    micro_dice_31 = reshape_micro_prediction(micro_dice_flat_31, prob_map_31.shape)
    micro_iou_32 = reshape_micro_prediction(micro_iou_flat_32, prob_map_32.shape)
    micro_dice_32 = reshape_micro_prediction(micro_dice_flat_32, prob_map_32.shape)
    micro_iou = expand_predictions_by_mask(micro_iou_31, micro_iou_32, sim31_mask, prob_map_31.expand(len(sim31_mask), -1, -1, -1).shape)
    micro_dice = expand_predictions_by_mask(micro_dice_31, micro_dice_32, sim31_mask, prob_map_31.expand(len(sim31_mask), -1, -1, -1).shape)

    return {
        "macro-IoU": macro_iou,
        "macro-Dice": macro_dice,
        "micro-IoU": micro_iou,
        "micro-Dice": micro_dice,
    }


def simulate_rankseg(sample_size=100, width=10, height=1, exact=False):
    sim31_mask = torch.rand(sample_size, device=DEVICE) < 0.5
    prob_map_31 = build_probability_map_sim_3_1(width=width, height=height)
    prob_map_32 = build_probability_map_sim_3_2(width=width, height=height)
    prob = torch.where(
        sim31_mask.view(sample_size, 1, 1, 1),
        prob_map_31.expand(sample_size, -1, -1, -1),
        prob_map_32.expand(sample_size, -1, -1, -1),
    )
    target = torch.bernoulli(prob)
    predictions = run_mixed_rankseg_predictions(prob_map_31, prob_map_32, sim31_mask, exact)
    predict_iou = predictions["macro-IoU"]
    predict_dice = predictions["macro-Dice"]

    iou_score = {
        "dice": macro_dice_coeff(predict_iou, target),
        "iou": macro_iou(predict_iou, target),
    }
    dice_score = {
        "dice": macro_dice_coeff(predict_dice, target),
        "iou": macro_iou(predict_dice, target),
    }

    predict_micro_iou = predictions["micro-IoU"]
    predict_micro_dice = predictions["micro-Dice"]

    micro_iou_score = {
        "dice": dice_coeff(predict_micro_iou, target),
        "iou": iou(predict_micro_iou, target),
    }
    micro_dice_score = {
        "dice": dice_coeff(predict_micro_dice, target),
        "iou": iou(predict_micro_dice, target),
    }

    return {
        "target": target,
        "sim31_mask": sim31_mask,
        "scores": {
            "macro-IoU": iou_score,
            "macro-Dice": dice_score,
            "micro-IoU": micro_iou_score,
            "micro-Dice": micro_dice_score,
        },
        "predictions": {
            "macro-IoU": predict_iou,
            "macro-Dice": predict_dice,
            "micro-IoU": predict_micro_iou,
            "micro-Dice": predict_micro_dice,
        },
    }


def flip_prediction_iid(prediction, flip_prob):
    flip_mask = torch.rand(prediction.shape, device=prediction.device) < flip_prob
    return torch.logical_xor(prediction, flip_mask)


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


def summarize_prediction_scores(prediction, target, use_micro_metrics):
    if use_micro_metrics:
        dice_values = dice_coeff(prediction, target)
        iou_values = iou(prediction, target)
    else:
        dice_values = macro_dice_coeff(prediction, target)
        iou_values = macro_iou(prediction, target)
    return {
        "dice": summarize_metric_values(dice_values),
        "iou": summarize_metric_values(iou_values),
    }


def build_group_flip_summaries(base_prediction, target, flip_probs, use_micro_metrics):
    summaries = {}
    for flip_prob in flip_probs:
        flipped_prediction = flip_prediction_iid(base_prediction, float(flip_prob))
        summaries[float(flip_prob)] = summarize_prediction_scores(
            flipped_prediction,
            target,
            use_micro_metrics=use_micro_metrics,
        )
    return summaries


def est_delta_micro(width=10, height=1):
    eta_tensor = torch.tensor(NUM_CLASSES * ETA, device=DEVICE)
    prob_31 = build_probability_map_sim_3_1(width=width, height=height)
    prob_32 = build_probability_map_sim_3_2(width=width, height=height)
    sum_31 = prob_31.sum(dim=(1, 2, 3))
    sum_32 = prob_32.sum(dim=(1, 2, 3))
    value_31 = 1.0 / torch.maximum(sum_31, eta_tensor)
    value_32 = 1.0 / torch.maximum(sum_32, eta_tensor)
    return float((0.5 * value_31 + 0.5 * value_32).item())


def est_delta_macro(width=10, height=1):
    eta_tensor = torch.tensor(ETA, device=DEVICE)
    prob_31 = build_probability_map_sim_3_1(width=width, height=height)
    prob_32 = build_probability_map_sim_3_2(width=width, height=height)
    class_sum_31 = prob_31.sum(dim=(2, 3))
    class_sum_32 = prob_32.sum(dim=(2, 3))
    value_31 = (1.0 / torch.maximum(class_sum_31, eta_tensor)).sum(dim=1)
    value_32 = (1.0 / torch.maximum(class_sum_32, eta_tensor)).sum(dim=1)
    return float((0.5 * value_31 + 0.5 * value_32).item())


def summarize_metric(metric_values):
    values = np.asarray(metric_values, dtype=float)
    return float(values.mean()), float(values.std() / np.sqrt(len(values)))


def print_rankseg_summary(width, height, scores):
    print("#" * 20)
    print(f"width: {width}; height: {height}")
    for optimizer in ["macro-IoU", "macro-Dice", "micro-IoU", "micro-Dice"]:
        optimizer_scores = scores[optimizer]
        dice_mean, dice_std = summarize_metric(optimizer_scores["dice"].cpu().numpy())
        iou_mean, iou_std = summarize_metric(optimizer_scores["iou"].cpu().numpy())
        print(
            "%s: dice score: %.3f(%.3f); iou score: %.3f(%.3f)"
            % (optimizer, dice_mean, dice_std, iou_mean, iou_std)
        )


def run_flip_sweep(sample_size=1000, width=10000, height=1):
    flip_rows = []
    exact = use_exact_rankseg(width)
    simulation = simulate_rankseg(
        sample_size=sample_size,
        width=width,
        height=height,
        exact=exact,
    )
    target = simulation["target"]
    sim31_mask = simulation["sim31_mask"]
    sim32_mask = ~sim31_mask
    scores = simulation["scores"]
    predictions = simulation["predictions"]
    base_optimizers = ["macro-Dice", "macro-IoU", "micro-Dice", "micro-IoU"]
    flip_probs = np.arange(0.0, 1.0001, 0.02)
    baselines = {}
    num_sim_31 = int(sim31_mask.sum().item())
    num_sim_32 = int(sim32_mask.sum().item())

    for base_optimizer in base_optimizers:
        base_prediction = predictions[base_optimizer]
        use_micro_metrics = base_optimizer.startswith("micro-")
        baseline_metrics = {
            "dice_mean": float(scores[base_optimizer]["dice"].mean().item()),
            "iou_mean": float(scores[base_optimizer]["iou"].mean().item()),
        }
        baselines[base_optimizer] = baseline_metrics
        print("#" * 20)
        print(f"base_optimizer: {base_optimizer}; width: {width}; height: {height}")
        print(f"Precomputing sim-3-1 flip sweep for {num_sim_31} samples")
        sim31_summaries = build_group_flip_summaries(
            base_prediction[sim31_mask],
            target[sim31_mask],
            flip_probs,
            use_micro_metrics=use_micro_metrics,
        )
        print(f"Precomputing sim-3-2 flip sweep for {num_sim_32} samples")
        sim32_summaries = build_group_flip_summaries(
            base_prediction[sim32_mask],
            target[sim32_mask],
            flip_probs,
            use_micro_metrics=use_micro_metrics,
        )
        for flip_prob_31 in flip_probs:
            for flip_prob_32 in flip_probs:
                sim31_score = sim31_summaries[float(flip_prob_31)]
                sim32_score = sim32_summaries[float(flip_prob_32)]
                dice_summary = combine_metric_summaries(sim31_score["dice"], sim32_score["dice"])
                iou_summary = combine_metric_summaries(sim31_score["iou"], sim32_score["iou"])
                row = {
                    "base_optimizer": base_optimizer,
                    "flip_prob_31": float(flip_prob_31),
                    "flip_prob_32": float(flip_prob_32),
                    "width": width,
                    "height": height,
                    "num_sim_31": num_sim_31,
                    "num_sim_32": num_sim_32,
                    "dice_mean": dice_summary["mean"],
                    "iou_mean": iou_summary["mean"],
                    "dice_std": dice_summary["std_error"],
                    "iou_std": iou_summary["std_error"],
                }
                flip_rows.append(row)
        print(
            f"Finished {base_optimizer}: combined {len(flip_probs) * len(flip_probs)} flip pairs"
        )

    return flip_rows, baselines, scores


def build_excess_risk_rows(flip_rows, baselines):
    excess_rows = []
    for index, row in enumerate(flip_rows):
        baseline = baselines[row["base_optimizer"]]
        excess_rows.append(
            {
                "index": index,
                "base_optimizer": row["base_optimizer"],
                "flip_prob_31": row["flip_prob_31"],
                "flip_prob_32": row["flip_prob_32"],
                "width": row["width"],
                "height": row["height"],
                "excess_dice": baseline["dice_mean"] - row["dice_mean"],
                "excess_iou": baseline["iou_mean"] - row["iou_mean"],
            }
        )
    return excess_rows


def infer_baselines_from_flip_rows(flip_rows):
    baselines = {}
    for row in flip_rows:
        if np.isclose(row["flip_prob_31"], 0.0) and np.isclose(row["flip_prob_32"], 0.0):
            baselines[row["base_optimizer"]] = {
                "dice_mean": row["dice_mean"],
                "iou_mean": row["iou_mean"],
            }
    missing = sorted({row["base_optimizer"] for row in flip_rows} - set(baselines))
    if missing:
        raise ValueError(f"Cannot infer baselines from flip CSV; missing zero-flip rows for {missing}")
    return baselines


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
    int_fields = {"index", "width", "height", "num_sim_31", "num_sim_32"}
    str_fields = {"base_optimizer"}
    coerced = {}
    for key, value in row.items():
        if key in str_fields:
            coerced[key] = value
        elif key in int_fields:
            coerced[key] = int(float(value))
        else:
            coerced[key] = float(value)
    return coerced


def infer_width_height(rows, default_width, default_height):
    if not rows:
        return default_width, default_height
    return int(rows[0].get("width", default_width)), int(rows[0].get("height", default_height))


def filter_excess_rows_by_scope(excess_rows, scope_label):
    return [row for row in excess_rows if row["base_optimizer"].startswith(f"{scope_label}-")]


def bound_diagnostics(excess_iou, excess_dice, bound_constant, tolerance=1e-6):
    """Report violations without changing or filtering any simulated point."""
    x_values = np.asarray(excess_iou, dtype=float)
    y_values = np.asarray(excess_dice, dtype=float)
    lower_violation = np.maximum(0.5 * (x_values - bound_constant) - y_values, 0.0)
    upper_violation = np.maximum(y_values - (2.0 * x_values + bound_constant), 0.0)
    violation = np.maximum(lower_violation, upper_violation)
    return {
        "num_points": int(x_values.size),
        "num_violations": int(np.count_nonzero(violation > tolerance)),
        "num_lower_violations": int(np.count_nonzero(lower_violation > tolerance)),
        "num_upper_violations": int(np.count_nonzero(upper_violation > tolerance)),
        "max_violation": float(violation.max(initial=0.0)),
        "tolerance": float(tolerance),
    }


def plot_excess_risk_panel(ax, excess_rows, delta, scope_label):
    excess_dice = np.array([row["excess_dice"] for row in excess_rows], dtype=float)
    excess_iou = np.array([row["excess_iou"] for row in excess_rows], dtype=float)
    upper_bound_style = {"color": "red", "linestyle": "--"}
    lower_bound_style = {"color": "green", "linestyle": "-."}
    if scope_label == "micro":
        bound_constant = C1 * NUM_CLASSES * delta
    elif scope_label == "macro":
        bound_constant = (C1 / NUM_CLASSES) * delta
    else:
        raise ValueError(f"Unknown scope_label: {scope_label}")

    diagnostics = bound_diagnostics(excess_iou, excess_dice, bound_constant)
    ax.scatter(excess_iou, excess_dice, s=10, alpha=0.35, linewidths=0)
    ax.set_xlabel(f"Excess {scope_label}-IoU risk", fontsize=20)
    ax.set_ylabel(f"Excess {scope_label}-Dice risk", fontsize=20)
    x_min = min(0.0, float(excess_iou.min(initial=0.0)))
    x_max = max(0.1, float(excess_iou.max(initial=0.0)))
    x_line = np.linspace(x_min, x_max, 200)
    ax.plot(x_line, 2 * x_line + bound_constant, label="upper bound", **upper_bound_style)
    ax.plot(
        x_line,
        0.5 * (x_line - bound_constant),
        label="lower bound",
        **lower_bound_style,
    )
    ax.legend()
    ax.grid(False)
    return diagnostics


def plot_excess_risks(macro_excess_rows, micro_excess_rows, delta, output_path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    diagnostics = []
    for ax, rows, scope in ((ax0, macro_excess_rows, "macro"), (ax1, micro_excess_rows, "micro")):
        row = plot_excess_risk_panel(ax, rows, delta[scope], scope_label=scope)
        row["scope"] = scope
        diagnostics.append(row)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return diagnostics


def main(refit=False, sample_size=10000, width=50000, height=1):
    seed_everything(42)
    print(f"Running sim-4 on {DEVICE}")

    flip_csv = (SCRIPT_DIR / "../tmp/data/sim-4-flip.csv").resolve()
    excess_csv = (SCRIPT_DIR / "../tmp/data/sim-4-excess.csv").resolve()
    figure_path = PROJECT_ROOT / "figures" / "sim-4.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata("sim-4", {"sample_size": sample_size, "width": width, "height": height, "flip_grid": [0.0, 1.0, 0.02]})

    if not refit and cache_matches(excess_csv, metadata):
        print(f"Loading existing excess-risk CSV from {excess_csv}")
        excess_rows = read_csv_rows(excess_csv)
        width, height = infer_width_height(excess_rows, default_width=width, default_height=height)
    elif not refit and cache_matches(flip_csv, metadata):
        print(f"Loading existing flip CSV from {flip_csv}")
        flip_rows = read_csv_rows(flip_csv)
        baselines = infer_baselines_from_flip_rows(flip_rows)
        excess_rows = build_excess_risk_rows(flip_rows, baselines)
        width, height = infer_width_height(excess_rows, default_width=width, default_height=height)
        save_csv(
            excess_rows,
            excess_csv,
            [
                "index",
                "base_optimizer",
                "flip_prob_31",
                "flip_prob_32",
                "width",
                "height",
                "excess_dice",
                "excess_iou",
            ],
        )
        write_cache_metadata(flip_csv, metadata)
        write_cache_metadata(excess_csv, metadata)
    else:
        if refit:
            print("refit=True, rerunning simulation even if CSV files already exist")
        else:
            print("No existing CSV found, running simulation")
        flip_rows, baselines, scores = run_flip_sweep(
            sample_size=sample_size,
            width=width,
            height=height,
        )
        print_rankseg_summary(width, height, scores)

        excess_rows = build_excess_risk_rows(flip_rows, baselines)
        save_csv(
            flip_rows,
            flip_csv,
            [
                "base_optimizer",
                "flip_prob_31",
                "flip_prob_32",
                "width",
                "height",
                "num_sim_31",
                "num_sim_32",
                "dice_mean",
                "iou_mean",
                "dice_std",
                "iou_std",
            ],
        )
        save_csv(
            excess_rows,
            excess_csv,
            [
                "index",
                "base_optimizer",
                "flip_prob_31",
                "flip_prob_32",
                "width",
                "height",
                "excess_dice",
                "excess_iou",
            ],
        )
        write_cache_metadata(flip_csv, metadata)
        write_cache_metadata(excess_csv, metadata)

    delta = {
        "macro": est_delta_macro(width=width, height=height),
        "micro": est_delta_micro(width=width, height=height),
    }
    print(f"delta_macro: {delta['macro']:.6f}")
    print(f"delta_micro: {delta['micro']:.6f}")
    macro_excess_rows = filter_excess_rows_by_scope(excess_rows, scope_label="macro")
    micro_excess_rows = filter_excess_rows_by_scope(excess_rows, scope_label="micro")
    diagnostics = plot_excess_risks(macro_excess_rows, micro_excess_rows, delta, figure_path)
    diagnostics_csv = (SCRIPT_DIR / "../tmp/data/sim-4-bound-diagnostics.csv").resolve()
    save_csv(diagnostics, diagnostics_csv, ["scope", "num_points", "num_violations", "num_lower_violations", "num_upper_violations", "max_violation", "tolerance"])
    for row in diagnostics:
        print(f"{row['scope']} bound violations: {row['num_violations']}/{row['num_points']}; max={row['max_violation']:.3e}; tolerance={row['tolerance']:.1e}")

    print(f"Flip CSV: {flip_csv}")
    print(f"Excess-risk CSV: {excess_csv}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    simulation_config = paper_config()["simulation"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true", help="rerun simulation instead of loading existing CSV files")
    parser.add_argument("--sample-size", type=int, default=int(simulation_config["repetitions"]))
    parser.add_argument("--width", type=int, default=int(simulation_config["excess_risk"]["dimension"]))
    parser.add_argument("--height", type=int, default=1)
    args = parser.parse_args()
    main(refit=args.refit, sample_size=args.sample_size, width=args.width, height=args.height)
