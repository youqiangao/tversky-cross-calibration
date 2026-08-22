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

from metrics import accuracy, dice_coeff, iou
from tversky_cross_calibration.compat import rank_dice, rank_iou
from tversky_cross_calibration.config import PROJECT_ROOT, paper_config
from tversky_cross_calibration.reproducibility import cache_matches, cache_metadata, seed_everything, write_cache_metadata

DEVICE = "cpu"
C1 = 72.0
ETA = 12.0


def build_prob_map_sim_1_1(width, height):
    prob = torch.zeros((1, 1, width, height), device=DEVICE)
    block_width = int(width / 10)
    prob[:, :, 0:block_width, :] = 0.4
    prob[:, :, block_width : (2 * block_width), :] = 0.24
    return prob


def build_prob_map_sim_1_2(width, height):
    prob = torch.zeros((1, 1, width, height), device=DEVICE)
    nonzero_num_pixel = int(width / 2)
    prob[:, :, 0:nonzero_num_pixel, :] = 0.8
    prob[:, :, nonzero_num_pixel:width, :] = 0.4
    return prob


def solve_rank_predictions(prob_single, exact):
    predict_iou_single, _, _ = rank_iou(prob_single, device=DEVICE, verbose=0, exact=exact)
    predict_dice_single, _, _ = rank_dice(prob_single, device=DEVICE, verbose=0, exact=exact)
    return predict_iou_single, predict_dice_single


def sim(sample_size=100, width=10, height=1, exact=False):
    sim11_mask = torch.rand(sample_size, device=DEVICE) < 0.5
    sim12_mask = ~sim11_mask

    prob_map_11 = build_prob_map_sim_1_1(width=width, height=height)
    prob_map_12 = build_prob_map_sim_1_2(width=width, height=height)
    prob = torch.where(
        sim11_mask.view(sample_size, 1, 1, 1),
        prob_map_11.expand(sample_size, -1, -1, -1),
        prob_map_12.expand(sample_size, -1, -1, -1),
    )
    target = torch.bernoulli(prob)

    predict_iou_11, predict_dice_11 = solve_rank_predictions(prob_map_11, exact=exact)
    predict_iou_12, predict_dice_12 = solve_rank_predictions(prob_map_12, exact=exact)

    predict_iou = torch.empty_like(prob, dtype=torch.bool)
    predict_dice = torch.empty_like(prob, dtype=torch.bool)
    if sim11_mask.any():
        predict_iou[sim11_mask] = predict_iou_11.expand(int(sim11_mask.sum().item()), -1, -1, -1)
        predict_dice[sim11_mask] = predict_dice_11.expand(int(sim11_mask.sum().item()), -1, -1, -1)
    if sim12_mask.any():
        predict_iou[sim12_mask] = predict_iou_12.expand(int(sim12_mask.sum().item()), -1, -1, -1)
        predict_dice[sim12_mask] = predict_dice_12.expand(int(sim12_mask.sum().item()), -1, -1, -1)
    predict_t = prob > 0.5

    iou_score = {
        "dice": dice_coeff(predict_iou, target),
        "iou": iou(predict_iou, target),
        "acc": accuracy(predict_iou, target),
    }
    dice_score = {
        "dice": dice_coeff(predict_dice, target),
        "iou": iou(predict_dice, target),
        "acc": accuracy(predict_dice, target),
    }
    threshold_score = {
        "dice": dice_coeff(predict_t, target),
        "iou": iou(predict_t, target),
        "acc": accuracy(predict_t, target),
    }

    return {
        "prob": prob,
        "target": target,
        "sim11_mask": sim11_mask,
        "sim12_mask": sim12_mask,
        "iou_score": iou_score,
        "dice_score": dice_score,
        "threshold_score": threshold_score,
        "predict_iou": predict_iou,
        "predict_dice": predict_dice,
        "predict_t": predict_t,
    }


def flip_prediction_iid(prediction, flip_prob):
    flip_mask = torch.rand(prediction.shape, device=prediction.device) < flip_prob
    return torch.logical_xor(prediction, flip_mask)


def conditional_flip_prediction(prediction, sim11_mask, flip_prob_11, flip_prob_12):
    flip_prob_per_sample = torch.where(
        sim11_mask,
        torch.full((prediction.shape[0],), flip_prob_11, device=prediction.device),
        torch.full((prediction.shape[0],), flip_prob_12, device=prediction.device),
    )
    flip_mask = torch.rand(prediction.shape, device=prediction.device) < flip_prob_per_sample.view(-1, 1, 1, 1)
    return torch.logical_xor(prediction, flip_mask)


def est_delta(width=10, height=1):
    prob_11 = build_prob_map_sim_1_1(width=width, height=height)
    prob_12 = build_prob_map_sim_1_2(width=width, height=height)
    eta_tensor = torch.tensor(ETA, device=DEVICE)
    sum_11 = prob_11.sum(dim=(1, 2, 3))
    sum_12 = prob_12.sum(dim=(1, 2, 3))
    value_11 = 1.0 / torch.maximum(sum_11, eta_tensor)
    value_12 = 1.0 / torch.maximum(sum_12, eta_tensor)
    return float((0.5 * value_11 + 0.5 * value_12).item())


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


def summarize_prediction_scores(prediction, target):
    return {
        "dice": summarize_metric_values(dice_coeff(prediction, target)),
        "iou": summarize_metric_values(iou(prediction, target)),
        "acc": summarize_metric_values(accuracy(prediction, target)),
    }


def build_group_flip_summaries(base_prediction, target, flip_probs):
    summaries = {}
    for flip_prob in flip_probs:
        flipped_prediction = flip_prediction_iid(base_prediction, float(flip_prob))
        summaries[float(flip_prob)] = summarize_prediction_scores(flipped_prediction, target)
    return summaries


def run_flip_sweep(sample_size=1000, width=10000, height=1, exact=False):
    flip_rows = []
    simulation = sim(
        sample_size=sample_size,
        width=width,
        height=height,
        exact=exact,
    )
    target = simulation["target"]
    sim11_mask = simulation["sim11_mask"]
    sim12_mask = simulation["sim12_mask"]
    base_predictions = [
        (
            "Dice",
            simulation["predict_dice"],
            {
                "dice_mean": float(simulation["dice_score"]["dice"].mean().item()),
                "iou_mean": float(simulation["dice_score"]["iou"].mean().item()),
            },
        ),
        (
            "IoU",
            simulation["predict_iou"],
            {
                "dice_mean": float(simulation["iou_score"]["dice"].mean().item()),
                "iou_mean": float(simulation["iou_score"]["iou"].mean().item()),
            },
        ),
    ]
    flip_probs = np.arange(0.0, 1.0001, 0.02)
    baselines = {}
    num_sim_11 = int(sim11_mask.sum().item())
    num_sim_12 = int(sim12_mask.sum().item())

    for base_optimizer, base_prediction, baseline_metrics in base_predictions:
        baselines[base_optimizer] = baseline_metrics
        print("#" * 20)
        print(f"base_optimizer: {base_optimizer}; width: {width}; height: {height}")
        print(f"Precomputing sim-1-1 flip sweep for {num_sim_11} samples")
        sim11_summaries = build_group_flip_summaries(
            base_prediction[sim11_mask],
            target[sim11_mask],
            flip_probs,
        )
        print(f"Precomputing sim-1-2 flip sweep for {num_sim_12} samples")
        sim12_summaries = build_group_flip_summaries(
            base_prediction[sim12_mask],
            target[sim12_mask],
            flip_probs,
        )
        for flip_prob_11 in flip_probs:
            for flip_prob_12 in flip_probs:
                sim11_score = sim11_summaries[float(flip_prob_11)]
                sim12_score = sim12_summaries[float(flip_prob_12)]
                dice_summary = combine_metric_summaries(sim11_score["dice"], sim12_score["dice"])
                iou_summary = combine_metric_summaries(sim11_score["iou"], sim12_score["iou"])
                acc_summary = combine_metric_summaries(sim11_score["acc"], sim12_score["acc"])
                row = {
                    "base_optimizer": base_optimizer,
                    "flip_prob_11": float(flip_prob_11),
                    "flip_prob_12": float(flip_prob_12),
                    "width": width,
                    "height": height,
                    "num_sim_11": num_sim_11,
                    "num_sim_12": num_sim_12,
                    "dice_mean": dice_summary["mean"],
                    "iou_mean": iou_summary["mean"],
                    "acc_mean": acc_summary["mean"],
                    "dice_std": dice_summary["std_error"],
                    "iou_std": iou_summary["std_error"],
                    "acc_std": acc_summary["std_error"],
                }
                flip_rows.append(row)
        print(f"Finished {base_optimizer}: combined {len(flip_probs) * len(flip_probs)} flip pairs")

    return flip_rows, baselines


def build_excess_risk_rows(flip_rows, baselines):
    excess_rows = []
    for index, row in enumerate(flip_rows):
        baseline = baselines[row["base_optimizer"]]
        excess_rows.append(
            {
                "index": index,
                "base_optimizer": row["base_optimizer"],
                "flip_prob_11": row["flip_prob_11"],
                "flip_prob_12": row["flip_prob_12"],
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
        if np.isclose(row["flip_prob_11"], 0.0) and np.isclose(row["flip_prob_12"], 0.0):
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
    int_fields = {"index", "width", "height", "num_sim_11", "num_sim_12"}
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


def get_zoom_upper(values):
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if len(sorted_values) == 0:
        return 1.0
    candidate_index = max(4, int(np.ceil(0.2 * len(sorted_values))) - 1)
    candidate_index = min(candidate_index, len(sorted_values) - 1)
    candidate = sorted_values[candidate_index]
    max_value = sorted_values[-1]
    if candidate <= 0:
        candidate = max_value * 0.1 if max_value > 0 else 1.0
    zoom_upper = candidate * 1.1
    if max_value > 0:
        zoom_upper = min(zoom_upper, max_value * 0.75)
        zoom_upper = max(zoom_upper, max_value * 0.05)
    return zoom_upper


def get_curve_rows(excess_rows, curve_name, base_optimizer="Dice"):
    optimizer_rows = [row for row in excess_rows if row["base_optimizer"] == base_optimizer]
    if curve_name == "flip_prob_12 = 0":
        curve_rows = [row for row in optimizer_rows if np.isclose(row["flip_prob_12"], 0.0)]
        return sorted(curve_rows, key=lambda row: row["flip_prob_11"])
    if curve_name == "flip_prob_11 = 0":
        curve_rows = [row for row in optimizer_rows if np.isclose(row["flip_prob_11"], 0.0)]
        return sorted(curve_rows, key=lambda row: row["flip_prob_12"])
    if curve_name == "flip_prob_11 = 1":
        curve_rows = [row for row in optimizer_rows if np.isclose(row["flip_prob_11"], 1.0)]
        return sorted(curve_rows, key=lambda row: row["flip_prob_12"])
    if curve_name.startswith("flip_prob_11 = "):
        flip_prob_11 = float(curve_name.removeprefix("flip_prob_11 = "))
        curve_rows = [
            row
            for row in optimizer_rows
            if np.isclose(row["flip_prob_11"], flip_prob_11)
        ]
        return sorted(curve_rows, key=lambda row: row["flip_prob_12"])
    if curve_name == "flip_prob_12 = 1":
        curve_rows = [row for row in optimizer_rows if np.isclose(row["flip_prob_12"], 1.0)]
        return sorted(curve_rows, key=lambda row: row["flip_prob_11"])
    if curve_name.startswith("flip_prob_12 = "):
        flip_prob_12 = float(curve_name.removeprefix("flip_prob_12 = "))
        curve_rows = [
            row
            for row in optimizer_rows
            if np.isclose(row["flip_prob_12"], flip_prob_12)
        ]
        return sorted(curve_rows, key=lambda row: row["flip_prob_11"])
    if curve_name == "diagonal average":
        curve_rows = [
            row
            for row in optimizer_rows
            if np.isclose(row["flip_prob_11"], row["flip_prob_12"])
        ]
        return sorted(curve_rows, key=lambda row: row["flip_prob_11"])
    raise ValueError(f"Unknown curve name: {curve_name}")


def plot_reference_curves(ax, excess_rows):
    curve_styles = {
        0.0: {"color": "#d62728", "label": "flip_prob_11 = 0"},
        0.1: {"color": "#ff7f0e", "label": "flip_prob_11 = 0.1"},
    }
    for flip_prob_11, style in curve_styles.items():
        curve_rows = get_curve_rows(excess_rows, f"flip_prob_11 = {flip_prob_11:.1f}")
        curve_rows = [
            row
            for row in curve_rows
            if np.isclose(row["flip_prob_12"] % 0.05, 0.0)
            or np.isclose(row["flip_prob_12"] % 0.05, 0.05)
        ]
        x_values = np.array([row["excess_iou"] for row in curve_rows], dtype=float)
        y_values = np.array([row["excess_dice"] for row in curve_rows], dtype=float)
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            linestyle="-",
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=style["label"],
        )


def plot_excess_risks(excess_rows, delta, output_path):
    excess_iou = np.array([row["excess_iou"] for row in excess_rows], dtype=float)
    excess_dice = np.array([row["excess_dice"] for row in excess_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    upper_bound_style = {"color": "red", "linestyle": "--"}
    lower_bound_style = {"color": "green", "linestyle": "-."}
    upper_delta_term = C1 * delta
    lower_delta_term = 0.5 * C1 * delta

    x_zoom_min = 0.0
    x_zoom_max = 0.1
    y_zoom_min = -0.004
    y_zoom_max = 0.2
    ax.scatter(excess_iou, excess_dice, s=16)
    ax.set_xlabel("Excess IoU risk", fontsize=20)
    ax.set_ylabel("Excess Dice risk", fontsize=20)
    x_line1 = np.linspace(x_zoom_min, x_zoom_max, 100)
    ax.plot(x_line1, 2 * x_line1 + upper_delta_term, label="upper bound", **upper_bound_style)
    ax.plot(
        x_line1,
        0.5 * x_line1 - lower_delta_term,
        label="lower bound",
        **lower_bound_style,
    )
    ax.set_xlim(-0.003, x_zoom_max)
    ax.set_ylim(-0.006, y_zoom_max)
    ax.set_xticks(np.arange(0.0, 0.1001, 0.02))
    ax.set_yticks(np.arange(0.0, 0.2001, 0.05))
    ax.legend()
    ax.grid(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(refit=False, sample_size=10000, width=50000, height=1):
    seed_everything(42)
    print(f"Running sim-2 on {DEVICE}")

    flip_csv = (SCRIPT_DIR / "../tmp/data/sim-2-flip.csv").resolve()
    excess_csv = (SCRIPT_DIR / "../tmp/data/sim-2-excess.csv").resolve()
    figure_path = PROJECT_ROOT / "figures" / "sim-2.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata("sim-2", {"sample_size": sample_size, "width": width, "height": height, "flip_grid": [0.0, 1.0, 0.02]})

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
            ["index", "base_optimizer", "flip_prob_11", "flip_prob_12", "width", "height", "excess_dice", "excess_iou"],
        )
        write_cache_metadata(flip_csv, metadata)
        write_cache_metadata(excess_csv, metadata)
    else:
        if refit:
            print("refit=True, rerunning simulation even if CSV files already exist")
        else:
            print("No existing CSV found, running simulation")
        flip_rows, baselines = run_flip_sweep(sample_size=sample_size, width=width, height=height)
        excess_rows = build_excess_risk_rows(flip_rows, baselines)
        save_csv(
            flip_rows,
            flip_csv,
            [
                "base_optimizer",
                "flip_prob_11",
                "flip_prob_12",
                "width",
                "height",
                "num_sim_11",
                "num_sim_12",
                "dice_mean",
                "iou_mean",
                "acc_mean",
                "dice_std",
                "iou_std",
                "acc_std",
            ],
        )
        save_csv(
            excess_rows,
            excess_csv,
            ["index", "base_optimizer", "flip_prob_11", "flip_prob_12", "width", "height", "excess_dice", "excess_iou"],
        )
        write_cache_metadata(flip_csv, metadata)
        write_cache_metadata(excess_csv, metadata)

    delta = est_delta(width=width, height=height)
    print(f"delta: {delta:.6f}")
    plot_excess_risks(excess_rows, delta, figure_path)

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
