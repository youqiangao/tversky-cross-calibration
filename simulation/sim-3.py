import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))
MPLCONFIG_DIR = (SCRIPT_DIR / "../tmp/matplotlib").resolve()
FONTCONFIG_DIR = (SCRIPT_DIR / "../tmp/fontconfig").resolve()
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
FONTCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
os.environ["XDG_CACHE_HOME"] = str(FONTCONFIG_DIR.parent)

from metrics import dice_coeff, iou, macro_dice_coeff, macro_iou
from tversky_cross_calibration import predict_rank
from tversky_cross_calibration.config import PROJECT_ROOT, paper_config
from tversky_cross_calibration.reproducibility import cache_matches, cache_metadata, seed_everything, write_cache_metadata

DEVICE = "cpu"
SIMULATION_CONFIG = paper_config()["simulation"]
SAMPLE_SIZE = int(SIMULATION_CONFIG["repetitions"])
WIDTHS = list(SIMULATION_CONFIG["dimensions"])
OUTPUT_CSV = (SCRIPT_DIR / "../tmp/data/sim-3.csv").resolve()
OUTPUT_MACRO_FIGURE = PROJECT_ROOT / "figures" / "sim-3-macro.png"
OUTPUT_MICRO_FIGURE = PROJECT_ROOT / "figures" / "sim-3-micro.png"
OUTPUT_COMBINED_FIGURE = PROJECT_ROOT / "figures" / "sim-3.png"
CONFIDENCE_MULTIPLIER = 1.96


def build_probability_map(sample_size, width, height):
    num_class = 3
    prob = torch.zeros((sample_size, num_class, width, height), device=DEVICE)
    block_width = int(width / 10)

    prob[:, 0, 0:block_width, :] = 0.9
    prob[:, 1, 0:block_width, :] = 0.5
    prob[:, 2, 0:block_width, :] = 0.4

    prob[:, 0, block_width : (2 * block_width), :] = 0.8
    prob[:, 1, block_width : (2 * block_width), :] = 0.29
    prob[:, 2, block_width : (2 * block_width), :] = 0.24

    prob[:, 0, 2 * block_width :, :] = 0.5
    return prob


def evaluate_predictions(target, prediction_iou, prediction_dice):
    return {
        "macro-IoU": {
            "dice": macro_dice_coeff(prediction_iou, target),
            "iou": macro_iou(prediction_iou, target),
        },
        "macro-Dice": {
            "dice": macro_dice_coeff(prediction_dice, target),
            "iou": macro_iou(prediction_dice, target),
        },
    }


def evaluate_micro_predictions(target, prediction_iou, prediction_dice):
    return {
        "micro-IoU": {
            "dice": dice_coeff(prediction_iou, target),
            "iou": iou(prediction_iou, target),
        },
        "micro-Dice": {
            "dice": dice_coeff(prediction_dice, target),
            "iou": iou(prediction_dice, target),
        },
    }


def run_rankseg_once(prob_single):
    return predict_rank(prob_single, metric="iou"), predict_rank(prob_single, metric="dice")


def reshape_micro_prediction(prediction_flat, reference_shape):
    return prediction_flat.reshape(reference_shape)


def run_single_simulation(sample_size=100, width=10, height=1):
    prob = build_probability_map(sample_size=sample_size, width=width, height=height)
    target = torch.bernoulli(prob)
    prob_single = prob[:1]
    # Macro optimization: solve rankseg independently per class.
    macro_prediction_iou_single, macro_prediction_dice_single = run_rankseg_once(prob_single)
    macro_prediction_iou = macro_prediction_iou_single.expand(sample_size, -1, -1, -1)
    macro_prediction_dice = macro_prediction_dice_single.expand(sample_size, -1, -1, -1)
    macro_scores = evaluate_predictions(target, macro_prediction_iou, macro_prediction_dice)

    # Micro optimization: flatten class and spatial dimensions into one channel,
    # solve rankseg once, then reshape the prediction back.
    prob_micro_single = prob_single.reshape(1, 1, -1, 1)
    micro_prediction_iou_flat, micro_prediction_dice_flat = run_rankseg_once(prob_micro_single)
    micro_prediction_iou_single = reshape_micro_prediction(micro_prediction_iou_flat, prob_single.shape)
    micro_prediction_dice_single = reshape_micro_prediction(micro_prediction_dice_flat, prob_single.shape)
    micro_prediction_iou = micro_prediction_iou_single.expand(sample_size, -1, -1, -1)
    micro_prediction_dice = micro_prediction_dice_single.expand(sample_size, -1, -1, -1)
    micro_scores = evaluate_micro_predictions(target, micro_prediction_iou, micro_prediction_dice)

    return {
        "width": width,
        "height": height,
        "macro_scores": macro_scores,
        "micro_scores": micro_scores,
    }


def summarize_metric(metric_values):
    values = np.asarray(metric_values, dtype=float)
    return float(values.mean()), float(values.std() / np.sqrt(len(values)))


def print_simulation_summary(sim_result):
    width = sim_result["width"]
    height = sim_result["height"]

    print("#" * 20)
    print(f"width: {width}; height: {height}")
    for scope_label, scores in [("macro", sim_result["macro_scores"]), ("micro", sim_result["micro_scores"])]:
        for optimizer in [f"{scope_label}-IoU", f"{scope_label}-Dice"]:
            dice_mean, dice_std = summarize_metric(scores[optimizer]["dice"].cpu().numpy())
            iou_mean, iou_std = summarize_metric(scores[optimizer]["iou"].cpu().numpy())
            print(
                f"{optimizer}: "
                f"dice score: {dice_mean:.3f}({dice_std:.3f}); "
                f"iou score: {iou_mean:.3f}({iou_std:.3f})"
            )


def run_experiments(sample_size, widths):
    results = []
    for width in widths:
        sim_result = run_single_simulation(sample_size=sample_size, width=width, height=1)
        print_simulation_summary(sim_result)
        results.append(sim_result)
    return results


def build_plot_rows(results, score_key, scope_label):
    plot_rows = []
    for result in results:
        width = str(result["width"])
        for optimizer, score_dict in result[score_key].items():
            dice_mean, dice_std = summarize_metric(score_dict["dice"].cpu().numpy())
            iou_mean, iou_std = summarize_metric(score_dict["iou"].cpu().numpy())
            plot_rows.append(
                {
                    "d": width,
                    "scope": scope_label,
                    "optimizer": optimizer,
                    "dice_mean": dice_mean,
                    "dice_std": dice_std,
                    "iou_mean": iou_mean,
                    "iou_std": iou_std,
                }
            )
    return plot_rows


def save_plot_rows_csv(plot_rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["d", "scope", "optimizer", "dice_mean", "dice_std", "iou_mean", "iou_std"]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plot_rows)


def load_plot_rows_csv(output_path):
    with output_path.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for row in reader:
            rows.append(
                {
                    "d": row["d"],
                    "scope": row["scope"],
                    "optimizer": row["optimizer"],
                    "dice_mean": float(row["dice_mean"]),
                    "dice_std": float(row["dice_std"]),
                    "iou_mean": float(row["iou_mean"]),
                    "iou_std": float(row["iou_std"]),
                }
            )
    return rows


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


def plot_metric_panels(
    plot_rows,
    axes,
    scope_label,
    left_axis_override=None,
    right_axis_override=None,
    legend_loc="center right",
):
    metrics = [
        ("dice_mean", "dice_std", "Dice"),
        ("iou_mean", "iou_std", "IoU"),
    ]
    widths = sorted({row["d"] for row in plot_rows}, key=int)
    optimizers = sorted({row["optimizer"] for row in plot_rows}, key=lambda name: ("IoU" in name, name))
    optimizer_styles = {
        "macro-Dice": {"linestyle": "-", "color": "#1f77b4"},
        "macro-IoU": {"linestyle": "--", "color": "#ff7f0e"},
        "micro-Dice": {"linestyle": "-", "color": "#1f77b4"},
        "micro-IoU": {"linestyle": "--", "color": "#ff7f0e"},
    }

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        x_values = np.arange(len(widths))
        all_bounds = []
        for optimizer in optimizers:
            sub = [row for row in plot_rows if row["optimizer"] == optimizer]
            sub = sorted(sub, key=lambda row: int(row["d"]))
            means = np.array([row[mean_col] for row in sub], dtype=float)
            stds = np.array([row[std_col] for row in sub], dtype=float)
            lower = means - CONFIDENCE_MULTIPLIER * stds
            upper = means + CONFIDENCE_MULTIPLIER * stds
            all_bounds.extend(lower.tolist())
            all_bounds.extend(upper.tolist())
            style = optimizer_styles[optimizer]
            ax.plot(
                x_values,
                means,
                marker="o",
                label=optimizer,
                linestyle=style["linestyle"],
                color=style["color"],
            )
            ax.fill_between(
                x_values,
                lower,
                upper,
                alpha=0.2,
                color=style["color"],
            )

        ax.grid(False)
        ax.set_ylabel(f"{scope_label}-{title}", fontsize=20)
        ax.set_xlabel("d", fontsize=20)
        ax.set_xticks(x_values)
        ax.set_xticklabels(widths)
        ax.legend(
            loc=legend_loc,
            title="Optimizer",
            fontsize=12,
            title_fontsize=12,
        )
        y_min, y_max = get_axis_limits(all_bounds)
        ax.set_ylim(y_min, y_max)

    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if left_axis_override is not None:
        y_min, y_max, y_ticks = left_axis_override
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_yticks(y_ticks)
    else:
        axes[0].set_yticks(build_two_decimal_ticks(*axes[0].get_ylim(), step=0.01))
    axes[0].minorticks_off()
    if right_axis_override is not None:
        y_min, y_max, y_ticks = right_axis_override
        axes[1].set_ylim(y_min, y_max)
        axes[1].set_yticks(y_ticks)
    else:
        axes[1].set_yticks(build_two_decimal_ticks(*axes[1].get_ylim(), step=0.01))
    axes[1].minorticks_off()

    return axes


def plot_results(
    plot_rows,
    output_path,
    scope_label,
    left_axis_override=None,
    right_axis_override=None,
    legend_loc="center right",
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    plot_metric_panels(
        plot_rows,
        axes,
        scope_label=scope_label,
        left_axis_override=left_axis_override,
        right_axis_override=right_axis_override,
        legend_loc=legend_loc,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_results(micro_plot_rows, macro_plot_rows, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True)
    plot_metric_panels(
        micro_plot_rows,
        axes[0],
        scope_label="micro",
        legend_loc="lower right",
    )
    plot_metric_panels(
        macro_plot_rows,
        axes[1],
        scope_label="macro",
        left_axis_override=(0.50, 0.60, np.arange(0.50, 0.601, 0.02)),
        right_axis_override=(0.42, 0.46, np.arange(0.42, 0.461, 0.01)),
        legend_loc="lower right",
    )
    for ax in axes[0]:
        ax.tick_params(axis="x", which="both", labelbottom=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Run sim-3 and generate plots.")
    parser.add_argument("--refit", action="store_true", help="Rerun the simulation even if cached CSV exists.")
    return parser.parse_args()


def main(refit=False):
    seed_everything(42)
    OUTPUT_COMBINED_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata("sim-3", {"sample_size": SAMPLE_SIZE, "dimensions": WIDTHS, "num_classes": 3})
    if cache_matches(OUTPUT_CSV, metadata) and not refit:
        print(f"Loading cached sim-3 results from {OUTPUT_CSV}")
        plot_rows = load_plot_rows_csv(OUTPUT_CSV)
        macro_plot_rows = [row for row in plot_rows if row["scope"] == "macro"]
        micro_plot_rows = [row for row in plot_rows if row["scope"] == "micro"]
    else:
        print(f"Running sim-3 on {DEVICE}")
        results = run_experiments(sample_size=SAMPLE_SIZE, widths=WIDTHS)
        macro_plot_rows = build_plot_rows(results, score_key="macro_scores", scope_label="macro")
        micro_plot_rows = build_plot_rows(results, score_key="micro_scores", scope_label="micro")
        save_plot_rows_csv(macro_plot_rows + micro_plot_rows, OUTPUT_CSV)
        write_cache_metadata(OUTPUT_CSV, metadata)

    plot_results(
        macro_plot_rows,
        OUTPUT_MACRO_FIGURE,
        scope_label="macro",
        left_axis_override=(0.50, 0.60, np.arange(0.50, 0.601, 0.02)),
        right_axis_override=(0.42, 0.46, np.arange(0.42, 0.461, 0.01)),
        legend_loc="lower right",
    )
    plot_results(micro_plot_rows, OUTPUT_MICRO_FIGURE, scope_label="micro", legend_loc="lower right")
    plot_combined_results(micro_plot_rows, macro_plot_rows, OUTPUT_COMBINED_FIGURE)
    print(f"Saved CSV to {OUTPUT_CSV}")
    print(f"Saved macro figure to {OUTPUT_MACRO_FIGURE}")
    print(f"Saved micro figure to {OUTPUT_MICRO_FIGURE}")
    print(f"Saved combined figure to {OUTPUT_COMBINED_FIGURE}")


if __name__ == "__main__":
    args = parse_args()
    main(refit=args.refit)
