import argparse
import os
import sys
import csv
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
import seaborn as sns
import torch
from matplotlib.ticker import FormatStrFormatter

from metrics import accuracy, dice_coeff, iou
from tversky_cross_calibration.compat import rank_dice, rank_iou
from tversky_cross_calibration.config import PROJECT_ROOT, paper_config
from tversky_cross_calibration.reproducibility import cache_matches, cache_metadata, seed_everything, write_cache_metadata

DEVICE = "cpu"


def sim(sample_size=100, width=10, height=10):
    num_class = 1

    prob = torch.zeros((sample_size, num_class, width, height), device=DEVICE)
    nonzero_num_pixel = int(width / 10)
    prob[:, :, 0:nonzero_num_pixel, :] = 0.4
    prob[:, :, nonzero_num_pixel : (2 * nonzero_num_pixel), :] = 0.24
    target = torch.bernoulli(prob)

    # Every sample shares the same probability map, so we only need to solve
    # the rank-based inference once and then expand it across the batch.
    prob_single = prob[:1]
    if width < 100:
        predict_iou_single, _, _ = rank_iou(prob_single, device=DEVICE, verbose=0)
        predict_dice_single, _, _ = rank_dice(prob_single, device=DEVICE, verbose=0)
    else:
        predict_iou_single, _, _ = rank_iou(prob_single, device=DEVICE, verbose=0, exact=False)
        predict_dice_single, _, _ = rank_dice(prob_single, device=DEVICE, verbose=0, exact=False)

    predict_iou = predict_iou_single.expand(sample_size, -1, -1, -1)
    predict_dice = predict_dice_single.expand(sample_size, -1, -1, -1)

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

    return prob, iou_score, dice_score, threshold_score, predict_iou, predict_dice, predict_t


def build_plot_dataframe(dice_result, iou_result, threshold_result):
    methods = ["Dice", "IoU", "Pixel Accuracy"]
    results = [dice_result, iou_result, threshold_result]
    plot_data = []

    for method, res in zip(methods, results):
        for result in res:
            width = str(result["width"])

            dice_arr = np.asarray(result["dice"])
            iou_arr = np.asarray(result["iou"])
            acc_arr = np.asarray(result["acc"])

            plot_data.append(
                {
                    "d": width,
                    "optimizer": method,
                    "dice_mean": float(dice_arr.mean()),
                    "dice_std": float(dice_arr.std() / np.sqrt(len(dice_arr))),
                    "iou_mean": float(iou_arr.mean()),
                    "iou_std": float(iou_arr.std() / np.sqrt(len(iou_arr))),
                    "acc_mean": float(acc_arr.mean()),
                    "acc_std": float(acc_arr.std() / np.sqrt(len(acc_arr))),
                }
            )

    return plot_data


def plot_results(plot_data, output_path):
    metrics = [
        ("dice_mean", "dice_std", "Dice"),
        ("iou_mean", "iou_std", "IoU"),
        ("acc_mean", "acc_std", "Pixel Accuracy"),
    ]
    widths = sorted({row["d"] for row in plot_data}, key=int)
    optimizers = ["Dice", "IoU", "Pixel Accuracy"]
    optimizer_styles = {
        "Dice": {"linestyle": "-", "color": "#1f77b4"},
        "IoU": {"linestyle": "--", "color": "#ff7f0e"},
        "Pixel Accuracy": {"linestyle": ":", "color": "#2ca02c"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        x_values = np.arange(len(widths))
        for optimizer in optimizers:
            sub = [row for row in plot_data if row["optimizer"] == optimizer]
            sub = sorted(sub, key=lambda row: int(row["d"]))
            means = np.array([row[mean_col] for row in sub], dtype=float)
            stds = np.array([row[std_col] for row in sub], dtype=float)
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
                means - 1.96 * stds,
                means + 1.96 * stds,
                alpha=0.2,
                color=style["color"],
            )

        ax.grid(False)
        ax.set_ylabel(title, fontsize=20)
        ax.set_xlabel("d", fontsize=20)
        ax.set_xticks(x_values)
        ax.set_xticklabels(widths)

    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].set_yticks(np.arange(0.0, 0.51, 0.10))
    axes[1].set_yticks(np.arange(0.0, 0.36, 0.07))
    axes[2].set_yticks(np.arange(0.86, 0.95, 0.02))
    legend_kwargs = {
        "title": "Optimizer",
        "fontsize": 13,
        "title_fontsize": 13,
    }
    axes[0].legend(loc="center right", **legend_kwargs)
    axes[1].legend(loc="center right", **legend_kwargs)
    axes[2].legend(loc="lower right", **legend_kwargs)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_plot_data_csv(plot_data, output_path):
    fieldnames = [
        "d",
        "optimizer",
        "dice_mean",
        "dice_std",
        "iou_mean",
        "iou_std",
        "acc_mean",
        "acc_std",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plot_data)


def read_plot_data_csv(input_path):
    with input_path.open(newline="") as csv_file:
        return [
            {
                "d": row["d"],
                "optimizer": row["optimizer"],
                "dice_mean": float(row["dice_mean"]),
                "dice_std": float(row["dice_std"]),
                "iou_mean": float(row["iou_mean"]),
                "iou_std": float(row["iou_std"]),
                "acc_mean": float(row["acc_mean"]),
                "acc_std": float(row["acc_std"]),
            }
            for row in csv.DictReader(csv_file)
        ]


def main(refit=False):
    seed_everything(42)
    simulation_config = paper_config()["simulation"]
    sample_size = int(simulation_config["repetitions"])
    dimensions = list(simulation_config["dimensions"])
    csv_path = (SCRIPT_DIR / "../tmp/data/sim-1-1.csv").resolve()
    figure_path = PROJECT_ROOT / "figures" / "sim-1-1.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running simulation on {DEVICE}")

    metadata = cache_metadata("sim-1-1", {"sample_size": sample_size, "dimensions": dimensions})
    if cache_matches(csv_path, metadata) and not refit:
        print(f"Loading existing CSV from {csv_path}")
        plot_results(read_plot_data_csv(csv_path), figure_path)
        print(f"Saved figure to {figure_path}")
        return

    dice_result, iou_result, threshold_result = [], [], []

    for prob_type in ["step"]:
        for width in dimensions:
            height = 1
            (
                _prob,
                iou_score,
                dice_score,
                threshold_score,
                _predict_iou,
                _predict_dice,
                _predict_t,
            ) = sim(sample_size=sample_size, width=width, height=height)

            dice_result.append(
                {
                    "width": width,
                    "height": height,
                    "dice": dice_score["dice"].cpu().numpy(),
                    "iou": dice_score["iou"].cpu().numpy(),
                    "acc": dice_score["acc"].cpu().numpy(),
                }
            )
            iou_result.append(
                {
                    "width": width,
                    "height": height,
                    "dice": iou_score["dice"].cpu().numpy(),
                    "iou": iou_score["iou"].cpu().numpy(),
                    "acc": iou_score["acc"].cpu().numpy(),
                }
            )
            threshold_result.append(
                {
                    "width": width,
                    "height": height,
                    "dice": threshold_score["dice"].cpu().numpy(),
                    "iou": threshold_score["iou"].cpu().numpy(),
                    "acc": threshold_score["acc"].cpu().numpy(),
                }
            )

            print("#" * 20)
            print(f"prob_type: {prob_type}; width: {width}; height: {height}")
            print(
                "rankiou: dice score: %.3f(%.3f); iou score: %.3f(%.3f); acc score: %.3f(%.3f)"
                % (
                    iou_score["dice"].mean(),
                    iou_score["dice"].std() / np.sqrt(sample_size),
                    iou_score["iou"].mean(),
                    iou_score["iou"].std() / np.sqrt(sample_size),
                    iou_score["acc"].mean(),
                    iou_score["acc"].std() / np.sqrt(sample_size),
                )
            )
            print(
                "rankdice: dice score: %.3f(%.3f); iou score: %.3f(%.3f); acc score: %.3f(%.3f)"
                % (
                    dice_score["dice"].mean(),
                    dice_score["dice"].std() / np.sqrt(sample_size),
                    dice_score["iou"].mean(),
                    dice_score["iou"].std() / np.sqrt(sample_size),
                    dice_score["acc"].mean(),
                    dice_score["acc"].std() / np.sqrt(sample_size),
                )
            )
            print(
                "T=0.5: dice score: %.3f(%.3f); iou score: %.3f(%.3f); acc score: %.3f(%.3f)"
                % (
                    threshold_score["dice"].mean(),
                    threshold_score["dice"].std() / np.sqrt(sample_size),
                    threshold_score["iou"].mean(),
                    threshold_score["iou"].std() / np.sqrt(sample_size),
                    threshold_score["acc"].mean(),
                    threshold_score["acc"].std() / np.sqrt(sample_size),
                )
            )

    plot_data = build_plot_dataframe(dice_result, iou_result, threshold_result)
    save_plot_data_csv(plot_data, csv_path)
    write_cache_metadata(csv_path, metadata)
    plot_results(plot_data, figure_path)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true", help="rerun simulation instead of loading existing CSV")
    args = parser.parse_args()
    main(refit=args.refit)
