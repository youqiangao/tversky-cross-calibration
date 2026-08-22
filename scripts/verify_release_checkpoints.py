#!/usr/bin/env python3
"""Evaluate the released checkpoints and build the four paper tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.binary.dataset import default_data_root_for as binary_data_root
from application.binary.delta import compute_delta_for_checkpoint
from application.binary.rank_eval import evaluate_checkpoint as evaluate_binary_rank
from application.binary.utils import resolve_device, save_csv
from application.checkpoints import PAPER_CHECKPOINTS, checkpoint_paths, hf_download_command
from application.multilabel_segmentation.dataset import default_data_root_for as multilabel_data_root
from application.multilabel_segmentation.delta import compute_checkpoint as compute_multilabel_delta
from application.multilabel_segmentation.rank_eval import evaluate_rank_checkpoint
from scripts.build_paper_tables import TABLE_SPECS, build_table
from scripts.check_paper_results import check_paper_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("."),
        help="Root containing downloaded outputs/application paths or an exported Hugging Face tree.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/release_verification"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)

    resolved_checkpoints = []
    missing = []
    for item in PAPER_CHECKPOINTS:
        remote_path, local_path = checkpoint_paths(item)
        local_candidate = args.checkpoint_root / local_path
        remote_candidate = args.checkpoint_root / remote_path
        checkpoint = local_candidate if local_candidate.exists() else remote_candidate
        if checkpoint.exists():
            resolved_checkpoints.append((item, checkpoint))
        else:
            missing.append(f"{local_candidate}\n  {hf_download_command(item)}")
    if missing:
        raise FileNotFoundError(
            "Missing released checkpoints:\n- " + "\n- ".join(missing)
        )

    binary_rank_rows = []
    binary_delta_rows = []
    multilabel_rank_rows = []
    multilabel_delta_rows = []
    for item, checkpoint in resolved_checkpoints:
        dataset = str(item["dataset"])
        model = str(item["model"])
        if item["task"] == "binary":
            binary_rank_rows.extend(
                evaluate_binary_rank(
                    dataset_name=dataset,
                    model_label=model,
                    checkpoint_path=checkpoint,
                    split=args.split,
                    num_workers=args.num_workers,
                    device=device,
                    output_dir=args.output_dir / "binary_rank",
                    max_samples=args.max_samples,
                    refit=args.refit,
                )
            )
            binary_delta_rows.append(
                compute_delta_for_checkpoint(
                    checkpoint, model, dataset, binary_data_root(dataset), args.split, 4,
                    args.num_workers, device, args.output_dir / "binary_delta" / dataset,
                    max_samples=args.max_samples, refit=args.refit,
                )
            )
        else:
            multilabel_rank_rows.extend(
                evaluate_rank_checkpoint(
                    dataset, model, checkpoint, args.split, 1, args.num_workers, device,
                    args.output_dir / "multilabel_rank", args.max_samples, args.refit, True,
                )
            )
            multilabel_delta_rows.append(
                compute_multilabel_delta(
                    checkpoint, model, dataset, multilabel_data_root(dataset), args.split, 2,
                    args.num_workers, device, args.output_dir / "multilabel_delta" / dataset,
                    max_samples=args.max_samples, refit=args.refit,
                )
            )

    summaries = {
        "binary_delta": binary_delta_rows,
        "binary_performance": binary_rank_rows,
        "multilabel_delta": multilabel_delta_rows,
        "multilabel_performance": multilabel_rank_rows,
    }
    summary_dir = args.output_dir / "summaries"
    table_dir = args.output_dir / "tables"
    for name, rows in summaries.items():
        source = summary_dir / f"{name}.csv"
        save_csv(rows, source)
        build_table(source, table_dir / f"{name}.tex", TABLE_SPECS[name])
        print(source)
    if args.max_samples is None:
        issues = check_paper_results(summaries)
        report = args.output_dir / "paper_result_check.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        if issues:
            report.write_text("FAIL\n" + "\n".join(f"- {issue}" for issue in issues) + "\n", encoding="utf-8")
            raise RuntimeError(f"Results differ from the displayed paper values; see {report}")
        report.write_text("PASS: all results match the values displayed in the paper.\n", encoding="utf-8")
        print(report)
    else:
        print("Skipped paper-value comparison because --max-samples was used.")


if __name__ == "__main__":
    main()
