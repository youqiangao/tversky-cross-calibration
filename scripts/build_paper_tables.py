#!/usr/bin/env python3
"""Build auditable LaTeX tables from real-application summary CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TABLE_SPECS = {
    "binary_delta": ("dataset_name", "model", "delta_hat"),
    "binary_performance": ("dataset_name", "model", "optimizer", "dice", "iou"),
    "multilabel_delta": ("dataset_name", "model", "delta_macro_hat", "delta_micro_hat"),
    "multilabel_performance": ("dataset_name", "model", "optimizer", "macro_dice", "macro_iou", "micro_dice", "micro_iou"),
}


def read_and_validate(path: Path, required_columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} contains no result rows.")
    missing = set(required_columns) - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "split" not in rows[0]:
        raise ValueError(f"{path} must record the evaluation split.")
    unexpected = sorted({row["split"] for row in rows if row["split"] != "test"})
    if unexpected:
        raise ValueError(f"{path} contains non-evaluation splits: {unexpected}")
    return rows


def latex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def render_table(rows: list[dict[str, str]], columns: tuple[str, ...]) -> str:
    alignment = "l" * len(columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(latex_escape(column) for column in columns) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(latex_escape(str(row[column])) for column in columns) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_table(source: Path, output: Path, columns: tuple[str, ...]) -> None:
    rows = read_and_validate(source, columns)
    rows.sort(key=lambda row: tuple(row[column] for column in columns[: min(3, len(columns))]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_table(rows, columns), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in TABLE_SPECS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/paper_tables"))
    args = parser.parse_args()
    for name, columns in TABLE_SPECS.items():
        build_table(getattr(args, name), args.output_dir / f"{name}.tex", columns)
        print(args.output_dir / f"{name}.tex")


if __name__ == "__main__":
    main()
