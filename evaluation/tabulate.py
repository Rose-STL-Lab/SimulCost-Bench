#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse evaluation logs and produce CSV / Excel summaries.

Usage
-----
Single task:
    python evaluation/tabulate.py -d 1D_heat_transfer -t cfl

All tasks under a dataset:
    python evaluation/tabulate.py -d 1D_heat_transfer
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from statistics import mean

import pandas as pd

# ------------------------------------------------------------
# Regex patterns (no emojis needed)
# ------------------------------------------------------------
SUMMARY_REGEX = re.compile(r"Evaluation summary:\s*({.*?})", re.S)
QID_REGEX = re.compile(r"QID:\s*(\d+)")

# Preferred ordering for some common metrics (will appear first)
PRIORITY_METRICS = [
    "converged_rate (converge does not guarantee success)",
    "success_rate",
    "model_cost_efficiency",
    "dummy_cost_efficiency",
    "relative_cost_efficiency",
]

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def parse_log(path: Path) -> Tuple[int, Dict[str, str]]:
    """Return (num_samples, metrics_dict) extracted from a single .log file."""
    text = path.read_text(encoding="utf-8", errors="ignore")

    # 1) number of samples = highest QID
    qids = [int(x) for x in QID_REGEX.findall(text)]
    num_samples = max(qids) if qids else 0

    # 2) evaluation summary block
    m = SUMMARY_REGEX.search(text)
    if not m:
        raise ValueError(f"No evaluation summary found in {path}")
    metrics = json.loads(m.group(1))
    return num_samples, metrics


def is_number(v: str) -> bool:
    """Return True if v can be cast to float."""
    try:
        float(v)
        return True
    except Exception:
        return False


def merge_metrics(metrics_list: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple test cases by averaging:
    - If all values can be converted to float, return the average
    - Otherwise, return the first non-empty string value
    """
    merged: Dict[str, float] = {}
    all_keys = set().union(*(m.keys() for m in metrics_list))

    for k in all_keys:
        vals = [m[k] for m in metrics_list if k in m]
        if all(is_number(v) for v in vals):
            merged[k] = mean(float(v) for v in vals)
        else:
            merged[k] = vals[0]  # Keep first value for non-numeric fields
    return merged


def collect_rows(dataset: str, tasks: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Gather all rows (one per log file) and union of metric columns.
    Special handling for burgers_1d: aggregate 5 cases by (model, task, mode).
    """
    base_root = Path("eval_results") / dataset
    rows: List[Dict[str, str]] = []
    metric_names = set()

    # ---------- burgers_1d and euler_1d (requires cross-case aggregation) ----------
    if dataset in ["burgers_1d", "euler_1d"]:
        # key: (model, task, mode)  → {"num_samples": int, "metrics": [dict, ...]}
        agg: Dict[Tuple[str, str, str], Dict[str, object]] = {}

        for task in tasks:
            task_dir = base_root / task
            # Five case directories
            for case_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                for log_path in sorted(case_dir.glob("*.log")):
                    fname = log_path.stem
                    MODE_RE = re.compile(r"^(iterative|zero_shot)_(.+)$")
                    match = MODE_RE.match(fname)
                    if not match:
                        raise ValueError(f"Unrecognized log filename: {fname}")
                    mode_key, model = match.groups()
                    mode = "Iterative" if mode_key == "iterative" else "Zero-shot"

                    num_samples, metrics = parse_log(log_path)
                    metric_names.update(metrics.keys())

                    key = (model, task, mode)
                    bucket = agg.setdefault(
                        key, {"num_samples": 0, "metrics_list": []}
                    )
                    bucket["num_samples"] += num_samples
                    bucket["metrics_list"].append(metrics)

        # Convert aggregated results to rows
        for (model, task, mode), info in agg.items():
            merged_metrics = merge_metrics(info["metrics_list"])  # Average values
            row = {
                "Model": model,
                "Task": task,
                "Inference Mode": mode,
                "Number of Samples": info["num_samples"],
                **merged_metrics,
            }
            rows.append(row)

        metric_columns = sorted(metric_names)
        return rows, metric_columns

    # ---------- Other datasets: original flat structure ----------
    for task in tasks:
        task_dir = base_root / task
        for log_path in sorted(task_dir.glob("*.log")):
            fname = log_path.stem  # e.g. iterative_anthropic.xxx
            MODE_RE = re.compile(r"^(iterative|zero_shot)_(.+)$")

            match = MODE_RE.match(fname)
            if not match:
                raise ValueError(f"Unrecognized log filename: {fname}")
            mode_key, model = match.groups()
            mode = "Iterative" if mode_key == "iterative" else "Zero-shot"

            num_samples, metrics = parse_log(log_path)

            row = {
                "Model": model,
                "Task": task,
                "Inference Mode": mode,
                "Number of Samples": num_samples,
                **metrics,
            }
            rows.append(row)
            metric_names.update(metrics.keys())

    metric_columns = sorted(metric_names)
    return rows, metric_columns


def _ordered_metric_cols(metric_cols: List[str]) -> List[str]:
    """Return metric_cols reordered with PRIORITY_METRICS first, rest alphabetically."""
    return [m for m in PRIORITY_METRICS if m in metric_cols] + [
        m for m in metric_cols if m not in PRIORITY_METRICS
    ]


def write_csv(rows: List[Dict[str, str]], metric_cols: List[str], outfile: Path) -> None:
    """Write rows to CSV."""
    ordered_metrics = _ordered_metric_cols(metric_cols)
    fieldnames = [
        "Model",
        "Task",
        "Inference Mode",
        "Number of Samples",
        *ordered_metrics,
    ]
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_excel(
    rows: List[Dict[str, str]], metric_cols: List[str], outfile: Path
) -> None:
    """Write rows to an Excel workbook (.xlsx) and add a blank row
    between different [Task, Inference Mode] groups.
    """
    ordered_metrics = _ordered_metric_cols(metric_cols)
    ordered_cols = [
        "Model",
        "Task",
        "Inference Mode",
        "Number of Samples",
        *ordered_metrics,
    ]
    df = pd.DataFrame(rows)[ordered_cols]

    # Convert to numeric for easier comparison
    for col in ("success_rate", "relative_cost_efficiency"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace NaN or INF values with the string 'nan'
    for col in df.columns:
        df[col] = df[col].apply(lambda x: 'nan' if pd.isna(x) or (isinstance(x, float) and (x == float('inf') or x == -float('inf'))) else x)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(outfile, engine="xlsxwriter") as writer:
        wb = writer.book
        ws = wb.add_worksheet("Summary")
        writer.sheets["Summary"] = ws

        # Formatting styles
        header_fmt = wb.add_format({"bold": True})
        bold_fmt = wb.add_format({"bold": True})
        green_bold_fmt = wb.add_format({"bold": True, "font_color": "green"})

        # Write header row
        ws.write_row(0, 0, ordered_cols, header_fmt)

        excel_row = 1  # Next Excel row to write to (0-based)

        # Write by group, leaving blank row between groups
        col_idx = {c: i for i, c in enumerate(ordered_cols)}
        for (task, mode), subdf in (
            df.sort_values(["Task", "Inference Mode"])
            .groupby(["Task", "Inference Mode"], sort=False)
        ):
            # Maximum values within group (may be empty)
            max_success = (
                subdf["success_rate"].max() if "success_rate" in subdf else None
            )
            max_rel_eff = (
                subdf["relative_cost_efficiency"].max()
                if "relative_cost_efficiency" in subdf
                else None
            )

            # Write row by row
            for _, row in subdf.iterrows():
                # Regular write operations
                ws.write(excel_row, col_idx["Model"], row["Model"])
                ws.write(excel_row, col_idx["Task"], row["Task"])
                ws.write(excel_row, col_idx["Inference Mode"], row["Inference Mode"])
                ws.write_number(
                    excel_row, col_idx["Number of Samples"], row["Number of Samples"]
                )

                # Other metric columns
                for m in ordered_metrics:
                    val = row[m]
                    if val == 'nan':
                        ws.write(excel_row, col_idx[m], 'nan')  # Write 'nan' as string
                    else:
                        ws.write(excel_row, col_idx[m], val)

                # Bold / green bold formatting logic (only when value is non-zero)
                is_top_success = (
                    max_success is not None
                    and row.get("success_rate", 0) == max_success
                    and row.get("success_rate", 0) != 0
                )
                is_top_rel_eff = (
                    max_rel_eff is not None
                    and row.get("relative_cost_efficiency", 0) == max_rel_eff
                    and row.get("relative_cost_efficiency", 0) != 0
                )

                if is_top_success:
                    ws.write(
                        excel_row,
                        col_idx["success_rate"],
                        row["success_rate"],
                        bold_fmt,
                    )
                if is_top_rel_eff:
                    ws.write(
                        excel_row,
                        col_idx["relative_cost_efficiency"],
                        row["relative_cost_efficiency"],
                        bold_fmt,
                    )

                if is_top_success and is_top_rel_eff:
                    # Both metrics are highest and non-zero → Model in green bold
                    ws.write(excel_row, col_idx["Model"], row["Model"], green_bold_fmt)
                elif is_top_success or is_top_rel_eff:
                    # Only one metric is highest and non-zero → Model in black bold
                    ws.write(excel_row, col_idx["Model"], row["Model"], bold_fmt)

                excel_row += 1

            # End of group → leave one blank row
            excel_row += 1

        # Auto filter + freeze header row
        ws.autofilter(0, 0, excel_row - 1, len(ordered_cols) - 1)
        ws.freeze_panes(1, 0)

        # Auto-adjust column width
        for idx, col in enumerate(ordered_cols):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            ws.set_column(idx, idx, min(max_len, 50))


# ------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse evaluation logs to CSV & Excel"
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="problem type (e.g. 1D_heat_transfer)"
    )
    parser.add_argument(
        "-t",
        "--task",
        help="task type (e.g. cfl). If omitted, process all tasks under the dataset.",
    )
    parser.add_argument("--csv-output", help="custom CSV output filename")
    parser.add_argument("--xlsx-output", help="custom Excel output filename")
    args = parser.parse_args()

    # Validate dataset directory
    eval_root = Path("eval_results")
    dataset_dir = eval_root / args.dataset
    if not dataset_dir.is_dir():
        available = ", ".join(sorted(p.name for p in eval_root.iterdir() if p.is_dir()))
        print(f"❌ Dataset '{args.dataset}' not found in {eval_root}")
        print(f"✅ Available datasets: {available}")
        return

    # Determine task list
    if args.task:
        task_dirs = [args.task]
        if not (dataset_dir / args.task).is_dir():
            available = ", ".join(
                sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
            )
            print(f"❌ Task '{args.task}' not found under dataset '{args.dataset}'")
            print(f"✅ Available tasks for '{args.dataset}': {available}")
            return
    else:
        # gather all subdirectories as tasks
        task_dirs = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
        if not task_dirs:
            print(f"⚠️  No task subdirectories found in '{dataset_dir}'")
            return

    # Collect rows
    rows, metric_cols = collect_rows(args.dataset, task_dirs)
    if not rows:
        print("⚠️  No .log files found - nothing to parse.")
        return

    # Output filenames
    if args.task:
        default_prefix = (
            (dataset_dir / args.task) / f"{args.dataset}_{args.task}_summary"
        )
    else:
        default_prefix = dataset_dir / f"{args.dataset}_summary"

    csv_path = Path(args.csv_output) if args.csv_output else default_prefix.with_suffix(
        ".csv"
    )
    xlsx_path = Path(
        args.xlsx_output
    ) if args.xlsx_output else default_prefix.with_suffix(".xlsx")

    # Write outputs
    write_csv(rows, metric_cols, csv_path)
    write_excel(rows, metric_cols, xlsx_path)

    print("✅ Summary saved:")
    print("   CSV  :", csv_path)
    print("   Excel:", xlsx_path, "(ready for humans ✨)")


if __name__ == "__main__":
    main()
