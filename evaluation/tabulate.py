#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse evaluation logs and produce CSV / Excel summaries.

Usage
-----
Single task:
    python evaluation/tabulate.py -d heat_1d -t cfl

All tasks under a dataset:
    python evaluation/tabulate.py -d heat_1d
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
# Regex patterns and field standardization
# ------------------------------------------------------------
SUMMARY_REGEX = re.compile(r"🧾 Evaluation [Ss]ummary.*?:\s*({.*?})", re.S)
QID_REGEX = re.compile(r"QID:\s*(\d+)")

# Field name standardization mapping: maps variant names to canonical names
FIELD_NAME_MAPPING = {
    # Converged rate variations
    "converged_rate (converge does not guarantee success)": "converged_rate",
    
    # RMSE variations
    "mean_RMSE": "mean_rmse",
    
    # Tolerance/threshold variations  
    "tolerances": "rmse_tolerance",
    
}

# Preferred ordering for some common metrics (will appear first)
PRIORITY_METRICS = [
    "converged_rate",
    "success_rate",
    "mean_efficiency",
    "mean_rmse",
    "rmse_tolerance",
]

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def clean_model_names(model_name: str) -> str:
    """
    Clean and shorten model names for better display.

    Args:
        model_name: Original model name

    Returns:
        Cleaned model name
    """
    name_mapping = {
        # Support both underscore and colon formats
        'amazon.nova-premier-v1:0': 'Nova-Premier',
        'amazon.nova-premier-v1_0': 'Nova-Premier',
        'anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude-3.7-Sonnet',
        'anthropic.claude-3-7-sonnet-20250219-v1_0': 'Claude-3.7-Sonnet',
        'mistral.mistral-large-2402-v1:0': 'Mistral-Large',
        'mistral.mistral-large-2402-v1_0': 'Mistral-Large',
        'meta.llama3-70b-instruct-v1:0': 'Llama-3-70B-Instruct',
        'meta.llama3-70b-instruct-v1_0': 'Llama-3-70B-Instruct',
        'gpt-5-2025-08-07': 'GPT-5',
        'qwen3_32b': 'Qwen3-32B',

        'qwen3_0_6b': 'Qwen3-0.6B',
        'qwen3_8b': 'Qwen3-8B',
        'anthropic.claude-3-5-haiku-20241022-v1:0': 'Claude-3.5-Haiku',
        'anthropic.claude-3-5-haiku-20241022-v1_0': 'Claude-3.5-Haiku',
        'anthropic.claude-3-5-sonnet-20240620-v1:0': 'Claude-3.5-Sonnet',
        'anthropic.claude-3-5-sonnet-20240620-v1_0': 'Claude-3.5-Sonnet',
    }

    return name_mapping.get(model_name, model_name)

def normalize_metrics(metrics: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize field names to canonical forms using FIELD_NAME_MAPPING.
    
    Args:
        metrics: Raw metrics dictionary with potentially inconsistent field names
        
    Returns:
        Normalized metrics dictionary with standardized field names
    """
    normalized = {}
    for key, value in metrics.items():
        # Use canonical name if mapping exists, otherwise keep original
        canonical_key = FIELD_NAME_MAPPING.get(key, key)
        normalized[canonical_key] = value
    return normalized
def parse_log(path: Path) -> Tuple[int, Dict[str, str]] | None:
    """
    Extract metrics from a single .log file with field name standardization.
    
    Args:
        path: Path to the .log file
        
    Returns:
        Tuple of (num_samples, normalized_metrics_dict) or None if parsing fails
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")

        # 1) Number of samples = highest QID found
        qids = [int(x) for x in QID_REGEX.findall(text)]
        num_samples = max(qids) if qids else 0

        # 2) Extract evaluation summary block
        match = SUMMARY_REGEX.search(text)
        if not match:
            print(f"Warning: No evaluation summary found in {path}, skipping...")
            return None
            
        # 3) Parse JSON and normalize field names
        raw_metrics = json.loads(match.group(1))
        normalized_metrics = normalize_metrics(raw_metrics)
        
        return num_samples, normalized_metrics
        
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {path}: {e}, skipping...")
        return None
    except Exception as e:
        print(f"Warning: Failed to parse {path}: {e}, skipping...")
        return None


def is_number(v) -> bool:
    """Return True if v can be cast to float."""
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def merge_metrics(metrics_list: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple test cases with robust handling.
    
    For each metric:
    - If all values are numeric, return the average
    - If values are mixed (numeric/string), prioritize numeric and average them
    - If all values are non-numeric strings, return the first non-empty value
    
    Args:
        metrics_list: List of metrics dictionaries from different test cases
        
    Returns:
        Merged metrics dictionary with aggregated values
    """
    if not metrics_list:
        return {}
        
    merged = {}
    all_keys = set().union(*(m.keys() for m in metrics_list))

    for key in all_keys:
        # Collect all values for this key across test cases
        all_values = [m[key] for m in metrics_list if key in m and m[key] is not None]
        
        if not all_values:
            continue
            
        # Separate numeric and non-numeric values
        numeric_values = [float(v) for v in all_values if is_number(v)]
        non_numeric_values = [v for v in all_values if not is_number(v)]
        
        if numeric_values:
            # If we have numeric values, use their average
            merged[key] = mean(numeric_values)
        elif non_numeric_values:
            # If only non-numeric values, use the first non-empty one
            merged[key] = next(v for v in non_numeric_values if str(v).strip())
            
    return merged


def collect_rows(dataset: str, tasks: List[str], precision_level: str = None) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Gather all rows (one per log file) and union of metric columns.
    All datasets now support precision levels with cross-case aggregation.
    If precision_level is specified, only process that precision level.
    """
    base_root = Path("eval_results") / dataset
    rows: List[Dict[str, str]] = []
    metric_names = set()

    # key: (model, task, mode)  → {"num_samples": int, "metrics": [dict, ...]}
    agg: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    for task in tasks:
        task_dir = base_root / task
        # Filter by precision_level if specified, otherwise process all precision levels
        if precision_level:
            case_dirs = [p for p in task_dir.iterdir() if p.is_dir() and p.name == precision_level]
        else:
            case_dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())
        
        for case_dir in case_dirs:
            for log_path in sorted(case_dir.glob("*.log")):
                fname = log_path.stem
                MODE_RE = re.compile(r"^(iterative|zero_shot)_(.+)$")
                match = MODE_RE.match(fname)
                if not match:
                    raise ValueError(f"Unrecognized log filename: {fname}")
                mode_key, model = match.groups()
                mode = "Iterative" if mode_key == "iterative" else "Zero-shot"

                # Clean model name
                model = clean_model_names(model)

                result = parse_log(log_path)
                if result is None:
                    continue
                num_samples, metrics = result
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

    # Sort rows: zero-shot first, then iterative, then by task
    def sort_key(row):
        mode_order = 0 if row["Inference Mode"] == "Zero-shot" else 1
        return (mode_order, row["Task"], row["Model"])

    rows.sort(key=sort_key)

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
    
    # Format specific metrics to 2 decimal places
    formatted_rows = []
    for row in rows:
        formatted_row = row.copy()
        for metric in ['converged_rate', 'success_rate', 'mean_efficiency']:
            if metric in formatted_row and isinstance(formatted_row[metric], (int, float)):
                formatted_row[metric] = f"{formatted_row[metric]:.2f}"
        formatted_rows.append(formatted_row)
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(formatted_rows)


def write_csv_by_mode(rows: List[Dict[str, str]], metric_cols: List[str], base_outfile: Path) -> None:
    """Write rows to CSV, split by inference mode into separate folders."""
    # Group rows by inference mode
    mode_groups = {}
    for row in rows:
        mode = row["Inference Mode"].lower().replace("-", "_")
        if mode not in mode_groups:
            mode_groups[mode] = []
        mode_groups[mode].append(row)
    
    # Write each mode to its own folder
    for mode, mode_rows in mode_groups.items():
        mode_dir = base_outfile.parent / mode
        mode_file = mode_dir / base_outfile.name
        write_csv(mode_rows, metric_cols, mode_file)


def write_excel_by_mode(rows: List[Dict[str, str]], metric_cols: List[str], base_outfile: Path) -> None:
    """Write rows to Excel, split by inference mode into separate folders."""
    # Group rows by inference mode
    mode_groups = {}
    for row in rows:
        mode = row["Inference Mode"].lower().replace("-", "_")
        if mode not in mode_groups:
            mode_groups[mode] = []
        mode_groups[mode].append(row)
    
    # Write each mode to its own folder
    for mode, mode_rows in mode_groups.items():
        mode_dir = base_outfile.parent / mode
        mode_file = mode_dir / base_outfile.name
        write_excel(mode_rows, metric_cols, mode_file)


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
    
    # Format specific metrics to 2 decimal places for Excel
    formatted_rows = []
    for row in rows:
        formatted_row = row.copy()
        for metric in ['converged_rate', 'success_rate', 'mean_efficiency']:
            if metric in formatted_row and isinstance(formatted_row[metric], (int, float)):
                formatted_row[metric] = f"{formatted_row[metric]:.2f}"
        formatted_rows.append(formatted_row)
    
    df = pd.DataFrame(formatted_rows)[ordered_cols]

    # Convert to numeric for easier comparison
    for col in ["mean_efficiency"]:
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

        # Formatting styles with color coding for inference modes
        header_fmt = wb.add_format({"bold": True})
        bold_fmt = wb.add_format({"bold": True})
        
        # Iterative mode rows - light green background
        iterative_fmt = wb.add_format({
            'border': 1,
            'bg_color': '#E8F5E8'  # Light green
        })
        
        # Zero-shot mode rows - light blue background
        zeroshot_fmt = wb.add_format({
            'border': 1,
            'bg_color': '#E8F0FF'  # Light blue
        })
        
        # Best performance highlighting with color coding
        best_iterative_fmt = wb.add_format({
            'bold': True,
            'border': 1,
            'bg_color': '#E8F5E8'  # Light green
        })
        
        best_zeroshot_fmt = wb.add_format({
            'bold': True,
            'border': 1,
            'bg_color': '#E8F0FF'  # Light blue
        })

        # Write header row
        ws.write_row(0, 0, ordered_cols, header_fmt)

        excel_row = 1  # Next Excel row to write to (0-based)

        # Write by group, leaving blank row between groups
        col_idx = {c: i for i, c in enumerate(ordered_cols)}
        # Sort by Task, then Inference Mode (Zero-shot first, then Iterative)
        df_sorted = df.copy()
        df_sorted["mode_order"] = df_sorted["Inference Mode"].map({"Zero-shot": 0, "Iterative": 1})
        df_sorted = df_sorted.sort_values(["Task", "mode_order", "Model"])

        for (task, mode), subdf in (
            df_sorted.groupby(["Task", "Inference Mode"], sort=False)
        ):
            # Maximum efficiency within group (may be empty)
            max_efficiency = (
                subdf["mean_efficiency"][subdf["mean_efficiency"] != 'nan'].max()
                if "mean_efficiency" in subdf and len(subdf["mean_efficiency"][subdf["mean_efficiency"] != 'nan']) > 0
                else None
            )

            # Write row by row
            for _, row in subdf.iterrows():
                # Determine formatting based on inference mode
                inference_mode = row["Inference Mode"]
                
                # Bold formatting logic for highest efficiency (only when value is non-zero)
                is_top_efficiency = (
                    max_efficiency is not None
                    and row.get("mean_efficiency", 0) == max_efficiency
                    and row.get("mean_efficiency", 0) != 0
                )
                
                # Choose background color format based on inference mode only
                if inference_mode.lower() == 'iterative':
                    bg_fmt = iterative_fmt
                elif inference_mode.lower() == 'zero-shot':
                    bg_fmt = zeroshot_fmt
                else:
                    # Fallback to no format
                    bg_fmt = None

                # Write cells with background color formatting
                ws.write(excel_row, col_idx["Model"], row["Model"], bg_fmt)
                ws.write(excel_row, col_idx["Task"], row["Task"], bg_fmt)
                ws.write(excel_row, col_idx["Inference Mode"], row["Inference Mode"], bg_fmt)
                ws.write_number(
                    excel_row, col_idx["Number of Samples"], row["Number of Samples"], bg_fmt
                )

                # Other metric columns
                for m in ordered_metrics:
                    val = row[m]
                    if val == 'nan':
                        ws.write(excel_row, col_idx[m], 'nan', bg_fmt)  # Write 'nan' as string
                    else:
                        ws.write(excel_row, col_idx[m], val, bg_fmt)

                # Apply bold formatting ONLY to model name and efficiency for top performers
                if is_top_efficiency:
                    # Choose bold format with appropriate background color
                    if inference_mode.lower() == 'iterative':
                        bold_fmt_with_bg = best_iterative_fmt
                    elif inference_mode.lower() == 'zero-shot':
                        bold_fmt_with_bg = best_zeroshot_fmt
                    else:
                        bold_fmt_with_bg = bold_fmt

                    # Overwrite model name and efficiency with bold formatting
                    ws.write(excel_row, col_idx["Model"], row["Model"], bold_fmt_with_bg)
                    if "mean_efficiency" in col_idx:
                        efficiency_val = row["mean_efficiency"]
                        if efficiency_val != 'nan':
                            ws.write(excel_row, col_idx["mean_efficiency"], efficiency_val, bold_fmt_with_bg)

                excel_row += 1

            # End of group → leave one blank row
            excel_row += 1

        # Auto filter + freeze header row
        ws.autofilter(0, 0, excel_row - 1, len(ordered_cols) - 1)
        ws.freeze_panes(1, 0)

        # Add a legend at the bottom
        legend_row = excel_row + 1
        legend_fmt = wb.add_format({'italic': True, 'font_size': 9})
        iterative_legend_fmt = wb.add_format({'italic': True, 'font_size': 9, 'bg_color': '#E8F5E8'})
        zeroshot_legend_fmt = wb.add_format({'italic': True, 'font_size': 9, 'bg_color': '#E8F0FF'})
        
        ws.write(legend_row, 0, "Legend:", legend_fmt)
        ws.write(legend_row + 1, 0, "Iterative mode", iterative_legend_fmt)
        ws.write(legend_row + 2, 0, "Zero-shot mode", zeroshot_legend_fmt)
        ws.write(legend_row + 3, 0, "Best efficiency model in each task/mode group shown in bold", legend_fmt)
        ws.write(legend_row + 4, 0, "Blank rows separate task and inference mode groups", legend_fmt)

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
        "-d", "--dataset", required=True, help="problem type (e.g. heat_1d)"
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

    # All datasets now support precision levels
    precision_levels = ["high", "medium", "low"]
    for precision in precision_levels:
        # Collect rows for this precision level
        rows, metric_cols = collect_rows(args.dataset, task_dirs, precision_level=precision)
        if not rows:
            print(f"⚠️  No .log files found for precision level '{precision}' - skipping.")
            continue

        # Output filenames with precision level
        if args.task:
            default_prefix = (
                (dataset_dir / args.task) / f"{args.dataset}_{args.task}_{precision}_summary"
            )
        else:
            default_prefix = dataset_dir / f"{args.dataset}_{precision}_summary"

        csv_path = Path(args.csv_output.replace('.csv', f'_{precision}.csv')) if args.csv_output else default_prefix.with_suffix(".csv")
        xlsx_path = Path(args.xlsx_output.replace('.xlsx', f'_{precision}.xlsx')) if args.xlsx_output else default_prefix.with_suffix(".xlsx")

        # Write outputs
        write_csv(rows, metric_cols, csv_path)
        write_excel(rows, metric_cols, xlsx_path)
        
        # Write outputs by inference mode
        write_csv_by_mode(rows, metric_cols, csv_path)
        write_excel_by_mode(rows, metric_cols, xlsx_path)

        print(f"✅ Summary for {precision} precision saved:")
        print("   CSV  :", csv_path)
        print("   Excel:", xlsx_path, "(ready for humans ✨)")
        print(f"   Mode-specific files created in: {csv_path.parent}/zero_shot/ and {csv_path.parent}/iterative/")


if __name__ == "__main__":
    main()
