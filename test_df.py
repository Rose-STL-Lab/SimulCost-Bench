#!/usr/bin/env python3
"""
DataFrame Inspector for SimulCost-Bench Evaluation Results

This script provides convenient inspection of parquet evaluation results.
"""

import pandas as pd
import json
import argparse
import sys
from pathlib import Path


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def show_basic_info(df: pd.DataFrame):
    """Display basic DataFrame information."""
    print_separator()
    print("📊 DATAFRAME BASIC INFO")
    print_separator()
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"  {i:2d}. {col:30s} ({dtype}) - {non_null}/{len(df)} non-null")
    print()


def show_column_list(df: pd.DataFrame):
    """Display just the column names."""
    print_separator()
    print("📋 COLUMN LIST")
    print_separator()
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    print()


def show_row_by_index(df: pd.DataFrame, index: int):
    """Display a specific row by index with formatting."""
    print_separator()
    print(f"🔍 ROW AT INDEX {index}")
    print_separator()

    if index < 0 or index >= len(df):
        print(f"❌ Error: Index {index} out of range [0, {len(df)-1}]")
        return

    row = df.iloc[index]
    for col, val in row.items():
        # Special formatting for attempt_history
        if col == 'attempt_history' and pd.notna(val):
            print(f"\n{col}:")
            try:
                history = json.loads(val)
                print(f"  Number of attempts: {len(history)}")
                print(f"  (Use --attempt-history to see details)")
            except:
                print(f"  {val}")
        else:
            print(f"{col}: {val}")
    print()


def show_row_by_qid(df: pd.DataFrame, qid: int):
    """Display rows matching a specific QID."""
    print_separator()
    print(f"🔍 ROWS WITH QID = {qid}")
    print_separator()

    matches = df[df['qid'] == qid]
    if len(matches) == 0:
        print(f"❌ No rows found with QID = {qid}")
        return

    print(f"Found {len(matches)} matching row(s)\n")
    for idx, (_, row) in enumerate(matches.iterrows()):
        print(f"--- Match {idx + 1} ---")
        for col, val in row.items():
            if col == 'attempt_history' and pd.notna(val):
                print(f"{col}:")
                try:
                    history = json.loads(val)
                    print(f"  Number of attempts: {len(history)}")
                except:
                    print(f"  {val}")
            else:
                print(f"{col}: {val}")
        print()


def show_attempt_history(df: pd.DataFrame, qid: int = None, index: int = None, task: str = None):
    """Display detailed attempt history."""
    print_separator()
    print("🎯 ATTEMPT HISTORY DETAILS")
    print_separator()

    # Get the row(s)
    if qid is not None:
        matches = df[df['qid'] == qid]
        if len(matches) == 0:
            print(f"❌ No rows found with QID = {qid}")
            return

        # Apply task filter if specified
        if task is not None:
            if 'task' in df.columns:
                matches = matches[matches['task'] == task]
                if len(matches) == 0:
                    print(f"❌ No rows found with QID = {qid} and task = {task}")
                    return
            else:
                print(f"⚠️  'task' column not found in DataFrame, ignoring --task filter")

        # Process all matching rows
        print(f"Found {len(matches)} matching row(s) for QID = {qid}")
        if task is not None:
            print(f"Filtered by task = {task}")
        print()

        for match_idx, (row_idx, row) in enumerate(matches.iterrows(), 1):
            _display_single_attempt_history(
                df, row, row_idx,
                identifier=f"QID {qid}, Match {match_idx}/{len(matches)}",
                show_separator=(match_idx > 1)
            )

    elif index is not None:
        if index < 0 or index >= len(df):
            print(f"❌ Error: Index {index} out of range [0, {len(df)-1}]")
            return
        row = df.iloc[index]
        _display_single_attempt_history(df, row, index, identifier=f"Index {index}")
    else:
        print("❌ Must specify either --qid or --index")
        return


def _display_single_attempt_history(df: pd.DataFrame, row: pd.Series, row_idx: int,
                                     identifier: str, show_separator: bool = False):
    """Helper function to display attempt history for a single row."""
    if show_separator:
        print("\n" + "█" * 80 + "\n")

    # Parse attempt history
    if 'attempt_history' not in df.columns:
        print("❌ 'attempt_history' column not found in this DataFrame")
        return

    history_json = row['attempt_history']
    if pd.isna(history_json):
        print(f"⚠️  No attempt history for {identifier}")
        return

    try:
        history = json.loads(history_json)
    except Exception as e:
        print(f"❌ Error parsing attempt_history: {e}")
        return

    # Display row information
    print(f"Row: {identifier}")
    print(f"DataFrame Index: {row_idx}")

    # Show task info if available
    if 'task' in row:
        print(f"Task: {row['task']}")
    if 'task_type' in row:
        print(f"Task Type: {row['task_type']}")

    print(f"Total attempts: {len(history)}\n")

    for attempt in history:
        print(f"{'='*60}")
        print(f"Attempt {attempt.get('attempt_number', 'N/A')}")
        print(f"{'='*60}")
        print(f"Tool: {attempt.get('tool_name', 'N/A')}")
        print(f"Args: {attempt.get('tool_args', 'N/A')}")
        print(f"Reason: {attempt.get('tool_reason', 'N/A')}")
        print(f"\nResults:")
        print(f"  RMSE: {attempt.get('RMSE', 'N/A')}")
        print(f"  Converged: {attempt.get('is_converged', 'N/A')}")
        print(f"  Accumulated Cost: {attempt.get('accumulated_cost', 'N/A')}")
        print(f"  Simulation Cost: {attempt.get('The cost of the solver simulating the environment', 'N/A')}")
        print(f"  Verification Cost: {attempt.get('The cost of the solver verifying convergence (This will not be included in your accumulated_cost)', 'N/A')}")
        print()


def show_mode_comparison(df: pd.DataFrame):
    """Display tasks that support multiple inference modes."""
    print_separator()
    print("🔄 INFERENCE MODE COMPARISON")
    print_separator()

    # Check if required columns exist
    if 'inference_mode' not in df.columns:
        print("❌ 'inference_mode' column not found in this DataFrame")
        return

    # Show overall statistics
    print(f"Overall dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nUnique inference modes: {sorted(df['inference_mode'].unique())}")
    print_separator()

    # Find tasks that support multiple modes
    groupby_cols = []
    if 'dataset' in df.columns:
        groupby_cols.append('dataset')
    if 'task' in df.columns:
        groupby_cols.append('task')

    if not groupby_cols:
        print("❌ Neither 'dataset' nor 'task' columns found - cannot group tasks")
        return

    combo_df = df.groupby(groupby_cols)['inference_mode'].apply(
        lambda x: sorted(x.unique())
    ).reset_index()

    # Filter for tasks with both iterative and zero_shot modes
    both_modes = combo_df[
        combo_df['inference_mode'].apply(
            lambda x: 'iterative' in x and 'zero_shot' in x
        )
    ]

    if len(both_modes) > 0:
        print("\n✅ Tasks that support BOTH iterative and zero_shot modes:")
        print(both_modes.to_string(index=False))
        print(f"\nTotal: {len(both_modes)} task combination(s)")
    else:
        print("\n⚠️  No tasks found that support both iterative and zero_shot modes")

    # Show tasks with only one mode
    single_mode = combo_df[
        combo_df['inference_mode'].apply(lambda x: len(x) == 1)
    ]
    if len(single_mode) > 0:
        print(f"\n📊 Tasks with only one inference mode: {len(single_mode)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SimulCost-Bench evaluation parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show basic info about the DataFrame
  python test_df.py path/to/file.parquet --info

  # List all columns
  python test_df.py path/to/file.parquet --columns

  # Show row at index 5
  python test_df.py path/to/file.parquet --index 5

  # Show row with QID 6
  python test_df.py path/to/file.parquet --qid 6

  # Show attempt history for ALL tasks with QID 6
  python test_df.py path/to/file.parquet --attempt-history --qid 6

  # Show attempt history for specific task (e.g., 'cfl') with QID 6
  python test_df.py path/to/file.parquet --attempt-history --qid 6 --task cfl

  # Compare inference modes across tasks
  python test_df.py eval_results/merged_results.parquet --mode-comparison
        """
    )

    parser.add_argument(
        'parquet_file',
        type=str,
        nargs='?',
        default='eval_results/epoch_1d/dataframes/iterative_gpt-5-2025-08-07.parquet',
        help='Path to parquet file (default: epoch_1d iterative GPT-5)'
    )

    parser.add_argument('--info', action='store_true', help='Show basic DataFrame info')
    parser.add_argument('--columns', action='store_true', help='List all columns')
    parser.add_argument('--index', type=int, help='Show row at specific index')
    parser.add_argument('--qid', type=int, help='Show row(s) with specific QID')
    parser.add_argument('--task', type=str, help='Filter by specific task type (use with --qid)')
    parser.add_argument('--attempt-history', action='store_true', help='Show detailed attempt history')
    parser.add_argument('--mode-comparison', action='store_true', help='Compare inference modes across tasks')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.parquet_file).exists():
        print(f"❌ Error: File not found: {args.parquet_file}")
        sys.exit(1)

    # Load DataFrame
    try:
        df = pd.read_parquet(args.parquet_file)
        print(f"\n✅ Loaded: {args.parquet_file}")
        print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    except Exception as e:
        print(f"❌ Error loading parquet file: {e}")
        sys.exit(1)

    # If no specific action requested, show basic info
    if not any([args.info, args.columns, args.index is not None,
                args.qid is not None, args.attempt_history, args.mode_comparison]):
        args.info = True

    # Execute requested actions
    if args.info:
        show_basic_info(df)

    if args.columns:
        show_column_list(df)

    if args.index is not None:
        if args.attempt_history:
            show_attempt_history(df, index=args.index)
        else:
            show_row_by_index(df, args.index)

    if args.qid is not None:
        if args.attempt_history:
            show_attempt_history(df, qid=args.qid, task=args.task)
        else:
            show_row_by_qid(df, args.qid)

    if args.mode_comparison:
        show_mode_comparison(df)


if __name__ == "__main__":
    main()
