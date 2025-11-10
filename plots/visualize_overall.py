#!/usr/bin/env python3
"""
Unified visualization script for SimulCost-Bench results.
Supports overall results, per-dataset results, and paired-only analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import (
    read_0shot_data,
    read_iterative_data,
    get_paired_data,
    compute_mean_metrics,
    create_overall_metric_plot,
)
from constants import BASE_DATASETS

# Set style
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14


def _apply_pairing_filter(df_zs, df_iter, verbose):
    """
    Apply pairing filter and return filtered dataframes.

    Args:
        df_zs: Zero-shot dataframe
        df_iter: Iterative dataframe
        verbose: Whether to print pairing statistics

    Returns:
        Tuple of (filtered_df_zs, filtered_df_iter)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BEFORE PAIRING:")
        print("=" * 60)
        print(f"Zero-shot entries: {len(df_zs)}")
        print(f"Iterative entries: {len(df_iter)}")

    print("\nGetting paired data (entries that exist in both modes)...")
    paired = get_paired_data(df_zs, df_iter)

    if verbose:
        print("\n" + "=" * 60)
        print("AFTER PAIRING:")
        print("=" * 60)
        print(f"Paired entries: {len(paired)}")

    # Split back into zero-shot and iterative views for metric calculation
    match_cols = [
        "dataset",
        "task",
        "precision_level",
        "model_name",
        "profile",
        "target_parameters",
        "non_target_parameters",
    ]

    # Extract columns with _zs suffix for zero-shot metrics
    zs_cols = match_cols + [col.replace("_zs", "") for col in paired.columns if col.endswith("_zs")]
    df_zs_filtered = paired[
        [
            col if col in paired.columns else col + "_zs"
            for col in zs_cols
            if col + "_zs" in paired.columns or col in paired.columns
        ]
    ].copy()
    df_zs_filtered.columns = [col.replace("_zs", "") for col in df_zs_filtered.columns]

    # Extract columns with _iter suffix for iterative metrics
    iter_cols = match_cols + [col.replace("_iter", "") for col in paired.columns if col.endswith("_iter")]
    df_iter_filtered = paired[
        [
            col if col in paired.columns else col + "_iter"
            for col in iter_cols
            if col + "_iter" in paired.columns or col in paired.columns
        ]
    ].copy()
    df_iter_filtered.columns = [col.replace("_iter", "") for col in df_iter_filtered.columns]

    return df_zs_filtered, df_iter_filtered


def _plot_metrics_for_data(df_zs, df_iter, output_dir, filename_prefix, first_groupby, paired_only):
    """
    Plot success rate and efficiency metrics.

    Args:
        df_zs: Zero-shot dataframe
        df_iter: Iterative dataframe
        output_dir: Directory to save plots
        filename_prefix: Prefix for output filenames (e.g., "overall" or "heat_2d")
        first_groupby: First-stage groupby columns
        paired_only: Whether using paired data (affects filename suffix)
    """
    second_groupby = ["model_name", "precision_level"]
    suffix = "_paired" if paired_only else ""

    # Calculate success rate metrics
    print(f"\nCalculating success rate metrics for {filename_prefix}...")
    success_zs = compute_mean_metrics(df_zs, "is_successful", first_groupby, second_groupby)
    success_zs["metric_value"] *= 100  # Convert to percentage

    success_iter = compute_mean_metrics(df_iter, "is_successful", first_groupby, second_groupby)
    success_iter["metric_value"] *= 100  # Convert to percentage

    create_overall_metric_plot(
        success_zs,
        success_iter,
        xlabel="Model",
        ylabel="Success Rate (%)",
        ylim=(0, 105),
        output_path=output_dir / f"{filename_prefix}_success_rate{suffix}.png",
    )

    # Calculate efficiency metrics
    print(f"\nCalculating efficiency metrics for {filename_prefix}...")
    efficiency_zs = compute_mean_metrics(df_zs, "efficiency", first_groupby, second_groupby)
    efficiency_iter = compute_mean_metrics(df_iter, "efficiency", first_groupby, second_groupby)

    max_eff = max(efficiency_zs["metric_value"].max(), efficiency_iter["metric_value"].max())
    create_overall_metric_plot(
        efficiency_zs,
        efficiency_iter,
        xlabel="Model",
        ylabel="Mean Efficiency",
        ylim=(0, max_eff * 1.25),
        output_path=output_dir / f"{filename_prefix}_efficiency{suffix}.png",
    )


def plot(parquet_path, output_dir, datasets, paired_only, per_dataset):
    """
    Generate visualizations.

    Args:
        parquet_path: Path to the merged parquet file
        output_dir: Directory to save output plots
        datasets: List of datasets to include
        paired_only: If True, only use entries that exist in both zero-shot and iterative modes
        per_dataset: If True, generate separate plots for each dataset; if False, generate overall plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_zs = read_0shot_data(parquet_path, datasets=datasets)
    df_iter = read_iterative_data(parquet_path, datasets=datasets)

    # Apply pairing filter if requested
    if paired_only:
        df_zs, df_iter = _apply_pairing_filter(df_zs, df_iter, verbose=(not per_dataset))

    mode_str = "PAIRED DATA ONLY" if paired_only else "ALL DATA"

    if per_dataset:
        # Process each dataset separately
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset}")
            print(f"{'='*60}")

            # Filter data for current dataset
            df_zs_dataset = df_zs[df_zs["dataset"] == dataset].copy()
            df_iter_dataset = df_iter[df_iter["dataset"] == dataset].copy()

            # Plot metrics (per-dataset, no task averaging in first groupby)
            _plot_metrics_for_data(
                df_zs=df_zs_dataset,
                df_iter=df_iter_dataset,
                output_dir=output_dir,
                filename_prefix=dataset,
                first_groupby=["dataset", "model_name", "precision_level"],
                paired_only=paired_only,
            )

        print("\n" + "=" * 60)
        print(f"Done! Generated per-dataset results visualizations ({mode_str}).")
        print("=" * 60)
    else:
        # Overall results across all datasets
        _plot_metrics_for_data(
            df_zs=df_zs,
            df_iter=df_iter,
            output_dir=output_dir,
            filename_prefix="overall",
            first_groupby=["dataset", "task", "model_name", "precision_level"],
            paired_only=paired_only,
        )
        print(f"\nDone! Generated overall results visualizations ({mode_str}).")


if __name__ == "__main__":
    # 1. Overall with all data
    plot(
        parquet_path="eval_results/merged_results.parquet",
        output_dir="plots/res/overall",
        datasets=BASE_DATASETS,
        paired_only=False,
        per_dataset=False,
    )

    # 2. Overall with paired data only
    plot(
        parquet_path="eval_results/merged_results.parquet",
        output_dir="plots/res/overall",
        datasets=BASE_DATASETS,
        paired_only=True,
        per_dataset=False,
    )

    # 3. Per-dataset with all data
    plot(
        parquet_path="eval_results/merged_results.parquet",
        output_dir="plots/res/overall/per_dataset",
        datasets=BASE_DATASETS,
        paired_only=False,
        per_dataset=True,
    )
