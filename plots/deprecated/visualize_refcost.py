#!/usr/bin/env python3
"""
Visualization script for reference cost (ref_cost) analysis.
Computes ref_cost = model_cost / zs_dummy_cost for both zero-shot and iterative modes,
split by success/failure status.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import read_0shot_data, read_iterative_data, get_paired_data
from barplot_utils import GroupedBarPlot
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


def compute_refcost_metrics(paired_df):
    """
    Compute ref_cost metrics for paired data.

    ref_cost = model_cost / zs_dummy_cost (using zero-shot's dummy_cost as reference for both modes)

    Returns DataFrame with ref_cost for both zero-shot and iterative modes,
    along with success status for filtering.
    """
    # # Extract relevant columns for zero-shot
    # zs_cols = [
    #     "dataset", "task", "precision_level", "model_name",
    #     "profile", "target_parameters", "non_target_parameters",
    #     "model_cost_zs", "dummy_cost_zs", "is_successful_zs"
    # ]

    # # Extract relevant columns for iterative
    # iter_cols = [
    #     "dataset", "task", "precision_level", "model_name",
    #     "profile", "target_parameters", "non_target_parameters",
    #     "model_cost_iter", "dummy_cost_iter", "is_successful_iter"
    # ]

    # # Check which columns exist in paired_df
    # available_cols = paired_df.columns.tolist()

    # Build the result dataframe
    result = paired_df[[
        "dataset", "task", "precision_level", "model_name",
        "profile", "target_parameters", "non_target_parameters"
    ]].copy()

    # Compute ref_cost for zero-shot: model_cost_zs / dummy_cost_zs
    result["ref_cost_zs"] = paired_df["model_cost_zs"] / paired_df["dummy_cost_zs"]
    result["is_successful_zs"] = paired_df["is_successful_zs"]

    # Compute ref_cost for iterative: model_cost_iter / dummy_cost_zs (use zs's dummy as reference!)
    result["ref_cost_iter"] = paired_df["model_cost_iter"] / paired_df["dummy_cost_zs"]
    result["is_successful_iter"] = paired_df["is_successful_iter"]

    return result


def compute_mean_refcost_by_precision(refcost_df, filter_success=None):
    """
    Compute mean ref_cost grouped by model, precision level, and mode.
    Uses two-stage averaging like in overall success rate plots.

    Args:
        refcost_df: DataFrame with ref_cost columns
        filter_success: None (all data), True (success only), False (failure only)

    Returns:
        dict: {model_name: {precision_name: {mode_name: value}}}
    """
    from utils import compute_mean_metrics

    # Apply filter if specified
    if filter_success is not None:
        if filter_success:
            # Keep only successful attempts for both modes
            refcost_df = refcost_df[
                (refcost_df["is_successful_zs"] == True) &
                (refcost_df["is_successful_iter"] == True)
            ].copy()
        else:
            # Keep only failed attempts for both modes
            refcost_df = refcost_df[
                (refcost_df["is_successful_zs"] == False) &
                (refcost_df["is_successful_iter"] == False)
            ].copy()

    # Prepare data for zero-shot
    zs_data = refcost_df[[
        "dataset", "task", "precision_level", "model_name", "ref_cost_zs"
    ]].copy()
    zs_data.columns = ["dataset", "task", "precision_level", "model_name", "ref_cost"]

    # Prepare data for iterative
    iter_data = refcost_df[[
        "dataset", "task", "precision_level", "model_name", "ref_cost_iter"
    ]].copy()
    iter_data.columns = ["dataset", "task", "precision_level", "model_name", "ref_cost"]

    # Compute two-stage averaged metrics for zero-shot
    zs_metrics = compute_mean_metrics(
        zs_data,
        metric_key="ref_cost",
        first_groupby=["dataset", "task", "model_name", "precision_level"],
        second_groupby=["model_name", "precision_level"]
    )

    # Compute two-stage averaged metrics for iterative
    iter_metrics = compute_mean_metrics(
        iter_data,
        metric_key="ref_cost",
        first_groupby=["dataset", "task", "model_name", "precision_level"],
        second_groupby=["model_name", "precision_level"]
    )

    # Build data structure: {model: {precision: {mode: value}}}
    data_dict = {}
    models = sorted(refcost_df["model_name"].unique())
    precision_levels = ["low", "medium", "high"]
    precision_names = {
        "low": "Low Precision",
        "medium": "Medium Precision",
        "high": "High Precision"
    }

    for model in models:
        data_dict[model] = {}
        for prec in precision_levels:
            # Get zero-shot value
            zs_row = zs_metrics[
                (zs_metrics["model_name"] == model) &
                (zs_metrics["precision_level"] == prec)
            ]
            zs_val = zs_row["metric_value"].values[0] if len(zs_row) > 0 else 0

            # Get iterative value
            iter_row = iter_metrics[
                (iter_metrics["model_name"] == model) &
                (iter_metrics["precision_level"] == prec)
            ]
            iter_val = iter_row["metric_value"].values[0] if len(iter_row) > 0 else 0

            data_dict[model][precision_names[prec]] = {
                "Single-Round": zs_val,
                "Multi-Round": iter_val
            }

    return data_dict


def plot_refcost(parquet_path, output_dir, datasets):
    """
    Generate ref_cost visualization with paired data only.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading zero-shot data...")
    df_zs = read_0shot_data(parquet_path, datasets=datasets)
    print(f"Loaded {len(df_zs)} zero-shot records from {len(datasets)} datasets")

    print("Loading iterative data...")
    df_iter = read_iterative_data(parquet_path, datasets=datasets)
    print(f"Loaded {len(df_iter)} iterative records from {len(datasets)} datasets")

    # Get paired data
    print("\nGetting paired data...")
    paired = get_paired_data(df_zs, df_iter)
    print(f"Paired entries: {len(paired)}")

    # Compute ref_cost metrics
    print("Computing ref_cost metrics...")
    refcost_df = compute_refcost_metrics(paired)

    # Create plot for successful attempts only
    print("\nPreparing plot data for successful attempts...")
    data_dict_success = compute_mean_refcost_by_precision(refcost_df, filter_success=True)

    print("Creating ref_cost plot for successful attempts...")
    plot_output_success = output_dir / "refcost_success.png"

    plotter_success = GroupedBarPlot()
    plotter_success.plot(
        data_dict=data_dict_success,
        xlabel="Model",
        ylabel="Reference Cost (Relative to ZS Single Run)",
        ylim=None,
        show_values=True,
        value_format=".1f",
    )
    plotter_success.save(plot_output_success)
    print(f"Saved ref_cost (success) plot to {plot_output_success}")

    # Create plot for failed attempts only
    print("\nPreparing plot data for failed attempts...")
    data_dict_fail = compute_mean_refcost_by_precision(refcost_df, filter_success=False)

    print("Creating ref_cost plot for failed attempts...")
    plot_output_fail = output_dir / "refcost_fail.png"

    plotter_fail = GroupedBarPlot()
    plotter_fail.plot(
        data_dict=data_dict_fail,
        xlabel="Model",
        ylabel="Reference Cost (Relative to ZS Single Run)",
        ylim=None,
        show_values=True,
        value_format=".1f",
    )
    plotter_fail.save(plot_output_fail)
    print(f"Saved ref_cost (fail) plot to {plot_output_fail}")


if __name__ == "__main__":
    plot_refcost(
        parquet_path="eval_results/merged_results.parquet",
        output_dir="plots/res/refcost",
        datasets=BASE_DATASETS,
    )

    print("\nRef_cost plot generated successfully!")
