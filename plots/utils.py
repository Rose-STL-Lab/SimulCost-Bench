"""
Utilities for visualization and data processing.
"""

import pandas as pd
from barplot_utils import GroupedBarPlot


def read_0shot_data(parquet_path: str, datasets=None):
    """Load zero-shot data for visualization."""
    print("Loading zero-shot data...")
    df = pd.read_parquet(parquet_path)

    if datasets is not None:
        df = df[df["dataset"].isin(datasets)].copy()

    df = df[df["inference_mode"] == "zero_shot"].copy()

    print(f"Loaded {len(df)} zero-shot records from {df['dataset'].nunique()} datasets")
    return df


def read_iterative_data(parquet_path: str, datasets=None):
    """Load iterative data for visualization."""
    print("Loading iterative data...")
    df = pd.read_parquet(parquet_path)

    if datasets is not None:
        df = df[df["dataset"].isin(datasets)].copy()

    df = df[df["inference_mode"] == "iterative"].copy()

    print(f"Loaded {len(df)} iterative records from {df['dataset'].nunique()} datasets")
    return df


def get_paired_data(df_zs, df_iter):
    """
    Merge zero-shot and iterative data on matching task configurations.

    Returns a single dataframe with both zero-shot and iterative values side-by-side
    for entries that exist in both modes.

    Args:
        df_zs: DataFrame with zero-shot results
        df_iter: DataFrame with iterative results

    Returns:
        DataFrame with columns: match_cols, all_zs_columns (with _zs suffix), all_iter_columns (with _iter suffix)
        Only includes rows that have matches in both dataframes.
    """
    # Define columns that identify the task/configuration
    match_cols = [
        "dataset",
        "task",
        "precision_level",
        "model_name",
        "profile",
        "target_parameters",
        "non_target_parameters",
    ]

    # Create match keys
    df_zs_copy = df_zs.copy()
    df_iter_copy = df_iter.copy()

    df_zs_copy["_match_key"] = df_zs_copy[match_cols].apply(lambda row: tuple(row), axis=1)
    df_iter_copy["_match_key"] = df_iter_copy[match_cols].apply(lambda row: tuple(row), axis=1)

    # Get columns to merge (exclude match_cols to avoid duplication, keep _match_key for merging)
    zs_value_cols = [col for col in df_zs_copy.columns if col not in match_cols]
    iter_value_cols = [col for col in df_iter_copy.columns if col not in match_cols]

    # Merge on match key
    merged = df_zs_copy[match_cols + zs_value_cols].merge(
        df_iter_copy[iter_value_cols],
        on="_match_key",
        suffixes=("_zs", "_iter"),
    )

    # Drop the match key column (no longer needed)
    merged = merged.drop(columns=["_match_key"])

    return merged


def compute_mean_metrics(df, metric_key, first_groupby, second_groupby):
    """
    Calculate mean metric with two-stage averaging.

    Two-stage averaging to ensure fairness:
    1. Average within each first_groupby combination
    2. Average across all first_groupby combinations (unweighted)

    Args:
        df: DataFrame with evaluation results (filtered to specific inference mode)
        metric_key: Column name to aggregate
        first_groupby: List of columns for stage 1 grouping (e.g., ['dataset', 'task', 'model_name', 'precision_level'])
        second_groupby: List of columns for stage 2 grouping (e.g., ['model_name', 'precision_level'])

    Returns:
        DataFrame with columns from second_groupby + 'metric_value'
    """
    # Stage 1: Calculate mean within each first_groupby combination
    stage1_metrics = df.groupby(first_groupby)[metric_key].mean().reset_index()
    stage1_metrics.rename(columns={metric_key: "stage1_mean"}, inplace=True)

    # Stage 2: Average across first_groupby combinations
    metrics = stage1_metrics.groupby(second_groupby)["stage1_mean"].mean().reset_index()
    metrics.rename(columns={"stage1_mean": "metric_value"}, inplace=True)

    return metrics


def create_overall_metric_plot(zero_shot_metrics, iterative_metrics, xlabel, ylabel, ylim, output_path):
    """
    Create bar plot grouped by model with precision level subgroups.

    Args:
        zero_shot_metrics: DataFrame with columns [model_name, precision_level, metric_value]
        iterative_metrics: DataFrame with columns [model_name, precision_level, metric_value]
        xlabel: X-axis label
        ylabel: Y-axis label
        ylim: Tuple (ymin, ymax) for y-axis limits
        output_path: Path to save the figure
    """
    # Configuration: human-readable names
    precision_levels = ["low", "medium", "high"]
    precision_names = {"low": "Low Precision", "medium": "Medium Precision", "high": "High Precision"}

    # Get all unique models
    models = sorted(zero_shot_metrics["model_name"].unique())

    # Calculate overall average (unweighted across models)
    overall_zs = {}
    overall_iter = {}
    for precision in precision_levels:
        zs_data = zero_shot_metrics[zero_shot_metrics["precision_level"] == precision]
        overall_zs[precision] = zs_data["metric_value"].mean()

        iter_data = iterative_metrics[iterative_metrics["precision_level"] == precision]
        overall_iter[precision] = iter_data["metric_value"].mean()

    # Prepare data dict: model -> precision_name -> mode_name -> value
    data_dict = {}

    # Add individual models
    for model in models:
        data_dict[model] = {}
        for precision in precision_levels:
            precision_name = precision_names[precision]
            data_dict[model][precision_name] = {}

            # Zero-shot
            zs_value = zero_shot_metrics[
                (zero_shot_metrics["model_name"] == model) & (zero_shot_metrics["precision_level"] == precision)
            ]["metric_value"].values[0]
            data_dict[model][precision_name]["single-round"] = zs_value

            # Iterative
            iter_value = iterative_metrics[
                (iterative_metrics["model_name"] == model) & (iterative_metrics["precision_level"] == precision)
            ]["metric_value"].values[0]
            data_dict[model][precision_name]["multi-round"] = iter_value

    # Add overall average
    data_dict["Overall"] = {}
    for precision in precision_levels:
        precision_name = precision_names[precision]
        data_dict["Overall"][precision_name] = {
            "single-round": overall_zs[precision],
            "multi-round": overall_iter[precision],
        }

    plotter = GroupedBarPlot(figsize=(18, 7))
    plotter.plot(data_dict, xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    plotter.save(output_path)
