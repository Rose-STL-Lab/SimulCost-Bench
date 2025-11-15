"""
Utilities for visualization and data processing.
"""

import numpy as np
import pandas as pd
from scipy import stats
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


def get_paired_data(df_zs, df_iter, merge):
    """
    Filter to only include matching task configurations that exist in both dataframes.

    Args:
        df_zs: DataFrame with zero-shot (or first variant) results
        df_iter: DataFrame with iterative (or second variant) results
        merge: If True, merge into single dataframe with _zs/_iter suffixes.
               If False, return tuple of (filtered_df_zs, filtered_df_iter) with only matching rows.

    Returns:
        If merge=True: Single DataFrame with both datasets merged (columns have _zs/_iter suffixes)
        If merge=False: Tuple of (filtered_df_zs, filtered_df_iter) containing only paired entries
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

    # Find common match keys
    common_keys = set(df_zs_copy["_match_key"]) & set(df_iter_copy["_match_key"])

    # Filter both dataframes to only common keys
    df_zs_filtered = df_zs_copy[df_zs_copy["_match_key"].isin(common_keys)].copy()
    df_iter_filtered = df_iter_copy[df_iter_copy["_match_key"].isin(common_keys)].copy()

    if not merge:
        # Return filtered dataframes without merging
        df_zs_filtered = df_zs_filtered.drop(columns=["_match_key"])
        df_iter_filtered = df_iter_filtered.drop(columns=["_match_key"])
        return df_zs_filtered, df_iter_filtered

    # Merge into single dataframe
    zs_value_cols = [col for col in df_zs_filtered.columns if col not in match_cols]
    iter_value_cols = [col for col in df_iter_filtered.columns if col not in match_cols]

    merged = df_zs_filtered[match_cols + zs_value_cols].merge(
        df_iter_filtered[iter_value_cols],
        on="_match_key",
        suffixes=("_zs", "_iter"),
    )

    # Drop the match key column
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


def create_overall_metric_plot(
    zero_shot_metrics, iterative_metrics, xlabel, ylabel, ylim, output_path, include_overall
):
    """
    Create bar plot grouped by model with precision level subgroups.

    Args:
        zero_shot_metrics: DataFrame with columns [model_name, precision_level, metric_value]
        iterative_metrics: DataFrame with columns [model_name, precision_level, metric_value]
        xlabel: X-axis label
        ylabel: Y-axis label
        ylim: Tuple (ymin, ymax) for y-axis limits
        output_path: Path to save the figure
        include_overall: Whether to include "Overall" group averaging across all models
    """
    # Configuration: human-readable names
    precision_levels = ["low", "medium", "high"]
    precision_names = {"low": "Low Precision", "medium": "Medium Precision", "high": "High Precision"}

    # Get all unique models
    models = sorted(zero_shot_metrics["model_name"].unique())

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

    # Add overall average if requested
    if include_overall:
        overall_zs = {}
        overall_iter = {}
        for precision in precision_levels:
            zs_data = zero_shot_metrics[zero_shot_metrics["precision_level"] == precision]
            overall_zs[precision] = zs_data["metric_value"].mean()

            iter_data = iterative_metrics[iterative_metrics["precision_level"] == precision]
            overall_iter[precision] = iter_data["metric_value"].mean()

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


def compute_paired_differences(paired_df, metric_col, pairing_cols):
    """
    Extract paired differences for a specific metric.

    Args:
        paired_df: DataFrame from get_paired_data() with _zs and _iter suffixes
        metric_col: Column name to compute differences for (without suffix)
        pairing_cols: List of columns that identify unique pairs (e.g., ["dataset", "task", "model_name", "precision_level"])

    Returns:
        DataFrame with pairing_cols, metric_col_zs, metric_col_iter, difference
    """
    zs_col = f"{metric_col}_zs"
    iter_col = f"{metric_col}_iter"

    result = paired_df[pairing_cols + [zs_col, iter_col]].copy()

    # Convert to numeric if boolean (for subtraction)
    if result[zs_col].dtype == "bool":
        result[zs_col] = result[zs_col].astype(int)
    if result[iter_col].dtype == "bool":
        result[iter_col] = result[iter_col].astype(int)

    # Compute difference (iterative - zero_shot)
    result["difference"] = result[iter_col] - result[zs_col]

    return result


def compute_paired_ttest(diff_df):
    """
    Perform paired t-test on differences.

    Tests H0: mean(difference) = 0 vs H1: mean(difference) ≠ 0

    Args:
        diff_df: DataFrame with 'difference' column (from compute_paired_differences)

    Returns:
        dict with t_statistic, p_value, df, mean_diff, std_diff, se_diff, ci_95, n_pairs, significance flags
    """
    differences = diff_df["difference"].values
    n = len(differences)

    # Perform one-sample t-test on differences
    t_statistic, p_value = stats.ttest_1samp(differences, 0)

    # Calculate statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n)

    # 95% confidence interval
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_95 = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

    return {
        "t_statistic": t_statistic,
        "p_value": p_value,
        "df": n - 1,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "ci_95": ci_95,
        "n_pairs": n,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "significant_001": p_value < 0.001,
    }
