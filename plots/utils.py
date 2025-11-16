"""
Utilities for visualization and data processing.
"""

import numpy as np
import pandas as pd


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
