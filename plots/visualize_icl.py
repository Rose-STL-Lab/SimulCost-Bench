#!/usr/bin/env python3
"""
Visualization script for In-Context Learning (ICL) ablation study.
Compares normal evaluation vs ICL variants (full, accuracy-focused, cost-excluded).
"""

import pandas as pd
from pathlib import Path

from utils import read_0shot_data, read_iterative_data, get_paired_data
from visualize_overall import _plot_metrics_for_data
from constants import ICL_BASE_DATASETS, ICL_VARIANT_ORDER, ICL_MODEL


def _prepare_icl_data(parquet_path, datasets):
    """
    Read ICL data and transform it to match the structure expected by overall plotting.

    Transform ICL variants from dataset-level distinction to model-level distinction:
    - Read all ICL datasets in one go (Normal + variants)
    - Rename model to include variant suffix based on dataset name
    - Strip ICL suffix from dataset names

    Args:
        parquet_path: Path to merged_results.parquet
        datasets: List of base datasets to include

    Returns:
        df_zs, df_iter: DataFrames ready for overall plotting logic
    """
    # Build combined dataset list (all variants)
    all_datasets = []
    all_datasets.extend(datasets)  # Normal
    all_datasets.extend([f"{ds}_icl_full" for ds in datasets])
    all_datasets.extend([f"{ds}_icl_accuracy_focused" for ds in datasets])
    all_datasets.extend([f"{ds}_icl_cost_excluded" for ds in datasets])

    # Read all data in one IO and filter for ICL model only
    df_zs = read_0shot_data(parquet_path, datasets=all_datasets)
    df_iter = read_iterative_data(parquet_path, datasets=all_datasets)

    df_zs = df_zs[df_zs["model_name"] == ICL_MODEL].copy()
    df_iter = df_iter[df_iter["model_name"] == ICL_MODEL].copy()

    # Transform: move variant from dataset name to model name
    # Add variant suffix to model name based on dataset suffix
    df_zs["model_name"] = df_zs.apply(
        lambda row: (
            row["model_name"] + "-ICL-Full"
            if "_icl_full" in row["dataset"]
            else (
                row["model_name"] + "-ICL-Accuracy"
                if "_icl_accuracy_focused" in row["dataset"]
                else row["model_name"] + "-ICL-NoCost" if "_icl_cost_excluded" in row["dataset"] else row["model_name"]
            )
        ),
        axis=1,
    )
    df_iter["model_name"] = df_iter.apply(
        lambda row: (
            row["model_name"] + "-ICL-Full"
            if "_icl_full" in row["dataset"]
            else (
                row["model_name"] + "-ICL-Accuracy"
                if "_icl_accuracy_focused" in row["dataset"]
                else row["model_name"] + "-ICL-NoCost" if "_icl_cost_excluded" in row["dataset"] else row["model_name"]
            )
        ),
        axis=1,
    )

    # Strip ICL suffix from dataset names
    df_zs["dataset"] = df_zs["dataset"].str.replace("_icl_full", "", regex=False)
    df_zs["dataset"] = df_zs["dataset"].str.replace("_icl_accuracy_focused", "", regex=False)
    df_zs["dataset"] = df_zs["dataset"].str.replace("_icl_cost_excluded", "", regex=False)

    df_iter["dataset"] = df_iter["dataset"].str.replace("_icl_full", "", regex=False)
    df_iter["dataset"] = df_iter["dataset"].str.replace("_icl_accuracy_focused", "", regex=False)
    df_iter["dataset"] = df_iter["dataset"].str.replace("_icl_cost_excluded", "", regex=False)

    return df_zs, df_iter


def plot(parquet_path, output_dir, paired_only, per_dataset):
    """
    Generate ICL comparison plots.

    Args:
        parquet_path: Path to merged_results.parquet
        output_dir: Directory to save plots
        paired_only: If True, filter to only tasks that exist in both zero-shot and iterative modes
        per_dataset: If True, generate separate plots for each dataset; if False, generate overall plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "PAIRED DATA ONLY" if paired_only else "ALL DATA"

    # Prepare ICL data (transform variant info from dataset to model)
    if per_dataset:
        # Process each dataset separately
        for dataset in ICL_BASE_DATASETS:
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset}")
            print(f"{'='*60}")

            # Load and transform data for this dataset
            df_zs, df_iter = _prepare_icl_data(parquet_path, datasets=[dataset])

            # Apply pairing filter if requested
            if paired_only:
                df_zs, df_iter = get_paired_data(df_zs, df_iter, merge=False)

            # Filter data for current dataset (after transformation, all have same base dataset name)
            df_zs_dataset = df_zs[df_zs["dataset"] == dataset].copy()
            df_iter_dataset = df_iter[df_iter["dataset"] == dataset].copy()

            # Plot metrics using overall plotting logic
            _plot_metrics_for_data(
                df_zs=df_zs_dataset,
                df_iter=df_iter_dataset,
                output_dir=output_dir,
                filename_prefix=f"icl_{dataset}",
                first_groupby=["dataset", "task", "model_name", "precision_level"],
                paired_only=paired_only,
                include_overall=False,
            )

        print("\n" + "=" * 60)
        print(f"Done! Generated per-dataset ICL visualizations ({mode_str}).")
        print("=" * 60)
    else:
        # Overall results across all datasets
        df_zs, df_iter = _prepare_icl_data(parquet_path, datasets=ICL_BASE_DATASETS)

        print(f"Loaded {len(df_zs)} zero-shot records")
        print(f"Loaded {len(df_iter)} iterative records")

        # Apply pairing filter if requested
        if paired_only:
            df_zs, df_iter = get_paired_data(df_zs, df_iter, merge=False)
            print(f"Paired zero-shot: {len(df_zs)}")
            print(f"Paired iterative: {len(df_iter)}")

        # Plot overall metrics using overall plotting logic
        _plot_metrics_for_data(
            df_zs=df_zs,
            df_iter=df_iter,
            output_dir=output_dir,
            filename_prefix="icl",
            first_groupby=["dataset", "task", "model_name", "precision_level"],
            paired_only=paired_only,
            include_overall=False,
        )

        print(f"\nDone! Generated overall ICL visualizations ({mode_str}).")


if __name__ == "__main__":
    # 1. Overall with all data
    plot(
        parquet_path="../eval_results/merged_results.parquet",
        output_dir="res/icl",
        paired_only=False,
        per_dataset=False,
    )

    # 2. Overall with paired data only
    plot(
        parquet_path="../eval_results/merged_results.parquet",
        output_dir="res/icl",
        paired_only=True,
        per_dataset=False,
    )

    # 3. Per-dataset plots
    plot(
        parquet_path="../eval_results/merged_results.parquet",
        output_dir="res/icl/per_dataset",
        paired_only=False,
        per_dataset=True,
    )
