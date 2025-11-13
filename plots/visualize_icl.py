#!/usr/bin/env python3
"""
Visualization script for In-Context Learning (ICL) ablation study.
Compares normal evaluation vs ICL variants (full, accuracy-focused, cost-excluded).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import compute_mean_metrics
from barplot_utils import GroupedBarPlot
from constants import ICL_BASE_DATASETS, ICL_VARIANTS, ICL_VARIANT_ORDER

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


def load_icl_data(parquet_path, inference_mode):
    """
    Load data for ICL analysis, including base datasets and their ICL variants.

    Args:
        parquet_path: Path to merged_results.parquet
        inference_mode: "zero_shot" or "iterative"

    Returns:
        DataFrame with all ICL-related data
    """
    df = pd.read_parquet(parquet_path)

    # Filter by inference mode
    df = df[df["inference_mode"] == inference_mode].copy()

    # Build list of all datasets to include (base + variants)
    datasets_to_include = []
    for base in ICL_BASE_DATASETS:
        datasets_to_include.append(base)  # Normal version
        for suffix in ICL_VARIANT_ORDER:
            if suffix:  # Skip empty string (already added base)
                datasets_to_include.append(base + suffix)

    # Filter to only include relevant datasets
    df = df[df["dataset"].isin(datasets_to_include)].copy()

    # Add a column to identify the base dataset and variant type
    df["base_dataset"] = df["dataset"].apply(lambda x: get_base_dataset(x))
    df["icl_variant"] = df["dataset"].apply(lambda x: get_icl_variant(x))

    print(f"Loaded {len(df)} {inference_mode} records from {len(datasets_to_include)} datasets")
    print(f"Base datasets: {ICL_BASE_DATASETS}")
    print(f"Unique variants found: {sorted(df['icl_variant'].unique())}")

    return df


def get_base_dataset(dataset_name):
    """Extract base dataset name from dataset that may have ICL suffix."""
    for base in ICL_BASE_DATASETS:
        if dataset_name.startswith(base):
            return base
    return dataset_name


def get_icl_variant(dataset_name):
    """Extract ICL variant suffix from dataset name."""
    for suffix in ICL_VARIANT_ORDER:
        if suffix == "":
            # Check if it's exactly the base dataset (no suffix)
            if dataset_name in ICL_BASE_DATASETS:
                return ""
        elif dataset_name.endswith(suffix):
            return suffix
    return ""


def get_paired_icl_data(df):
    """
    Filter to only include tasks that exist in ALL ICL variants.

    Args:
        df: DataFrame with ICL data (already filtered by inference mode)

    Returns:
        Filtered DataFrame containing only paired entries
    """
    # Match columns define what makes a task unique (excluding ICL variant)
    match_cols = [
        "base_dataset",
        "task",
        "precision_level",
        "model_name",
        "profile",
        "target_parameters",
        "non_target_parameters",
    ]

    # Create match key for each row
    df_copy = df.copy()
    df_copy["_match_key"] = df_copy[match_cols].apply(lambda row: tuple(row), axis=1)

    # For each match key, count how many ICL variants exist
    variant_counts = df_copy.groupby("_match_key")["icl_variant"].nunique()

    # Keep only match keys that have all 4 ICL variants
    paired_keys = variant_counts[variant_counts == len(ICL_VARIANT_ORDER)].index

    # Filter to only paired entries
    df_paired = df_copy[df_copy["_match_key"].isin(paired_keys)].copy()
    df_paired = df_paired.drop(columns=["_match_key"])

    print(f"  Original entries: {len(df)}")
    print(f"  Paired entries (all 4 ICL variants): {len(df_paired)}")
    print(f"  Reduction: {len(df) - len(df_paired)} entries ({(len(df) - len(df_paired))/len(df)*100:.1f}%)")

    return df_paired


def compute_icl_metrics(df, metric_key, paired_only=False):
    """
    Compute metrics for ICL comparison with two-stage averaging.

    Args:
        df: DataFrame with ICL data (already filtered by inference mode)
        metric_key: 'is_successful' or 'efficiency'
        paired_only: If True, filter to only tasks that exist in all ICL variants

    Returns:
        dict: {model_name: {precision_name: {variant_name: value}}}
    """
    # Apply pairing filter if requested
    if paired_only:
        print(f"  Applying pairing filter for {metric_key}...")
        df = get_paired_icl_data(df)

    data_dict = {}
    models = sorted(df["model_name"].unique())
    precision_levels = ["low", "medium", "high"]
    precision_names = {"low": "Low Precision", "medium": "Medium Precision", "high": "High Precision"}

    for model in models:
        data_dict[model] = {}
        model_data = df[df["model_name"] == model]

        for prec in precision_levels:
            data_dict[model][precision_names[prec]] = {}
            prec_data = model_data[model_data["precision_level"] == prec]

            # Compute metric for each ICL variant
            for variant_suffix in ICL_VARIANT_ORDER:
                variant_data = prec_data[prec_data["icl_variant"] == variant_suffix]

                if len(variant_data) == 0:
                    data_dict[model][precision_names[prec]][ICL_VARIANTS[variant_suffix]] = 0
                    continue

                # Two-stage averaging
                metrics = compute_mean_metrics(
                    variant_data,
                    metric_key=metric_key,
                    first_groupby=["base_dataset", "task", "model_name", "precision_level"],
                    second_groupby=["model_name", "precision_level"],
                )

                if len(metrics) > 0:
                    value = metrics["metric_value"].values[0]
                    # Convert to percentage for success rate
                    if metric_key == "is_successful":
                        value *= 100
                    data_dict[model][precision_names[prec]][ICL_VARIANTS[variant_suffix]] = value
                else:
                    data_dict[model][precision_names[prec]][ICL_VARIANTS[variant_suffix]] = 0

    # Compute overall average across all models (add at the end)
    overall_dict = {}
    for prec in precision_levels:
        overall_dict[precision_names[prec]] = {}
        for variant_suffix in ICL_VARIANT_ORDER:
            variant_name = ICL_VARIANTS[variant_suffix]
            # Average the values across all models for this precision and variant
            values = [
                data_dict[model][precision_names[prec]][variant_name]
                for model in models
                if data_dict[model][precision_names[prec]][variant_name] > 0
            ]
            overall_dict[precision_names[prec]][variant_name] = sum(values) / len(values) if values else 0

    # Add Overall at the end (rightmost position)
    data_dict["Overall"] = overall_dict

    return data_dict


def plot_icl_per_dataset(parquet_path, output_dir):
    """
    Generate ICL comparison plots per dataset for zero-shot and iterative modes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for both modes
    print("\n" + "=" * 70)
    print("LOADING DATA FOR PER-DATASET ICL PLOTS")
    print("=" * 70)

    df_zs = load_icl_data(parquet_path, "zero_shot")
    df_iter = load_icl_data(parquet_path, "iterative")

    # Generate plots for each base dataset
    for base_dataset in ICL_BASE_DATASETS:
        print("\n" + "=" * 70)
        print(f"PROCESSING DATASET: {base_dataset.upper()}")
        print("=" * 70)

        # Filter data for this base dataset
        df_zs_dataset = df_zs[df_zs["base_dataset"] == base_dataset].copy()
        df_iter_dataset = df_iter[df_iter["base_dataset"] == base_dataset].copy()

        print(f"Zero-shot entries: {len(df_zs_dataset)}")
        print(f"Iterative entries: {len(df_iter_dataset)}")

        # Success rate - Zero-shot
        print(f"\nComputing zero-shot success rate metrics for {base_dataset}...")
        data_dict_zs_success = compute_icl_metrics(df_zs_dataset, "is_successful")

        print(f"Creating zero-shot success rate plot for {base_dataset}...")
        plotter_zs_success = GroupedBarPlot(figsize=(18, 7))
        plotter_zs_success.plot(
            data_dict=data_dict_zs_success,
            xlabel="Model",
            ylabel="Success Rate (%)",
            ylim=(0, 105),
            show_values=True,
            value_format=".1f",
        )
        output_path = output_dir / f"icl_{base_dataset}_zero_shot_success_rate.png"
        plotter_zs_success.save(output_path)

        # Success rate - Iterative
        print(f"\nComputing iterative success rate metrics for {base_dataset}...")
        data_dict_iter_success = compute_icl_metrics(df_iter_dataset, "is_successful")

        print(f"Creating iterative success rate plot for {base_dataset}...")
        plotter_iter_success = GroupedBarPlot(figsize=(18, 7))
        plotter_iter_success.plot(
            data_dict=data_dict_iter_success,
            xlabel="Model",
            ylabel="Success Rate (%)",
            ylim=(0, 105),
            show_values=True,
            value_format=".1f",
        )
        output_path = output_dir / f"icl_{base_dataset}_iterative_success_rate.png"
        plotter_iter_success.save(output_path)

        # Efficiency - Zero-shot
        print(f"Computing zero-shot efficiency metrics for {base_dataset}...")
        data_dict_zs_eff = compute_icl_metrics(df_zs_dataset, "efficiency")

        max_eff = 0
        for model_data in data_dict_zs_eff.values():
            for prec_data in model_data.values():
                for val in prec_data.values():
                    if val > max_eff:
                        max_eff = val

        print(f"Creating zero-shot efficiency plot for {base_dataset}...")
        plotter_zs_eff = GroupedBarPlot(figsize=(18, 7))
        plotter_zs_eff.plot(
            data_dict=data_dict_zs_eff,
            xlabel="Model",
            ylabel="Efficiency",
            ylim=(0, max_eff * 1.25),
            show_values=True,
            value_format=".1f",
        )
        output_path = output_dir / f"icl_{base_dataset}_zero_shot_efficiency.png"
        plotter_zs_eff.save(output_path)

        # Efficiency - Iterative
        print(f"Computing iterative efficiency metrics for {base_dataset}...")
        data_dict_iter_eff = compute_icl_metrics(df_iter_dataset, "efficiency")

        max_eff = 0
        for model_data in data_dict_iter_eff.values():
            for prec_data in model_data.values():
                for val in prec_data.values():
                    if val > max_eff:
                        max_eff = val

        print(f"Creating iterative efficiency plot for {base_dataset}...")
        plotter_iter_eff = GroupedBarPlot(figsize=(18, 7))
        plotter_iter_eff.plot(
            data_dict=data_dict_iter_eff,
            xlabel="Model",
            ylabel="Efficiency",
            ylim=(0, max_eff * 1.25),
            show_values=True,
            value_format=".1f",
        )
        output_path = output_dir / f"icl_{base_dataset}_iterative_efficiency.png"
        plotter_iter_eff.save(output_path)

    print("\n" + "=" * 70)
    print("ALL PER-DATASET ICL PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 70)


def plot_icl_comparison(parquet_path, output_dir, paired_only=False):
    """
    Generate ICL comparison plots for zero-shot and iterative modes (overall).

    Args:
        parquet_path: Path to merged_results.parquet
        output_dir: Directory to save plots
        paired_only: If True, filter to only tasks that exist in all ICL variants
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_paired" if paired_only else ""
    mode_str = "PAIRED" if paired_only else "ALL DATA"

    # Plot for zero-shot mode
    print("\n" + "=" * 70)
    print(f"ZERO-SHOT ICL COMPARISON ({mode_str})")
    print("=" * 70)

    df_zs = load_icl_data(parquet_path, "zero_shot")

    # Success rate plot
    print("\nComputing success rate metrics...")
    data_dict_zs_success = compute_icl_metrics(df_zs, "is_successful", paired_only=paired_only)

    print("Creating zero-shot success rate plot...")
    plotter_zs_success = GroupedBarPlot(figsize=(18, 7))
    plotter_zs_success.plot(
        data_dict=data_dict_zs_success,
        xlabel="Model",
        ylabel="Success Rate (%)",
        ylim=(0, 105),
        show_values=True,
        value_format=".1f",
    )
    output_path_zs_success = output_dir / f"icl_zero_shot_success_rate{suffix}.png"
    plotter_zs_success.save(output_path_zs_success)

    # Efficiency plot
    print("Computing efficiency metrics...")
    data_dict_zs_eff = compute_icl_metrics(df_zs, "efficiency", paired_only=paired_only)

    print("Creating zero-shot efficiency plot...")
    plotter_zs_eff = GroupedBarPlot(figsize=(18, 7))

    # Compute max efficiency for ylim
    max_eff = 0
    for model_data in data_dict_zs_eff.values():
        for prec_data in model_data.values():
            for val in prec_data.values():
                if val > max_eff:
                    max_eff = val

    plotter_zs_eff.plot(
        data_dict=data_dict_zs_eff,
        xlabel="Model",
        ylabel="Efficiency",
        ylim=(0, max_eff * 1.25),
        show_values=True,
        value_format=".1f",
    )
    output_path_zs_eff = output_dir / f"icl_zero_shot_efficiency{suffix}.png"
    plotter_zs_eff.save(output_path_zs_eff)

    # Plot for iterative mode
    print("\n" + "=" * 70)
    print(f"ITERATIVE ICL COMPARISON ({mode_str})")
    print("=" * 70)

    df_iter = load_icl_data(parquet_path, "iterative")

    # Success rate plot
    print("\nComputing success rate metrics...")
    data_dict_iter_success = compute_icl_metrics(df_iter, "is_successful", paired_only=paired_only)

    print("Creating iterative success rate plot...")
    plotter_iter_success = GroupedBarPlot(figsize=(18, 7))
    plotter_iter_success.plot(
        data_dict=data_dict_iter_success,
        xlabel="Model",
        ylabel="Success Rate (%)",
        ylim=(0, 105),
        show_values=True,
        value_format=".1f",
    )
    output_path_iter_success = output_dir / f"icl_iterative_success_rate{suffix}.png"
    plotter_iter_success.save(output_path_iter_success)

    # Efficiency plot
    print("Computing efficiency metrics...")
    data_dict_iter_eff = compute_icl_metrics(df_iter, "efficiency", paired_only=paired_only)

    print("Creating iterative efficiency plot...")
    plotter_iter_eff = GroupedBarPlot(figsize=(18, 7))

    # Compute max efficiency for ylim
    max_eff = 0
    for model_data in data_dict_iter_eff.values():
        for prec_data in model_data.values():
            for val in prec_data.values():
                if val > max_eff:
                    max_eff = val

    plotter_iter_eff.plot(
        data_dict=data_dict_iter_eff,
        xlabel="Model",
        ylabel="Efficiency",
        ylim=(0, max_eff * 1.25),
        show_values=True,
        value_format=".1f",
    )
    output_path_iter_eff = output_dir / f"icl_iterative_efficiency{suffix}.png"
    plotter_iter_eff.save(output_path_iter_eff)

    print("\n" + "=" * 70)
    print(f"ALL ICL {mode_str} PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    # Generate overall ICL comparison plots (all data)
    plot_icl_comparison(
        parquet_path="eval_results/merged_results.parquet",
        output_dir="plots/res/icl",
        paired_only=False,
    )

    # Generate overall ICL comparison plots (paired only)
    plot_icl_comparison(
        parquet_path="eval_results/merged_results.parquet",
        output_dir="plots/res/icl",
        paired_only=True,
    )

    # Generate per-dataset ICL comparison plots
    plot_icl_per_dataset(
        parquet_path="eval_results/merged_results.parquet",
        output_dir="plots/res/icl/per_dataset",
    )
