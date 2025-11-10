#!/usr/bin/env python3
"""
Visualize efficiency metrics by dataset and precision level.
Creates bar charts with different colors for zero-shot and iterative modes.

Efficiency is calculated as: success * (dummy_cost / model_cost)
- Higher values indicate better efficiency (achieving accuracy with lower cost)
- Zero values indicate failure to meet accuracy requirements
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16


def load_and_prepare_data(parquet_path: str):
    """Load and prepare data for visualization."""
    print("Loading data...")
    df = pd.read_parquet(parquet_path)

    # Filter out ICL datasets (only keep base datasets)
    base_datasets = [
        "burgers_1d",
        "diff_react_1d",
        "euler_1d",
        "heat_1d",
        "heat_2d",
        "ns_transient_2d",
        "mpm_2d",
        "epoch_1d",
        "euler_2d",
    ]
    df = df[df["dataset"].isin(base_datasets)].copy()

    print(f"Loaded {len(df)} records from {df['dataset'].nunique()} datasets")

    return df


def calculate_efficiency_metrics(df):
    """Calculate mean efficiency for each dataset, precision level, inference mode, and model."""
    # Group by dataset, precision level, inference mode, and model
    efficiency_metrics = (
        df.groupby(["dataset", "precision_level", "inference_mode", "model_name"])["efficiency"].mean().reset_index()
    )
    efficiency_metrics.rename(columns={"efficiency": "mean_efficiency"}, inplace=True)

    return efficiency_metrics


def create_visualizations_by_dataset(efficiency_metrics, output_dir="plots/res"):
    """Create separate visualizations for each dataset with three subplots (precision levels)."""

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    datasets = sorted(efficiency_metrics["dataset"].unique())
    precision_levels = ["low", "medium", "high"]
    models = sorted(efficiency_metrics["model_name"].unique())

    # Color palette
    colors = {"zero_shot": "#3498db", "iterative": "#e74c3c"}

    for dataset in datasets:
        # Create a figure for this dataset with 3 subplots (one for each precision level)
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))

        for precision_idx, precision in enumerate(precision_levels):
            ax = axes[precision_idx]

            dataset_data = efficiency_metrics[
                (efficiency_metrics["dataset"] == dataset) & (efficiency_metrics["precision_level"] == precision)
            ]

            # Prepare data for plotting
            x = np.arange(len(models))
            width = 0.35

            zero_shot_values = []
            iterative_values = []

            for model in models:
                zero_data = dataset_data[
                    (dataset_data["model_name"] == model) & (dataset_data["inference_mode"] == "zero_shot")
                ]
                zero_shot_values.append(zero_data["mean_efficiency"].values[0] if not zero_data.empty else 0)

                iter_data = dataset_data[
                    (dataset_data["model_name"] == model) & (dataset_data["inference_mode"] == "iterative")
                ]
                iterative_values.append(iter_data["mean_efficiency"].values[0] if not iter_data.empty else 0)

            # Plot bars
            bars1 = ax.bar(
                x - width / 2, zero_shot_values, width, label="Zero-shot", color=colors["zero_shot"], alpha=0.8
            )
            bars2 = ax.bar(
                x + width / 2, iterative_values, width, label="Iterative", color=colors["iterative"], alpha=0.8
            )

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                    )

            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                    )

            # Formatting
            ax.set_title(f"{precision.title()} Precision", fontsize=16, fontweight="bold")
            ax.set_xlabel("Model", fontsize=16)
            if precision_idx == 0:  # Only show ylabel on leftmost plot
                ax.set_ylabel("Mean Efficiency", fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=16)

            # Dynamically set y-axis limit based on data
            max_val = max(max(zero_shot_values, default=0), max(iterative_values, default=0))
            ax.set_ylim(0, max_val * 1.15)  # Add 15% padding

            ax.grid(axis="y", alpha=0.3, linestyle="--")
            if precision_idx == 0:  # Only show legend on leftmost plot
                ax.legend(loc="upper right", fontsize=16)

        plt.suptitle(
            f'{dataset.replace("_", " ").title()} - Mean Efficiency by Precision Level',
            fontsize=18,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        output_path = Path(output_dir) / f"efficiency_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


def create_precision_summary(efficiency_metrics, output_dir="plots/res"):
    """Create a summary figure showing average efficiency by precision level, averaged across all datasets."""

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    precision_levels = ["low", "medium", "high"]
    models = sorted(efficiency_metrics["model_name"].unique())

    # Create a figure with 3 subplots (one for each precision level)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Color palette
    colors = {"zero_shot": "#3498db", "iterative": "#e74c3c"}

    for precision_idx, precision in enumerate(precision_levels):
        ax = axes[precision_idx]

        # Filter data for this precision level
        precision_data = efficiency_metrics[efficiency_metrics["precision_level"] == precision]

        # Prepare data for plotting
        x = np.arange(len(models))
        width = 0.35

        # Calculate average across all datasets for each model and inference mode
        zero_shot_values = []
        iterative_values = []

        for model in models:
            model_data = precision_data[precision_data["model_name"] == model]

            zero_data = model_data[model_data["inference_mode"] == "zero_shot"]
            zero_avg = zero_data["mean_efficiency"].mean() if not zero_data.empty else 0
            zero_shot_values.append(zero_avg)

            iter_data = model_data[model_data["inference_mode"] == "iterative"]
            iter_avg = iter_data["mean_efficiency"].mean() if not iter_data.empty else 0
            iterative_values.append(iter_avg)

        # Plot bars
        bars1 = ax.bar(x - width / 2, zero_shot_values, width, label="Zero-shot", color=colors["zero_shot"], alpha=0.8)
        bars2 = ax.bar(x + width / 2, iterative_values, width, label="Iterative", color=colors["iterative"], alpha=0.8)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=11
                )

        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=11
                )

        # Formatting
        ax.set_title(f"{precision.title()} Precision", fontsize=16, fontweight="bold")
        ax.set_xlabel("Model", fontsize=16)
        if precision_idx == 0:  # Only show ylabel on leftmost plot
            ax.set_ylabel("Mean Efficiency", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=14)

        # Dynamically set y-axis limit
        max_val = max(max(zero_shot_values, default=0), max(iterative_values, default=0))
        ax.set_ylim(0, max_val * 1.15)

        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if precision_idx == 0:  # Only show legend on leftmost plot
            ax.legend(loc="upper right", fontsize=14)

    plt.suptitle(
        "Mean Efficiency by Precision Level (Averaged Across All Datasets)", fontsize=20, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    output_path = Path(output_dir) / "efficiency_by_precision_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def create_comparison_visualization(efficiency_metrics, output_dir="plots/res"):
    """Create a single visualization comparing all datasets and precision levels."""

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Average across models for each dataset, precision level, and inference mode
    aggregated = (
        efficiency_metrics.groupby(["dataset", "precision_level", "inference_mode"])["mean_efficiency"]
        .mean()
        .reset_index()
    )

    datasets = sorted(aggregated["dataset"].unique())
    precision_levels = ["low", "medium", "high"]

    fig, ax = plt.subplots(figsize=(18, 8))

    x = np.arange(len(datasets))
    width = 0.12  # Narrower bars for better spacing

    colors = {
        ("low", "zero_shot"): "#5dade2",  # Light blue
        ("low", "iterative"): "#ec7063",  # Light red
        ("medium", "zero_shot"): "#3498db",  # Medium blue
        ("medium", "iterative"): "#e74c3c",  # Medium red
        ("high", "zero_shot"): "#2471a3",  # Dark blue
        ("high", "iterative"): "#c0392b",  # Dark red
    }

    for i, precision in enumerate(precision_levels):
        # Zero-shot bars
        zero_values = []
        for dataset in datasets:
            data = aggregated[
                (aggregated["dataset"] == dataset)
                & (aggregated["precision_level"] == precision)
                & (aggregated["inference_mode"] == "zero_shot")
            ]
            zero_values.append(data["mean_efficiency"].values[0] if not data.empty else 0)

        ax.bar(
            x + i * width * 2 - width * 2.5,
            zero_values,
            width,
            label=f"{precision.title()} - Zero-shot",
            color=colors[(precision, "zero_shot")],
            alpha=0.8,
        )

        # Iterative bars
        iter_values = []
        for dataset in datasets:
            data = aggregated[
                (aggregated["dataset"] == dataset)
                & (aggregated["precision_level"] == precision)
                & (aggregated["inference_mode"] == "iterative")
            ]
            iter_values.append(data["mean_efficiency"].values[0] if not data.empty else 0)

        ax.bar(
            x + i * width * 2 - width * 2.5 + width,
            iter_values,
            width,
            label=f"{precision.title()} - Iterative",
            color=colors[(precision, "iterative")],
            alpha=0.8,
        )

    ax.set_title(
        "Average Mean Efficiency by Dataset, Precision Level, and Inference Mode",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Dataset", fontsize=16)
    ax.set_ylabel("Mean Efficiency", fontsize=16)
    ax.set_xticks(x - width * 1.5)
    ax.set_xticklabels([d.replace("_", " ").title() for d in datasets], rotation=45, ha="right", fontsize=16)
    ax.set_ylim(0, None)  # Auto-scale y-axis
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", ncol=3, fontsize=16)

    plt.tight_layout()
    output_path = Path(output_dir) / "efficiency_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parquet_path = "eval_results/merged_results.parquet"

    # Load data
    df = load_and_prepare_data(parquet_path)

    # Calculate efficiency metrics
    efficiency_metrics = calculate_efficiency_metrics(df)

    # Create visualizations
    print("\nCreating visualizations by dataset...")
    create_visualizations_by_dataset(efficiency_metrics)

    print("\nCreating precision summary visualization...")
    create_precision_summary(efficiency_metrics)

    print("\nCreating overall comparison visualization...")
    create_comparison_visualization(efficiency_metrics)

    print("\nDone! Check the generated PNG files.")


if __name__ == "__main__":
    main()
