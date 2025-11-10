#!/usr/bin/env python3
"""
Visualization script for paired t-test results.
Creates forest plot to visualize mean differences and confidence intervals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import read_0shot_data, read_iterative_data, get_paired_data
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


def plot_forest_plot(paired_df, output_path):
    """
    Forest plot showing mean differences and 95% CIs for each model.
    Uses different colors for different models from barplot color scheme.
    """
    from analysis_paired_ttest import compute_paired_ttest
    from barplot_utils import SUBGROUP_COLORS

    models = sorted(paired_df["model_name"].unique())
    results = []

    for model in models:
        model_paired = paired_df[paired_df["model_name"] == model]
        ttest_res = compute_paired_ttest(model_paired, "is_successful")
        results.append(
            {
                "model": model,
                "mean_diff": ttest_res["mean_diff"],
                "ci_lower": ttest_res["ci_95"][0],
                "ci_upper": ttest_res["ci_95"][1],
                "p_value": ttest_res["p_value"],
            }
        )

    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(len(results_df))

    # Plot points and error bars with different colors per model
    for i, row in results_df.iterrows():
        color = SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]

        # Plot error bar with horizontal ticks at boundaries
        ax.plot(
            [row["ci_lower"], row["ci_upper"]],
            [y_positions[i], y_positions[i]],
            "-",
            color=color,
            linewidth=2,
            zorder=2,
        )

        # Add horizontal ticks at interval boundaries
        ax.plot(
            [row["ci_lower"], row["ci_lower"]],
            [y_positions[i] - 0.15, y_positions[i] + 0.15],
            "-",
            color=color,
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            [row["ci_upper"], row["ci_upper"]],
            [y_positions[i] - 0.15, y_positions[i] + 0.15],
            "-",
            color=color,
            linewidth=2,
            zorder=2,
        )

        # Plot point
        ax.plot(
            row["mean_diff"],
            y_positions[i],
            "o",
            color=color,
            markersize=10,
            zorder=3,
        )

    # Reference line at 0
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Set y-axis with colored labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(results_df["model"])

    # Color each y-tick label to match its corresponding model
    for i, tick_label in enumerate(ax.get_yticklabels()):
        color = SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    ax.set_xlabel("Mean Difference (Iterative - Zero-Shot)")
    ax.set_title("Forest Plot: Success Rate Improvement by Model")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved forest plot to {output_path}")


if __name__ == "__main__":
    parquet_path = "eval_results/merged_results.parquet"
    output_dir = Path("plots/res/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_zs = read_0shot_data(parquet_path, datasets=BASE_DATASETS)
    df_iter = read_iterative_data(parquet_path, datasets=BASE_DATASETS)

    # Get paired data
    print("Getting paired data...")
    paired = get_paired_data(df_zs, df_iter)
    print(f"Paired entries: {len(paired)}")

    # Generate forest plot
    plot_forest_plot(paired, output_dir / "forest_plot.png")

    print("\nForest plot generated successfully!")
