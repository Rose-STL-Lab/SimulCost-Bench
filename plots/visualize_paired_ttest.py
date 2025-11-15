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

from utils import (
    read_0shot_data,
    read_iterative_data,
    get_paired_data,
    compute_paired_differences,
    compute_paired_ttest,
)
from constants import BASE_DATASETS, PAIRING_COLS

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


def compute_ttest_by_group(paired_df, group_by, include_overall):
    """
    Compute paired t-test statistics for each group.

    Args:
        paired_df: Paired data DataFrame
        group_by: Column to group by (e.g., "model_name" or "precision_level")
        include_overall: Whether to include an "Overall" row for all data combined

    Returns:
        DataFrame with columns: {group_by}, t_statistic, p_value, mean_diff, ci_lower, ci_upper, n_pairs, mean_zs, mean_iter
    """
    groups = sorted(paired_df[group_by].unique())

    # Custom order for precision_level to get high-medium-low
    if group_by == "precision_level":
        groups = ["High", "Medium", "Low"]

    results = []

    for group in groups:
        group_paired = paired_df[paired_df[group_by] == group]
        diff_df = compute_paired_differences(group_paired, "is_successful", PAIRING_COLS)
        ttest_res = compute_paired_ttest(diff_df)

        mean_zs = diff_df["is_successful_zs"].mean()
        mean_iter = diff_df["is_successful_iter"].mean()

        results.append(
            {
                group_by: group,
                "t_statistic": ttest_res["t_statistic"],
                "p_value": ttest_res["p_value"],
                "mean_diff": ttest_res["mean_diff"],
                "ci_lower": ttest_res["ci_95"][0],
                "ci_upper": ttest_res["ci_95"][1],
                "n_pairs": ttest_res["n_pairs"],
                "mean_zs": mean_zs,
                "mean_iter": mean_iter,
            }
        )

    # Add overall if requested
    if include_overall:
        diff_df = compute_paired_differences(paired_df, "is_successful", PAIRING_COLS)
        ttest_res = compute_paired_ttest(diff_df)

        mean_zs = diff_df["is_successful_zs"].mean()
        mean_iter = diff_df["is_successful_iter"].mean()

        results.append(
            {
                group_by: "Overall",
                "t_statistic": ttest_res["t_statistic"],
                "p_value": ttest_res["p_value"],
                "mean_diff": ttest_res["mean_diff"],
                "ci_lower": ttest_res["ci_95"][0],
                "ci_upper": ttest_res["ci_95"][1],
                "n_pairs": ttest_res["n_pairs"],
                "mean_zs": mean_zs,
                "mean_iter": mean_iter,
            }
        )

    return pd.DataFrame(results)


def plot_forest_plot(results_df, output_path):
    """
    Create forest plot from t-test results.
    Uses different colors for different groups from barplot color scheme.

    Args:
        results_df: DataFrame from compute_ttest_by_group()
        output_path: Path to save figure
    """
    from barplot_utils import SUBGROUP_COLORS

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reverse y_positions so last row (Overall) is at bottom
    y_positions = np.arange(len(results_df))[::-1]

    # Plot points and error bars with different colors per model
    for i, (idx, row) in enumerate(results_df.iterrows()):
        # Use black color for Overall, otherwise use SUBGROUP_COLORS
        if row.iloc[0] == "Overall":
            color = "black"
        else:
            color = SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]

        # Convert to percentage
        mean_diff_pct = row["mean_diff"] * 100
        ci_lower_pct = row["ci_lower"] * 100
        ci_upper_pct = row["ci_upper"] * 100

        # Plot error bar with horizontal ticks at boundaries
        ax.plot(
            [ci_lower_pct, ci_upper_pct],
            [y_positions[i], y_positions[i]],
            "-",
            color=color,
            linewidth=2,
            zorder=2,
        )

        # Add horizontal ticks at interval boundaries
        ax.plot(
            [ci_lower_pct, ci_lower_pct],
            [y_positions[i] - 0.15, y_positions[i] + 0.15],
            "-",
            color=color,
            linewidth=2,
            zorder=2,
        )
        ax.plot(
            [ci_upper_pct, ci_upper_pct],
            [y_positions[i] - 0.15, y_positions[i] + 0.15],
            "-",
            color=color,
            linewidth=2,
            zorder=2,
        )

        # Plot point
        ax.plot(
            mean_diff_pct,
            y_positions[i],
            "o",
            color=color,
            markersize=10,
            zorder=3,
        )

        # Add p-value annotation below the confidence interval
        p_value = row["p_value"]
        if p_value < 1e-10:
            p_text = f"p < 1e{int(np.floor(np.log10(p_value)))}"
        elif p_value < 0.001:
            p_text = f"p = {p_value:.2e}"
        else:
            p_text = f"p = {p_value:.3f}"

        ax.text(
            mean_diff_pct,  # Center on the mean difference point
            y_positions[i] - 0.2,  # Below the bar
            p_text,
            va="top",
            ha="center",
            fontsize=12,
            color=color,
        )

    # Reference line at 0
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Set y-axis with colored labels (use first column which is the group identifier)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(results_df.iloc[:, 0])

    # Color each y-tick label to match its corresponding group
    for i, (tick_label, (idx, row)) in enumerate(zip(ax.get_yticklabels(), results_df.iterrows())):
        if row.iloc[0] == "Overall":
            color = "black"
        else:
            color = SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    ax.set_xlabel("Success Rate Improvement (%)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def print_ttest_summary(results_df, group_by):
    """
    Print brief summary of t-test results.

    Args:
        results_df: DataFrame from compute_ttest_by_group()
        group_by: Column name used for grouping
    """
    n_significant = (results_df["p_value"] < 0.05).sum()
    mean_improvement = results_df["mean_diff"].mean()

    print(f"  {len(results_df)} groups analyzed by {group_by}")
    print(f"  {n_significant}/{len(results_df)} groups show significant improvement (p < 0.05)")
    print(f"  Average improvement: {mean_improvement:.4f}")


if __name__ == "__main__":
    parquet_path = "../eval_results/merged_results.parquet"
    output_dir = Path("res/ttest")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_zs = read_0shot_data(parquet_path, datasets=BASE_DATASETS)
    df_iter = read_iterative_data(parquet_path, datasets=BASE_DATASETS)

    # Get paired data
    print("Getting paired data...")
    paired = get_paired_data(df_zs, df_iter, merge=True)
    print(f"Paired entries: {len(paired)}")

    # Overall t-test
    print("\nOVERALL PAIRED T-TEST:")
    diff_df = compute_paired_differences(paired, "is_successful", PAIRING_COLS)
    ttest_result = compute_paired_ttest(diff_df)

    sig = (
        "***"
        if ttest_result["p_value"] < 0.001
        else "**" if ttest_result["p_value"] < 0.01 else "*" if ttest_result["p_value"] < 0.05 else "ns"
    )
    print(f"  Mean improvement: {ttest_result['mean_diff']:.4f} (p = {ttest_result['p_value']:.2e} {sig})")
    print(f"  95% CI: [{ttest_result['ci_95'][0]:.4f}, {ttest_result['ci_95'][1]:.4f}], n = {ttest_result['n_pairs']}")

    # Per-model analysis
    print("\nPER-MODEL ANALYSIS:")
    model_results = compute_ttest_by_group(paired, group_by="model_name", include_overall=True)
    print_ttest_summary(model_results, "model_name")

    csv_path = output_dir / "paired_ttest_by_model.csv"
    model_results.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    plot_forest_plot(
        model_results,
        output_dir / "forest_plot_by_model.png",
    )

    # Per-precision analysis
    print("\nPER-PRECISION ANALYSIS:")
    precision_results = compute_ttest_by_group(paired, group_by="precision_level", include_overall=True)
    print_ttest_summary(precision_results, "precision_level")

    csv_path = output_dir / "paired_ttest_by_precision.csv"
    precision_results.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    plot_forest_plot(
        precision_results,
        output_dir / "forest_plot_by_precision.png",
    )

    print("\nDone!")
