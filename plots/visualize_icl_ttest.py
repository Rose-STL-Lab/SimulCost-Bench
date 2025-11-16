#!/usr/bin/env python3
"""
T-test analysis for ICL variants compared to base (normal) model.
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
)
from ttest_utils import compute_paired_differences, compute_paired_ttest
from constants import ICL_BASE_DATASETS, ICL_MODEL, PAIRING_COLS
from visualize_icl import _prepare_icl_data

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


def compute_icl_ttest_by_variant(df, base_model_name):
    """
    Compute paired t-test for each ICL variant compared to base model.

    Args:
        df: DataFrame with is_successful column (either zero-shot or iterative, NOT merged)
        base_model_name: Name of the base model (e.g., "Claude-3.7-Sonnet")

    Returns:
        DataFrame with columns: variant, t_statistic, p_value, mean_diff, ci_lower, ci_upper, n_pairs, mean_base, mean_variant
    """
    # Get all model variants (ICL variants)
    all_models = sorted(df["model_name"].unique())
    icl_variants = [m for m in all_models if m != base_model_name and m.startswith(base_model_name)]

    results = []

    for variant in icl_variants:
        # Filter for base and variant
        base_data = df[df["model_name"] == base_model_name].copy()
        variant_data = df[df["model_name"] == variant].copy()

        # Merge on pairing columns to get matched pairs
        merge_cols = [col for col in PAIRING_COLS if col != "model_name"]
        merged = base_data.merge(
            variant_data,
            on=merge_cols,
            suffixes=("_base", "_variant"),
        )

        if len(merged) == 0:
            continue

        # Compute differences (variant - base)
        # Convert to int to avoid boolean subtraction issue
        merged["difference"] = merged["is_successful_variant"].astype(int) - merged["is_successful_base"].astype(int)

        # Compute t-test
        differences = merged["difference"].values
        n = len(differences)

        from scipy import stats

        t_statistic, p_value = stats.ttest_1samp(differences, 0)

        # Calculate statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_diff = std_diff / np.sqrt(n)

        # 95% confidence interval
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_95 = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

        mean_base = merged["is_successful_base"].mean()
        mean_variant = merged["is_successful_variant"].mean()

        # Extract variant name (remove base model prefix)
        variant_short = variant.replace(base_model_name + "-", "")

        results.append(
            {
                "variant": variant_short,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "mean_diff": mean_diff,
                "ci_lower": ci_95[0],
                "ci_upper": ci_95[1],
                "n_pairs": n,
                "mean_base": mean_base,
                "mean_variant": mean_variant,
            }
        )

    return pd.DataFrame(results)


def plot_icl_forest_plot(results_df, output_path):
    """
    Create forest plot for ICL variant comparisons.

    Args:
        results_df: DataFrame from compute_icl_ttest_by_variant()
        output_path: Path to save figure
    """
    from barplot_utils import SUBGROUP_COLORS

    fig, ax = plt.subplots(figsize=(10, 6))

    # Reverse y_positions so we go top to bottom
    y_positions = np.arange(len(results_df))[::-1]

    # Plot points and error bars with different colors per variant
    for i, (idx, row) in enumerate(results_df.iterrows()):
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
            mean_diff_pct,
            y_positions[i] - 0.25,
            p_text,
            va="top",
            ha="center",
            fontsize=12,
            color=color,
        )

    # Reference line at 0
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Set y-axis with colored labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(results_df["variant"])

    # Color each y-tick label to match its corresponding variant
    for i, (tick_label, (idx, row)) in enumerate(zip(ax.get_yticklabels(), results_df.iterrows())):
        color = SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    ax.set_xlabel("Success Rate Difference vs Base (%)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def print_icl_summary(results_df):
    """
    Print brief summary of ICL t-test results.

    Args:
        results_df: DataFrame from compute_icl_ttest_by_variant()
    """
    n_significant = (results_df["p_value"] < 0.05).sum()
    n_positive = (results_df["mean_diff"] > 0).sum()
    n_negative = (results_df["mean_diff"] < 0).sum()

    print(f"  {len(results_df)} ICL variants analyzed")
    print(f"  {n_significant}/{len(results_df)} variants show significant difference (p < 0.05)")
    print(f"  {n_positive} variants improve, {n_negative} variants degrade")


if __name__ == "__main__":
    parquet_path = "../eval_results/merged_results.parquet"
    output_dir = Path("res/icl")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and transform ICL data
    df_zs, df_iter = _prepare_icl_data(parquet_path, datasets=ICL_BASE_DATASETS)

    print(f"Zero-shot entries: {len(df_zs)}")
    print(f"Iterative entries: {len(df_iter)}")

    # Single-round (zero-shot) analysis
    print("\nSINGLE-ROUND ICL VARIANT T-TEST (vs Base):")
    zs_results = compute_icl_ttest_by_variant(df_zs, base_model_name=ICL_MODEL)
    print_icl_summary(zs_results)

    csv_path = output_dir / "icl_ttest_single_round.csv"
    zs_results.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    plot_icl_forest_plot(
        zs_results,
        output_dir / "icl_forest_plot_single_round.png",
    )

    # Multi-round (iterative) analysis
    print("\nMULTI-ROUND ICL VARIANT T-TEST (vs Base):")
    iter_results = compute_icl_ttest_by_variant(df_iter, base_model_name=ICL_MODEL)
    print_icl_summary(iter_results)

    csv_path = output_dir / "icl_ttest_multi_round.csv"
    iter_results.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    plot_icl_forest_plot(
        iter_results,
        output_dir / "icl_forest_plot_multi_round.png",
    )

    print("\nDone!")
