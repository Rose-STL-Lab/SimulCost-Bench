#!/usr/bin/env python3
"""
McNemar's test analysis for ICL variants compared to base (normal) model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from mcnemar_utils import compute_mcnemar_test, plot_forest_plot
from constants import ICL_BASE_DATASETS, ICL_MODEL, PAIRING_COLS, SUBGROUP_COLORS
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


def compute_icl_mcnemar_by_variant(df, base_model_name):
    """
    Compute McNemar's test for each ICL variant compared to base model.

    Args:
        df: DataFrame with is_successful column (either zero-shot or iterative, NOT merged)
        base_model_name: Name of the base model (e.g., "Claude-3.7-Sonnet")

    Returns:
        DataFrame with columns: variant, statistic, p_value, mean_diff, ci_lower, ci_upper, n_pairs, mean_base, mean_variant
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

        # Compute McNemar's test using utility function
        mcnemar_res = compute_mcnemar_test(merged["is_successful_base"].values, merged["is_successful_variant"].values)

        # Extract variant name (remove base model prefix)
        variant_short = variant.replace(base_model_name + "-", "")

        results.append(
            {
                "variant": variant_short,
                "statistic": mcnemar_res["statistic"],
                "p_value": mcnemar_res["p_value"],
                "mean_diff": mcnemar_res["mean_diff"],
                "ci_lower": mcnemar_res["ci_95"][0],
                "ci_upper": mcnemar_res["ci_95"][1],
                "n_pairs": mcnemar_res["n_pairs"],
                "n_improved": mcnemar_res["n_improved"],
                "n_degraded": mcnemar_res["n_degraded"],
                "mean_base": mcnemar_res["mean_base"],
                "mean_variant": mcnemar_res["mean_variant"],
            }
        )

    return pd.DataFrame(results)


def plot_icl_forest_plot(results_df, output_path):
    """
    Create forest plot for ICL variant comparisons.
    Uses plot_forest_plot with variant as group column and no subgroup.

    Args:
        results_df: DataFrame from compute_icl_mcnemar_by_variant()
        output_path: Path to save figure
    """
    # Rename variant column to use plot_forest_plot
    plot_df = results_df.copy()
    plot_df = plot_df.rename(columns={"variant": "model_name"})
    plot_forest_plot(plot_df, "model_name", None, output_path)


def print_icl_summary(results_df):
    """
    Print brief summary of ICL McNemar test results.

    Args:
        results_df: DataFrame from compute_icl_mcnemar_by_variant()
    """
    n_significant = (results_df["p_value"] < 0.05).sum()
    n_positive = (results_df["mean_diff"] > 0).sum()
    n_negative = (results_df["mean_diff"] < 0).sum()

    print(f"  {len(results_df)} ICL variants analyzed")
    print(f"  {n_significant}/{len(results_df)} variants show significant difference (p < 0.05)")
    print(f"  {n_positive} variants improve, {n_negative} variants degrade")


if __name__ == "__main__":
    parquet_path = "../eval_results/merged_results.parquet"
    output_dir = Path("res/mcnemar")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and transform ICL data
    df_zs, df_iter = _prepare_icl_data(parquet_path, datasets=ICL_BASE_DATASETS)

    print(f"Zero-shot entries: {len(df_zs)}")
    print(f"Iterative entries: {len(df_iter)}")

    # Single-round (zero-shot) analysis
    print("\nSINGLE-ROUND ICL VARIANT MCNEMAR TEST (vs Base):")
    zs_results = compute_icl_mcnemar_by_variant(df_zs, base_model_name=ICL_MODEL)
    print_icl_summary(zs_results)

    csv_path = output_dir / "icl_mcnemar_single_round.csv"
    zs_results.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    plot_icl_forest_plot(
        zs_results,
        output_dir / "icl_forest_plot_single_round.png",
    )

    # Multi-round (iterative) analysis
    print("\nMULTI-ROUND ICL VARIANT MCNEMAR TEST (vs Base):")
    iter_results = compute_icl_mcnemar_by_variant(df_iter, base_model_name=ICL_MODEL)
    print_icl_summary(iter_results)

    csv_path = output_dir / "icl_mcnemar_multi_round.csv"
    iter_results.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    plot_icl_forest_plot(
        iter_results,
        output_dir / "icl_forest_plot_multi_round.png",
    )

    print("\nDone!")
