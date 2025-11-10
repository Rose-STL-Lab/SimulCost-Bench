#!/usr/bin/env python3
"""
Statistical analysis of paired zero-shot vs iterative results.
Performs paired t-tests on success rates and efficiency.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from utils import read_0shot_data, read_iterative_data, get_paired_data
from constants import BASE_DATASETS


def compute_paired_differences(paired_df, metric_col="is_successful"):
    """
    Extract paired differences for a specific metric.

    Args:
        paired_df: DataFrame from get_paired_data()
        metric_col: Column name to compute differences for

    Returns:
        DataFrame with match_cols, metric_col_zs, metric_col_iter, difference
    """
    match_cols = [
        "dataset",
        "task",
        "precision_level",
        "model_name",
        "profile",
        "target_parameters",
        "non_target_parameters",
    ]

    zs_col = f"{metric_col}_zs"
    iter_col = f"{metric_col}_iter"

    result = paired_df[match_cols + [zs_col, iter_col]].copy()

    # Convert to numeric if boolean (for subtraction)
    if result[zs_col].dtype == "bool":
        result[zs_col] = result[zs_col].astype(int)
    if result[iter_col].dtype == "bool":
        result[iter_col] = result[iter_col].astype(int)

    # Compute difference (iterative - zero_shot)
    result["difference"] = result[iter_col] - result[zs_col]

    return result


def compute_paired_ttest(paired_df, metric_col="is_successful"):
    """
    Perform paired t-test on differences.

    Tests H0: mean(difference) = 0 vs H1: mean(difference) ≠ 0

    Args:
        paired_df: DataFrame from get_paired_data()
        metric_col: Column name to test

    Returns:
        dict with t_statistic, p_value, df, mean_diff, std_diff, se_diff, ci_95, n_pairs, significance flags
    """
    # Get differences
    diff_df = compute_paired_differences(paired_df, metric_col)
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




def main():
    parquet_path = "eval_results/merged_results.parquet"
    output_dir = Path("plots/res/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_zs = read_0shot_data(parquet_path, datasets=BASE_DATASETS)
    df_iter = read_iterative_data(parquet_path, datasets=BASE_DATASETS)

    # Get paired data
    print("\nGetting paired data...")
    paired = get_paired_data(df_zs, df_iter)
    print(f"Paired entries: {len(paired)}")

    # Overall analysis
    print("\n" + "=" * 70)
    print("OVERALL PAIRED T-TEST (is_successful)")
    print("=" * 70)

    ttest_result = compute_paired_ttest(paired, "is_successful")
    diff_df = compute_paired_differences(paired, "is_successful")

    mean_zs = diff_df["is_successful_zs"].mean()
    mean_iter = diff_df["is_successful_iter"].mean()

    print(f"\nHypothesis Test:")
    print(f"  H0: mean(iter - zs) = 0")
    print(f"  H1: mean(iter - zs) ≠ 0")

    print(f"\nPaired t-test:")
    print(f"  t({ttest_result['df']}) = {ttest_result['t_statistic']:.4f}")
    print(f"  p-value = {ttest_result['p_value']:.2e}")
    sig = (
        "***"
        if ttest_result["p_value"] < 0.001
        else "**" if ttest_result["p_value"] < 0.01 else "*" if ttest_result["p_value"] < 0.05 else "ns"
    )
    print(f"  Significance: {sig}")

    print(f"\nEstimates:")
    print(f"  Mean zero-shot: {mean_zs:.4f}")
    print(f"  Mean iterative: {mean_iter:.4f}")
    print(f"  Mean difference: {ttest_result['mean_diff']:.4f}")
    print(f"  95% CI: [{ttest_result['ci_95'][0]:.4f}, {ttest_result['ci_95'][1]:.4f}]")
    print(f"  Sample size: {ttest_result['n_pairs']} pairs")

    # Per-model analysis
    print("\n" + "=" * 70)
    print("PER-MODEL PAIRED T-TESTS")
    print("=" * 70)
    print(
        f"\n{'Model':<25s} {'t':>7s} {'p-value':>10s} {'Sig':>4s} {'Mean Diff':>10s} {'95% CI':>20s} {'n':>6s}"
    )
    print("-" * 90)

    results_list = []
    for model in sorted(paired["model_name"].unique()):
        model_paired = paired[paired["model_name"] == model]

        ttest_res = compute_paired_ttest(model_paired, "is_successful")
        diff_df = compute_paired_differences(model_paired, "is_successful")

        mean_zs = diff_df["is_successful_zs"].mean()
        mean_iter = diff_df["is_successful_iter"].mean()

        sig = (
            "***"
            if ttest_res["p_value"] < 0.001
            else "**" if ttest_res["p_value"] < 0.01 else "*" if ttest_res["p_value"] < 0.05 else "ns"
        )

        print(
            f"{model:<25s} {ttest_res['t_statistic']:7.3f} {ttest_res['p_value']:10.2e} {sig:>4s} "
            f"{ttest_res['mean_diff']:10.4f} [{ttest_res['ci_95'][0]:6.4f}, {ttest_res['ci_95'][1]:6.4f}] "
            f"{ttest_res['n_pairs']:6d}"
        )

        results_list.append(
            {
                "model": model,
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

    # Save results to CSV
    results_df = pd.DataFrame(results_list)
    output_file = output_dir / "paired_ttest_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
