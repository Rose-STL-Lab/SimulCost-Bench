#!/usr/bin/env python3
"""
Statistical analysis of paired zero-shot vs iterative results.
Performs paired t-tests on success rates and efficiency.
"""

import pandas as pd
from pathlib import Path

from utils import (
    read_0shot_data,
    read_iterative_data,
    get_paired_data,
    compute_paired_differences,
    compute_paired_ttest,
)
from constants import BASE_DATASETS, PAIRING_COLS


def main():
    parquet_path = "../eval_results/merged_results.parquet"
    output_dir = Path("res/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_zs = read_0shot_data(parquet_path, datasets=BASE_DATASETS)
    df_iter = read_iterative_data(parquet_path, datasets=BASE_DATASETS)

    # Get paired data
    print("\nGetting paired data...")
    paired = get_paired_data(df_zs, df_iter, merge=True)
    print(f"Paired entries: {len(paired)}")

    # Overall analysis
    print("\n" + "=" * 70)
    print("OVERALL PAIRED T-TEST (is_successful)")
    print("=" * 70)

    diff_df = compute_paired_differences(paired, "is_successful", PAIRING_COLS)
    ttest_result = compute_paired_ttest(diff_df)

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
    print(f"\n{'Model':<25s} {'t':>7s} {'p-value':>10s} {'Sig':>4s} {'Mean Diff':>10s} {'95% CI':>20s} {'n':>6s}")
    print("-" * 90)

    results_list = []
    for model in sorted(paired["model_name"].unique()):
        model_paired = paired[paired["model_name"] == model]

        diff_df = compute_paired_differences(model_paired, "is_successful", PAIRING_COLS)
        ttest_res = compute_paired_ttest(diff_df)

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

    # Save per-model results to CSV
    results_df = pd.DataFrame(results_list)
    output_file = output_dir / "paired_ttest_per_model.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\nPer-model results saved to: {output_file}")

    # Per-precision analysis
    print("\n" + "=" * 70)
    print("PER-PRECISION PAIRED T-TESTS")
    print("=" * 70)
    print(f"\n{'Precision':<15s} {'t':>7s} {'p-value':>10s} {'Sig':>4s} {'Mean Diff':>10s} {'95% CI':>20s} {'n':>6s}")
    print("-" * 90)

    precision_results_list = []
    for precision in ["low", "medium", "high"]:
        precision_paired = paired[paired["precision_level"] == precision]

        diff_df = compute_paired_differences(precision_paired, "is_successful", PAIRING_COLS)
        ttest_res = compute_paired_ttest(diff_df)

        mean_zs = diff_df["is_successful_zs"].mean()
        mean_iter = diff_df["is_successful_iter"].mean()

        sig = (
            "***"
            if ttest_res["p_value"] < 0.001
            else "**" if ttest_res["p_value"] < 0.01 else "*" if ttest_res["p_value"] < 0.05 else "ns"
        )

        print(
            f"{precision.capitalize():<15s} {ttest_res['t_statistic']:7.3f} {ttest_res['p_value']:10.2e} {sig:>4s} "
            f"{ttest_res['mean_diff']:10.4f} [{ttest_res['ci_95'][0]:6.4f}, {ttest_res['ci_95'][1]:6.4f}] "
            f"{ttest_res['n_pairs']:6d}"
        )

        precision_results_list.append(
            {
                "precision_level": precision,
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

    # Save per-precision results to CSV
    precision_results_df = pd.DataFrame(precision_results_list)
    precision_output_file = output_dir / "paired_ttest_per_precision.csv"
    precision_results_df.to_csv(precision_output_file, index=False)
    print(f"\n\nPer-precision results saved to: {precision_output_file}")


if __name__ == "__main__":
    main()
