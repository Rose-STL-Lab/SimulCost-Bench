#!/usr/bin/env python3
"""
Visualization script for paired t-test results.
Creates forest plot to visualize mean differences and confidence intervals.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import read_0shot_data, read_iterative_data, get_paired_data
from constants import BASE_DATASETS, PAIRING_COLS, PRECISION_ORDER
from ttest_utils import compute_ttest_results, print_ttest_summary, plot_forest_plot

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

    # Per-model-precision granular analysis
    print("\nPER-MODEL-PRECISION ANALYSIS:")
    model_precision_results = []
    # Use lowercase for data filtering, capitalize for display
    precision_data_order = ["high", "medium", "low"]
    for model in sorted(paired["model_name"].unique()):
        for precision in precision_data_order:
            subset = paired[(paired["model_name"] == model) & (paired["precision_level"] == precision)]
            if len(subset) == 0:
                continue
            result = compute_ttest_results(subset, "is_successful", PAIRING_COLS)
            result["model_name"] = model
            result["precision_level"] = precision.capitalize()
            model_precision_results.append(result)
    # Add overall model (aggregating all models) for each precision
    for precision in precision_data_order:
        subset = paired[paired["precision_level"] == precision]
        result = compute_ttest_results(subset, "is_successful", PAIRING_COLS)
        result["model_name"] = "Overall"
        result["precision_level"] = precision.capitalize()
        model_precision_results.append(result)
    model_precision_results_df = pd.DataFrame(model_precision_results)

    print_ttest_summary(model_precision_results_df)
    csv_path = output_dir / "paired_ttest_by_model_precision.csv"
    model_precision_results_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")
    plot_forest_plot(
        model_precision_results_df, "model_name", "precision_level", output_dir / "forest_plot_by_model_precision.png"
    )

    print("\nDone!")
