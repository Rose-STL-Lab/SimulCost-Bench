#!/usr/bin/env python3
"""
Visualize overall results across all datasets.
Creates bar charts grouped by model with precision level subgroups.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import read_0shot_data, read_iterative_data, compute_mean_metrics, create_overall_metric_plot

# Configuration constants
BASE_DATASETS = [
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


def main():
    parquet_path = "eval_results/merged_results.parquet"
    output_dir = Path("plots/res/overall")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for both modes
    df_zs = read_0shot_data(parquet_path, datasets=BASE_DATASETS)
    df_iter = read_iterative_data(parquet_path, datasets=BASE_DATASETS)

    # Calculate success rate metrics
    print("\nCalculating success rate metrics...")
    success_zs = compute_mean_metrics(
        df_zs,
        "is_successful",
        first_groupby=["dataset", "task", "model_name", "precision_level"],
        second_groupby=["model_name", "precision_level"],
    )
    success_zs["metric_value"] *= 100  # Convert to percentage

    success_iter = compute_mean_metrics(
        df_iter,
        "is_successful",
        first_groupby=["dataset", "task", "model_name", "precision_level"],
        second_groupby=["model_name", "precision_level"],
    )
    success_iter["metric_value"] *= 100  # Convert to percentage

    create_overall_metric_plot(
        success_zs,
        success_iter,
        xlabel="Model",
        ylabel="Success Rate (%)",
        ylim=(0, 105),
        output_path=output_dir / "overall_success_rate.png",
    )

    # Calculate efficiency metrics
    print("\nCalculating efficiency metrics...")
    efficiency_zs = compute_mean_metrics(
        df_zs,
        "efficiency",
        first_groupby=["dataset", "task", "model_name", "precision_level"],
        second_groupby=["model_name", "precision_level"],
    )
    efficiency_iter = compute_mean_metrics(
        df_iter,
        "efficiency",
        first_groupby=["dataset", "task", "model_name", "precision_level"],
        second_groupby=["model_name", "precision_level"],
    )

    max_eff = max(efficiency_zs["metric_value"].max(), efficiency_iter["metric_value"].max())
    create_overall_metric_plot(
        efficiency_zs,
        efficiency_iter,
        xlabel="Model",
        ylabel="Mean Efficiency",
        ylim=(0, max_eff * 1.25),
        output_path=output_dir / "overall_efficiency.png",
    )

    print("\nDone! Generated overall results visualizations.")


if __name__ == "__main__":
    main()
