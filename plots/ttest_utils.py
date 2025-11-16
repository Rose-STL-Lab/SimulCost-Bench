"""
Utilities for paired t-test analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from constants import PRECISION_ORDER, PRECISION_MARKERS, SUBGROUP_COLORS


def compute_paired_differences(paired_df, metric_col, pairing_cols):
    """
    Extract paired differences for a specific metric.

    Args:
        paired_df: DataFrame from get_paired_data() with _zs and _iter suffixes
        metric_col: Column name to compute differences for (without suffix)
        pairing_cols: List of columns that identify unique pairs (e.g., ["dataset", "task", "model_name", "precision_level"])

    Returns:
        DataFrame with pairing_cols, metric_col_zs, metric_col_iter, difference
    """
    zs_col = f"{metric_col}_zs"
    iter_col = f"{metric_col}_iter"

    result = paired_df[pairing_cols + [zs_col, iter_col]].copy()

    # Convert to numeric if boolean (for subtraction)
    if result[zs_col].dtype == "bool":
        result[zs_col] = result[zs_col].astype(int)
    if result[iter_col].dtype == "bool":
        result[iter_col] = result[iter_col].astype(int)

    # Compute difference (iterative - zero_shot)
    result["difference"] = result[iter_col] - result[zs_col]

    return result


def compute_paired_ttest(diff_df):
    """
    Perform paired t-test on differences.

    Tests H0: mean(difference) = 0 vs H1: mean(difference) != 0

    Args:
        diff_df: DataFrame with 'difference' column (from compute_paired_differences)

    Returns:
        dict with t_statistic, p_value, df, mean_diff, std_diff, se_diff, ci_95, n_pairs, significance flags
    """
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


def compute_ttest_results(paired_df, metric_col, pairing_cols):
    """
    Compute t-test statistics for paired data.

    Combines compute_paired_differences and compute_paired_ttest into a single call,
    returning a flat dictionary suitable for DataFrame construction.

    Args:
        paired_df: Paired data DataFrame with _zs and _iter suffixes
        metric_col: Column name to compute differences for (without suffix)
        pairing_cols: List of columns that identify unique pairs

    Returns:
        dict with t-test results and mean success rates
    """
    diff_df = compute_paired_differences(paired_df, metric_col, pairing_cols)
    ttest_res = compute_paired_ttest(diff_df)

    zs_col = f"{metric_col}_zs"
    iter_col = f"{metric_col}_iter"

    return {
        "t_statistic": ttest_res["t_statistic"],
        "p_value": ttest_res["p_value"],
        "mean_diff": ttest_res["mean_diff"],
        "ci_lower": ttest_res["ci_95"][0],
        "ci_upper": ttest_res["ci_95"][1],
        "n_pairs": ttest_res["n_pairs"],
        "mean_zs": diff_df[zs_col].mean(),
        "mean_iter": diff_df[iter_col].mean(),
    }


def print_ttest_summary(results_df):
    """
    Print brief summary of t-test results.

    Args:
        results_df: DataFrame with t-test results (must have p_value and mean_diff columns)
    """
    n_significant = (results_df["p_value"] < 0.05).sum()
    mean_improvement = results_df["mean_diff"].mean()

    print(f"  {len(results_df)} groups analyzed")
    print(f"  {n_significant}/{len(results_df)} groups show significant improvement (p < 0.05)")
    print(f"  Average improvement: {mean_improvement:.4f}")


def plot_forest_plot(results_df, group_col, subgroup_col, output_path):
    """
    Create forest plot from t-test results with group-subgroup hierarchy.

    Args:
        results_df: DataFrame with t-test results (must have mean_diff, ci_lower, ci_upper, p_value columns)
        group_col: Column name for main grouping (y-axis labels)
        subgroup_col: Column name for subgrouping (different markers/colors), or None for single-level
        output_path: Path to save figure
    """
    # Get groups in order, with "Overall" at bottom
    all_groups = list(results_df[group_col].unique())
    if "Overall" in all_groups:
        all_groups.remove("Overall")
        groups = sorted(all_groups) + ["Overall"]
    else:
        groups = sorted(all_groups)

    # Subgroup configuration (single-level is just two-level with 1 element)
    if subgroup_col is None:
        subgroups = ["single"]
        subgroup_markers = {"single": "o"}
    else:
        subgroups = PRECISION_ORDER
        subgroup_markers = PRECISION_MARKERS

    # Balanced offsets above and below center
    n_subgroups = len(subgroups)
    if n_subgroups == 1:
        subgroup_offsets = {subgroups[0]: 0.0}
    else:
        offset_range = 0.4
        subgroup_offsets = {s: offset_range / 2 - i * offset_range / (n_subgroups - 1) for i, s in enumerate(subgroups)}

    fig, ax = plt.subplots(figsize=(12, max(6, len(groups) * 1.0)))

    # Reverse y_positions so first group is at top, Overall at bottom
    y_positions = {group: i for i, group in enumerate(reversed(groups))}

    # Plot each subgroup for each group
    for subgroup_idx, subgroup in enumerate(subgroups):
        if subgroup_col is None:
            subgroup_data = results_df
        else:
            subgroup_data = results_df[results_df[subgroup_col] == subgroup]

        color = SUBGROUP_COLORS[subgroup_idx % len(SUBGROUP_COLORS)]
        marker = subgroup_markers[subgroup]

        for _, row in subgroup_data.iterrows():
            group = row[group_col]
            y_pos = y_positions[group] + subgroup_offsets[subgroup]

            # Override color to black for Overall group in single-level
            if subgroup_col is None and group == "Overall":
                plot_color = "black"
            else:
                plot_color = color

            # Convert to percentage
            mean_diff_pct = row["mean_diff"] * 100
            ci_lower_pct = row["ci_lower"] * 100
            ci_upper_pct = row["ci_upper"] * 100

            # Plot error bar with ticks at boundaries
            ax.plot([ci_lower_pct, ci_upper_pct], [y_pos, y_pos], "-", color=plot_color, linewidth=2, zorder=2)
            ax.plot(
                [ci_lower_pct, ci_lower_pct], [y_pos - 0.1, y_pos + 0.1], "-", color=plot_color, linewidth=2, zorder=2
            )
            ax.plot(
                [ci_upper_pct, ci_upper_pct], [y_pos - 0.1, y_pos + 0.1], "-", color=plot_color, linewidth=2, zorder=2
            )

            # Plot point
            ax.plot(mean_diff_pct, y_pos, marker, color=plot_color, markersize=10, zorder=3)

            # Add p-value annotation to the right
            p_value = row["p_value"]
            if p_value >= 0.1:
                p_text = f"p={p_value:.2f}"
            else:
                p_text = f"p<1e{int(np.floor(np.log10(p_value)))}"

            ax.text(ci_upper_pct + 0.5, y_pos, p_text, va="center", ha="left", fontsize=12, color=plot_color)

    # Reference line at 0
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Set y-axis with bold labels
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(reversed(groups)))

    # Bold y-tick labels, all black for consistency
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("black")

    # Add legend only for two-level (subgroups)
    if subgroup_col is not None:
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker=subgroup_markers[s],
                color=SUBGROUP_COLORS[i],
                linestyle="None",
                markersize=8,
                label=s.capitalize(),
            )
            for i, s in enumerate(subgroups)
        ]
        ax.legend(handles=legend_elements, loc="upper right", title="Precision Level")

    ax.set_xlabel("Success Rate Improvement (%)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")
