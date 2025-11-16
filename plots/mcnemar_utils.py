"""
Utilities for McNemar's test analysis (paired binary data).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from constants import PRECISION_ORDER, PRECISION_MARKERS, SUBGROUP_COLORS


def compute_mcnemar_test(base_vals, variant_vals):
    """
    Perform McNemar's test on paired binary data.

    Tests H0: P(base=0, variant=1) = P(base=1, variant=0) (no difference in proportions)

    Args:
        base_vals: Array of binary values for base condition
        variant_vals: Array of binary values for variant condition

    Returns:
        dict with statistic, p_value, mean_diff, ci_95, n_pairs, contingency table counts
    """
    # Convert to numeric if boolean
    base_vals = np.asarray(base_vals)
    variant_vals = np.asarray(variant_vals)
    if base_vals.dtype == bool:
        base_vals = base_vals.astype(int)
    if variant_vals.dtype == bool:
        variant_vals = variant_vals.astype(int)

    n = len(base_vals)

    # Build contingency table
    # a: both fail (0,0), b: base fail, variant success (0,1)
    # c: base success, variant fail (1,0), d: both success (1,1)
    a = np.sum((base_vals == 0) & (variant_vals == 0))  # both fail
    b = np.sum((base_vals == 0) & (variant_vals == 1))  # base fail, variant success (improvement)
    c = np.sum((base_vals == 1) & (variant_vals == 0))  # base success, variant fail (degradation)
    d = np.sum((base_vals == 1) & (variant_vals == 1))  # both success

    # McNemar's test statistic (with continuity correction)
    if b + c == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        # Chi-squared with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = stats.chi2.sf(statistic, df=1)  # survival function = 1 - cdf

    # Mean difference in proportions
    mean_diff = (b - c) / n

    # 95% confidence interval for difference in proportions
    # Using Wald interval for paired proportions
    se_diff = np.sqrt((b + c) - (b - c) ** 2 / n) / n
    z_crit = 1.96
    ci_95 = (mean_diff - z_crit * se_diff, mean_diff + z_crit * se_diff)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_95": ci_95,
        "n_pairs": n,
        "n_both_fail": int(a),
        "n_improved": int(b),  # base fail -> variant success
        "n_degraded": int(c),  # base success -> variant fail
        "n_both_success": int(d),
        "mean_base": float(np.mean(base_vals)),
        "mean_variant": float(np.mean(variant_vals)),
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
