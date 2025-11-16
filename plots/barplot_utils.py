"""
Utilities for creating grouped bar plots and forest plots.
"""

import matplotlib.pyplot as plt
import numpy as np

from constants import PRECISION_ORDER, PRECISION_MARKERS, SUBGROUP_COLORS


class GroupedBarPlot:
    """
    General plotting class for grouped bar charts.

    Structure:
    - X-axis: Main groups (e.g., LLM models)
    - Within each group: Subgroups (e.g., precision levels)
    - Within each subgroup: One or more bars (e.g., different experiment settings)

    Color scheme:
    - Different subgroups have different colors (MATLAB-like palette)
    - Within each subgroup, bars share the same color but differ by pattern/legend
    """

    def __init__(self, figsize=(14, 7)):
        self.figsize = figsize
        self.fig = None
        self.ax = None

    def plot(self, data_dict, xlabel, ylabel, ylim=None, show_values=True, value_format=".1f"):
        """
        Create grouped bar plot.

        Args:
            data_dict: Dict with structure: {group_name: {subgroup_name: {bar_type: value}}}
                      Example: {'Model1': {'low': {'zero_shot': 0.5, 'iterative': 0.6}, ...}, ...}
            xlabel: X-axis label
            ylabel: Y-axis label
            ylim: Tuple (ymin, ymax) or None for auto
            show_values: Whether to show value labels on bars
            value_format: Format string for value labels (default '.1f')

        Returns:
            fig, ax
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Extract structure from data
        groups = list(data_dict.keys())
        subgroups = list(list(data_dict.values())[0].keys())
        bar_types = list(list(list(data_dict.values())[0].values())[0].keys())

        n_groups = len(groups)
        n_subgroups = len(subgroups)
        n_bars_per_subgroup = len(bar_types)

        # Width allocation - clearer approach:
        # Each group gets 1.0 unit, divide it among subgroups with gaps
        group_width = 0.90  # Use 85% of space for bars, 15% natural gap between groups

        # Within each subgroup: small gap between bars (zero-shot/iterative)
        gap_within_subgroup = 0.00

        # Between subgroups: larger gap to visually separate precision levels
        gap_between_subgroups = 0.025

        # Calculate widths
        total_gaps = (n_subgroups - 1) * gap_between_subgroups
        available_for_bars = group_width - total_gaps
        subgroup_width = available_for_bars / n_subgroups
        bar_width = (subgroup_width - (n_bars_per_subgroup - 1) * gap_within_subgroup) / n_bars_per_subgroup

        # Plot bars
        for i, subgroup in enumerate(subgroups):
            color = SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]

            for j, bar_type in enumerate(bar_types):
                # Collect values for all groups
                values = [data_dict[group][subgroup][bar_type] for group in groups]

                # Position calculation - simpler and clearer:
                x_positions = []
                for group_idx in range(n_groups):
                    # Start from left edge of group
                    group_start = group_idx - group_width / 2

                    # Position of current subgroup start
                    subgroup_start = group_start + i * (subgroup_width + gap_between_subgroups)

                    # Position of current bar within subgroup
                    bar_pos = subgroup_start + j * (bar_width + gap_within_subgroup)

                    x_positions.append(bar_pos + bar_width / 2)  # Center of bar

                # Use hatching pattern and fill to differentiate bars within subgroup
                # Support up to 4 bars: solid, forward hatch, dots, hollow
                if j == 0:
                    hatch = None
                    edgecolor = color
                    linewidth = 0
                    facecolor = color
                    alpha = 0.85
                elif j == 1:
                    hatch = "//"
                    edgecolor = "black"
                    linewidth = 0.5
                    facecolor = color
                    alpha = 0.85
                elif j == 2:
                    hatch = "..."
                    edgecolor = "black"
                    linewidth = 0.5
                    facecolor = color
                    alpha = 0.85
                else:  # j == 3 or higher
                    hatch = None
                    edgecolor = color
                    linewidth = 1.5
                    facecolor = "none"
                    alpha = 1.0

                # Create label that combines bar_type and subgroup for proper legend
                label = f"{bar_type}: {subgroup}"

                bars = self.ax.bar(
                    x_positions,
                    values,
                    bar_width,
                    label=label,
                    color=facecolor,
                    alpha=alpha,
                    hatch=hatch,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )

                # Add value labels
                if show_values:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            self.ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height,
                                f"{height:{value_format}}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                            )

        # Formatting
        self.ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
        self.ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
        self.ax.set_xticks(np.arange(n_groups))
        self.ax.set_xticklabels(groups, rotation=45, ha="right", fontweight="bold")
        # Create legend organized by bar type (rows) and subgroup (columns)
        handles, labels = self.ax.get_legend_handles_labels()
        # The handles/labels are currently: [bar0_sub0, bar0_sub1, bar0_sub2, bar1_sub0, bar1_sub1, ...]
        # We want rows of bar types, columns of subgroups
        self.ax.legend(handles, labels, loc="upper right", framealpha=0.9, ncol=n_subgroups)
        self.ax.grid(axis="y", alpha=0.3, linestyle="--")

        if ylim is not None:
            self.ax.set_ylim(ylim)

        plt.tight_layout()
        return self.fig, self.ax

    def save(self, output_path, dpi=300):
        """Save figure to file."""
        if self.fig is None:
            raise ValueError("No figure to save. Call plot() first.")
        self.fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close(self.fig)


def plot_metric_comparison_barplot(
    zero_shot_metrics, iterative_metrics, xlabel, ylabel, ylim, output_path, include_overall
):
    """
    Create bar plot grouped by model with precision level subgroups.

    Args:
        zero_shot_metrics: DataFrame with columns [model_name, precision_level, metric_value]
        iterative_metrics: DataFrame with columns [model_name, precision_level, metric_value]
        xlabel: X-axis label
        ylabel: Y-axis label
        ylim: Tuple (ymin, ymax) for y-axis limits
        output_path: Path to save the figure
        include_overall: Whether to include "Overall" group averaging across all models
    """
    # Configuration: human-readable names
    precision_levels = ["low", "medium", "high"]
    precision_names = {"low": "Low Precision", "medium": "Medium Precision", "high": "High Precision"}

    # Get all unique models
    models = sorted(zero_shot_metrics["model_name"].unique())

    # Prepare data dict: model -> precision_name -> mode_name -> value
    data_dict = {}

    # Add individual models
    for model in models:
        data_dict[model] = {}
        for precision in precision_levels:
            precision_name = precision_names[precision]
            data_dict[model][precision_name] = {}

            # Zero-shot
            zs_value = zero_shot_metrics[
                (zero_shot_metrics["model_name"] == model) & (zero_shot_metrics["precision_level"] == precision)
            ]["metric_value"].values[0]
            data_dict[model][precision_name]["single-round"] = zs_value

            # Iterative
            iter_value = iterative_metrics[
                (iterative_metrics["model_name"] == model) & (iterative_metrics["precision_level"] == precision)
            ]["metric_value"].values[0]
            data_dict[model][precision_name]["multi-round"] = iter_value

    # Add overall average if requested
    if include_overall:
        overall_zs = {}
        overall_iter = {}
        for precision in precision_levels:
            zs_data = zero_shot_metrics[zero_shot_metrics["precision_level"] == precision]
            overall_zs[precision] = zs_data["metric_value"].mean()

            iter_data = iterative_metrics[iterative_metrics["precision_level"] == precision]
            overall_iter[precision] = iter_data["metric_value"].mean()

        data_dict["Overall"] = {}
        for precision in precision_levels:
            precision_name = precision_names[precision]
            data_dict["Overall"][precision_name] = {
                "single-round": overall_zs[precision],
                "multi-round": overall_iter[precision],
            }

    plotter = GroupedBarPlot(figsize=(18, 7))
    plotter.plot(data_dict, xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    plotter.save(output_path)
