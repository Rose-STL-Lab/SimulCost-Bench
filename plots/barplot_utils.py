"""
Utilities for creating grouped bar plots.
"""

import matplotlib.pyplot as plt
import numpy as np

# MATLAB-like default colors for subgroups
SUBGROUP_COLORS = [
    "#0072BD",  # Blue
    "#D95319",  # Orange
    "#EDB120",  # Yellow
    "#7E2F8E",  # Purple
    "#77AC30",  # Green
    "#4DBEEE",  # Cyan
    "#A2142F",  # Red
]


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

                # Use hatching pattern to differentiate bars within subgroup
                hatch = None if j == 0 else "//" if j == 1 else "\\\\"
                edgecolor = color if j == 0 else "black"
                linewidth = 0 if j == 0 else 0.5

                # Label: only mention bar_type for first subgroup, otherwise just subgroup name
                if i == 0:
                    label = f"{bar_type.title()}: {subgroup.title()}"
                else:
                    label = subgroup.title()

                bars = self.ax.bar(
                    x_positions,
                    values,
                    bar_width,
                    label=label,
                    color=color,
                    alpha=0.85,
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
        # Create legend with rows organized by bar type
        handles, labels = self.ax.get_legend_handles_labels()
        # Reorganize: first n_subgroups handles are for first bar_type, next n_subgroups for second bar_type, etc.
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
