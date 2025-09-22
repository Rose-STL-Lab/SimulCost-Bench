#!/usr/bin/env python3
"""
Task Difficulty Analysis Script

This script analyzes task difficulty across all datasets and precision levels by:
1. Categorizing tasks into Common, Occasional, and Rare based on expected model performance
2. Aggregating statistics for each task category across all datasets
3. Generating visualizations and summary tables

Categories:
- Common: cfl, dx, n_space, resolution, mesh_x, mesh_y, dt_multiplier
- Occasional: *_threshold, relax*
- Rare: all other tasks (case by case)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

try:
    from adjustText import adjust_text
    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False
    print("Warning: adjustText package not available. Text labels may overlap.")


class TaskDifficultyAnalyzer:
    """Analyzer for task difficulty across SimulCost-Bench datasets"""

    def __init__(self, eval_results_dir: str = "eval_results", datasets: List[str] = None):
        """
        Initialize the analyzer

        Args:
            eval_results_dir: Path to evaluation results directory
            datasets: List of dataset names to analyze. If None, analyze all datasets.
        """
        self.eval_results_dir = Path(eval_results_dir)
        self.output_dir = Path("eval_results/stats/task_difficulty")
        self.target_datasets = datasets

        # Task difficulty categories
        self.task_categories = {
            'Common': ['cfl', 'dx', 'n_space', 'resolution', 'mesh_x', 'mesh_y', 'dt_multiplier', 'nx', 'npart'],
            'Uncommon': []  # All other tasks will be classified as uncommon
        }

        # Metrics to analyze
        self.metrics = ['converged_rate', 'success_rate', 'mean_efficiency', 'mean_rmse']

        # Precision levels
        self.precision_levels = ['low', 'medium', 'high']

        # Dataset names
        self.datasets = []

    def discover_datasets_and_tasks(self) -> None:
        """Discover all available datasets and tasks from evaluation results"""
        print("Discovering datasets and tasks...")

        # Find all dataset directories
        all_available_datasets = []
        for dataset_dir in self.eval_results_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name not in ['overall', 'stats']:
                all_available_datasets.append(dataset_dir.name)

        # Filter datasets based on target_datasets if specified
        if self.target_datasets:
            self.datasets = [d for d in all_available_datasets if d in self.target_datasets]
            print(f"Using specified datasets: {self.datasets}")
            if not self.datasets:
                raise ValueError(f"None of the specified datasets {self.target_datasets} were found!")
        else:
            self.datasets = all_available_datasets
            print(f"Found {len(self.datasets)} datasets: {self.datasets}")

        # Collect all unique tasks from subdirectories only
        all_tasks = set()

        for dataset in self.datasets:
            for method_type in ['zero_shot', 'iterative']:
                for precision in self.precision_levels:
                    summary_file = self.eval_results_dir / dataset / method_type / f"{dataset}_{precision}_summary.csv"
                    if summary_file.exists():
                        df = pd.read_csv(summary_file)
                        if 'Task' in df.columns:
                            all_tasks.update(df['Task'].unique())

        # Categorize tasks: Common tasks stay Common, all others become Uncommon
        for task in all_tasks:
            if task not in self.task_categories['Common']:
                self.task_categories['Uncommon'].append(task)

        print(f"Task categorization:")
        for category, tasks in self.task_categories.items():
            print(f"  {category}: {tasks}")

    def load_all_summary_data(self) -> pd.DataFrame:
        """
        Load and combine all summary data from different datasets and precision levels

        Returns:
            Combined DataFrame with all evaluation results
        """
        print("Loading all summary data...")

        all_data = []

        # Only load summary files from method subdirectories
        for dataset in self.datasets:
            for method_type in ['zero_shot', 'iterative']:
                for precision in self.precision_levels:
                    summary_file = self.eval_results_dir / dataset / method_type / f"{dataset}_{precision}_summary.csv"

                    if not summary_file.exists():
                        continue

                    try:
                        df = pd.read_csv(summary_file)

                        df['Dataset'] = dataset
                        df['Precision'] = precision
                        df['MethodType'] = method_type
                        df['FilePath'] = str(summary_file)
                        all_data.append(df)
                        print(f"  Loaded {len(df)} records from {dataset}_{precision}_{method_type}")
                    except Exception as e:
                        print(f"  Error loading {summary_file}: {e}")

        if not all_data:
            raise ValueError("No summary data found!")

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total records loaded: {len(combined_df)}")

        return combined_df

    def categorize_tasks_in_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add task category column to the DataFrame

        Args:
            df: DataFrame with Task column

        Returns:
            DataFrame with added TaskCategory column
        """
        def get_task_category(task):
            for category, tasks in self.task_categories.items():
                if task in tasks:
                    return category
            return 'Uncommon'  # Default fallback

        df['TaskCategory'] = df['Task'].apply(get_task_category)
        return df

    def aggregate_task_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate statistics by task category across all datasets and precision levels

        Args:
            df: Combined DataFrame with all evaluation results

        Returns:
            Aggregated statistics DataFrame
        """
        print("Aggregating task statistics...")

        # Group by task category and calculate statistics
        agg_stats = []

        for category in ['Common', 'Uncommon']:
            category_data = df[df['TaskCategory'] == category]

            if len(category_data) == 0:
                continue

            stats = {
                'TaskCategory': category,
                'NumTasks': len(category_data['Task'].unique()),
                'TotalSamples': len(category_data),
            }

            # Calculate mean and std for each metric
            for metric in self.metrics:
                if metric in category_data.columns:
                    values = category_data[metric].dropna()
                    stats[f'{metric}_mean'] = values.mean()
                    stats[f'{metric}_std'] = values.std()
                    stats[f'{metric}_median'] = values.median()
                    stats[f'{metric}_count'] = len(values)

            agg_stats.append(stats)

        agg_df = pd.DataFrame(agg_stats)
        return agg_df

    def generate_detailed_task_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate detailed statistics for each individual task by method type

        Args:
            df: Combined DataFrame with all evaluation results

        Returns:
            Detailed task statistics DataFrame
        """
        print("Generating detailed task statistics...")

        detailed_stats = []

        # Generate stats for each task and method type combination
        for task in df['Task'].unique():
            for method_type in df['MethodType'].unique():
                task_method_data = df[(df['Task'] == task) & (df['MethodType'] == method_type)]

                if len(task_method_data) == 0:
                    continue

                task_category = task_method_data['TaskCategory'].iloc[0]

                stats = {
                    'Task': task,
                    'TaskCategory': task_category,
                    'MethodType': method_type,
                    'NumDatasets': len(task_method_data['Dataset'].unique()),
                    'TotalSamples': len(task_method_data),
                }

                # Calculate statistics for each metric
                for metric in self.metrics:
                    if metric in task_method_data.columns:
                        values = task_method_data[metric].dropna()
                        stats[f'{metric}_mean'] = values.mean()
                        stats[f'{metric}_std'] = values.std()
                        stats[f'{metric}_median'] = values.median()
                        stats[f'{metric}_count'] = len(values)

                detailed_stats.append(stats)

        detailed_df = pd.DataFrame(detailed_stats)
        detailed_df = detailed_df.sort_values(['MethodType', 'TaskCategory', f'{self.metrics[0]}_mean'],
                                             ascending=[True, True, False])

        return detailed_df

    def create_visualizations(self, agg_df: pd.DataFrame, detailed_df: pd.DataFrame, combined_df: pd.DataFrame) -> None:
        """
        Create bar charts and visualizations for task difficulty analysis

        Args:
            agg_df: Aggregated statistics by category
            detailed_df: Detailed statistics by individual task
            combined_df: Combined DataFrame with all evaluation results
        """
        print("Creating visualizations...")

        # Set up the plotting style
        plt.style.use('default')

        # Create separate charts for success rate and efficiency
        self._create_success_rate_chart(combined_df)
        self._create_efficiency_chart(combined_df)

        # Create detailed task performance scatter plots for each method type
        for method_type in ['zero_shot', 'iterative']:
            self._create_success_rate_scatter(detailed_df, method_type)
            self._create_efficiency_scatter(detailed_df, method_type)

    def _create_success_rate_chart(self, combined_df: pd.DataFrame) -> None:
        """Create bar chart for success rate by task categories for different method types"""
        # Create charts for each method type
        for method_type in ['zero_shot', 'iterative']:
            self._create_method_success_rate_chart(combined_df, method_type)

    def _create_method_success_rate_chart(self, combined_df: pd.DataFrame, method_type: str) -> None:
        """Create success rate chart for a specific method type by precision levels"""

        # Filter data for this method type and sufficient data
        method_data = combined_df[combined_df['MethodType'] == method_type].copy()

        # Calculate aggregated success rate by precision and category
        precision_category_stats = method_data.groupby(['Precision', 'TaskCategory'])['success_rate'].mean().reset_index()

        if len(precision_category_stats) == 0:
            print(f"Not enough data for {method_type} success rate chart")
            return

        # Define colors for task categories
        category_colors = {'Common': '#2E8B57', 'Uncommon': '#B22222'}

        # Prepare data for grouped bar chart
        precision_order = ['low', 'medium', 'high']
        categories = ['Common', 'Uncommon']

        # Create data structure for plotting
        data_dict = {}
        for precision in precision_order:
            data_dict[precision] = {}
            for category in categories:
                subset = precision_category_stats[
                    (precision_category_stats['Precision'] == precision) &
                    (precision_category_stats['TaskCategory'] == category)
                ]
                if len(subset) > 0:
                    data_dict[precision][category] = subset['success_rate'].iloc[0]
                else:
                    data_dict[precision][category] = 0

        # Create the grouped bar chart
        fig, ax = plt.subplots(figsize=(5, 5))

        x = np.arange(len(precision_order)) * 0.6  # the label locations (closer spacing)
        width = 0.15  # the width of the bars (narrower)

        common_values = [data_dict[p]['Common'] for p in precision_order]
        uncommon_values = [data_dict[p]['Uncommon'] for p in precision_order]

        gap = 0.05  # gap between the two bars
        bars1 = ax.bar(x - width/2 - gap/2, common_values, width, label='Common',
                       color=category_colors['Common'], alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2 + gap/2, uncommon_values, width, label='Uncommon',
                       color=category_colors['Uncommon'], alpha=0.7, hatch='///', edgecolor='black')

        # Customize the chart
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_xlabel('Precision Level', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(precision_order)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        def add_value_labels(bars, values):
            for bar, val in zip(bars, values):
                if val > 0:  # Only add label if there's data
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        add_value_labels(bars1, common_values)
        add_value_labels(bars2, uncommon_values)

        # Save the plot
        plt.tight_layout()
        output_file = self.output_dir / f"success_rate_by_tasks_{method_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {method_type} success rate chart to {output_file}")
        plt.show()

    def _create_efficiency_chart(self, combined_df: pd.DataFrame) -> None:
        """Create bar chart for efficiency by task categories for different method types"""
        # Create charts for each method type
        for method_type in ['zero_shot', 'iterative']:
            self._create_method_efficiency_chart(combined_df, method_type)

    def _create_method_efficiency_chart(self, combined_df: pd.DataFrame, method_type: str) -> None:
        """Create efficiency chart for a specific method type by precision levels"""

        # Filter data for this method type and sufficient data
        method_data = combined_df[combined_df['MethodType'] == method_type].copy()

        # Calculate aggregated efficiency by precision and category
        precision_category_stats = method_data.groupby(['Precision', 'TaskCategory'])['mean_efficiency'].mean().reset_index()

        if len(precision_category_stats) == 0:
            print(f"Not enough data for {method_type} efficiency chart")
            return

        # Define colors for task categories
        category_colors = {'Common': '#2E8B57', 'Uncommon': '#B22222'}

        # Prepare data for grouped bar chart
        precision_order = ['low', 'medium', 'high']
        categories = ['Common', 'Uncommon']

        # Create data structure for plotting
        data_dict = {}
        for precision in precision_order:
            data_dict[precision] = {}
            for category in categories:
                subset = precision_category_stats[
                    (precision_category_stats['Precision'] == precision) &
                    (precision_category_stats['TaskCategory'] == category)
                ]
                if len(subset) > 0:
                    data_dict[precision][category] = subset['mean_efficiency'].iloc[0]
                else:
                    data_dict[precision][category] = 0

        # Create the grouped bar chart
        fig, ax = plt.subplots(figsize=(5, 5))

        x = np.arange(len(precision_order)) * 0.6  # the label locations (closer spacing)
        width = 0.15  # the width of the bars (narrower)

        common_values = [data_dict[p]['Common'] for p in precision_order]
        uncommon_values = [data_dict[p]['Uncommon'] for p in precision_order]

        gap = 0.05  # gap between the two bars
        bars1 = ax.bar(x - width/2 - gap/2, common_values, width, label='Common',
                       color=category_colors['Common'], alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2 + gap/2, uncommon_values, width, label='Uncommon',
                       color=category_colors['Uncommon'], alpha=0.7, hatch='///', edgecolor='black')

        # Customize the chart
        ax.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
        ax.set_xlabel('Precision Level', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(precision_order)

        # Set ylim based on data with extra margin
        all_values = common_values + uncommon_values
        max_val = max([v for v in all_values if v > 0]) if any(v > 0 for v in all_values) else 1
        ax.set_ylim(0, max_val * 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        def add_value_labels(bars, values):
            for bar, val in zip(bars, values):
                if val > 0:  # Only add label if there's data
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max_val * 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        add_value_labels(bars1, common_values)
        add_value_labels(bars2, uncommon_values)

        # Save the plot
        plt.tight_layout()
        output_file = self.output_dir / f"efficiency_by_tasks_{method_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {method_type} efficiency chart to {output_file}")
        plt.show()

    def _create_success_rate_scatter(self, detailed_df: pd.DataFrame, method_type: str) -> None:
        """Create scatter plot showing success rate distribution by task category for a specific method type"""

        # Filter data for this method type and sufficient data
        method_data = detailed_df[detailed_df['MethodType'] == method_type].copy()
        filtered_df = method_data[method_data['success_rate_count'] >= 5].copy()

        if len(filtered_df) == 0:
            print(f"Not enough data for {method_type} success rate scatter plot")
            return

        # Use the filtered data directly (no need to aggregate since it's already per method type)
        task_aggregated = filtered_df[['Task', 'TaskCategory', 'success_rate_mean']].copy()

        plt.figure(figsize=(12, 8))

        # Create scatter plot with task categories - separate regions for each category
        colors = {'Common': '#2E8B57', 'Uncommon': '#B22222'}

        # Separate data by category
        common_data = task_aggregated[task_aggregated['TaskCategory'] == 'Common'].copy()
        uncommon_data = task_aggregated[task_aggregated['TaskCategory'] == 'Uncommon'].copy()

        # Sort each category by success rate for better organization
        common_data = common_data.sort_values('success_rate_mean') if len(common_data) > 0 else common_data
        uncommon_data = uncommon_data.sort_values('success_rate_mean') if len(uncommon_data) > 0 else uncommon_data

        # Assign y positions with clear separation between categories
        y_spacing = 0.8  # Space between individual points
        category_gap = 2.0  # Gap between categories

        texts = []

        # Plot Common tasks (bottom half)
        if len(common_data) > 0:
            y_start_common = 0
            for i, (_, row) in enumerate(common_data.iterrows()):
                y_pos = y_start_common + i * y_spacing
                plt.scatter(row['success_rate_mean'], y_pos, c=colors['Common'],
                           label='Common' if i == 0 else "", alpha=0.7, s=120,
                           edgecolors='black', linewidth=0.5)

                # Add text annotation
                text = plt.annotate(row['Task'], (row['success_rate_mean'], y_pos),
                                   xytext=(8, 0), textcoords='offset points', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                                   ha='left', va='center')
                texts.append(text)

        # Plot Uncommon tasks (top half)
        if len(uncommon_data) > 0:
            y_start_uncommon = (len(common_data) * y_spacing) + category_gap
            for i, (_, row) in enumerate(uncommon_data.iterrows()):
                y_pos = y_start_uncommon + i * y_spacing
                plt.scatter(row['success_rate_mean'], y_pos, c=colors['Uncommon'],
                           label='Uncommon' if i == 0 else "", alpha=0.7, s=120,
                           edgecolors='black', linewidth=0.5)

                # Add text annotation
                text = plt.annotate(row['Task'], (row['success_rate_mean'], y_pos),
                                   xytext=(8, 0), textcoords='offset points', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                                   ha='left', va='center')
                texts.append(text)

        # Add category separators
        if len(common_data) > 0 and len(uncommon_data) > 0:
            separator_y = (len(common_data) * y_spacing) + category_gap/2
            plt.axhline(y=separator_y, color='gray', linestyle='--', alpha=0.5, linewidth=1)


        plt.xlabel('Success Rate', fontsize=12, fontweight='bold')
        plt.ylabel('')  # Remove y-axis label
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, axis='x')
        plt.xlim(-0.05, 1.1)  # Add margins on both sides

        # Remove y-axis ticks and labels
        plt.yticks([])

        # Save the scatter plot
        plt.tight_layout()
        output_file = self.output_dir / f"success_rate_scatter_{method_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {method_type} success rate scatter plot to {output_file}")
        plt.show()

    def _create_efficiency_scatter(self, detailed_df: pd.DataFrame, method_type: str) -> None:
        """Create scatter plot showing efficiency distribution by task category for a specific method type"""

        # Filter data for this method type and sufficient data
        method_data = detailed_df[detailed_df['MethodType'] == method_type].copy()
        filtered_df = method_data[method_data['mean_efficiency_count'] >= 5].copy()

        if len(filtered_df) == 0:
            print(f"Not enough data for {method_type} efficiency scatter plot")
            return

        # Use the filtered data directly (no need to aggregate since it's already per method type)
        task_aggregated = filtered_df[['Task', 'TaskCategory', 'mean_efficiency_mean']].copy()

        plt.figure(figsize=(12, 8))

        # Create scatter plot with task categories - separate regions for each category
        colors = {'Common': '#2E8B57', 'Uncommon': '#B22222'}

        # Separate data by category
        common_data = task_aggregated[task_aggregated['TaskCategory'] == 'Common'].copy()
        uncommon_data = task_aggregated[task_aggregated['TaskCategory'] == 'Uncommon'].copy()

        # Sort each category by efficiency for better organization
        common_data = common_data.sort_values('mean_efficiency_mean') if len(common_data) > 0 else common_data
        uncommon_data = uncommon_data.sort_values('mean_efficiency_mean') if len(uncommon_data) > 0 else uncommon_data

        # Assign y positions with clear separation between categories
        y_spacing = 0.8  # Space between individual points
        category_gap = 2.0  # Gap between categories

        texts = []

        # Plot Common tasks (bottom half)
        if len(common_data) > 0:
            y_start_common = 0
            for i, (_, row) in enumerate(common_data.iterrows()):
                y_pos = y_start_common + i * y_spacing
                plt.scatter(row['mean_efficiency_mean'], y_pos, c=colors['Common'],
                           label='Common' if i == 0 else "", alpha=0.7, s=120,
                           edgecolors='black', linewidth=0.5)

                # Add text annotation
                text = plt.annotate(row['Task'], (row['mean_efficiency_mean'], y_pos),
                                   xytext=(8, 0), textcoords='offset points', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                                   ha='left', va='center')
                texts.append(text)

        # Plot Uncommon tasks (top half)
        if len(uncommon_data) > 0:
            y_start_uncommon = (len(common_data) * y_spacing) + category_gap
            for i, (_, row) in enumerate(uncommon_data.iterrows()):
                y_pos = y_start_uncommon + i * y_spacing
                plt.scatter(row['mean_efficiency_mean'], y_pos, c=colors['Uncommon'],
                           label='Uncommon' if i == 0 else "", alpha=0.7, s=120,
                           edgecolors='black', linewidth=0.5)

                # Add text annotation
                text = plt.annotate(row['Task'], (row['mean_efficiency_mean'], y_pos),
                                   xytext=(8, 0), textcoords='offset points', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                                   ha='left', va='center')
                texts.append(text)

        # Add category separators
        if len(common_data) > 0 and len(uncommon_data) > 0:
            separator_y = (len(common_data) * y_spacing) + category_gap/2
            plt.axhline(y=separator_y, color='gray', linestyle='--', alpha=0.5, linewidth=1)


        # Get the efficiency range and add margins
        min_eff = task_aggregated['mean_efficiency_mean'].min()
        max_eff = task_aggregated['mean_efficiency_mean'].max()
        margin = (max_eff - min_eff) * 0.1  # 10% margin on each side

        plt.xlabel('Efficiency', fontsize=12, fontweight='bold')
        plt.ylabel('')  # Remove y-axis label
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, axis='x')
        # Increase margins to prevent points from touching edges
        margin = max((max_eff - min_eff) * 0.15, 0.5)  # At least 15% margin or 0.5, whichever is larger
        plt.xlim(min_eff - margin, max_eff + margin)

        # Remove y-axis ticks and labels
        plt.yticks([])

        # Save the scatter plot
        plt.tight_layout()
        output_file = self.output_dir / f"efficiency_scatter_{method_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {method_type} efficiency scatter plot to {output_file}")
        plt.show()

    def save_results(self, agg_df: pd.DataFrame, detailed_df: pd.DataFrame,
                    combined_df: pd.DataFrame) -> None:
        """
        Save analysis results to CSV files

        Args:
            agg_df: Aggregated statistics by category
            detailed_df: Detailed statistics by task
            combined_df: Original combined data with categories
        """
        print("Saving results...")

        # Save aggregated statistics
        agg_output = self.output_dir / "task_category_aggregated_stats.csv"
        agg_df.to_csv(agg_output, index=False)
        print(f"Saved aggregated stats to {agg_output}")

        # Save detailed task statistics
        detailed_output = self.output_dir / "detailed_task_stats.csv"
        detailed_df.to_csv(detailed_output, index=False)
        print(f"Saved detailed stats to {detailed_output}")

        # Save combined data with categories
        combined_output = self.output_dir / "combined_data_with_categories.csv"
        combined_df.to_csv(combined_output, index=False)
        print(f"Saved combined data to {combined_output}")

    def print_summary(self, agg_df: pd.DataFrame, detailed_df: pd.DataFrame) -> None:
        """Print summary of the analysis"""
        print("\n" + "="*60)
        print("TASK DIFFICULTY ANALYSIS SUMMARY")
        print("="*60)

        print(f"\nDatasets analyzed: {len(self.datasets)}")
        print(f"Total tasks found: {len(detailed_df)}")

        print("\nTask distribution by category:")
        for category in ['Common', 'Uncommon']:
            count = len([t for t in detailed_df['TaskCategory'] if t == category])
            tasks = [t for t, c in zip(detailed_df['Task'], detailed_df['TaskCategory']) if c == category]
            print(f"  {category}: {count} tasks - {tasks}")

        print("\nPerformance by category (mean ± std):")
        for _, row in agg_df.iterrows():
            category = row['TaskCategory']
            print(f"\n{category}:")
            if 'success_rate_mean' in row:
                print(f"  Success Rate: {row['success_rate_mean']:.3f} ± {row['success_rate_std']:.3f}")
            if 'mean_efficiency_mean' in row:
                print(f"  Efficiency: {row['mean_efficiency_mean']:.3f} ± {row['mean_efficiency_std']:.3f}")
            if 'converged_rate_mean' in row:
                print(f"  Convergence Rate: {row['converged_rate_mean']:.3f} ± {row['converged_rate_std']:.3f}")
            if 'mean_rmse_mean' in row:
                print(f"  RMSE: {row['mean_rmse_mean']:.4f} ± {row['mean_rmse_std']:.4f}")

        print("\nBest performing tasks (by success rate):")
        top_tasks = detailed_df.nlargest(5, 'success_rate_mean')
        for _, task in top_tasks.iterrows():
            print(f"  {task['Task']} ({task['TaskCategory']}): {task['success_rate_mean']:.3f}")

        print("\nWorst performing tasks (by success rate):")
        bottom_tasks = detailed_df.nsmallest(5, 'success_rate_mean')
        for _, task in bottom_tasks.iterrows():
            print(f"  {task['Task']} ({task['TaskCategory']}): {task['success_rate_mean']:.3f}")

    def run_analysis(self) -> None:
        """Run the complete task difficulty analysis"""
        print("Starting Task Difficulty Analysis...")
        print("="*50)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Discover datasets and tasks
        self.discover_datasets_and_tasks()

        # Step 2: Load all data
        combined_df = self.load_all_summary_data()

        # Step 3: Categorize tasks in the data
        combined_df = self.categorize_tasks_in_data(combined_df)

        # Step 4: Aggregate statistics by category
        agg_df = self.aggregate_task_statistics(combined_df)

        # Step 5: Generate detailed task statistics
        detailed_df = self.generate_detailed_task_stats(combined_df)

        # Step 6: Create visualizations
        self.create_visualizations(agg_df, detailed_df, combined_df)

        # Step 7: Save results
        self.save_results(agg_df, detailed_df, combined_df)

        # Step 8: Print summary
        self.print_summary(agg_df, detailed_df)

        print("\nAnalysis completed successfully!")


def main():
    """Main function to run the task difficulty analysis"""
    parser = argparse.ArgumentParser(description='Analyze task difficulty across SimulCost-Bench datasets')
    parser.add_argument('--eval-results-dir', type=str, default='eval_results',
                       help='Path to evaluation results directory')
    parser.add_argument('-d', '--datasets', type=str, nargs='+', default=None,
                       help='List of dataset names to analyze (e.g., burgers_1d euler_1d). If not specified, analyze all datasets.')

    args = parser.parse_args()

    # Initialize and run the analyzer
    analyzer = TaskDifficultyAnalyzer(eval_results_dir=args.eval_results_dir, datasets=args.datasets)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()