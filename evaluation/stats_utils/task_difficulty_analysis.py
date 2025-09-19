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


class TaskDifficultyAnalyzer:
    """Analyzer for task difficulty across SimulCost-Bench datasets"""

    def __init__(self, eval_results_dir: str = "eval_results"):
        """
        Initialize the analyzer

        Args:
            eval_results_dir: Path to evaluation results directory
        """
        self.eval_results_dir = Path(eval_results_dir)
        self.output_dir = Path("eval_results/stats/task_difficulty")

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
        for dataset_dir in self.eval_results_dir.iterdir():
            if dataset_dir.is_dir() and dataset_dir.name not in ['overall', 'stats']:
                self.datasets.append(dataset_dir.name)

        print(f"Found {len(self.datasets)} datasets: {self.datasets}")

        # Collect all unique tasks
        all_tasks = set()

        for dataset in self.datasets:
            for precision in self.precision_levels:
                summary_file = self.eval_results_dir / dataset / f"{dataset}_{precision}_summary.csv"
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

        # Search for all summary files recursively
        summary_files = list(self.eval_results_dir.glob("**/*_summary.csv"))
        print(f"Found {len(summary_files)} summary files")

        for summary_file in summary_files:
            # Skip overall and stats directories
            if 'overall' in str(summary_file) or 'stats' in str(summary_file):
                continue

            try:
                df = pd.read_csv(summary_file)

                # Extract dataset, precision, and method type from file path
                file_parts = summary_file.stem.split('_')
                if len(file_parts) >= 3:
                    precision = file_parts[-1]  # last part should be precision
                    dataset = '_'.join(file_parts[:-1])  # everything before precision
                else:
                    continue

                # Determine method type (zero_shot, iterative, or other)
                method_type = "other"  # default
                if 'zero_shot' in str(summary_file):
                    method_type = "zero_shot"
                elif 'iterative' in str(summary_file):
                    method_type = "iterative"

                df['Dataset'] = dataset
                df['Precision'] = precision
                df['MethodType'] = method_type
                df['FilePath'] = str(summary_file)
                all_data.append(df)
                print(f"  Loaded {len(df)} records from {dataset}_{precision}")
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

    def create_visualizations(self, agg_df: pd.DataFrame, detailed_df: pd.DataFrame) -> None:
        """
        Create bar charts and visualizations for task difficulty analysis

        Args:
            agg_df: Aggregated statistics by category
            detailed_df: Detailed statistics by individual task
        """
        print("Creating visualizations...")

        # Set up the plotting style
        plt.style.use('default')

        # Create separate charts for success rate and efficiency
        self._create_success_rate_chart(detailed_df)
        self._create_efficiency_chart(detailed_df)

        # Create detailed task performance scatter plots
        self._create_success_rate_scatter(detailed_df)
        self._create_efficiency_scatter(detailed_df)

    def _create_success_rate_chart(self, detailed_df: pd.DataFrame) -> None:
        """Create bar chart for success rate by task categories for different method types"""
        # Create charts for each method type
        for method_type in ['zero_shot', 'iterative']:
            self._create_method_success_rate_chart(detailed_df, method_type)

    def _create_method_success_rate_chart(self, detailed_df: pd.DataFrame, method_type: str) -> None:
        """Create success rate chart for a specific method type"""
        # Filter data for this method type and sufficient data
        method_data = detailed_df[detailed_df['MethodType'] == method_type].copy()
        filtered_df = method_data[method_data['success_rate_count'] >= 5].copy()

        if len(filtered_df) == 0:
            print(f"Not enough data for {method_type} success rate chart")
            return

        # Calculate aggregated success rate by category
        category_stats = filtered_df.groupby('TaskCategory')['success_rate_mean'].mean().reset_index()
        category_stats = category_stats.sort_values('TaskCategory')  # Ensure Common comes first

        # Define colors for task categories
        category_colors = {'Common': 'green', 'Uncommon': 'blue'}
        colors = [category_colors[cat] for cat in category_stats['TaskCategory']]

        # Create the bar chart with smaller size
        plt.figure(figsize=(4, 4))
        bars = plt.bar(category_stats['TaskCategory'], category_stats['success_rate_mean'], color=colors, alpha=0.7, width=0.4)

        # Customize the chart
        plt.ylabel('Success Rate', fontsize=12, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, category_stats['success_rate_mean']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Save the plot
        plt.tight_layout()
        output_file = self.output_dir / f"success_rate_by_tasks_{method_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {method_type} success rate chart to {output_file}")
        plt.show()

    def _create_efficiency_chart(self, detailed_df: pd.DataFrame) -> None:
        """Create bar chart for efficiency by task categories for different method types"""
        # Create charts for each method type
        for method_type in ['zero_shot', 'iterative']:
            self._create_method_efficiency_chart(detailed_df, method_type)

    def _create_method_efficiency_chart(self, detailed_df: pd.DataFrame, method_type: str) -> None:
        """Create efficiency chart for a specific method type"""
        # Filter data for this method type and sufficient data
        method_data = detailed_df[detailed_df['MethodType'] == method_type].copy()
        filtered_df = method_data[method_data['mean_efficiency_count'] >= 5].copy()

        if len(filtered_df) == 0:
            print(f"Not enough data for {method_type} efficiency chart")
            return

        # Calculate aggregated efficiency by category
        category_stats = filtered_df.groupby('TaskCategory')['mean_efficiency_mean'].mean().reset_index()
        category_stats = category_stats.sort_values('TaskCategory')  # Ensure Common comes first

        # Define colors for task categories
        category_colors = {'Common': 'green', 'Uncommon': 'blue'}
        colors = [category_colors[cat] for cat in category_stats['TaskCategory']]

        # Create the bar chart with smaller size
        plt.figure(figsize=(4, 4))
        bars = plt.bar(category_stats['TaskCategory'], category_stats['mean_efficiency_mean'], color=colors, alpha=0.7, width=0.4)

        # Customize the chart
        plt.ylabel('Efficiency', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, category_stats['mean_efficiency_mean']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Save the plot
        plt.tight_layout()
        output_file = self.output_dir / f"efficiency_by_tasks_{method_type}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved {method_type} efficiency chart to {output_file}")
        plt.show()

    def _create_success_rate_scatter(self, detailed_df: pd.DataFrame) -> None:
        """Create scatter plot showing success rate distribution by task category"""

        # Filter out tasks with insufficient data and aggregate by task (average across method types)
        filtered_df = detailed_df[detailed_df['success_rate_count'] >= 5].copy()

        if len(filtered_df) == 0:
            print("Not enough data for success rate scatter plot")
            return

        # Aggregate by task name to get one point per task
        task_aggregated = filtered_df.groupby(['Task', 'TaskCategory']).agg({
            'success_rate_mean': 'mean'
        }).reset_index()

        plt.figure(figsize=(9, 6))

        # Create scatter plot with task categories
        colors = {'Common': 'green', 'Uncommon': 'blue'}
        y_positions = {}
        y_counter = 0

        # Assign y positions to avoid overlap with reduced spacing
        for category in ['Common', 'Uncommon']:
            cat_data = task_aggregated[task_aggregated['TaskCategory'] == category]
            if len(cat_data) > 0:
                # Sort by success rate to create better visual separation
                cat_data = cat_data.sort_values('success_rate_mean')

                for i, (_, row) in enumerate(cat_data.iterrows()):
                    # Add some vertical jitter to prevent overlap
                    y_positions[row['Task']] = y_counter * 0.3 + (i % 2) * 0.1
                    y_counter += 1

        # Create the scatter plot
        for category in ['Common', 'Uncommon']:
            cat_data = task_aggregated[task_aggregated['TaskCategory'] == category]
            if len(cat_data) > 0:
                x_values = cat_data['success_rate_mean']
                y_values = [y_positions[task] for task in cat_data['Task']]
                plt.scatter(x_values, y_values, c=colors[category], label=category,
                           alpha=0.7, s=120, edgecolors='black', linewidth=0.5)

        # Add task labels
        for _, row in task_aggregated.iterrows():
            plt.annotate(row['Task'],
                       (row['success_rate_mean'], y_positions[row['Task']]),
                       xytext=(10, 0), textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       ha='left', va='center')

        plt.xlabel('Success Rate', fontsize=12, fontweight='bold')
        plt.ylabel('')  # Remove y-axis label
        plt.title('Task Success Rate Distribution by Category', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, axis='x')
        plt.xlim(0, 1.05)  # Add margin on right side only

        # Remove y-axis ticks and labels
        plt.yticks([])

        # Save the scatter plot
        plt.tight_layout()
        output_file = self.output_dir / "success_rate_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved success rate scatter plot to {output_file}")
        plt.show()

    def _create_efficiency_scatter(self, detailed_df: pd.DataFrame) -> None:
        """Create scatter plot showing efficiency distribution by task category"""

        # Filter out tasks with insufficient data and aggregate by task (average across method types)
        filtered_df = detailed_df[detailed_df['mean_efficiency_count'] >= 5].copy()

        if len(filtered_df) == 0:
            print("Not enough data for efficiency scatter plot")
            return

        # Aggregate by task name to get one point per task
        task_aggregated = filtered_df.groupby(['Task', 'TaskCategory']).agg({
            'mean_efficiency_mean': 'mean'
        }).reset_index()

        plt.figure(figsize=(9, 6))

        # Create scatter plot with task categories
        colors = {'Common': 'green', 'Uncommon': 'blue'}
        y_positions = {}
        y_counter = 0

        # Assign y positions to avoid overlap with reduced spacing
        for category in ['Common', 'Uncommon']:
            cat_data = task_aggregated[task_aggregated['TaskCategory'] == category]
            if len(cat_data) > 0:
                # Sort by efficiency to create better visual separation
                cat_data = cat_data.sort_values('mean_efficiency_mean')

                for i, (_, row) in enumerate(cat_data.iterrows()):
                    # Add some vertical jitter to prevent overlap
                    y_positions[row['Task']] = y_counter * 0.3 + (i % 2) * 0.1
                    y_counter += 1

        # Create the scatter plot
        for category in ['Common', 'Uncommon']:
            cat_data = task_aggregated[task_aggregated['TaskCategory'] == category]
            if len(cat_data) > 0:
                x_values = cat_data['mean_efficiency_mean']
                y_values = [y_positions[task] for task in cat_data['Task']]
                plt.scatter(x_values, y_values, c=colors[category], label=category,
                           alpha=0.7, s=120, edgecolors='black', linewidth=0.5)

        # Add task labels
        for _, row in task_aggregated.iterrows():
            plt.annotate(row['Task'],
                       (row['mean_efficiency_mean'], y_positions[row['Task']]),
                       xytext=(10, 0), textcoords='offset points', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       ha='left', va='center')

        # Get the efficiency range and add margins
        min_eff = task_aggregated['mean_efficiency_mean'].min()
        max_eff = task_aggregated['mean_efficiency_mean'].max()
        margin = (max_eff - min_eff) * 0.1  # 10% margin on each side

        plt.xlabel('Efficiency', fontsize=12, fontweight='bold')
        plt.ylabel('')  # Remove y-axis label
        plt.title('Task Efficiency Distribution by Category', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3, axis='x')
        plt.xlim(min_eff - margin, max_eff + margin)  # Add margins based on data range

        # Remove y-axis ticks and labels
        plt.yticks([])

        # Save the scatter plot
        plt.tight_layout()
        output_file = self.output_dir / "efficiency_scatter.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved efficiency scatter plot to {output_file}")
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
        self.create_visualizations(agg_df, detailed_df)

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

    args = parser.parse_args()

    # Initialize and run the analyzer
    analyzer = TaskDifficultyAnalyzer(eval_results_dir=args.eval_results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()