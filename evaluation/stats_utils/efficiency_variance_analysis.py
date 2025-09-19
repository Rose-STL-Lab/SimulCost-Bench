#!/usr/bin/env python3
"""
Efficiency Variance Analysis for SimulCost-Bench Evaluation Results

This script analyzes efficiency variance across different precision levels by extracting
individual QID-level efficiency values from log files and computing statistical metrics
including variance, standard deviation, and coefficient of variation.

Usage:
    python evaluation/stats_utils/efficiency_variance_analysis.py -d burgers_1d -t beta
    python evaluation/stats_utils/efficiency_variance_analysis.py -d euler_1d -t k --models qwen3_32b,gpt-5
"""

import argparse
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class EfficiencyVarianceAnalyzer:
    """Analyze efficiency variance across precision levels from individual log files."""

    def __init__(self, base_dir: str = "eval_results", output_dir: str = "eval_results/stats/efficiency_variance"):
        """
        Initialize the efficiency variance analyzer.

        Args:
            base_dir: Base directory containing evaluation results
            output_dir: Output directory for generated visualizations and reports
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure matplotlib for professional plots
        plt.style.use('default')
        sns.set_palette("husl")

    def extract_efficiency_from_log(self, log_path: Path) -> List[float]:
        """
        Extract all efficiency values from a single log file.

        Args:
            log_path: Path to the log file

        Returns:
            List of efficiency values for each QID
        """
        efficiency_values = []
        efficiency_pattern = r'⚡ Efficiency: ([\d.-]+)'

        try:
            with open(log_path, 'r', encoding='utf-8') as file:
                for line in file:
                    match = re.search(efficiency_pattern, line)
                    if match:
                        efficiency_values.append(float(match.group(1)))
        except Exception as e:
            print(f"Warning: Failed to read {log_path}: {e}")

        return efficiency_values

    def collect_efficiency_data(self, dataset: str, task_type: str,
                              models: Optional[List[str]] = None) -> Dict[str, Dict[str, List[float]]]:
        """
        Collect efficiency data from all relevant log files (iterative and zero-shot).

        Args:
            dataset: Dataset name (e.g., 'burgers_1d', 'euler_1d')
            task_type: Task type (e.g., 'beta', 'k', 'cfl')
            models: Optional list of specific models to analyze

        Returns:
            Nested dictionary: {precision_level: {model_mode: [efficiency_values]}}
        """
        data = defaultdict(lambda: defaultdict(list))
        precision_levels = ['low', 'medium', 'high']

        dataset_dir = self.base_dir / dataset / task_type

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        for precision in precision_levels:
            precision_dir = dataset_dir / precision
            if not precision_dir.exists():
                print(f"Warning: Precision directory not found: {precision_dir}")
                continue

            # Find all log files in this precision directory
            iterative_logs = list(precision_dir.glob("iterative_*.log"))
            zero_shot_logs = list(precision_dir.glob("zero_shot_*.log"))

            # Process iterative logs
            for log_file in iterative_logs:
                model_name = log_file.name.replace("iterative_", "").replace(".log", "")
                if models and not any(model in model_name for model in models):
                    continue

                efficiency_values = self.extract_efficiency_from_log(log_file)
                if efficiency_values:
                    key = f"{model_name}_iterative"
                    data[precision][key] = efficiency_values
                    print(f"✅ Loaded {len(efficiency_values)} efficiency values from {log_file.name}")

            # Process zero-shot logs
            for log_file in zero_shot_logs:
                model_name = log_file.name.replace("zero_shot_", "").replace(".log", "")
                if models and not any(model in model_name for model in models):
                    continue

                efficiency_values = self.extract_efficiency_from_log(log_file)
                if efficiency_values:
                    key = f"{model_name}_zero_shot"
                    data[precision][key] = efficiency_values
                    print(f"✅ Loaded {len(efficiency_values)} efficiency values from {log_file.name}")

        return dict(data)

    def calculate_variance_statistics(self, data: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
        """
        Calculate variance statistics for each precision level and model.

        Args:
            data: Efficiency data by precision and model

        Returns:
            DataFrame with variance statistics
        """
        stats_data = []

        for precision, models in data.items():
            # Calculate within-precision statistics across all models
            all_values = []
            for model_values in models.values():
                all_values.extend(model_values)

            if all_values:
                overall_stats = {
                    'precision': precision,
                    'model': 'ALL_MODELS',
                    'count': len(all_values),
                    'mean': np.mean(all_values),
                    'std': np.std(all_values, ddof=1) if len(all_values) > 1 else 0,
                    'variance': np.var(all_values, ddof=1) if len(all_values) > 1 else 0,
                    'cv': np.std(all_values, ddof=1) / np.mean(all_values) if np.mean(all_values) != 0 else np.inf,
                    'min': np.min(all_values),
                    'max': np.max(all_values),
                    'median': np.median(all_values),
                    'q25': np.percentile(all_values, 25),
                    'q75': np.percentile(all_values, 75)
                }
                stats_data.append(overall_stats)

            # Calculate individual model statistics
            for model, values in models.items():
                if values:
                    model_stats = {
                        'precision': precision,
                        'model': model,
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                        'variance': np.var(values, ddof=1) if len(values) > 1 else 0,
                        'cv': np.std(values, ddof=1) / np.mean(values) if np.mean(values) != 0 else np.inf,
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75)
                    }
                    stats_data.append(model_stats)

        return pd.DataFrame(stats_data)

    def create_box_plots(self, data: Dict[str, Dict[str, List[float]]],
                        dataset: str, task_type: str) -> None:
        """
        Create box plots showing efficiency distribution by precision level.

        Args:
            data: Efficiency data by precision and model
            dataset: Dataset name for title
            task_type: Task type for title
        """
        # Prepare data for plotting
        plot_data = []
        for precision, models in data.items():
            for model, values in models.items():
                for value in values:
                    plot_data.append({
                        'precision': precision.capitalize(),
                        'model': model,
                        'efficiency': value
                    })

        df = pd.DataFrame(plot_data)

        if df.empty:
            print("No data available for box plots")
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Overall box plot by precision
        precision_order = ['Low', 'Medium', 'High']
        sns.boxplot(data=df, x='precision', y='efficiency', order=precision_order, ax=ax1)
        ax1.set_title(f'Efficiency Distribution by Precision Level\n{dataset.replace("_", " ").title()} - {task_type.upper()}',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Precision Level', fontsize=12)
        ax1.set_ylabel('Efficiency', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Box plot by model and precision
        sns.boxplot(data=df, x='precision', y='efficiency', hue='model', order=precision_order, ax=ax2)
        ax2.set_title(f'Efficiency Distribution by Model and Precision\n{dataset.replace("_", " ").title()} - {task_type.upper()}',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Precision Level', fontsize=12)
        ax2.set_ylabel('Efficiency', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f'{dataset}_{task_type}_efficiency_boxplots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Box plots saved to: {output_path}")
        plt.close()

    def create_variance_comparison(self, stats_df: pd.DataFrame,
                                 dataset: str, task_type: str) -> None:
        """
        Create variance comparison plots across precision levels.

        Args:
            stats_df: DataFrame with variance statistics
            dataset: Dataset name for title
            task_type: Task type for title
        """
        # Filter for overall statistics
        overall_stats = stats_df[stats_df['model'] == 'ALL_MODELS'].copy()

        if overall_stats.empty:
            print("No overall statistics available for variance comparison")
            return

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Precision order
        precision_order = ['low', 'medium', 'high']
        overall_stats = overall_stats.set_index('precision').reindex(precision_order).reset_index()

        # 1. Variance comparison
        sns.barplot(data=overall_stats, x='precision', y='variance', ax=axes[0,0])
        axes[0,0].set_title('Variance by Precision Level', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Precision Level')
        axes[0,0].set_ylabel('Variance')

        # 2. Standard deviation comparison
        sns.barplot(data=overall_stats, x='precision', y='std', ax=axes[0,1])
        axes[0,1].set_title('Standard Deviation by Precision Level', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Precision Level')
        axes[0,1].set_ylabel('Standard Deviation')

        # 3. Coefficient of variation
        # Handle infinite CV values
        cv_data = overall_stats.copy()
        cv_data['cv'] = cv_data['cv'].replace([np.inf, -np.inf], np.nan)
        sns.barplot(data=cv_data, x='precision', y='cv', ax=axes[1,0])
        axes[1,0].set_title('Coefficient of Variation by Precision Level', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Precision Level')
        axes[1,0].set_ylabel('CV (std/mean)')

        # 4. Mean efficiency with error bars
        axes[1,1].bar(overall_stats['precision'], overall_stats['mean'],
                     yerr=overall_stats['std'], capsize=5)
        axes[1,1].set_title('Mean Efficiency ± Standard Deviation', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Precision Level')
        axes[1,1].set_ylabel('Efficiency')

        # Apply styling to all subplots
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle(f'Efficiency Variance Analysis - {dataset.replace("_", " ").title()} ({task_type.upper()})',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f'{dataset}_{task_type}_variance_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Variance comparison saved to: {output_path}")
        plt.close()

    def create_distribution_plots(self, data: Dict[str, Dict[str, List[float]]],
                                dataset: str, task_type: str) -> None:
        """
        Create distribution density plots for efficiency across precision levels.

        Args:
            data: Efficiency data by precision and model
            dataset: Dataset name for title
            task_type: Task type for title
        """
        # Prepare data
        plot_data = []
        for precision, models in data.items():
            all_values = []
            for values in models.values():
                all_values.extend(values)
            if all_values:
                for value in all_values:
                    plot_data.append({
                        'precision': precision.capitalize(),
                        'efficiency': value
                    })

        df = pd.DataFrame(plot_data)

        if df.empty:
            print("No data available for distribution plots")
            return

        # Create distribution plots
        plt.figure(figsize=(12, 8))

        precision_order = ['Low', 'Medium', 'High']
        colors = ['skyblue', 'lightgreen', 'salmon']

        for i, precision in enumerate(precision_order):
            data_subset = df[df['precision'] == precision]['efficiency']
            if not data_subset.empty:
                plt.hist(data_subset, bins=30, alpha=0.7, label=f'{precision} Precision',
                        color=colors[i], density=True)

        plt.xlabel('Efficiency', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Efficiency Distribution by Precision Level\n{dataset.replace("_", " ").title()} - {task_type.upper()}',
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        output_path = self.output_dir / f'{dataset}_{task_type}_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Distribution plot saved to: {output_path}")
        plt.close()

    def save_statistics_report(self, stats_df: pd.DataFrame,
                              dataset: str, task_type: str) -> None:
        """
        Save detailed statistics report to CSV and generate summary.

        Args:
            stats_df: DataFrame with variance statistics
            dataset: Dataset name
            task_type: Task type
        """
        # Save detailed statistics
        csv_path = self.output_dir / f'{dataset}_{task_type}_efficiency_stats.csv'
        stats_df.to_csv(csv_path, index=False)
        print(f"📄 Detailed statistics saved to: {csv_path}")

        # Generate summary report
        summary_path = self.output_dir / f'{dataset}_{task_type}_summary.txt'

        with open(summary_path, 'w') as f:
            f.write(f"Efficiency Variance Analysis Summary\n")
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Task Type: {task_type}\n")
            f.write(f"=" * 50 + "\n\n")

            # Overall statistics by precision
            overall_stats = stats_df[stats_df['model'] == 'ALL_MODELS']

            if not overall_stats.empty:
                f.write("Overall Statistics by Precision Level:\n")
                f.write("-" * 40 + "\n")

                for _, row in overall_stats.iterrows():
                    f.write(f"\n{row['precision'].upper()} PRECISION:\n")
                    f.write(f"  Sample Size: {row['count']}\n")
                    f.write(f"  Mean: {row['mean']:.4f}\n")
                    f.write(f"  Std Dev: {row['std']:.4f}\n")
                    f.write(f"  Variance: {row['variance']:.4f}\n")
                    if np.isfinite(row['cv']):
                        f.write(f"  CV: {row['cv']:.4f}\n")
                    else:
                        f.write(f"  CV: Undefined (mean = 0)\n")
                    f.write(f"  Range: [{row['min']:.4f}, {row['max']:.4f}]\n")
                    f.write(f"  Median: {row['median']:.4f}\n")
                    f.write(f"  IQR: [{row['q25']:.4f}, {row['q75']:.4f}]\n")

                # Variance comparison
                f.write(f"\nVariance Comparison:\n")
                f.write("-" * 20 + "\n")
                variances = overall_stats.set_index('precision')['variance']
                if len(variances) > 1:
                    max_var_precision = variances.idxmax()
                    min_var_precision = variances.idxmin()
                    f.write(f"Highest Variance: {max_var_precision} ({variances[max_var_precision]:.4f})\n")
                    f.write(f"Lowest Variance: {min_var_precision} ({variances[min_var_precision]:.4f})\n")
                    f.write(f"Variance Ratio (max/min): {variances[max_var_precision]/variances[min_var_precision]:.2f}\n")

        print(f"📄 Summary report saved to: {summary_path}")

    def discover_available_datasets(self) -> List[Tuple[str, str]]:
        """
        Discover all available dataset/task_type combinations.

        Returns:
            List of (dataset, task_type) tuples
        """
        combinations = []

        if not self.base_dir.exists():
            return combinations

        for dataset_dir in self.base_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name

                for task_dir in dataset_dir.iterdir():
                    if task_dir.is_dir():
                        task_name = task_dir.name

                        # Check if this task has precision subdirectories with log files
                        has_data = False
                        log_count = 0
                        for precision in ['low', 'medium', 'high']:
                            precision_dir = task_dir / precision
                            if precision_dir.exists():
                                all_logs = list(precision_dir.glob("*.log"))
                                if all_logs:
                                    has_data = True
                                    log_count += len(all_logs)

                        if has_data and log_count >= 3:  # Require at least some logs
                            combinations.append((dataset_name, task_name))

        return combinations

    def analyze_efficiency_variance(self, dataset: str, task_type: str,
                                  models: Optional[List[str]] = None) -> None:
        """
        Main analysis function that orchestrates the complete efficiency variance analysis.

        Args:
            dataset: Dataset name (e.g., 'burgers_1d', 'euler_1d')
            task_type: Task type (e.g., 'beta', 'k', 'cfl')
            models: Optional list of specific models to analyze
        """
        print(f"🔍 Starting efficiency variance analysis for {dataset}/{task_type}")

        # Collect efficiency data
        data = self.collect_efficiency_data(dataset=dataset, task_type=task_type, models=models)

        if not data:
            print(f"❌ No efficiency data found for {dataset}/{task_type}")
            return

        # Calculate statistics
        stats_df = self.calculate_variance_statistics(data)

        # Generate visualizations
        self.create_box_plots(data, dataset, task_type)
        self.create_variance_comparison(stats_df, dataset, task_type)
        self.create_distribution_plots(data, dataset, task_type)

        # Save reports
        self.save_statistics_report(stats_df, dataset, task_type)

        print(f"✅ Analysis complete! Results saved to: {self.output_dir}")

    def analyze_all_datasets(self, models: Optional[List[str]] = None) -> None:
        """
        Analyze overall efficiency variance across all datasets and task types.

        Args:
            models: Optional list of specific models to analyze
        """
        print("🔍 Discovering available datasets and task types...")

        combinations = self.discover_available_datasets()

        if not combinations:
            print("❌ No datasets with efficiency data found!")
            return

        print(f"📊 Found {len(combinations)} dataset/task combinations:")
        for dataset, task_type in combinations:
            print(f"  - {dataset}/{task_type}")
        print()

        # Collect all efficiency data across datasets/tasks
        print("🔍 Collecting efficiency data from all datasets...")
        overall_data = defaultdict(list)

        for dataset, task_type in combinations:
            try:
                data = self.collect_efficiency_data(dataset=dataset, task_type=task_type, models=models)

                # Aggregate by precision level, distinguishing modes
                for precision, model_dict in data.items():
                    for model_key, model_values in model_dict.items():
                        if '_iterative' in model_key:
                            mode = 'iterative'
                        elif '_zero_shot' in model_key:
                            mode = 'zero_shot'
                        else:
                            mode = 'unknown'

                        # Store values with mode info
                        for value in model_values:
                            overall_data[precision].append({
                                'efficiency': value,
                                'mode': mode
                            })

                print(f"✅ Processed {dataset}/{task_type}")
            except Exception as e:
                print(f"❌ Failed to process {dataset}/{task_type}: {e}")

        if not overall_data:
            print("❌ No efficiency data found!")
            return

        # Generate overall analysis
        self.generate_overall_efficiency_analysis(overall_data, combinations)

    def generate_overall_efficiency_analysis(self, data: Dict[str, List[Dict]],
                                           combinations: List[Tuple[str, str]]) -> None:
        """
        Generate comprehensive analysis of overall efficiency variance across all datasets.

        Args:
            data: Dictionary with precision levels as keys and all efficiency values as lists
            combinations: List of (dataset, task_type) tuples for reference
        """
        print("\n📊 Generating overall efficiency variance analysis...")

        # Calculate statistics for each precision level
        stats = {}
        for precision, data_list in data.items():
            if data_list:
                # Extract efficiency values
                efficiency_values = [item['efficiency'] for item in data_list]

                stats[precision] = {
                    'count': len(efficiency_values),
                    'mean': np.mean(efficiency_values),
                    'std': np.std(efficiency_values, ddof=1) if len(efficiency_values) > 1 else 0,
                    'variance': np.var(efficiency_values, ddof=1) if len(efficiency_values) > 1 else 0,
                    'cv': np.std(efficiency_values, ddof=1) / np.mean(efficiency_values) if np.mean(efficiency_values) != 0 else np.inf,
                    'min': np.min(efficiency_values),
                    'max': np.max(efficiency_values),
                    'median': np.median(efficiency_values),
                    'q25': np.percentile(efficiency_values, 25),
                    'q75': np.percentile(efficiency_values, 75)
                }

        # Create visualizations
        self.create_overall_visualizations(data)

        # Save comprehensive report
        self.save_overall_report(stats, combinations, data)

        print(f"✅ Overall analysis complete! Results saved to: {self.output_dir}")

    def create_overall_visualizations(self, data: Dict[str, List[Dict]]) -> None:
        """
        Create simplified visualizations: box plot and scatter plot showing iterative vs zero-shot modes.

        Args:
            data: Dictionary with precision levels as keys and data_list with mode info as values
        """
        # Prepare data for plotting
        plot_data = []
        for precision, data_list in data.items():
            for i, item in enumerate(data_list):
                mode_name = item['mode'].replace('_', ' ').title()
                if mode_name == 'Zero Shot':
                    mode_name = 'Zero-shot'
                plot_data.append({
                    'precision': precision.capitalize(),
                    'efficiency': item['efficiency'],
                    'mode': mode_name,
                    'sample_index': i
                })

        df = pd.DataFrame(plot_data)
        precision_order = ['Low', 'Medium', 'High']

        # Reorder mode data to put Zero-shot first, then Iterative
        mode_order = ['Zero-shot', 'Iterative']

        # 1. Create and save box plot
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        sns.boxplot(data=df, x='precision', y='efficiency', hue='mode',
                   order=precision_order, hue_order=mode_order, ax=ax1)
        ax1.set_xlabel('Accuracy Level', fontsize=12)
        ax1.set_ylabel('Efficiency', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Move legend to upper right
        legend = ax1.legend(title='Mode', loc='upper right')
        legend.set_title('')

        plt.tight_layout()
        boxplot_path = self.output_dir / 'efficiency_boxplot.png'
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Box plot saved to: {boxplot_path}")
        plt.close()

        # 2. Create and save scatter plot
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
        mode_colors = {'Iterative': 'blue', 'Zero-shot': 'hotpink', 'Unknown': 'gray'}
        mode_markers = {'Iterative': 'o', 'Zero-shot': 's', 'Unknown': '^'}

        # Configurable spacing parameters
        precision_spacing = 0.3  # Controls distance between precision levels
        mode_spacing_ratio = 0.15  # Mode separation relative to precision spacing
        jitter_ratio = 0.04  # Random noise relative to precision spacing

        # Process modes in the desired order for legend
        for precision in precision_order:
            for mode in mode_order:
                subset = df[(df['precision'] == precision) & (df['mode'] == mode)]
                if not subset.empty:
                    # Add jitter to x-axis for better visibility with proportional spacing
                    x_pos = precision_order.index(precision) * precision_spacing
                    offset = (precision_spacing * mode_spacing_ratio) if mode == 'Iterative' else -(precision_spacing * mode_spacing_ratio) if mode == 'Zero-shot' else 0
                    x_jitter = x_pos + offset + np.random.normal(0, precision_spacing * jitter_ratio, len(subset))

                    ax2.scatter(x_jitter, subset['efficiency'],
                               alpha=0.7, s=25,
                               color=mode_colors.get(mode, 'gray'),
                               marker=mode_markers.get(mode, 'o'),
                               label=f'{mode}' if precision == 'Low' else "",
                               edgecolors='black', linewidth=0.5)

        ax2.set_xticks([i * precision_spacing for i in range(len(precision_order))])
        ax2.set_xticklabels(precision_order)
        ax2.set_xlabel('Accuracy Level', fontsize=12)
        ax2.set_ylabel('Efficiency', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Move legend to upper right and remove title
        legend = ax2.legend(loc='upper right')
        legend.set_title('')

        plt.tight_layout()
        scatterplot_path = self.output_dir / 'efficiency_scatterplot.png'
        plt.savefig(scatterplot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Scatter plot saved to: {scatterplot_path}")
        plt.close()

    def save_overall_report(self, stats: Dict[str, Dict], combinations: List[Tuple[str, str]], data: Dict[str, List[Dict]] = None) -> None:
        """
        Save comprehensive overall report with mode analysis.

        Args:
            stats: Statistics dictionary by precision level
            combinations: List of analyzed dataset/task combinations
            data: Raw data for mode-specific analysis
        """
        report_path = self.output_dir / 'overall_efficiency_variance_report.txt'

        with open(report_path, 'w') as f:
            f.write("Overall Efficiency Variance Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Analysis Overview:\n")
            f.write(f"- Datasets analyzed: {len(set(combo[0] for combo in combinations))}\n")
            f.write(f"- Task types analyzed: {len(set(combo[1] for combo in combinations))}\n")
            f.write(f"- Total dataset/task combinations: {len(combinations)}\n")

            # Add mode information if available
            if data:
                total_iterative = sum(len([item for item in data_list if item['mode'] == 'iterative'])
                                     for data_list in data.values())
                total_zero_shot = sum(len([item for item in data_list if item['mode'] == 'zero_shot'])
                                     for data_list in data.values())
                f.write(f"- Total iterative samples: {total_iterative:,}\n")
                f.write(f"- Total zero-shot samples: {total_zero_shot:,}\n")

            f.write("\n")

            f.write("Dataset/Task Combinations Analyzed:\n")
            for dataset, task in combinations:
                f.write(f"  - {dataset}/{task}\n")
            f.write("\n")

            f.write("Overall Efficiency Statistics by Precision Level:\n")
            f.write("-" * 50 + "\n\n")

            precision_order = ['low', 'medium', 'high']
            for precision in precision_order:
                if precision in stats:
                    stat = stats[precision]
                    f.write(f"{precision.upper()} PRECISION:\n")
                    f.write(f"  Total Samples: {stat['count']:,}\n")
                    f.write(f"  Mean Efficiency: {stat['mean']:.4f}\n")
                    f.write(f"  Standard Deviation: {stat['std']:.4f}\n")
                    f.write(f"  Variance: {stat['variance']:.4f}\n")
                    if np.isfinite(stat['cv']):
                        f.write(f"  Coefficient of Variation: {stat['cv']:.4f}\n")
                    else:
                        f.write(f"  Coefficient of Variation: Undefined (mean = 0)\n")
                    f.write(f"  Range: [{stat['min']:.4f}, {stat['max']:.4f}]\n")
                    f.write(f"  Median: {stat['median']:.4f}\n")
                    f.write(f"  Interquartile Range: [{stat['q25']:.4f}, {stat['q75']:.4f}]\n\n")

            # Mode-specific analysis
            if data:
                f.write("Mode-Specific Analysis:\n")
                f.write("-" * 25 + "\n\n")

                for precision in precision_order:
                    if precision in data:
                        data_list = data[precision]
                        iterative_values = [item['efficiency'] for item in data_list if item['mode'] == 'iterative']
                        zero_shot_values = [item['efficiency'] for item in data_list if item['mode'] == 'zero_shot']

                        f.write(f"{precision.upper()} PRECISION BY MODE:\n")

                        if iterative_values:
                            f.write(f"  Iterative Mode:\n")
                            f.write(f"    Samples: {len(iterative_values):,}\n")
                            f.write(f"    Mean: {np.mean(iterative_values):.4f}\n")
                            f.write(f"    Variance: {np.var(iterative_values, ddof=1) if len(iterative_values) > 1 else 0:.4f}\n")

                        if zero_shot_values:
                            f.write(f"  Zero-Shot Mode:\n")
                            f.write(f"    Samples: {len(zero_shot_values):,}\n")
                            f.write(f"    Mean: {np.mean(zero_shot_values):.4f}\n")
                            f.write(f"    Variance: {np.var(zero_shot_values, ddof=1) if len(zero_shot_values) > 1 else 0:.4f}\n")

                        f.write("\n")

            # Comparative analysis
            f.write("Comparative Analysis:\n")
            f.write("-" * 20 + "\n")

            if len(stats) > 1:
                variances = {prec: stat['variance'] for prec, stat in stats.items()}
                max_var_prec = max(variances, key=variances.get)
                min_var_prec = min(variances, key=variances.get)

                f.write(f"Most Variable: {max_var_prec.upper()} precision (variance: {variances[max_var_prec]:.4f})\n")
                f.write(f"Most Stable: {min_var_prec.upper()} precision (variance: {variances[min_var_prec]:.4f})\n")
                f.write(f"Variance Ratio (max/min): {variances[max_var_prec]/variances[min_var_prec]:.2f}x\n\n")

                means = {prec: stat['mean'] for prec, stat in stats.items()}
                f.write("Mean Efficiency Ranking:\n")
                sorted_means = sorted(means.items(), key=lambda x: x[1], reverse=True)
                for i, (prec, mean_val) in enumerate(sorted_means, 1):
                    f.write(f"  {i}. {prec.upper()}: {mean_val:.4f}\n")

        print(f"📄 Overall report saved to: {report_path}")



def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Analyze efficiency variance across precision levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze all available datasets and task types
    python evaluation/stats_utils/efficiency_variance_analysis.py

    # Analyze specific dataset/task combination
    python evaluation/stats_utils/efficiency_variance_analysis.py -d burgers_1d -t beta

    # Analyze specific models only
    python evaluation/stats_utils/efficiency_variance_analysis.py --models qwen3_32b,gpt-5
        """
    )

    parser.add_argument(
        '-d', '--dataset',
        help='Dataset name (e.g., burgers_1d, euler_1d). If not provided, analyze all datasets.'
    )

    parser.add_argument(
        '-t', '--task-type',
        help='Task type (e.g., beta, k, cfl). If not provided, analyze all task types.'
    )

    parser.add_argument(
        '--models',
        help='Comma-separated list of specific models to analyze (optional)'
    )

    parser.add_argument(
        '--output-dir',
        default='eval_results/stats/efficiency_variance',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Parse models list
    models = None
    if args.models:
        models = [model.strip() for model in args.models.split(',')]

    # Initialize analyzer
    analyzer = EfficiencyVarianceAnalyzer(output_dir=args.output_dir)

    # Run analysis
    try:
        if args.dataset and args.task_type:
            # Analyze specific dataset/task combination
            analyzer.analyze_efficiency_variance(
                dataset=args.dataset,
                task_type=args.task_type,
                models=models
            )
        else:
            # Analyze all available datasets
            analyzer.analyze_all_datasets(models=models)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()