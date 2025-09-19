#!/usr/bin/env python3
"""
First Cost Distribution Analysis for SimulCost-Bench Evaluation Results

This script analyzes the first cost attempt distribution between zero-shot and iterative approaches.
It generates distribution plots to understand how aggressive the first attempts are in different scenarios.

Usage:
    python evaluation/stats_utils/first_cost_distribution.py -d heat_1d
    python evaluation/stats_utils/first_cost_distribution.py  # Process all available datasets
"""

import argparse
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import glob


class FirstCostAnalyzer:
    """Analyze first cost distributions from model attempt results."""

    def __init__(self, base_dir: str = "results_model_attempt", output_dir: str = "eval_results/stats/first_cost_distribution"):
        """
        Initialize the first cost analyzer.

        Args:
            base_dir: Base directory containing model attempt results
            output_dir: Directory to save generated plots
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up professional plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def find_available_datasets(self) -> List[str]:
        """
        Find all available datasets with JSON result files.

        Returns:
            List of dataset names
        """
        datasets = []
        if not self.base_dir.exists():
            return datasets

        for item in self.base_dir.iterdir():
            if item.is_dir() and any(item.glob("**/*.json")):
                datasets.append(item.name)

        return sorted(datasets)

    def load_first_costs(self, dataset_name: str) -> Tuple[List[float], List[float]]:
        """
        Load first cost data for zero-shot and iterative approaches.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Tuple of (zero_shot_first_costs, iterative_first_costs)
        """
        dataset_path = self.base_dir / dataset_name

        if not dataset_path.exists():
            print(f"Warning: Dataset directory not found: {dataset_path}")
            return [], []

        zero_shot_costs = []
        iterative_costs = []

        # Walk through all subdirectories (difficulty/parameter combinations)
        for difficulty_dir in dataset_path.iterdir():
            if not difficulty_dir.is_dir():
                continue

            difficulty = difficulty_dir.name

            for param_dir in difficulty_dir.iterdir():
                if not param_dir.is_dir():
                    continue

                param_type = param_dir.name
                print(f"  Processing {difficulty}/{param_type}")

                # Find JSON files in this parameter directory
                json_files = list(param_dir.glob("*.json"))

                # Group files by model name
                zero_shot_files = {}
                iterative_files = {}

                for json_file in json_files:
                    filename = json_file.name
                    if filename.startswith('zero_shot_'):
                        model_name = filename.replace('zero_shot_', '').replace('.json', '')
                        zero_shot_files[model_name] = json_file
                    elif filename.startswith('iterative_'):
                        model_name = filename.replace('iterative_', '').replace('.json', '')
                        iterative_files[model_name] = json_file

                # Find models that have both zero-shot and iterative files
                common_models = set(zero_shot_files.keys()) & set(iterative_files.keys())

                if not common_models:
                    print(f"    No models with both data types in {difficulty}/{param_type}")
                    continue

                print(f"    Found {len(common_models)} models with both data types: {sorted(common_models)}")

                # Process paired files
                for model_name in sorted(common_models):
                    zs_file = zero_shot_files[model_name]
                    it_file = iterative_files[model_name]

                    try:
                        # Load zero-shot data
                        with open(zs_file, 'r') as f:
                            zs_data = json.load(f)

                        zs_costs = []
                        for experiment in zs_data:
                            if 'cost_sequence' in experiment and experiment['cost_sequence']:
                                zs_costs.append(experiment['cost_sequence'][0])

                        # Load iterative data
                        with open(it_file, 'r') as f:
                            it_data = json.load(f)

                        it_costs = []
                        for experiment in it_data:
                            if 'cost_sequence' in experiment and experiment['cost_sequence']:
                                it_costs.append(experiment['cost_sequence'][0])

                        if zs_costs and it_costs:
                            zero_shot_costs.extend(zs_costs)
                            iterative_costs.extend(it_costs)
                            print(f"    Including {model_name}: {len(zs_costs)} zero-shot, {len(it_costs)} iterative")
                        else:
                            print(f"    Skipping {model_name}: missing data in one or both files")

                    except Exception as e:
                        print(f"    Warning: Error processing {model_name}: {e}")
                        continue

        print(f"Dataset {dataset_name}: {len(zero_shot_costs)} zero-shot, {len(iterative_costs)} iterative first costs")
        return zero_shot_costs, iterative_costs

    def create_distribution_plots(self, zero_shot_costs: List[float], iterative_costs: List[float],
                                dataset_name: str) -> None:
        """
        Create three separate distribution plots comparing zero-shot vs iterative first costs.

        Args:
            zero_shot_costs: List of first costs from zero-shot approaches
            iterative_costs: List of first costs from iterative approaches
            dataset_name: Name of the dataset
        """
        if not zero_shot_costs and not iterative_costs:
            print(f"No data found for dataset {dataset_name}")
            return

        # Set modern style
        plt.style.use('default')
        sns.set_palette("Set2")
        colors = ['#FF6B6B', '#4ECDC4']  # Red for zero-shot, Teal for iterative

        # Prepare data
        all_costs = zero_shot_costs + iterative_costs
        labels = ['Zero-shot'] * len(zero_shot_costs) + ['Iterative'] * len(iterative_costs)
        df_plot = pd.DataFrame({'First Cost': all_costs, 'Approach': labels})

        # Plot 1: Violin plots
        fig1, ax1 = plt.subplots(figsize=(8, 6))

        violin_parts = ax1.violinplot([zero_shot_costs, iterative_costs],
                                     positions=[0, 1], showmeans=True, showmedians=True)

        # Customize violin plot colors
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1)

        # Customize other elements
        violin_parts['cmeans'].set_color('black')
        violin_parts['cmeans'].set_linewidth(2)
        violin_parts['cmedians'].set_color('white')
        violin_parts['cmedians'].set_linewidth(3)
        violin_parts['cbars'].set_color('black')
        violin_parts['cmaxes'].set_color('black')
        violin_parts['cmins'].set_color('black')

        ax1.set_yscale('log')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Zero-shot', 'Iterative'], fontsize=14, fontweight='bold')
        ax1.set_ylabel('First Attempt Cost (log scale)', fontsize=14, fontweight='bold')
        ax1.set_title(f'First Cost Distribution - {dataset_name}', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=12)

        # Save violin plot
        output_file1 = self.output_dir / f"{dataset_name}_violin_plot.png"
        plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved violin plot to: {output_file1}")
        plt.close()

        # Plot 2: Histogram comparison
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        bins = np.logspace(np.log10(min(all_costs)), np.log10(max(all_costs)), 30)
        ax2.hist(zero_shot_costs, bins=bins, alpha=0.7, label='Zero-shot',
                color=colors[0], density=True)
        ax2.hist(iterative_costs, bins=bins, alpha=0.7, label='Iterative',
                color=colors[1], density=True)

        ax2.set_xscale('log')
        ax2.set_xlabel('First Attempt Cost (log scale)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
        ax2.set_title(f'First Cost Histogram - {dataset_name}', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=12)

        # Save histogram plot
        output_file2 = self.output_dir / f"{dataset_name}_histogram.png"
        plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved histogram to: {output_file2}")
        plt.close()

        # Plot 3: Statistics table
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.axis('off')

        # Calculate comprehensive statistics
        zs_mean = np.mean(zero_shot_costs)
        zs_median = np.median(zero_shot_costs)
        zs_q75 = np.percentile(zero_shot_costs, 75)
        zs_q25 = np.percentile(zero_shot_costs, 25)

        it_mean = np.mean(iterative_costs)
        it_median = np.median(iterative_costs)
        it_q75 = np.percentile(iterative_costs, 75)
        it_q25 = np.percentile(iterative_costs, 25)

        ratio_mean = zs_mean / it_mean
        ratio_median = zs_median / it_median

        # Create beautiful statistics table
        stats_data = [
            ['Metric', 'Zero-shot', 'Iterative', 'Ratio (ZS/IT)'],
            ['Median', f'{zs_median:.2e}', f'{it_median:.2e}', f'{ratio_median:.2f}×'],
            ['Mean', f'{zs_mean:.2e}', f'{it_mean:.2e}', f'{ratio_mean:.2f}×'],
            ['Q75', f'{zs_q75:.2e}', f'{it_q75:.2e}', f'{zs_q75/it_q75:.2f}×'],
            ['Q25', f'{zs_q25:.2e}', f'{it_q25:.2e}', f'{zs_q25/it_q25:.2f}×']
        ]

        # Create table
        table = ax3.table(cellText=stats_data[1:], colLabels=stats_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.15, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 3)

        # Style the table
        for i in range(len(stats_data)):
            for j in range(len(stats_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold', color='white')
                elif j == 3:  # Ratio column
                    cell.set_facecolor('#FFE66D')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#F8F8F8')

        ax3.set_title(f'First Cost Statistics - {dataset_name}', fontsize=16, fontweight='bold', pad=20)

        # Save statistics plot
        output_file3 = self.output_dir / f"{dataset_name}_statistics.png"
        plt.savefig(output_file3, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved statistics table to: {output_file3}")
        plt.close()

    def create_detailed_breakdown(self, dataset_name: str) -> None:
        """
        Create detailed breakdown by difficulty level and parameter type.

        Args:
            dataset_name: Name of the dataset
        """
        dataset_path = self.base_dir / dataset_name

        if not dataset_path.exists():
            return

        # Collect data by difficulty and parameter
        breakdown_data = {}

        # Find all subdirectories (difficulty levels)
        for difficulty_dir in dataset_path.iterdir():
            if not difficulty_dir.is_dir():
                continue

            difficulty = difficulty_dir.name
            breakdown_data[difficulty] = {}

            # Find parameter subdirectories
            for param_dir in difficulty_dir.iterdir():
                if not param_dir.is_dir():
                    continue

                param = param_dir.name
                breakdown_data[difficulty][param] = {'zero_shot': [], 'iterative': []}

                # Process JSON files in this parameter directory
                for json_file in param_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)

                        for experiment in data:
                            if 'cost_sequence' not in experiment or not experiment['cost_sequence']:
                                continue

                            first_cost = experiment['cost_sequence'][0]
                            is_zero_shot = experiment.get('zero_shot', False)

                            if is_zero_shot:
                                breakdown_data[difficulty][param]['zero_shot'].append(first_cost)
                            else:
                                breakdown_data[difficulty][param]['iterative'].append(first_cost)

                    except Exception as e:
                        print(f"Warning: Error processing {json_file}: {e}")
                        continue

        # Create breakdown plot
        fig, axes = plt.subplots(len(breakdown_data), len(list(breakdown_data.values())[0]),
                               figsize=(5 * len(list(breakdown_data.values())[0]), 4 * len(breakdown_data)))
        fig.suptitle(f'First Cost Distribution Breakdown - {dataset_name}', fontsize=16, fontweight='bold')

        if len(breakdown_data) == 1:
            axes = [axes]
        if len(list(breakdown_data.values())[0]) == 1:
            axes = [[ax] for ax in axes]

        for i, (difficulty, params) in enumerate(breakdown_data.items()):
            for j, (param, costs) in enumerate(params.items()):
                ax = axes[i][j]

                zero_shot = costs['zero_shot']
                iterative = costs['iterative']

                if zero_shot or iterative:
                    all_costs = zero_shot + iterative
                    labels = ['Zero-shot'] * len(zero_shot) + ['Iterative'] * len(iterative)

                    if all_costs:
                        df_plot = pd.DataFrame({'First Cost': all_costs, 'Approach': labels})
                        sns.boxplot(data=df_plot, x='Approach', y='First Cost', ax=ax)
                        ax.set_yscale('log')
                        ax.set_title(f'{difficulty} - {param}')
                        ax.set_ylabel('First Cost (log scale)')
                else:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                           ha='center', va='center', fontsize=12)
                    ax.set_title(f'{difficulty} - {param}')

        plt.tight_layout()

        # Save the breakdown plot
        output_file = self.output_dir / f"{dataset_name}_first_cost_breakdown.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved breakdown plot to: {output_file}")

        plt.show()
        plt.close()

    def analyze_dataset(self, dataset_name: str) -> None:
        """
        Perform complete first cost analysis for a dataset.

        Args:
            dataset_name: Name of the dataset to analyze
        """
        print(f"\\nAnalyzing first cost distribution for dataset: {dataset_name}")

        # Load data
        zero_shot_costs, iterative_costs = self.load_first_costs(dataset_name)

        if not zero_shot_costs and not iterative_costs:
            print(f"No data found for dataset {dataset_name}")
            return

        # Create main distribution plots
        self.create_distribution_plots(zero_shot_costs, iterative_costs, dataset_name)

    def analyze_all_datasets(self) -> None:
        """Analyze all available datasets."""
        datasets = self.find_available_datasets()

        if not datasets:
            print("No datasets found.")
            return

        print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")

        for dataset in datasets:
            self.analyze_dataset(dataset)


def main():
    """Main function to run the first cost distribution analysis."""
    parser = argparse.ArgumentParser(description="Analyze first cost distributions from model attempt results")
    parser.add_argument('-d', '--dataset', type=str,
                       help='Specific dataset to analyze (e.g., heat_1d)')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = FirstCostAnalyzer()

    if args.dataset:
        analyzer.analyze_dataset(args.dataset)
    else:
        analyzer.analyze_all_datasets()


if __name__ == "__main__":
    main()