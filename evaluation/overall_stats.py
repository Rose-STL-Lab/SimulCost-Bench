#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overall Performance Statistics Generator for SimulCost-Bench

This script aggregates model performance across all datasets in SimulCost-Bench,
providing comprehensive statistics and visualizations for model comparison using simple averages.

Usage
-----
python evaluation/overall_stats.py
python evaluation/overall_stats.py -d heat_1d_bo  # Compare models with Bayesian optimization

Output: Creates comprehensive statistics and visualizations in eval_results/overall/:
- overall_summary.csv (aggregated performance across all datasets)
- overall_summary.xlsx (beautifully formatted Excel file)
- success_rate_overall.png (success rate bar chart)
- efficiency_overall.png (efficiency bar chart)
- line_plot_zero_shot.png (success rate & efficiency line plots for zero-shot mode)
- line_plot_iterative.png (success rate & efficiency line plots for iterative mode)

For heat_1d_bo comparison:
- heat_1d_bo_comparison.csv (model vs Bayesian optimization comparison)
- heat_1d_bo_comparison.xlsx (formatted comparison table)
- heat_1d_bo_comparison_plots.png (comparison visualizations)
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from statistics import mean

# Optional import for smart text positioning
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("Warning: adjustText library not found. Text labels may overlap. Install with: pip install adjustText")


class OverallStatsGenerator:
    """Generate comprehensive statistics and visualizations across all datasets."""

    def __init__(self, base_dir: str = "eval_results", output_dir: str = "eval_results/overall", target_dataset: Optional[str] = None):
        """
        Initialize the overall statistics generator.

        Args:
            base_dir: Base directory containing individual dataset results
            output_dir: Directory to save overall statistics and visualizations
            target_dataset: Specific dataset to process (e.g., 'heat_1d_bo')
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_dataset = target_dataset

        # Set up professional plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        # Model name mapping for clean display names
        self.name_mapping = {
            'amazon.nova-premier-v1:0': 'Nova-Premier',
            'anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude-3.7-Sonnet',
            'mistral.mistral-large-2402-v1:0': 'Mistral-Large',
            'meta.llama3-70b-instruct-v1:0': 'Llama-3-70B-Instruct',
            'gpt-5-2025-08-07': 'GPT-5',
            'qwen3_32b': 'Qwen3-32B',
            'qwen3_0_6b': 'Qwen3-0.6B',
            'qwen3_8b': 'Qwen3-8B',
            'anthropic.claude-3-5-haiku-20241022-v1:0': 'Claude-3.5-Haiku',
            'anthropic.claude-3-5-sonnet-20240620-v1:0': 'Claude-3.5-Sonnet',
            'bayesian_optimization': 'Bayesian Optimization',
        }

    def clean_model_name(self, model_name: str) -> str:
        """
        Clean model name using the name mapping.

        Args:
            model_name: Raw model name from the data

        Returns:
            Clean, display-friendly model name
        """
        return self.name_mapping.get(model_name, model_name)

    def find_available_datasets(self) -> List[str]:
        """
        Find all available datasets with summary CSV files.
        If target_dataset is specified, only include that dataset.

        Returns:
            List of dataset names that have summary files
        """
        datasets = []
        if not self.base_dir.exists():
            return datasets

        # If target dataset is specified, only look for that one
        if self.target_dataset:
            if self.target_dataset == 'heat_1d_bo':
                # Special handling for heat_1d_bo comparison
                heat_1d_file = self.base_dir / "heat_1d" / "heat_1d_sum.csv"
                heat_1d_bo_file = self.base_dir / "heat_1d_bo" / "heat_1d_bo_sum.csv"
                if heat_1d_file.exists() and heat_1d_bo_file.exists():
                    return ['heat_1d', 'heat_1d_bo']
            else:
                # Standard single dataset processing
                target_dir = self.base_dir / self.target_dataset
                if target_dir.exists() and target_dir.is_dir():
                    summary_file = target_dir / f"{self.target_dataset}_sum.csv"
                    if summary_file.exists():
                        return [self.target_dataset]
            return []

        # Default behavior: find all available datasets
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name not in ['overall', 'stats']:
                summary_file = item / f"{item.name}_sum.csv"
                if summary_file.exists():
                    datasets.append(item.name)

        return sorted(datasets)

    def load_all_datasets(self) -> pd.DataFrame:
        """
        Load and combine evaluation results from all available datasets.

        Returns:
            Combined DataFrame containing results from all datasets
        """
        datasets = self.find_available_datasets()
        if not datasets:
            raise ValueError("No datasets found with summary CSV files")

        print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")

        all_data = []
        total_records = 0

        for dataset in datasets:
            csv_path = self.base_dir / dataset / f"{dataset}_sum.csv"

            try:
                df = pd.read_csv(csv_path)

                # Apply model name mapping
                if 'Model' in df.columns:
                    df['Model'] = df['Model'].apply(self.clean_model_name)

                records_count = len(df)
                total_records += records_count
                all_data.append(df)
                print(f"  - Loaded {records_count} records from {dataset}")
            except Exception as e:
                print(f"  - Warning: Failed to load {dataset}: {e}")
                continue

        if not all_data:
            raise ValueError("No valid dataset files could be loaded")

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total combined records: {len(combined_df)}")

        return combined_df


    def calculate_overall_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate overall statistics aggregated across all datasets using simple averages.

        Args:
            df: Combined DataFrame from all datasets

        Returns:
            DataFrame with overall statistics per model/mode/precision combination
        """
        # Group by Model, Precision Level, and Inference Mode
        # Calculate simple averages across all simulations
        aggregated_results = []

        grouping_cols = ['Model', 'Precision Level', 'Inference Mode']
        for group_keys, group in df.groupby(grouping_cols):
            model, precision, mode = group_keys

            # Calculate total samples across all simulations
            total_samples = group['Number of Samples'].sum()

            # Number of simulations this model was tested on
            num_simulations = len(group['Simulation'].unique())

            # Calculate simple averages for metrics

            # Success rate: simple average across all simulations
            success_values = [float(val) for val in group['success_rate'] if pd.notnull(val)]
            avg_success_rate = self._simple_average(success_values) if success_values else 0.0

            # Efficiency: simple average across all simulations
            efficiency_values = [float(val) for val in group['mean_efficiency'] if pd.notnull(val)]
            avg_efficiency = self._simple_average(efficiency_values) if efficiency_values else 0.0

            # Create aggregated row
            agg_row = {
                'Model': model,
                'Precision Level': precision,
                'Inference Mode': mode,
                'Total Samples': total_samples,
                'Number of Solvers': num_simulations,
                'Success Rate': round(avg_success_rate, 2),
                'Efficiency': round(avg_efficiency, 2),
                # Add individual simulation performance for reference
                'Simulations': ', '.join(sorted(group['Simulation'].unique()))
            }

            aggregated_results.append(agg_row)

        # Sort by inference mode (Zero-shot first, then Iterative), then precision level, then model name
        result_df = pd.DataFrame(aggregated_results)
        result_df['mode_order'] = result_df['Inference Mode'].map({'Zero-shot': 0, 'Iterative': 1})
        precision_order = {'low': 0, 'medium': 1, 'high': 2}
        result_df['precision_order'] = result_df['Precision Level'].map(precision_order)
        result_df = result_df.sort_values(['mode_order', 'precision_order', 'Model']).drop(['mode_order', 'precision_order'], axis=1)

        return result_df

    def _simple_average(self, values: List[float]) -> float:
        """Calculate simple average."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def calculate_aggregated_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregated statistics without precision level dimension using simple averages.
        Aggregates across all precision levels for each model/mode combination.

        Args:
            df: Combined DataFrame from all datasets

        Returns:
            DataFrame with aggregated statistics per model/mode combination (no precision level)
        """
        # Group by Model and Inference Mode only (no Precision Level)
        aggregated_results = []

        grouping_cols = ['Model', 'Inference Mode']
        for group_keys, group in df.groupby(grouping_cols):
            model, mode = group_keys

            # Calculate total samples across all simulations and precision levels
            total_samples = group['Number of Samples'].sum()

            # Number of simulations this model was tested on
            num_simulations = len(group['Simulation'].unique())

            # Number of precision levels tested
            num_precision_levels = len(group['Precision Level'].unique())

            # Calculate simple averages for metrics

            # Success rate: simple average across all simulations and precision levels
            success_values = [float(val) for val in group['success_rate'] if pd.notnull(val)]
            avg_success_rate = self._simple_average(success_values) if success_values else 0.0

            # Efficiency: simple average across all simulations and precision levels
            efficiency_values = [float(val) for val in group['mean_efficiency'] if pd.notnull(val)]
            avg_efficiency = self._simple_average(efficiency_values) if efficiency_values else 0.0

            # Create aggregated row
            agg_row = {
                'Model': model,
                'Inference Mode': mode,
                'Total Samples': total_samples,
                'Number of Solvers': num_simulations,
                'Precision Levels Tested': num_precision_levels,
                'Success Rate': round(avg_success_rate, 2),
                'Efficiency': round(avg_efficiency, 2),
                # Add individual simulation performance for reference
                'Simulations': ', '.join(sorted(group['Simulation'].unique())),
                'Precision Levels': ', '.join(sorted(group['Precision Level'].unique()))
            }

            aggregated_results.append(agg_row)

        # Sort by inference mode (Zero-shot first, then Iterative), then model name
        result_df = pd.DataFrame(aggregated_results)
        result_df['mode_order'] = result_df['Inference Mode'].map({'Zero-shot': 0, 'Iterative': 1})
        result_df = result_df.sort_values(['mode_order', 'Model']).drop(['mode_order'], axis=1)

        return result_df

    def save_csv_results(self, df: pd.DataFrame, filename: str = "overall_summary.csv") -> None:
        """
        Save results to CSV with proper formatting.

        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.output_dir / filename

        # Create ordered columns
        ordered_cols = [
            'Model', 'Precision Level', 'Inference Mode',
            'Total Samples', 'Number of Solvers', 'Success Rate',
            'Efficiency', 'Simulations'
        ]

        # Data is already formatted to 2 decimal places in calculate_overall_statistics
        formatted_df = df.copy()

        # Write CSV
        formatted_df[ordered_cols].to_csv(output_path, index=False)
        print(f"CSV results saved to: {output_path}")

    def save_aggregated_csv_results(self, df: pd.DataFrame, filename: str = "overall_summary_aggregated.csv") -> None:
        """
        Save aggregated results to CSV without precision level column.

        Args:
            df: DataFrame to save (from calculate_aggregated_statistics)
            filename: Output filename
        """
        output_path = self.output_dir / filename

        # Create ordered columns for aggregated data
        ordered_cols = [
            'Model', 'Inference Mode',
            'Total Samples', 'Number of Solvers', 'Precision Levels Tested',
            'Success Rate', 'Efficiency', 'Simulations', 'Precision Levels'
        ]

        # Data is already formatted to 2 decimal places in calculate_aggregated_statistics
        formatted_df = df.copy()

        # Write CSV
        formatted_df[ordered_cols].to_csv(output_path, index=False)
        print(f"Aggregated CSV results saved to: {output_path}")

    def save_excel_results(self, df: pd.DataFrame, filename: str = "overall_summary.xlsx") -> None:
        """
        Save results to Excel with beautiful formatting.

        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.output_dir / filename

        # Create ordered columns
        ordered_cols = [
            'Model', 'Precision Level', 'Inference Mode',
            'Total Samples', 'Number of Solvers', 'Success Rate',
            'Efficiency', 'Simulations'
        ]

        # Prepare formatted data
        excel_df = df[ordered_cols].copy()

        # Data is already sorted by inference mode (Zero-shot first, then Iterative) in calculate_overall_statistics

        # Write Excel with advanced formatting
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            wb = writer.book
            ws = wb.add_worksheet('Overall Summary')
            writer.sheets['Overall Summary'] = ws

            # Formatting styles
            header_fmt = wb.add_format({
                'bold': True,
                'border': 1,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#F0F0F0'
            })

            # Iterative mode rows - light green background
            iterative_fmt = wb.add_format({
                'border': 1,
                'bg_color': '#E8F5E8'
            })

            # Zero-shot mode rows - light blue background
            zeroshot_fmt = wb.add_format({
                'border': 1,
                'bg_color': '#E8F0FF'
            })

            # Best performance highlighting
            best_iterative_fmt = wb.add_format({
                'bold': True,
                'border': 1,
                'bg_color': '#E8F5E8'
            })

            best_zeroshot_fmt = wb.add_format({
                'bold': True,
                'border': 1,
                'bg_color': '#E8F0FF'
            })

            # Write header
            for i, col in enumerate(ordered_cols):
                display_name = col.replace('_', ' ')
                ws.write(0, i, display_name, header_fmt)

            excel_row = 1
            current_precision = None
            current_mode = None

            for _, row in excel_df.iterrows():
                precision = row['Precision Level']
                mode = row['Inference Mode']

                # Add visual separator when precision level changes
                if current_precision != precision and current_precision is not None:
                    excel_row += 1
                current_precision = precision

                # Add visual separator when inference mode changes within same precision
                if current_mode != mode and current_mode is not None:
                    excel_row += 1
                current_mode = mode

                # Find best efficiency in this precision/mode group
                same_group = excel_df[
                    (excel_df['Precision Level'] == precision) &
                    (excel_df['Inference Mode'] == mode)
                ]
                max_efficiency = same_group['Efficiency'].max()

                is_best_performance = (
                    pd.notnull(max_efficiency) and
                    row['Efficiency'] == max_efficiency and
                    row['Efficiency'] > 0
                )

                # Choose formatting
                if mode.lower() == 'iterative':
                    base_fmt = iterative_fmt
                    best_fmt = best_iterative_fmt
                else:  # zero-shot
                    base_fmt = zeroshot_fmt
                    best_fmt = best_zeroshot_fmt

                # Write row data
                for i, col in enumerate(ordered_cols):
                    val = row[col]

                    if is_best_performance and col in ['Model', 'Efficiency']:
                        cell_fmt = best_fmt
                    else:
                        cell_fmt = base_fmt

                    if col == 'Total Samples':
                        ws.write_number(excel_row, i, val, cell_fmt)
                    elif col == 'Number of Solvers':
                        ws.write_number(excel_row, i, val, cell_fmt)
                    elif col in ['Success Rate', 'Efficiency']:
                        ws.write_number(excel_row, i, val, cell_fmt)
                    else:
                        ws.write(excel_row, i, val, cell_fmt)

                excel_row += 1

            # Auto filter and freeze panes
            ws.autofilter(0, 0, excel_row - 1, len(ordered_cols) - 1)
            ws.freeze_panes(1, 0)

            # Auto-adjust column widths
            for idx, col in enumerate(ordered_cols):
                if col == 'Model':
                    ws.set_column(idx, idx, 20)
                elif col == 'Simulations':
                    ws.set_column(idx, idx, 30)
                elif col in ['Precision Level', 'Inference Mode']:
                    ws.set_column(idx, idx, 15)
                else:
                    max_len = max(len(str(excel_df[col].max())), len(col)) + 3
                    ws.set_column(idx, idx, min(max_len, 20))

            # Add legend
            legend_row = excel_row + 2
            legend_fmt = wb.add_format({'italic': True, 'font_size': 9})
            iterative_legend_fmt = wb.add_format({'italic': True, 'font_size': 9, 'bg_color': '#E8F5E8'})
            zeroshot_legend_fmt = wb.add_format({'italic': True, 'font_size': 9, 'bg_color': '#E8F0FF'})

            ws.write(legend_row, 0, "Legend:", legend_fmt)
            ws.write(legend_row + 1, 0, "Iterative mode", iterative_legend_fmt)
            ws.write(legend_row + 2, 0, "Zero-shot mode", zeroshot_legend_fmt)
            ws.write(legend_row + 3, 0, "Best efficiency model in each precision/mode group shown in bold", legend_fmt)

        print(f"Excel results saved to: {output_path}")

    def save_aggregated_excel_results(self, df: pd.DataFrame, filename: str = "overall_summary_aggregated.xlsx") -> None:
        """
        Save aggregated results to Excel with beautiful formatting (no precision level column).

        Args:
            df: DataFrame to save (from calculate_aggregated_statistics)
            filename: Output filename
        """
        output_path = self.output_dir / filename

        # Create ordered columns for aggregated data
        ordered_cols = [
            'Model', 'Inference Mode',
            'Total Samples', 'Number of Solvers', 'Precision Levels Tested',
            'Success Rate', 'Efficiency', 'Simulations', 'Precision Levels'
        ]

        # Prepare formatted data
        excel_df = df[ordered_cols].copy()

        # Data is already sorted by inference mode in calculate_aggregated_statistics

        # Write Excel with advanced formatting
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            wb = writer.book
            ws = wb.add_worksheet('Aggregated Summary')
            writer.sheets['Aggregated Summary'] = ws

            # Formatting styles
            header_fmt = wb.add_format({
                'bold': True,
                'border': 1,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#F0F0F0'
            })

            # Iterative mode rows - light green background
            iterative_fmt = wb.add_format({
                'border': 1,
                'bg_color': '#E8F5E8'
            })

            # Zero-shot mode rows - light blue background
            zeroshot_fmt = wb.add_format({
                'border': 1,
                'bg_color': '#E8F0FF'
            })

            # Best performance highlighting
            best_iterative_fmt = wb.add_format({
                'bold': True,
                'border': 1,
                'bg_color': '#E8F5E8'
            })

            best_zeroshot_fmt = wb.add_format({
                'bold': True,
                'border': 1,
                'bg_color': '#E8F0FF'
            })

            # Write header
            for i, col in enumerate(ordered_cols):
                display_name = col.replace('_', ' ')
                ws.write(0, i, display_name, header_fmt)

            excel_row = 1
            current_mode = None

            for _, row in excel_df.iterrows():
                mode = row['Inference Mode']

                # Add visual separator when inference mode changes
                if current_mode != mode and current_mode is not None:
                    excel_row += 1
                current_mode = mode

                # Find best efficiency in this mode group
                same_mode = excel_df[excel_df['Inference Mode'] == mode]
                max_efficiency = same_mode['Efficiency'].max()

                is_best_performance = (
                    pd.notnull(max_efficiency) and
                    row['Efficiency'] == max_efficiency and
                    row['Efficiency'] > 0
                )

                # Choose formatting
                if mode.lower() == 'iterative':
                    base_fmt = iterative_fmt
                    best_fmt = best_iterative_fmt
                else:  # zero-shot
                    base_fmt = zeroshot_fmt
                    best_fmt = best_zeroshot_fmt

                # Write row data
                for i, col in enumerate(ordered_cols):
                    val = row[col]

                    if is_best_performance and col in ['Model', 'Efficiency']:
                        cell_fmt = best_fmt
                    else:
                        cell_fmt = base_fmt

                    if col == 'Total Samples':
                        ws.write_number(excel_row, i, val, cell_fmt)
                    elif col in ['Number of Solvers', 'Precision Levels Tested']:
                        ws.write_number(excel_row, i, val, cell_fmt)
                    elif col in ['Success Rate', 'Efficiency']:
                        ws.write_number(excel_row, i, val, cell_fmt)
                    else:
                        ws.write(excel_row, i, val, cell_fmt)

                excel_row += 1

            # Auto filter and freeze panes
            ws.autofilter(0, 0, excel_row - 1, len(ordered_cols) - 1)
            ws.freeze_panes(1, 0)

            # Auto-adjust column widths
            for idx, col in enumerate(ordered_cols):
                if col == 'Model':
                    ws.set_column(idx, idx, 20)
                elif col in ['Simulations', 'Precision Levels']:
                    ws.set_column(idx, idx, 30)
                elif col == 'Inference Mode':
                    ws.set_column(idx, idx, 15)
                else:
                    max_len = max(len(str(excel_df[col].max())), len(col)) + 3
                    ws.set_column(idx, idx, min(max_len, 20))

            # Add legend
            legend_row = excel_row + 2
            legend_fmt = wb.add_format({'italic': True, 'font_size': 9})
            iterative_legend_fmt = wb.add_format({'italic': True, 'font_size': 9, 'bg_color': '#E8F5E8'})
            zeroshot_legend_fmt = wb.add_format({'italic': True, 'font_size': 9, 'bg_color': '#E8F0FF'})

            ws.write(legend_row, 0, "Legend:", legend_fmt)
            ws.write(legend_row + 1, 0, "Iterative mode", iterative_legend_fmt)
            ws.write(legend_row + 2, 0, "Zero-shot mode", zeroshot_legend_fmt)
            ws.write(legend_row + 3, 0, "Best efficiency model in each mode group shown in bold", legend_fmt)
            ws.write(legend_row + 4, 0, "Aggregated across all precision levels", legend_fmt)

        print(f"Aggregated Excel results saved to: {output_path}")

    def create_success_rate_chart(self, df: pd.DataFrame, filename: str = "success_rate_overall.png") -> None:
        """
        Create overall success rate bar chart across all datasets.

        Args:
            df: DataFrame containing overall statistics
            filename: Output filename for the chart
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Get unique models and precision levels
        models = sorted(df['Model'].unique())
        precision_levels = ['low', 'medium', 'high']
        inference_modes = ['Zero-shot', 'Iterative']

        # Chart parameters
        bar_width = 0.12
        precision_spacing = 0.12
        mode_spacing = 0.01
        model_spacing = 0.4

        # Colors
        zero_shot_color = '#BBBBBB'
        iterative_color = '#4169E1'

        # Calculate positions
        total_group_width = len(precision_levels) * (2 * bar_width + mode_spacing) + (len(precision_levels) - 1) * precision_spacing
        x_positions = np.arange(len(models)) * (total_group_width + model_spacing)
        precision_labels = ['L', 'M', 'H']

        # Get max value for y-axis
        max_val = df['Success Rate'].max()

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]

            for j, precision in enumerate(precision_levels):
                precision_data = model_data[model_data['Precision Level'] == precision]

                for k, mode in enumerate(inference_modes):
                    mode_data = precision_data[precision_data['Inference Mode'] == mode]

                    if len(mode_data) > 0:
                        value = mode_data['Success Rate'].iloc[0]

                        # Choose color and pattern
                        if mode == 'Zero-shot':
                            color = zero_shot_color
                            hatch = '///'
                        else:  # Iterative
                            color = iterative_color
                            hatch = None

                        # Calculate position
                        precision_offset = j * (2 * bar_width + mode_spacing + precision_spacing)
                        mode_offset = k * (bar_width + mode_spacing)
                        bar_pos = x_positions[i] + precision_offset + mode_offset - (len(precision_levels) * (2 * bar_width + mode_spacing + precision_spacing) - precision_spacing) / 2

                        # Create bar
                        ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                              hatch=hatch, edgecolor='lightgray', linewidth=0.3)

                        # Add precision label
                        ax.text(bar_pos, value/2, precision_labels[j],
                               ha='center', va='center', fontweight='bold',
                               color='black', fontsize=8)

                        # Add value label
                        ax.text(bar_pos, value + 0.01, f'{value:.2f}',
                               ha='center', va='bottom', fontsize=7)

        # Customize chart
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=10, fontweight='bold')
        ax.set_title('Success Rate Across All Datasets',
                    fontsize=10, fontweight='bold')  # Match y-axis font size
        ax.set_ylim(0, max_val * 1.1)  # Add 10% padding to the maximum value
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0)  # Horizontal labels
        ax.grid(True, alpha=0.3)

        # Create custom legend with patterns
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=zero_shot_color, hatch='///', edgecolor='lightgray', label='Zero-shot'),
            Patch(facecolor=iterative_color, label='Iterative')
        ]
        ax.legend(handles=legend_elements, title='Inference Mode',
                 title_fontsize=8, fontsize=7, loc='upper right', frameon=True)

        plt.tight_layout()

        # Save chart
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Success rate chart saved to: {output_path}")

    def create_aggregated_success_rate_chart(self, df: pd.DataFrame, filename: str = "success_rate_overall_aggregated.png") -> None:
        """
        Create aggregated success rate bar chart without precision level grouping.

        Args:
            df: DataFrame containing aggregated statistics (from calculate_aggregated_statistics)
            filename: Output filename for the chart
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Get unique models and inference modes
        models = sorted(df['Model'].unique())
        inference_modes = ['Zero-shot', 'Iterative']

        # Chart parameters
        bar_width = 0.35
        model_spacing = 0.8

        # Colors
        zero_shot_color = '#BBBBBB'
        iterative_color = '#4169E1'

        # Calculate positions
        x_positions = np.arange(len(models))

        # Get max value for y-axis
        max_val = df['Success Rate'].max()

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]

            for j, mode in enumerate(inference_modes):
                mode_data = model_data[model_data['Inference Mode'] == mode]

                if len(mode_data) > 0:
                    value = mode_data['Success Rate'].iloc[0]

                    # Choose color and pattern
                    if mode == 'Zero-shot':
                        color = zero_shot_color
                        hatch = '///'
                    else:  # Iterative
                        color = iterative_color
                        hatch = None

                    # Calculate position
                    bar_pos = x_positions[i] + (j - 0.5) * bar_width

                    # Create bar
                    ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                          hatch=hatch, edgecolor='lightgray', linewidth=0.3)

                    # Add value label
                    ax.text(bar_pos, value + 0.01, f'{value:.2f}',
                           ha='center', va='bottom', fontsize=8)

        # Customize chart
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_title('Success Rate Across All Datasets (Aggregated)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max_val * 1.15)  # Add 15% padding to the maximum value
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0)
        ax.grid(True, alpha=0.3)

        # Create custom legend with patterns
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=zero_shot_color, hatch='///', edgecolor='lightgray', label='Zero-shot'),
            Patch(facecolor=iterative_color, label='Iterative')
        ]
        ax.legend(handles=legend_elements, title='Inference Mode',
                 title_fontsize=10, fontsize=9, loc='upper right', frameon=True)

        plt.tight_layout()

        # Save chart
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Aggregated success rate chart saved to: {output_path}")

    def create_efficiency_chart(self, df: pd.DataFrame, filename: str = "efficiency_overall.png") -> None:
        """
        Create overall efficiency bar chart across all datasets.

        Args:
            df: DataFrame containing overall statistics
            filename: Output filename for the chart
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Get unique models and precision levels
        models = sorted(df['Model'].unique())
        precision_levels = ['low', 'medium', 'high']
        inference_modes = ['Zero-shot', 'Iterative']

        # Chart parameters
        bar_width = 0.12
        precision_spacing = 0.12
        mode_spacing = 0.01
        model_spacing = 0.4

        # Colors
        zero_shot_color = '#BBBBBB'
        iterative_color = '#4169E1'

        # Calculate positions
        total_group_width = len(precision_levels) * (2 * bar_width + mode_spacing) + (len(precision_levels) - 1) * precision_spacing
        x_positions = np.arange(len(models)) * (total_group_width + model_spacing)
        precision_labels = ['L', 'M', 'H']

        # Get max value for y-axis
        max_val = df['Efficiency'].max()

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]

            for j, precision in enumerate(precision_levels):
                precision_data = model_data[model_data['Precision Level'] == precision]

                for k, mode in enumerate(inference_modes):
                    mode_data = precision_data[precision_data['Inference Mode'] == mode]

                    if len(mode_data) > 0:
                        value = mode_data['Efficiency'].iloc[0]

                        # Choose color and pattern
                        if mode == 'Zero-shot':
                            color = zero_shot_color
                            hatch = '///'
                        else:  # Iterative
                            color = iterative_color
                            hatch = None

                        # Calculate position
                        precision_offset = j * (2 * bar_width + mode_spacing + precision_spacing)
                        mode_offset = k * (bar_width + mode_spacing)
                        bar_pos = x_positions[i] + precision_offset + mode_offset - (len(precision_levels) * (2 * bar_width + mode_spacing + precision_spacing) - precision_spacing) / 2

                        # Create bar
                        ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                              hatch=hatch, edgecolor='lightgray', linewidth=0.3)

                        # Add precision label
                        ax.text(bar_pos, value/2, precision_labels[j],
                               ha='center', va='center', fontweight='bold',
                               color='black', fontsize=8)

                        # Add value label
                        ax.text(bar_pos, value + max_val * 0.01, f'{value:.2f}',
                               ha='center', va='bottom', fontsize=7)

        # Customize chart
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Efficiency', fontsize=10, fontweight='bold')
        ax.set_title('Efficiency Across All Datasets',
                    fontsize=10, fontweight='bold')  # Match y-axis font size
        ax.set_ylim(0, max_val * 1.1)  # Add 10% padding to the maximum value
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0)  # Horizontal labels
        ax.grid(True, alpha=0.3)

        # Create custom legend with patterns
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=zero_shot_color, hatch='///', edgecolor='lightgray', label='Zero-shot'),
            Patch(facecolor=iterative_color, label='Iterative')
        ]
        ax.legend(handles=legend_elements, title='Inference Mode',
                 title_fontsize=8, fontsize=7, loc='upper right', frameon=True)

        plt.tight_layout()

        # Save chart
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Efficiency chart saved to: {output_path}")

    def create_aggregated_efficiency_chart(self, df: pd.DataFrame, filename: str = "efficiency_overall_aggregated.png") -> None:
        """
        Create aggregated efficiency bar chart without precision level grouping.

        Args:
            df: DataFrame containing aggregated statistics (from calculate_aggregated_statistics)
            filename: Output filename for the chart
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Get unique models and inference modes
        models = sorted(df['Model'].unique())
        inference_modes = ['Zero-shot', 'Iterative']

        # Chart parameters
        bar_width = 0.35
        model_spacing = 0.8

        # Colors
        zero_shot_color = '#BBBBBB'
        iterative_color = '#4169E1'

        # Calculate positions
        x_positions = np.arange(len(models))

        # Get max value for y-axis
        max_val = df['Efficiency'].max()

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]

            for j, mode in enumerate(inference_modes):
                mode_data = model_data[model_data['Inference Mode'] == mode]

                if len(mode_data) > 0:
                    value = mode_data['Efficiency'].iloc[0]

                    # Choose color and pattern
                    if mode == 'Zero-shot':
                        color = zero_shot_color
                        hatch = '///'
                    else:  # Iterative
                        color = iterative_color
                        hatch = None

                    # Calculate position
                    bar_pos = x_positions[i] + (j - 0.5) * bar_width

                    # Create bar
                    ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                          hatch=hatch, edgecolor='lightgray', linewidth=0.3)

                    # Add value label
                    ax.text(bar_pos, value + max_val * 0.01, f'{value:.2f}',
                           ha='center', va='bottom', fontsize=8)

        # Customize chart
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency Across All Datasets (Aggregated)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max_val * 1.15)  # Add 15% padding to the maximum value
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0)
        ax.grid(True, alpha=0.3)

        # Create custom legend with patterns
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=zero_shot_color, hatch='///', edgecolor='lightgray', label='Zero-shot'),
            Patch(facecolor=iterative_color, label='Iterative')
        ]
        ax.legend(handles=legend_elements, title='Inference Mode',
                 title_fontsize=10, fontsize=9, loc='upper right', frameon=True)

        plt.tight_layout()

        # Save chart
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Aggregated efficiency chart saved to: {output_path}")

    def create_combined_line_plots(self, df: pd.DataFrame) -> None:
        """
        Create a single combined line plot with 4 subplots:
        Top row: Zero-shot (Success Rate, Efficiency)
        Bottom row: Iterative (Success Rate, Efficiency)
        Uses clean styling with custom legend layout.

        Args:
            df: DataFrame containing overall statistics
        """
        # Get unique models and precision levels
        models = sorted(df['Model'].unique())
        precision_levels = ['low', 'medium', 'high']
        inference_modes = ['Zero-shot', 'Iterative']

        # Define three colors: blue, green, pink
        base_colors = ['#1f77b4', '#2ca02c', '#ff69b4']  # Blue, Green, Pink

        # Define markers: hollow circle and hollow square only
        base_markers = ['o', 's']

        # Define line styles: solid, dashed with shorter segments
        line_styles = ['-', (0, (3, 2))]  # solid, short dashed (3pt dash, 2pt gap)

        # Create style mapping for models (similar to the example image)
        model_styles = {}
        for i, model in enumerate(models):
            color_idx = i % len(base_colors)
            marker_idx = i % len(base_markers)
            line_idx = (i // len(base_colors)) % len(line_styles)

            model_styles[model] = {
                'color': base_colors[color_idx],
                'marker': base_markers[marker_idx],
                'linestyle': line_styles[line_idx]
            }

        # Create single figure with 4 subplots (1x4 horizontal layout)
        fig = plt.figure(figsize=(16, 5))

        # Create custom layout: legend at top, plots in middle, labels at bottom
        gs = fig.add_gridspec(3, 4, height_ratios=[0.1, 1, 0.25], hspace=0.2, wspace=0.18)

        # Create legend axis (spans full width at top)
        legend_ax = fig.add_subplot(gs[0, :])

        # Create 4 subplot axes (1x4 horizontal grid)
        axes = {
            ('Zero-shot', 'Success Rate'): fig.add_subplot(gs[1, 0]),
            ('Zero-shot', 'Efficiency'): fig.add_subplot(gs[1, 1]),
            ('Iterative', 'Success Rate'): fig.add_subplot(gs[1, 2]),
            ('Iterative', 'Efficiency'): fig.add_subplot(gs[1, 3])
        }

        # Create bottom label axes
        label_axes = [
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
            fig.add_subplot(gs[2, 3])
        ]

        metrics = ['Success Rate', 'Efficiency']

        # Plot data for all mode-metric combinations
        for mode in inference_modes:
            mode_data = df[df['Inference Mode'] == mode]

            for metric in metrics:
                ax = axes[(mode, metric)]

                for model in models:
                    model_data = mode_data[mode_data['Model'] == model]

                    if len(model_data) == 0:
                        continue

                    # Prepare data points for this model
                    x_values = []
                    y_values = []

                    for precision_idx, precision in enumerate(precision_levels):
                        precision_data = model_data[model_data['Precision Level'] == precision]
                        if len(precision_data) > 0:
                            x_values.append(precision_idx)
                            y_values.append(precision_data[metric].iloc[0])

                    # Plot line for this model
                    if len(x_values) > 0:
                        style = model_styles[model]

                        ax.plot(x_values, y_values,
                               marker=style['marker'],
                               color=style['color'],
                               linestyle=style['linestyle'],
                               linewidth=1.0,
                               markersize=3,
                               markerfacecolor='white',
                               markeredgecolor=style['color'],
                               markeredgewidth=1.0,
                               alpha=1.0,
                               clip_on=False)  # Prevent markers from being clipped at plot boundaries

                # Customize subplot
                ax.set_xlabel('Accuracy Level', fontsize=12)
                ax.set_ylabel(metric, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='gray')

                # Set x-axis labels and ticks with tighter spacing and extra padding for markers
                ax.set_xticks(range(len(precision_levels)))
                ax.set_xticklabels([p.capitalize() for p in precision_levels], fontsize=10)
                ax.set_xlim(-0.1, len(precision_levels) - 0.9)

                # Set y-axis limits with more padding to prevent marker cropping
                if len(mode_data) > 0:
                    y_min = mode_data[metric].min()
                    y_max = mode_data[metric].max()
                    y_range = y_max - y_min

                    if metric == 'Success Rate':
                        # Allow negative lower bound to show markers at 0 values properly
                        lower_bound = y_min - 0.08
                        if y_min <= 0.05:  # If minimum value is close to 0
                            lower_bound = -0.05
                        ax.set_ylim(lower_bound, min(1, y_max + 0.08))
                    else:  # Efficiency
                        # Allow negative lower bound to show markers at 0 values properly
                        lower_bound = y_min - 0.5
                        if y_min <= 1.0:  # If minimum value is close to 0
                            lower_bound = -1.0
                        ax.set_ylim(lower_bound, y_max + max(0.5, y_range * 0.1))

        # Create custom legend in top area
        legend_ax.axis('off')

        # Create legend elements for all models
        legend_elements = []
        for model in models:
            style = model_styles[model]
            from matplotlib.lines import Line2D

            legend_elements.append(Line2D([0], [0],
                                         marker=style['marker'],
                                         color=style['color'],
                                         linestyle=style['linestyle'],
                                         linewidth=1.0,
                                         markersize=3,
                                         markerfacecolor='white',
                                         markeredgecolor=style['color'],
                                         markeredgewidth=1.0,
                                         label=model))

        # Display legend horizontally
        ncols = len(models)
        if len(models) > 6:  # If too many models, split into 2 rows
            ncols = 3

        legend_ax.legend(handles=legend_elements,
                       loc='center',
                       ncol=ncols,
                       frameon=False,
                       fontsize=11,
                       columnspacing=2.0,
                       handlelength=2.5)

        # Add bottom labels for each subplot
        subplot_labels = [
            ('a', 'Zero-shot - Success Rate'),
            ('b', 'Zero-shot - Efficiency'),
            ('c', 'Iterative - Success Rate'),
            ('d', 'Iterative - Efficiency')
        ]

        for i, (letter, description) in enumerate(subplot_labels):
            label_ax = label_axes[i]
            label_ax.axis('off')
            label_ax.text(0.5, 0.5, f'({letter}) {description}',
                         ha='center', va='center',
                         fontsize=13, fontweight='bold',
                         transform=label_ax.transAxes)

        # Save single combined chart with extra padding
        filename = "line_plot_combined.png"
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()

        print(f"Combined line plot saved to: {output_path}")

    def generate_heat_1d_bo_comparison(self) -> None:
        """Generate comparison between models and Bayesian optimization for heat_1d_bo."""
        print("🚀 Starting heat_1d_bo comparison generation...")

        try:
            # Create heat_1d_bo specific output directory
            heat_1d_bo_output_dir = self.output_dir / "heat_1d_bo"
            heat_1d_bo_output_dir.mkdir(parents=True, exist_ok=True)

            # Load heat_1d data
            heat_1d_file = self.base_dir / "heat_1d" / "heat_1d_sum.csv"
            heat_1d_bo_file = self.base_dir / "heat_1d_bo" / "heat_1d_bo_sum.csv"

            if not heat_1d_file.exists() or not heat_1d_bo_file.exists():
                raise ValueError("Required files for heat_1d_bo comparison not found")

            # Load the datasets
            heat_1d_df = pd.read_csv(heat_1d_file)
            heat_1d_bo_df = pd.read_csv(heat_1d_bo_file)

            # Apply model name mapping
            heat_1d_df['Model'] = heat_1d_df['Model'].apply(self.clean_model_name)
            heat_1d_bo_df['Model'] = heat_1d_bo_df['Model'].apply(self.clean_model_name)

            print(f"  - Loaded {len(heat_1d_df)} records from heat_1d")
            print(f"  - Loaded {len(heat_1d_bo_df)} records from heat_1d_bo")

            # Generate comparison statistics and visualizations
            comparison_df = self.create_heat_1d_bo_comparison_table(heat_1d_df, heat_1d_bo_df)

            # Save results to heat_1d_bo subdirectory
            self.save_heat_1d_bo_comparison_results(comparison_df, heat_1d_bo_output_dir)
            self.create_heat_1d_bo_comparison_plots(heat_1d_df, heat_1d_bo_df, heat_1d_bo_output_dir)

            print(f"\n🎉 heat_1d_bo comparison generation completed!")
            print(f"📁 Results saved to: {heat_1d_bo_output_dir}")

        except Exception as e:
            print(f"❌ Error generating heat_1d_bo comparison: {e}")
            raise

    def create_heat_1d_bo_comparison_table(self, heat_1d_df: pd.DataFrame, heat_1d_bo_df: pd.DataFrame) -> pd.DataFrame:
        """Create comparison table between models and Bayesian optimization."""
        comparison_data = []

        # Get unique precision levels
        precision_levels = ['low', 'medium', 'high']

        for precision in precision_levels:
            # Get Bayesian optimization results for this precision
            bo_data = heat_1d_bo_df[heat_1d_bo_df['Precision Level'] == precision]
            if len(bo_data) == 0:
                continue

            bo_success_rate = bo_data['success_rate'].iloc[0]
            bo_efficiency = bo_data['mean_efficiency'].iloc[0]

            # Get model results for this precision level
            precision_models = heat_1d_df[heat_1d_df['Precision Level'] == precision]

            # Group by model and calculate averages across inference modes
            for model in precision_models['Model'].unique():
                model_data = precision_models[precision_models['Model'] == model]

                # Calculate averages across inference modes
                avg_success_rate = model_data['success_rate'].mean()
                avg_efficiency = model_data['mean_efficiency'].mean()

                # Calculate differences vs Bayesian optimization
                success_diff = avg_success_rate - bo_success_rate
                efficiency_diff = avg_efficiency - bo_efficiency

                comparison_data.append({
                    'Precision Level': precision,
                    'Model': model,
                    'Model Success Rate': round(avg_success_rate, 2),
                    'Model Efficiency': round(avg_efficiency, 2),
                    'BO Success Rate': round(bo_success_rate, 2),
                    'BO Efficiency': round(bo_efficiency, 2),
                    'Success Rate Diff': round(success_diff, 2),
                    'Efficiency Diff': round(efficiency_diff, 2),
                    'Model Better (Success)': 'Yes' if success_diff > 0 else 'No',
                    'Model Better (Efficiency)': 'Yes' if efficiency_diff > 0 else 'No'
                })

        # Create DataFrame and sort
        comparison_df = pd.DataFrame(comparison_data)
        precision_order = {'low': 0, 'medium': 1, 'high': 2}
        comparison_df['precision_order'] = comparison_df['Precision Level'].map(precision_order)
        comparison_df = comparison_df.sort_values(['precision_order', 'Model']).drop(['precision_order'], axis=1)

        return comparison_df

    def save_heat_1d_bo_comparison_results(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Save heat_1d_bo comparison results to CSV and Excel."""
        # Save CSV
        csv_path = output_dir / "heat_1d_bo_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"heat_1d_bo comparison CSV saved to: {csv_path}")

        # Save Excel with formatting
        excel_path = output_dir / "heat_1d_bo_comparison.xlsx"

        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            wb = writer.book
            ws = wb.add_worksheet('heat_1d_bo Comparison')
            writer.sheets['heat_1d_bo Comparison'] = ws

            # Formatting styles
            header_fmt = wb.add_format({
                'bold': True,
                'border': 1,
                'align': 'center',
                'valign': 'vcenter',
                'bg_color': '#F0F0F0'
            })

            # Model better formatting
            model_better_fmt = wb.add_format({
                'border': 1,
                'bg_color': '#E8F5E8'  # Light green
            })

            # BO better formatting
            bo_better_fmt = wb.add_format({
                'border': 1,
                'bg_color': '#FFE8E8'  # Light red
            })

            # Neutral formatting
            neutral_fmt = wb.add_format({
                'border': 1,
                'bg_color': '#F8F8F8'
            })

            # Write headers
            for i, col in enumerate(df.columns):
                ws.write(0, i, col, header_fmt)

            # Write data with conditional formatting
            for row_idx, (_, row) in enumerate(df.iterrows(), 1):
                for col_idx, col in enumerate(df.columns):
                    value = row[col]

                    # Choose formatting based on performance
                    if col in ['Model Better (Success)', 'Model Better (Efficiency)']:
                        if value == 'Yes':
                            cell_fmt = model_better_fmt
                        else:
                            cell_fmt = bo_better_fmt
                    elif col in ['Success Rate Diff', 'Efficiency Diff']:
                        if value > 0:
                            cell_fmt = model_better_fmt
                        elif value < 0:
                            cell_fmt = bo_better_fmt
                        else:
                            cell_fmt = neutral_fmt
                    else:
                        cell_fmt = neutral_fmt

                    ws.write(row_idx, col_idx, value, cell_fmt)

            # Auto-adjust column widths
            for idx, col in enumerate(df.columns):
                max_len = max(len(str(df[col].max())), len(col)) + 2
                ws.set_column(idx, idx, min(max_len, 20))

            # Add legend
            legend_row = len(df) + 3
            legend_fmt = wb.add_format({'italic': True, 'font_size': 9})

            ws.write(legend_row, 0, "Legend:", legend_fmt)
            ws.write(legend_row + 1, 0, "Green: Model performs better", model_better_fmt)
            ws.write(legend_row + 2, 0, "Red: Bayesian Optimization performs better", bo_better_fmt)

        print(f"heat_1d_bo comparison Excel saved to: {excel_path}")

    def create_heat_1d_bo_comparison_plots(self, heat_1d_df: pd.DataFrame, heat_1d_bo_df: pd.DataFrame, output_dir: Path) -> None:
        """Create comparison line plots between models and Bayesian optimization."""
        precision_levels = ['low', 'medium', 'high']
        metrics = ['Success Rate', 'Efficiency']
        inference_modes = ['Zero-shot', 'Iterative']

        # Get unique models
        models = sorted(heat_1d_df['Model'].unique())

        # Define three colors: blue, green, pink
        base_colors = ['#1f77b4', '#2ca02c', '#ff69b4']  # Blue, Green, Pink

        # Define markers: hollow circle and hollow square only
        base_markers = ['o', 's']

        # Define line styles: solid, dashed with shorter segments
        line_styles = ['-', (0, (3, 2))]  # solid, short dashed (3pt dash, 2pt gap)

        # Create style mapping for models (similar to the original line plot)
        model_styles = {}
        for i, model in enumerate(models):
            color_idx = i % len(base_colors)
            marker_idx = i % len(base_markers)
            line_idx = (i // len(base_colors)) % len(line_styles)

            model_styles[model] = {
                'color': base_colors[color_idx],
                'marker': base_markers[marker_idx],
                'linestyle': line_styles[line_idx]
            }

        # Add Bayesian Optimization style
        bo_style = {
            'color': '#FF6B35',
            'marker': 'D',
            'linestyle': '-'
        }

        # Create single figure with 4 subplots (1x4 horizontal layout)
        fig = plt.figure(figsize=(16, 5))

        # Create custom layout: legend at top, plots in middle, labels at bottom
        gs = fig.add_gridspec(3, 4, height_ratios=[0.1, 1, 0.25], hspace=0.2, wspace=0.18)

        # Create legend axis (spans full width at top)
        legend_ax = fig.add_subplot(gs[0, :])

        # Create 4 subplot axes (1x4 horizontal grid)
        axes = {
            ('Zero-shot', 'Success Rate'): fig.add_subplot(gs[1, 0]),
            ('Zero-shot', 'Efficiency'): fig.add_subplot(gs[1, 1]),
            ('Iterative', 'Success Rate'): fig.add_subplot(gs[1, 2]),
            ('Iterative', 'Efficiency'): fig.add_subplot(gs[1, 3])
        }

        # Create bottom label axes
        label_axes = [
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
            fig.add_subplot(gs[2, 3])
        ]

        # Plot data for all mode-metric combinations
        for mode in inference_modes:
            for metric in metrics:
                ax = axes[(mode, metric)]

                # Plot model lines for this specific inference mode
                for model in models:
                    x_values = []
                    y_values = []

                    for precision_idx, precision in enumerate(precision_levels):
                        # Get model data for this precision level and specific inference mode
                        model_data = heat_1d_df[
                            (heat_1d_df['Model'] == model) &
                            (heat_1d_df['Precision Level'] == precision) &
                            (heat_1d_df['Inference Mode'] == mode)
                        ]
                        if len(model_data) > 0:
                            if metric == 'Success Rate':
                                value = model_data['success_rate'].iloc[0]
                            else:  # Efficiency
                                value = model_data['mean_efficiency'].iloc[0]

                            x_values.append(precision_idx)
                            y_values.append(value)

                    # Plot line for this model
                    if len(x_values) > 0:
                        style = model_styles[model]

                        ax.plot(x_values, y_values,
                               marker=style['marker'],
                               color=style['color'],
                               linestyle=style['linestyle'],
                               linewidth=1.0,
                               markersize=3,
                               markerfacecolor='white',
                               markeredgecolor=style['color'],
                               markeredgewidth=1.0,
                               alpha=1.0,
                               clip_on=False)

                # Plot Bayesian Optimization line (same for all modes)
                bo_x_values = []
                bo_y_values = []

                for precision_idx, precision in enumerate(precision_levels):
                    bo_data = heat_1d_bo_df[heat_1d_bo_df['Precision Level'] == precision]
                    if len(bo_data) > 0:
                        if metric == 'Success Rate':
                            bo_value = bo_data['success_rate'].iloc[0]
                        else:  # Efficiency
                            bo_value = bo_data['mean_efficiency'].iloc[0]

                        bo_x_values.append(precision_idx)
                        bo_y_values.append(bo_value)

                if len(bo_x_values) > 0:
                    ax.plot(bo_x_values, bo_y_values,
                           marker=bo_style['marker'],
                           color=bo_style['color'],
                           linestyle=bo_style['linestyle'],
                           linewidth=2.0,
                           markersize=4,
                           markerfacecolor=bo_style['color'],
                           markeredgecolor=bo_style['color'],
                           markeredgewidth=1.0,
                           alpha=1.0,
                           clip_on=False)

                # Customize subplot
                ax.set_xlabel('Accuracy Level', fontsize=12)
                ax.set_ylabel(metric, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='gray')

                # Set x-axis labels and ticks with tighter spacing and extra padding for markers
                ax.set_xticks(range(len(precision_levels)))
                ax.set_xticklabels([p.capitalize() for p in precision_levels], fontsize=10)
                ax.set_xlim(-0.1, len(precision_levels) - 0.9)

                # Set y-axis limits with more padding to prevent marker cropping
                all_y_values = []

                # Collect all model values for this specific mode
                for model in models:
                    for precision in precision_levels:
                        model_data = heat_1d_df[
                            (heat_1d_df['Model'] == model) &
                            (heat_1d_df['Precision Level'] == precision) &
                            (heat_1d_df['Inference Mode'] == mode)
                        ]
                        if len(model_data) > 0:
                            if metric == 'Success Rate':
                                all_y_values.append(model_data['success_rate'].iloc[0])
                            else:
                                all_y_values.append(model_data['mean_efficiency'].iloc[0])

                # Collect BO values
                for precision in precision_levels:
                    bo_data = heat_1d_bo_df[heat_1d_bo_df['Precision Level'] == precision]
                    if len(bo_data) > 0:
                        if metric == 'Success Rate':
                            all_y_values.append(bo_data['success_rate'].iloc[0])
                        else:
                            all_y_values.append(bo_data['mean_efficiency'].iloc[0])

                if all_y_values:
                    y_min = min(all_y_values)
                    y_max = max(all_y_values)
                    y_range = y_max - y_min

                    if metric == 'Success Rate':
                        # Allow negative lower bound to show markers at 0 values properly
                        lower_bound = y_min - 0.08
                        if y_min <= 0.05:  # If minimum value is close to 0
                            lower_bound = -0.05
                        ax.set_ylim(lower_bound, min(1, y_max + 0.08))
                    else:  # Efficiency
                        # Allow negative lower bound to show markers at 0 values properly
                        lower_bound = y_min - 0.5
                        if y_min <= 1.0:  # If minimum value is close to 0
                            lower_bound = -1.0
                        ax.set_ylim(lower_bound, y_max + max(0.5, y_range * 0.1))

        # Create custom legend in top area
        legend_ax.axis('off')

        # Create legend elements for all models + BO
        legend_elements = []
        for model in models:
            style = model_styles[model]
            from matplotlib.lines import Line2D

            legend_elements.append(Line2D([0], [0],
                                         marker=style['marker'],
                                         color=style['color'],
                                         linestyle=style['linestyle'],
                                         linewidth=1.0,
                                         markersize=3,
                                         markerfacecolor='white',
                                         markeredgecolor=style['color'],
                                         markeredgewidth=1.0,
                                         label=model))

        # Add Bayesian Optimization to legend
        legend_elements.append(Line2D([0], [0],
                                     marker=bo_style['marker'],
                                     color=bo_style['color'],
                                     linestyle=bo_style['linestyle'],
                                     linewidth=2.0,
                                     markersize=4,
                                     markerfacecolor=bo_style['color'],
                                     markeredgecolor=bo_style['color'],
                                     markeredgewidth=1.0,
                                     label='Bayesian Optimization'))

        # Display legend horizontally
        ncols = len(legend_elements)
        if len(legend_elements) > 6:  # If too many models, split into 2 rows
            ncols = 4

        legend_ax.legend(handles=legend_elements,
                       loc='center',
                       ncol=ncols,
                       frameon=False,
                       fontsize=11,
                       columnspacing=2.0,
                       handlelength=2.5)

        # Add bottom labels for each subplot
        subplot_labels = [
            ('a', 'Zero-shot - Success Rate'),
            ('b', 'Zero-shot - Efficiency'),
            ('c', 'Iterative - Success Rate'),
            ('d', 'Iterative - Efficiency')
        ]

        for i, (letter, description) in enumerate(subplot_labels):
            label_ax = label_axes[i]
            label_ax.axis('off')
            label_ax.text(0.5, 0.5, f'({letter}) {description}',
                         ha='center', va='center',
                         fontsize=13, fontweight='bold',
                         transform=label_ax.transAxes)

        # Save line plot with extra padding
        plot_path = output_dir / "heat_1d_bo_comparison_line_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()

        print(f"heat_1d_bo comparison line plot saved to: {plot_path}")

    def generate_overall_statistics(self) -> None:
        """Generate comprehensive overall statistics and visualizations."""
        # Special handling for heat_1d_bo comparison
        if self.target_dataset == 'heat_1d_bo':
            self.generate_heat_1d_bo_comparison()
            return

        print("🚀 Starting overall statistics generation...")

        try:
            # Load all datasets
            combined_df = self.load_all_datasets()

            # Calculate detailed statistics (with precision levels)
            print("\n📊 Calculating detailed statistics...")
            overall_stats = self.calculate_overall_statistics(combined_df)

            # Calculate aggregated statistics (without precision levels)
            print("📊 Calculating aggregated statistics...")
            aggregated_stats = self.calculate_aggregated_statistics(combined_df)

            # Save detailed results
            print("\n💾 Saving detailed results...")
            self.save_csv_results(overall_stats)
            self.save_excel_results(overall_stats)

            # Save aggregated results
            print("💾 Saving aggregated results...")
            self.save_aggregated_csv_results(aggregated_stats)
            self.save_aggregated_excel_results(aggregated_stats)

            # Generate detailed visualizations
            print("\n📈 Generating detailed visualizations...")
            self.create_success_rate_chart(overall_stats)
            self.create_efficiency_chart(overall_stats)

            # Generate aggregated visualizations
            print("📈 Generating aggregated visualizations...")
            self.create_aggregated_success_rate_chart(aggregated_stats)
            self.create_aggregated_efficiency_chart(aggregated_stats)

            # Generate combined line plots
            print("📈 Generating combined line plots...")
            self.create_combined_line_plots(overall_stats)

            # Print summary
            print(f"\n🎉 Overall statistics generation completed!")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📋 Processed {len(overall_stats)} detailed model configurations and {len(aggregated_stats)} aggregated model configurations")
            print(f"📋 Across {len(combined_df['Simulation'].unique())} datasets")
            print(f"🔍 Available datasets: {', '.join(sorted(combined_df['Simulation'].unique()))}")
            print(f"📊 Generated both detailed (with precision levels) and aggregated (without precision levels) outputs")

        except Exception as e:
            print(f"❌ Error generating overall statistics: {e}")
            raise


def main():
    """Main function to generate overall performance statistics."""
    parser = argparse.ArgumentParser(description='Generate overall performance statistics for SimulCost-Bench')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                       help='Specific dataset to process (e.g., heat_1d_bo for model vs BO comparison)')

    args = parser.parse_args()

    # Create generator with target dataset
    generator = OverallStatsGenerator(target_dataset=args.dataset)
    generator.generate_overall_statistics()


if __name__ == "__main__":
    main()