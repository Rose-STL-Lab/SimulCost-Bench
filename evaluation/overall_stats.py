#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overall Performance Statistics Generator for SimulCost-Bench

This script aggregates model performance across all datasets in SimulCost-Bench,
providing comprehensive statistics and visualizations for model comparison using simple averages.

Usage
-----
python evaluation/overall_stats.py

Output: Creates comprehensive statistics and visualizations in eval_results/overall/:
- overall_summary.csv (aggregated performance across all datasets)
- overall_summary.xlsx (beautifully formatted Excel file)
- success_rate_overall.png (success rate bar chart)
- efficiency_overall.png (efficiency bar chart)
- line_plot_success_rate.png (combined zero-shot & iterative success rate line plots)
- line_plot_efficiency.png (combined zero-shot & iterative efficiency line plots)

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
            'anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude-3.7-Sonnet',
            'meta.llama3-70b-instruct-v1:0': 'Llama-3-70B-Instruct',
            'gpt-5-2025-08-07': 'GPT-5',
            'qwen3_32b': 'Qwen3-32B',

            # 'amazon.nova-premier-v1:0': 'Nova-Premier',
            # 'mistral.mistral-large-2402-v1:0': 'Mistral-Large',
            # 'qwen3_0_6b': 'Qwen3-0.6B',
            # 'qwen3_8b': 'Qwen3-8B',
            # 'anthropic.claude-3-5-haiku-20241022-v1:0': 'Claude-3.5-Haiku',
            # 'anthropic.claude-3-5-sonnet-20240620-v1:0': 'Claude-3.5-Sonnet',
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
        Find available datasets from fixed paths.
        If target_dataset is specified, only include that dataset.

        Returns:
            List of dataset names
        """
        # Fixed dataset paths to read from (now read from merged_results.parquet)
        fixed_datasets = ['burgers_1d', 'diff_react_1d', 'euler_1d', 'heat_1d', 'heat_2d', 'ns_transient_2d', 'mpm_2d']

        # If target dataset is specified, only return that one
        if self.target_dataset:
            if self.target_dataset in fixed_datasets:
                return [self.target_dataset]
            return []

        # Return all fixed datasets
        return sorted(fixed_datasets)

    def load_all_datasets(self) -> pd.DataFrame:
        """
        Load and combine evaluation results from all available datasets.
        Reads from merged_results.parquet and aggregates data.
        Only includes data from specific target models.

        Returns:
            Combined DataFrame containing aggregated results from all datasets
        """
        datasets = self.find_available_datasets()
        if not datasets:
            raise ValueError("No datasets found")

        print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")

        # Define target models to include (matching names in parquet file)
        target_models = [
            'Claude-3.7-Sonnet',
            'Llama-3-70B-Instruct',
            'GPT-5',
            'Qwen3-32B'
        ]

        # Read merged results from parquet
        parquet_path = self.base_dir / "merged_results.parquet"
        if not parquet_path.exists():
            raise ValueError(f"Merged results file not found: {parquet_path}")

        print(f"Loading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        # Filter to target datasets only (exclude ICL variants and epoch_1d)
        df = df[df['dataset'].isin(datasets)]
        print(f"  - Filtered to {len(datasets)} target datasets")

        # Filter to target models only
        df = df[df['model_name'].isin(target_models)]
        print(f"  - Filtered to {len(target_models)} target models")

        if len(df) == 0:
            raise ValueError("No data found after filtering")

        # Convert inference_mode format: zero_shot -> Zero-shot, iterative -> Iterative
        inference_mode_mapping = {
            'zero_shot': 'Zero-shot',
            'iterative': 'Iterative'
        }
        df['inference_mode'] = df['inference_mode'].map(inference_mode_mapping)

        print(f"  - Total filtered records: {len(df)}")

        # First-level aggregation: calculate metrics per (dataset, model, precision, mode, task)
        print("  - Performing first-level aggregation by task...")
        task_level = df.groupby(['dataset', 'model_name', 'precision_level', 'inference_mode', 'task']).agg({
            'is_successful': 'mean',  # success rate per task
            'efficiency': 'mean',      # mean efficiency per task
            'qid': 'count'             # number of samples per task
        }).reset_index()

        task_level.rename(columns={
            'is_successful': 'success_rate_task',
            'efficiency': 'mean_efficiency_task',
            'qid': 'num_samples_task'
        }, inplace=True)

        # Second-level aggregation: average across tasks
        print("  - Performing second-level aggregation across tasks...")
        aggregated_results = []

        for (dataset, model, precision, mode), group in task_level.groupby(
            ['dataset', 'model_name', 'precision_level', 'inference_mode']
        ):
            # Calculate simple averages across tasks
            avg_success_rate = group['success_rate_task'].mean()
            avg_efficiency = group['mean_efficiency_task'].mean()
            total_samples = group['num_samples_task'].sum()

            aggregated_results.append({
                'Model': model,
                'Simulation': dataset,
                'Precision Level': precision,
                'Inference Mode': mode,
                'Number of Samples': int(total_samples),
                'success_rate': round(avg_success_rate, 2),
                'mean_efficiency': round(avg_efficiency, 2)
            })

        combined_df = pd.DataFrame(aggregated_results)
        print(f"Total aggregated records: {len(combined_df)}")

        # Print summary by dataset
        for dataset in datasets:
            dataset_records = len(combined_df[combined_df['Simulation'] == dataset])
            print(f"  - {dataset}: {dataset_records} aggregated records")

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
        bar_width = 0.12 * 0.85
        precision_spacing = 0.12
        mode_spacing = 0.01
        model_spacing = 0.4

        # Colors
        zero_shot_color = '#4682B4'
        iterative_color = '#FF7F32'

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
                        else:  # Iterative
                            color = iterative_color

                        # Calculate position
                        precision_offset = j * (2 * bar_width + mode_spacing + precision_spacing)
                        mode_offset = k * (bar_width + mode_spacing)
                        bar_pos = x_positions[i] + precision_offset + mode_offset - (len(precision_levels) * (2 * bar_width + mode_spacing + precision_spacing) - precision_spacing) / 2

                        # Create bar with hatching for iterative mode
                        hatch = '///' if mode == 'Iterative' else None
                        ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                              edgecolor='black', linewidth=0.8, hatch=hatch)

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
        ax.set_ylim(0, max_val * 1.05)  # Add 10% padding to the maximum value
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0)  # Horizontal labels
        ax.grid(True, alpha=0.3)

        # Create custom legend with patterns
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=zero_shot_color, edgecolor='black', label='Zero-shot'),
            Patch(facecolor=iterative_color, edgecolor='black', hatch='///', label='Iterative')
        ]
        ax.legend(handles=legend_elements,
                 bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2,
                 fontsize=10, frameon=False)

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
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        # Get unique models and inference modes
        models = sorted(df['Model'].unique())
        inference_modes = ['Zero-shot', 'Iterative']

        # Chart parameters
        bar_width = 0.25
        model_spacing = 0.6

        # Colors
        zero_shot_color = '#4682B4'
        iterative_color = '#FF7F32'

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
                    else:  # Iterative
                        color = iterative_color

                    # Calculate position with spacing between bars
                    bar_spacing = 0.05  # Small gap between the two bars of same model
                    bar_pos = x_positions[i] + (j - 0.5) * (bar_width + bar_spacing)

                    # Create bar with hatching for iterative mode
                    hatch = '///' if mode == 'Iterative' else None
                    ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                          edgecolor='black', linewidth=0.8, hatch=hatch)

                    # Add value label
                    ax.text(bar_pos, value + 0.01, f'{value:.2f}',
                           ha='center', va='bottom', fontsize=8)

        # Customize chart
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max_val * 1.05)  # Add 15% padding to the maximum value
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0)
        ax.grid(True, alpha=0.3)

        # Create custom legend with patterns
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=zero_shot_color, edgecolor='black', label='Zero-shot'),
            Patch(facecolor=iterative_color, edgecolor='black', hatch='///', label='Iterative')
        ]
        ax.legend(handles=legend_elements,
                 bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2,
                 fontsize=10, frameon=False)

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
        bar_width = 0.12 * 0.85
        precision_spacing = 0.12
        mode_spacing = 0.01
        model_spacing = 0.4

        # Colors
        zero_shot_color = '#4682B4'
        iterative_color = '#FF7F32'

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
                        else:  # Iterative
                            color = iterative_color
                
                        # Calculate position
                        precision_offset = j * (2 * bar_width + mode_spacing + precision_spacing)
                        mode_offset = k * (bar_width + mode_spacing)
                        bar_pos = x_positions[i] + precision_offset + mode_offset - (len(precision_levels) * (2 * bar_width + mode_spacing + precision_spacing) - precision_spacing) / 2

                        # Create bar with hatching for iterative mode
                        hatch = '///' if mode == 'Iterative' else None
                        ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                              edgecolor='black', linewidth=0.8, hatch=hatch)

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
        ax.set_ylim(0, max_val * 1.05)  # Add 10% padding to the maximum value
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0)  # Horizontal labels
        ax.grid(True, alpha=0.3)

        # Create custom legend with patterns
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=zero_shot_color, edgecolor='black', label='Zero-shot'),
            Patch(facecolor=iterative_color, edgecolor='black', hatch='///', label='Iterative')
        ]
        ax.legend(handles=legend_elements,
                 bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2,
                 fontsize=10, frameon=False)

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
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        # Get unique models and inference modes
        models = sorted(df['Model'].unique())
        inference_modes = ['Zero-shot', 'Iterative']

        # Chart parameters
        bar_width = 0.25
        model_spacing = 0.6

        # Colors
        zero_shot_color = '#4682B4'
        iterative_color = '#FF7F32'

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
                    else:  # Iterative
                        color = iterative_color

                    # Calculate position with spacing between bars
                    bar_spacing = 0.05  # Small gap between the two bars of same model
                    bar_pos = x_positions[i] + (j - 0.5) * (bar_width + bar_spacing)

                    # Create bar with hatching for iterative mode
                    hatch = '///' if mode == 'Iterative' else None
                    ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                          edgecolor='black', linewidth=0.8, hatch=hatch)

                    # Add value label
                    ax.text(bar_pos, value + max_val * 0.01, f'{value:.2f}',
                           ha='center', va='bottom', fontsize=8)

        # Customize chart
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max_val * 1.05)  # Add 15% padding to the maximum value
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=0)
        ax.grid(True, alpha=0.3)

        # Create custom legend with patterns
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=zero_shot_color, edgecolor='black', label='Zero-shot'),
            Patch(facecolor=iterative_color, edgecolor='black', hatch='///', label='Iterative')
        ]
        ax.legend(handles=legend_elements,
                 bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2,
                 fontsize=10, frameon=False)

        plt.tight_layout()

        # Save chart
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Aggregated efficiency chart saved to: {output_path}")

    def create_combined_metric_plots(self, df: pd.DataFrame) -> None:
        """
        Create two combined line plots:
        - Success Rate: Zero-shot and Iterative side by side
        - Efficiency: Zero-shot and Iterative side by side
        Uses clean styling with two colors only.

        Args:
            df: DataFrame containing overall statistics
        """
        # Get unique models and precision levels
        models = sorted(df['Model'].unique())
        precision_levels = ['low', 'medium', 'high']
        inference_modes = ['Zero-shot', 'Iterative']

        # Define two colors: green and red
        base_colors = ['#2E8B57', '#B22222']  # SeaGreen, FireBrick

        # Define markers: hollow circle and hollow square only
        base_markers = ['o', 's']

        # Define line styles: solid, dashed with shorter segments
        line_styles = ['-', (0, (3, 2))]  # solid, short dashed (3pt dash, 2pt gap)

        # Create style mapping for models
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

        metrics = ['Success Rate', 'Efficiency']

        # Create two combined plots (one for each metric)
        for metric in metrics:
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            axes = [ax1, ax2]

            for mode_idx, mode in enumerate(inference_modes):
                ax = axes[mode_idx]
                ax.set_title(mode, fontsize=12, fontweight='bold')

                mode_data = df[df['Inference Mode'] == mode]

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
                               clip_on=False,
                               label=model)

                # Customize subplot
                ax.set_xlabel('Accuracy Level', fontsize=12)
                ax.set_ylabel(metric, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='gray')

                # Set x-axis labels and ticks with tighter spacing and extra padding for markers
                ax.set_xticks(range(len(precision_levels)))
                ax.set_xticklabels([p.capitalize() for p in precision_levels], fontsize=10)
                ax.set_xlim(-0.1, len(precision_levels) - 0.9)

                # Set y-axis limits with more padding to prevent marker cropping
                metric_data = df[metric]
                y_min = metric_data.min()
                y_max = metric_data.max()
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

            # Add shared legend at the top of the figure
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, fontsize=10,
                      bbox_to_anchor=(0.5, 0.96), loc='lower center', ncol=len(models))

            plt.tight_layout()

            # Save combined plot for this metric
            filename = f"line_plot_{metric.lower().replace(' ', '_')}.png"
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            print(f"{metric} combined plot saved to: {output_path}")


    def generate_overall_statistics(self) -> None:
        """Generate comprehensive overall statistics and visualizations."""
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

            # Generate combined metric line plots
            print("📈 Generating combined metric line plots...")
            self.create_combined_metric_plots(overall_stats)

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
                       help='Specific dataset to process')

    args = parser.parse_args()

    # Create generator with target dataset
    generator = OverallStatsGenerator(target_dataset=args.dataset)
    generator.generate_overall_statistics()


if __name__ == "__main__":
    main()