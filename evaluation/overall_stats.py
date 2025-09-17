#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overall Performance Statistics Generator for SimulCost-Bench

This script aggregates model performance across all datasets in SimulCost-Bench,
providing comprehensive statistics and visualizations for model comparison.

Usage
-----
python evaluation/overall_stats.py

Output: Creates comprehensive statistics and visualizations in eval_results/overall/:
- overall_summary.csv (aggregated performance across all datasets)
- overall_summary.xlsx (beautifully formatted Excel file)
- success_rate_overall.png (success rate bar chart)
- efficiency_overall.png (efficiency bar chart)
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from statistics import mean


class OverallStatsGenerator:
    """Generate comprehensive statistics and visualizations across all datasets."""

    def __init__(self, base_dir: str = "eval_results", output_dir: str = "eval_results/overall"):
        """
        Initialize the overall statistics generator.

        Args:
            base_dir: Base directory containing individual dataset results
            output_dir: Directory to save overall statistics and visualizations
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up professional plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")


    def find_available_datasets(self) -> List[str]:
        """
        Find all available datasets with summary CSV files.

        Returns:
            List of dataset names that have summary files
        """
        datasets = []
        if not self.base_dir.exists():
            return datasets

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
        Calculate overall statistics aggregated across all datasets.

        Args:
            df: Combined DataFrame from all datasets

        Returns:
            DataFrame with overall statistics per model/mode/precision combination
        """
        # Model names are already cleaned by upstream processing

        # Group by Model, Precision Level, and Inference Mode
        # Calculate weighted averages across all simulations
        aggregated_results = []

        grouping_cols = ['Model', 'Precision Level', 'Inference Mode']
        for group_keys, group in df.groupby(grouping_cols):
            model, precision, mode = group_keys

            # Calculate total samples across all simulations
            total_samples = group['Number of Samples'].sum()

            # Number of simulations this model was tested on
            num_simulations = len(group['Simulation'].unique())

            # Calculate weighted averages for metrics
            weights = group['Number of Samples'].tolist()

            # Success rate: weighted average across all simulations
            success_values = [float(val) for val in group['success_rate'] if pd.notnull(val)]
            success_weights = [weights[i] for i, val in enumerate(group['success_rate']) if pd.notnull(val)]
            avg_success_rate = self._weighted_average(success_values, success_weights) if success_values else 0.0

            # Efficiency: weighted average across all simulations
            efficiency_values = [float(val) for val in group['mean_efficiency'] if pd.notnull(val)]
            efficiency_weights = [weights[i] for i, val in enumerate(group['mean_efficiency']) if pd.notnull(val)]
            avg_efficiency = self._weighted_average(efficiency_values, efficiency_weights) if efficiency_values else 0.0

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

    def _weighted_average(self, values: List[float], weights: List[int]) -> float:
        """Calculate weighted average."""
        if not values or not weights or sum(weights) == 0:
            return 0.0
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    def calculate_aggregated_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregated statistics without precision level dimension.
        Aggregates across all precision levels for each model/mode combination.

        Args:
            df: Combined DataFrame from all datasets

        Returns:
            DataFrame with aggregated statistics per model/mode combination (no precision level)
        """
        # Model names are already cleaned by upstream processing

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

            # Calculate weighted averages for metrics
            weights = group['Number of Samples'].tolist()

            # Success rate: weighted average across all simulations and precision levels
            success_values = [float(val) for val in group['success_rate'] if pd.notnull(val)]
            success_weights = [weights[i] for i, val in enumerate(group['success_rate']) if pd.notnull(val)]
            avg_success_rate = self._weighted_average(success_values, success_weights) if success_values else 0.0

            # Efficiency: weighted average across all simulations and precision levels
            efficiency_values = [float(val) for val in group['mean_efficiency'] if pd.notnull(val)]
            efficiency_weights = [weights[i] for i, val in enumerate(group['mean_efficiency']) if pd.notnull(val)]
            avg_efficiency = self._weighted_average(efficiency_values, efficiency_weights) if efficiency_values else 0.0

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
    generator = OverallStatsGenerator()
    generator.generate_overall_statistics()


if __name__ == "__main__":
    main()