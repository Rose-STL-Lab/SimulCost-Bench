#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Optimization Analysis Tool for SimulCost-Bench

This script provides comprehensive analysis and comparison between models and
Bayesian optimization results for specific datasets in SimulCost-Bench.

Usage
-----
python evaluation/stats_utils/bayesian_optimization_analysis.py -d heat_1d
python evaluation/stats_utils/bayesian_optimization_analysis.py -d euler_1d

Output: Creates comparison results in eval_results/stats/bayesian_optimization/:
- {dataset}_bo_comparison.csv (model vs Bayesian optimization comparison)
- {dataset}_bo_comparison.xlsx (formatted comparison table)
- {dataset}_bo_comparison_line_plot.png (comparison visualizations)
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional import for smart text positioning
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("Warning: adjustText library not found. Text labels may overlap. Install with: pip install adjustText")


class BayesianOptimizationAnalyzer:
    """Analyze and compare model performance with Bayesian optimization."""

    def __init__(self, base_dir: str = "eval_results", dataset_name: str = None):
        """
        Initialize the Bayesian optimization analyzer.

        Args:
            base_dir: Base directory containing individual dataset results
            dataset_name: Specific dataset to analyze (e.g., 'heat_1d', 'euler_1d')
        """
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name

        # Create output directory for Bayesian optimization analysis
        self.output_dir = Path("eval_results/stats/bayesian_optimization")
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def load_bayesian_optimization_data(self, dataset_name: str) -> tuple[pd.DataFrame, str]:
        """
        Load bayesian optimization data for a specific dataset and convert to standard format.

        Args:
            dataset_name: Name of the dataset (e.g., 'heat_1d', 'euler_1d')

        Returns:
            Tuple of (DataFrame with bayesian optimization data in standard format, task_name)
        """
        bo_records = []
        task_name = None

        # Find all BO files for this dataset to determine task name
        bo_dir = self.base_dir / "bayesian_optimization"
        if bo_dir.exists():
            for bo_file in bo_dir.glob(f"{dataset_name}_*_iterative.csv"):
                # Extract task name from filename: {dataset}_{task}_iterative.csv
                filename_parts = bo_file.stem.split('_')
                dataset_parts = dataset_name.split('_')
                if len(filename_parts) >= len(dataset_parts) + 2:
                    # Remove dataset parts and mode suffix to get task name
                    task_name = '_'.join(filename_parts[len(dataset_parts):-1])
                    break

        if task_name is None:
            task_name = "unknown"

        # Load both iterative and zero-shot files
        for mode in ['iterative', 'zero_shot']:
            bo_file = self.base_dir / "bayesian_optimization" / f"{dataset_name}_{task_name}_{mode}.csv"

            if not bo_file.exists():
                continue

            df = pd.read_csv(bo_file, index_col=0)

            # Find BO_default row
            if 'BO_default' not in df.index:
                continue

            bo_row = df.loc['BO_default']

            # Extract data for each precision level
            precision_levels = ['low', 'medium', 'high']

            for precision in precision_levels:
                success_rate_col = f"{precision}_success_rate"
                efficiency_col = f"{precision}_mean_efficiency"

                if success_rate_col in bo_row.index and efficiency_col in bo_row.index:
                    success_rate = bo_row[success_rate_col]
                    efficiency = bo_row[efficiency_col]

                    # Skip if values are NaN
                    if pd.isna(success_rate) or pd.isna(efficiency):
                        continue

                    bo_records.append({
                        'Model': 'Bayesian Optimization',
                        'Precision Level': precision,
                        'Inference Mode': 'Iterative' if mode == 'iterative' else 'Zero-shot',
                        'Number of Samples': 100,  # Default value
                        'Simulation': dataset_name,
                        'success_rate': float(success_rate),
                        'mean_efficiency': float(efficiency)
                    })

        return pd.DataFrame(bo_records), task_name

    def clean_model_name(self, model_name: str) -> str:
        """
        Clean model name using the name mapping.

        Args:
            model_name: Raw model name from the data

        Returns:
            Clean, display-friendly model name
        """
        return self.name_mapping.get(model_name, model_name)

    def load_model_data(self, dataset_name: str, task_name: str) -> pd.DataFrame:
        """
        Load model data for the specified dataset and task across all precision levels.

        Args:
            dataset_name: Name of the dataset
            task_name: Name of the specific task

        Returns:
            DataFrame containing model performance data for the specific task across all precision levels
        """
        all_data = []
        precision_levels = ['low', 'medium', 'high']

        for precision in precision_levels:
            model_file = self.base_dir / dataset_name / f"{dataset_name}_{precision}_summary.csv"

            if not model_file.exists():
                print(f"Warning: {precision} precision file not found: {model_file}")
                continue

            df = pd.read_csv(model_file)

            # Filter for the specific task if task column exists
            if 'Task' in df.columns and task_name != "unknown":
                df = df[df['Task'] == task_name].copy()

            if len(df) == 0:
                print(f"Warning: No data found for task '{task_name}' in {precision} precision file")
                continue

            # Add metadata columns
            df['Simulation'] = dataset_name
            df['Precision Level'] = precision

            all_data.append(df)

        if not all_data:
            # Fallback to the averaged summary file if no detailed files exist
            model_file = self.base_dir / dataset_name / f"{dataset_name}_sum.csv"
            if model_file.exists():
                df = pd.read_csv(model_file)
                all_data.append(df)
            else:
                raise ValueError(f"No model files found for dataset: {dataset_name}")

        # Combine all precision levels
        combined_df = pd.concat(all_data, ignore_index=True)

        # Apply model name mapping
        if 'Model' in combined_df.columns:
            combined_df['Model'] = combined_df['Model'].apply(self.clean_model_name)

        return combined_df

    def create_bo_comparison_table(self, model_df: pd.DataFrame, bo_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Create comparison table between models and Bayesian optimization."""
        comparison_data = []

        # Get available precision levels from both model and BO data
        model_precision_levels = set()
        if 'Precision Level' in model_df.columns:
            model_precision_levels = set(model_df['Precision Level'].unique())
        else:
            # If no precision level in model data, assume high precision
            model_precision_levels = {'high'}

        bo_precision_levels = set()
        if 'Precision Level' in bo_df.columns:
            bo_precision_levels = set(bo_df['Precision Level'].unique())
        else:
            # Map from BO data columns which have precision prefixes
            bo_precision_levels = {'low', 'medium', 'high'}

        # Use intersection of available precision levels in the correct order
        available_precision = model_precision_levels.intersection(bo_precision_levels)
        precision_levels = [p for p in ['low', 'medium', 'high'] if p in available_precision]
        if not precision_levels:
            # If no intersection, use high as default
            precision_levels = ['high']

        for precision in precision_levels:
            # Get Bayesian optimization results for this precision (average across inference modes)
            if 'Precision Level' in bo_df.columns:
                bo_precision_data = bo_df[bo_df['Precision Level'] == precision]
            else:
                bo_precision_data = bo_df  # Use all BO data if no precision level column

            if len(bo_precision_data) == 0:
                continue

            bo_success_rate = bo_precision_data['success_rate'].mean()
            bo_efficiency = bo_precision_data['mean_efficiency'].mean()

            # Get model results for this precision level
            if 'Precision Level' in model_df.columns:
                precision_models = model_df[model_df['Precision Level'] == precision]
            else:
                precision_models = model_df  # Use all model data if no precision level column

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
        if len(comparison_df) > 0:
            precision_order = {'low': 0, 'medium': 1, 'high': 2}
            comparison_df['precision_order'] = comparison_df['Precision Level'].map(precision_order)
            comparison_df = comparison_df.sort_values(['precision_order', 'Model']).drop(['precision_order'], axis=1)

        return comparison_df

    def save_bo_comparison_results(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Save bayesian optimization comparison results to CSV and Excel."""
        # Save CSV
        csv_path = self.output_dir / f"{dataset_name}_bo_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"{dataset_name}_bo comparison CSV saved to: {csv_path}")

        # Save Excel with formatting
        excel_path = self.output_dir / f"{dataset_name}_bo_comparison.xlsx"

        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            wb = writer.book
            ws = wb.add_worksheet(f'{dataset_name}_bo Comparison')
            writer.sheets[f'{dataset_name}_bo Comparison'] = ws

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

        print(f"{dataset_name}_bo comparison Excel saved to: {excel_path}")

    def create_bo_comparison_plots(self, model_df: pd.DataFrame, bo_df: pd.DataFrame, dataset_name: str, task_name: str) -> None:
        """Create comparison line plots between models and Bayesian optimization."""
        # Get available precision levels from the data
        model_precision_levels = set()
        if 'Precision Level' in model_df.columns:
            model_precision_levels = set(model_df['Precision Level'].unique())
        else:
            model_precision_levels = {'high'}  # Default to high if no precision level

        bo_precision_levels = set()
        if 'Precision Level' in bo_df.columns:
            bo_precision_levels = set(bo_df['Precision Level'].unique())
        else:
            bo_precision_levels = {'low', 'medium', 'high'}  # BO data typically has all levels

        # Use intersection or default to available levels in the correct order
        available_precision = model_precision_levels.intersection(bo_precision_levels)
        precision_levels = [p for p in ['low', 'medium', 'high'] if p in available_precision]
        if not precision_levels:
            precision_levels = ['high']  # Default fallback

        metrics = ['Success Rate', 'Efficiency']

        # Get available inference modes
        inference_modes = set()
        if 'Inference Mode' in model_df.columns:
            inference_modes.update(model_df['Inference Mode'].unique())
        if 'Inference Mode' in bo_df.columns:
            inference_modes.update(bo_df['Inference Mode'].unique())

        # Default to common modes if none found
        if not inference_modes:
            inference_modes = {'Zero-shot', 'Iterative'}
        else:
            inference_modes = sorted(list(inference_modes))

        # Get unique models
        models = sorted(model_df['Model'].unique())

        # Define three colors: blue, green, pink
        base_colors = ['#1f77b4', '#2ca02c', '#ff69b4']  # Blue, Green, Pink

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
                        model_data = model_df[
                            (model_df['Model'] == model) &
                            (model_df['Precision Level'] == precision) &
                            (model_df['Inference Mode'] == mode)
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
                    bo_precision_data = bo_df[
                        (bo_df['Precision Level'] == precision) &
                        (bo_df['Inference Mode'] == mode)
                    ]
                    if len(bo_precision_data) > 0:
                        if metric == 'Success Rate':
                            bo_value = bo_precision_data['success_rate'].iloc[0]
                        else:  # Efficiency
                            bo_value = bo_precision_data['mean_efficiency'].iloc[0]

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
                        model_data = model_df[
                            (model_df['Model'] == model) &
                            (model_df['Precision Level'] == precision) &
                            (model_df['Inference Mode'] == mode)
                        ]
                        if len(model_data) > 0:
                            if metric == 'Success Rate':
                                all_y_values.append(model_data['success_rate'].iloc[0])
                            else:
                                all_y_values.append(model_data['mean_efficiency'].iloc[0])

                # Collect BO values for this specific mode
                for precision in precision_levels:
                    bo_data = bo_df[
                        (bo_df['Precision Level'] == precision) &
                        (bo_df['Inference Mode'] == mode)
                    ]
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

        # Add bottom labels for each subplot with new format
        subplot_labels = [
            ('a', f'Zero-shot - {task_name}'),
            ('b', f'Zero-shot - {task_name}'),
            ('c', f'Iterative - {task_name}'),
            ('d', f'Iterative - {task_name}')
        ]

        for i, (letter, description) in enumerate(subplot_labels):
            label_ax = label_axes[i]
            label_ax.axis('off')
            label_ax.text(0.5, 0.5, f'({letter}) {description}',
                         ha='center', va='center',
                         fontsize=13, fontweight='bold',
                         transform=label_ax.transAxes)

        # Save line plot with extra padding
        plot_path = self.output_dir / f"{dataset_name}_bo_comparison_line_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()

        print(f"{dataset_name}_bo comparison line plot saved to: {plot_path}")

    def analyze_bayesian_optimization(self) -> None:
        """Perform comprehensive Bayesian optimization analysis for the specified dataset."""
        if not self.dataset_name:
            raise ValueError("Dataset name must be specified for Bayesian optimization analysis")

        print(f"🚀 Starting Bayesian optimization analysis for {self.dataset_name}...")

        try:
            # Load Bayesian optimization data first to get the task name
            bo_df, task_name = self.load_bayesian_optimization_data(self.dataset_name)

            # Load model data using the task name
            model_df = self.load_model_data(self.dataset_name, task_name)

            print(f"  - Loaded {len(model_df)} records from {self.dataset_name} models (task: {task_name})")
            print(f"  - Loaded {len(bo_df)} records from {self.dataset_name} bayesian optimization (task: {task_name})")

            # Generate comparison statistics and visualizations
            comparison_df = self.create_bo_comparison_table(model_df, bo_df, self.dataset_name)

            # Save results
            self.save_bo_comparison_results(comparison_df, self.dataset_name)
            self.create_bo_comparison_plots(model_df, bo_df, self.dataset_name, task_name)

            print(f"\n🎉 Bayesian optimization analysis for {self.dataset_name} completed!")
            print(f"📁 Results saved to: {self.output_dir}")

        except Exception as e:
            print(f"❌ Error performing Bayesian optimization analysis: {e}")
            raise


def discover_datasets(base_dir: str = "eval_results") -> List[str]:
    """
    Automatically discover available datasets by scanning the bayesian optimization directory.

    Args:
        base_dir: Base directory containing evaluation results

    Returns:
        List of dataset names found in the bayesian optimization directory
    """
    datasets = set()
    bo_dir = Path(base_dir) / "bayesian_optimization"

    if bo_dir.exists():
        # Look for all BO files and extract dataset names
        for bo_file in bo_dir.glob("*.csv"):
            # Files are named like: {dataset}_{task}_{mode}.csv
            # Need to extract the dataset part correctly
            filename_parts = bo_file.stem.split('_')
            if len(filename_parts) >= 3:
                # Try to determine dataset name by checking common patterns
                # Most datasets end with dimension info like "_1d", "_2d"
                if len(filename_parts) >= 4 and filename_parts[2] in ['1d', '2d', '3d']:
                    # Three-part dataset name: e.g., "ns_transient_2d"
                    dataset_name = f"{filename_parts[0]}_{filename_parts[1]}_{filename_parts[2]}"
                elif len(filename_parts) >= 3 and filename_parts[1] in ['1d', '2d', '3d']:
                    # Two-part dataset name: e.g., "heat_1d"
                    dataset_name = f"{filename_parts[0]}_{filename_parts[1]}"
                else:
                    # Fallback: use first two parts
                    dataset_name = f"{filename_parts[0]}_{filename_parts[1]}"
                datasets.add(dataset_name)

    return sorted(list(datasets))


def main():
    """Main function to perform Bayesian optimization analysis."""
    parser = argparse.ArgumentParser(description='Perform Bayesian optimization analysis for SimulCost-Bench datasets')
    parser.add_argument('-d', '--dataset', type=str, required=False,
                       help='Specific dataset to analyze (e.g., heat_1d, euler_1d). If not provided, analyzes all available datasets.')

    args = parser.parse_args()

    # Discover available datasets
    available_datasets = discover_datasets()

    if not available_datasets:
        print("❌ No datasets found in eval_results/bayesian_optimization/")
        return

    print(f"📊 Found {len(available_datasets)} datasets: {', '.join(available_datasets)}")

    # Determine which datasets to analyze
    if args.dataset:
        if args.dataset not in available_datasets:
            print(f"❌ Dataset '{args.dataset}' not found. Available datasets: {', '.join(available_datasets)}")
            return
        datasets_to_analyze = [args.dataset]
    else:
        datasets_to_analyze = available_datasets

    # Analyze each dataset
    for dataset in datasets_to_analyze:
        print(f"\n{'='*60}")
        print(f"🔍 Analyzing dataset: {dataset}")
        print(f"{'='*60}")

        try:
            analyzer = BayesianOptimizationAnalyzer(dataset_name=dataset)
            analyzer.analyze_bayesian_optimization()
        except Exception as e:
            print(f"❌ Failed to analyze {dataset}: {e}")
            continue

    print(f"\n🎉 Bayesian optimization analysis completed for {len(datasets_to_analyze)} dataset(s)!")
    print(f"📁 Results saved to: eval_results/stats/bayesian_optimization/")


if __name__ == "__main__":
    main()