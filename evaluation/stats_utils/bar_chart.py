#!/usr/bin/env python3
"""
Bar Chart Generator for SimulCost-Bench Evaluation Results

This script generates professional bar charts from evaluation results CSV files.
It processes model performance metrics and creates visualizations for different
precision levels and inference modes.

Usage:
    python evaluation/stats_utils/bar_chart.py -d {dataset}
    python evaluation/stats_utils/bar_chart.py  # Process all available datasets
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


class BarChartGenerator:
    """Generate professional bar charts from evaluation results."""
    
    def __init__(self, base_dir: str = "eval_results", output_dir: str = "eval_results/stats/bar_chart"):
        """
        Initialize the bar chart generator.
        
        Args:
            base_dir: Base directory containing evaluation results
            output_dir: Directory to save generated charts
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
            List of dataset names
        """
        datasets = []
        if not self.base_dir.exists():
            return datasets
            
        for item in self.base_dir.iterdir():
            if item.is_dir():
                summary_file = item / f"{item.name}_sum.csv"
                if summary_file.exists():
                    datasets.append(item.name)
        
        return sorted(datasets)
    
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load evaluation results for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame containing the evaluation results, or None if not found
        """
        csv_path = self.base_dir / dataset_name / f"{dataset_name}_sum.csv"
        
        if not csv_path.exists():
            print(f"Warning: CSV file not found for dataset '{dataset_name}' at {csv_path}")
            return None
            
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} records for dataset '{dataset_name}'")
            return df
        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {e}")
            return None
    
    
    def create_success_rate_chart(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Create bar chart for success rates across models and configurations.
        All precision levels are shown in a single chart with 6 bars per model.

        Args:
            df: DataFrame containing evaluation results
            dataset_name: Name of the dataset
        """
        # Model name mapping for clean display names
        name_mapping = {
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
        }

        # Apply model name mapping
        df = df.copy()
        df['Model'] = df['Model'].apply(lambda x: name_mapping.get(x, x))

        # Create single figure for all precision levels
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Get unique models and precision levels
        models = sorted(df['Model'].unique())
        precision_levels = ['low', 'medium', 'high']  # Ensure consistent ordering
        inference_modes = ['Zero-shot', 'Iterative']

        # Prepare data for grouped bars
        bar_width = 0.12  # Width of individual bars
        precision_spacing = 0.12  # Spacing between different precision levels
        mode_spacing = 0.01  # Small spacing between zero-shot and iterative within same precision
        model_spacing = 0.32  # Spacing between different models

        # Fixed colors - light gray with stripes for Zero-shot, dark blue with dots for Iterative
        zero_shot_color = '#BBBBBB'  # Slightly darker gray
        iterative_color = '#4169E1'  # Dark blue

        # Get max value for y-axis scaling
        max_val = df['success_rate'].max()

        # Calculate model positions with spacing
        total_group_width = len(precision_levels) * (2 * bar_width + mode_spacing) + (len(precision_levels) - 1) * precision_spacing
        x_positions = np.arange(len(models)) * (total_group_width + model_spacing)
        precision_labels = ['L', 'M', 'H']

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]

            for j, precision in enumerate(precision_levels):
                precision_data = model_data[model_data['Precision Level'] == precision]

                for k, mode in enumerate(inference_modes):
                    mode_data = precision_data[precision_data['Inference Mode'] == mode]

                    if len(mode_data) > 0:
                        value = mode_data['success_rate'].iloc[0]

                        # Choose color and pattern based on inference mode
                        if mode == 'Zero-shot':
                            color = zero_shot_color
                            hatch = '///'  # Gray stripes
                        else:  # Iterative
                            color = iterative_color
                            hatch = None  # No pattern

                        # Calculate bar position with spacing between precision levels
                        precision_offset = j * (2 * bar_width + mode_spacing + precision_spacing)
                        mode_offset = k * (bar_width + mode_spacing)
                        bar_pos = x_positions[i] + precision_offset + mode_offset - (len(precision_levels) * (2 * bar_width + mode_spacing + precision_spacing) - precision_spacing) / 2

                        # Create bar with fine stripes
                        bar = ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8, hatch=hatch, edgecolor='lightgray', linewidth=0.3)

                        # Add precision label in the middle of the bar
                        ax.text(bar_pos, value/2, precision_labels[j],
                               ha='center', va='center', fontweight='bold',
                               color='black', fontsize=8)

                        # Add value label on top of the bar
                        ax.text(bar_pos, value + 0.01, f'{value:.2f}',
                               ha='center', va='bottom', fontsize=7)

        # Customize the chart
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=10, fontweight='bold')
        ax.set_title(f'{dataset_name.replace("_", " ").title()}',
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
        output_path = self.output_dir / dataset_name / "success_rate.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Success rate chart saved to: {output_path}")
    
    def create_efficiency_chart(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Create bar chart for efficiency metrics across models and configurations.
        All precision levels are shown in a single chart with 6 bars per model.

        Args:
            df: DataFrame containing evaluation results
            dataset_name: Name of the dataset
        """
        # Model name mapping for clean display names
        name_mapping = {
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
        }

        # Apply model name mapping
        df = df.copy()
        df['Model'] = df['Model'].apply(lambda x: name_mapping.get(x, x))

        # Focus only on efficiency metric
        metric = 'mean_efficiency'
        if metric not in df.columns:
            print(f"Warning: Hard efficiency metric not found in dataset '{dataset_name}'")
            return

        # Create single figure for all precision levels
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Get unique models and precision levels
        models = sorted(df['Model'].unique())
        precision_levels = ['low', 'medium', 'high']  # Ensure consistent ordering
        inference_modes = ['Zero-shot', 'Iterative']

        # Prepare data for grouped bars
        bar_width = 0.12  # Width of individual bars
        precision_spacing = 0.12  # Spacing between different precision levels
        mode_spacing = 0.01  # Small spacing between zero-shot and iterative within same precision
        model_spacing = 0.32  # Spacing between different models

        # Fixed colors - light gray with stripes for Zero-shot, dark blue with dots for Iterative
        zero_shot_color = '#BBBBBB'  # Slightly darker gray
        iterative_color = '#4169E1'  # Dark blue

        # Get max value for positioning labels
        all_values = df[metric].values
        max_val = np.max(all_values)

        # Calculate model positions with spacing
        total_group_width = len(precision_levels) * (2 * bar_width + mode_spacing) + (len(precision_levels) - 1) * precision_spacing
        x_positions = np.arange(len(models)) * (total_group_width + model_spacing)
        precision_labels = ['L', 'M', 'H']

        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]

            for j, precision in enumerate(precision_levels):
                precision_data = model_data[model_data['Precision Level'] == precision]

                for k, mode in enumerate(inference_modes):
                    mode_data = precision_data[precision_data['Inference Mode'] == mode]

                    if len(mode_data) > 0:
                        value = mode_data[metric].iloc[0]

                        # Choose color and pattern based on inference mode
                        if mode == 'Zero-shot':
                            color = zero_shot_color
                            hatch = '//'  # Gray sparse stripes
                        else:  # Iterative
                            color = iterative_color
                            hatch = None  # No pattern

                        # Calculate bar position with spacing between precision levels
                        precision_offset = j * (2 * bar_width + mode_spacing + precision_spacing)
                        mode_offset = k * (bar_width + mode_spacing)
                        bar_pos = x_positions[i] + precision_offset + mode_offset - (len(precision_levels) * (2 * bar_width + mode_spacing + precision_spacing) - precision_spacing) / 2

                        # Create bar with fine stripes
                        bar = ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8, hatch=hatch, edgecolor='lightgray', linewidth=0.3)

                        # Add precision label in the middle of the bar
                        ax.text(bar_pos, value/2, precision_labels[j],
                               ha='center', va='center', fontweight='bold',
                               color='black', fontsize=8)

                        # Add value label on top of the bar
                        ax.text(bar_pos, value + max_val * 0.01, f'{value:.2f}',
                               ha='center', va='bottom', fontsize=7)

        # Customize the chart
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel('Efficiency', fontsize=10, fontweight='bold')
        ax.set_title(f'{dataset_name.replace("_", " ").title()}',
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
        output_path = self.output_dir / dataset_name / "efficiency.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Hard efficiency chart saved to: {output_path}")

    def create_aggregated_success_rate_chart(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Create bar chart for success rates with aggregated precision levels.
        Each model shows only 2 bars (Zero-shot and Iterative).

        Args:
            df: DataFrame containing evaluation results
            dataset_name: Name of the dataset
        """
        # Model name mapping for clean display names
        name_mapping = {
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
        }

        # Aggregate data across precision levels
        aggregated_df = df.groupby(['Model', 'Inference Mode'])['success_rate'].mean().reset_index()

        # Apply model name mapping
        aggregated_df['Model'] = aggregated_df['Model'].apply(lambda x: name_mapping.get(x, x))

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Get unique models and inference modes
        models = sorted(aggregated_df['Model'].unique())
        inference_modes = ['Zero-shot', 'Iterative']

        # Prepare data for grouped bars
        bar_width = 0.35  # Width of individual bars
        model_spacing = 0.8  # Spacing between different models

        # Fixed colors
        zero_shot_color = '#BBBBBB'  # Light gray
        iterative_color = '#4169E1'  # Dark blue

        # Get max value for y-axis scaling
        max_val = aggregated_df['success_rate'].max()

        # Calculate model positions
        x_positions = np.arange(len(models))

        for i, model in enumerate(models):
            model_data = aggregated_df[aggregated_df['Model'] == model]

            for j, mode in enumerate(inference_modes):
                mode_data = model_data[model_data['Inference Mode'] == mode]

                if len(mode_data) > 0:
                    value = mode_data['success_rate'].iloc[0]

                    # Choose color and pattern based on inference mode
                    if mode == 'Zero-shot':
                        color = zero_shot_color
                        hatch = '///'  # Gray stripes
                    else:  # Iterative
                        color = iterative_color
                        hatch = None  # No pattern

                    # Calculate bar position
                    bar_pos = x_positions[i] + (j - 0.5) * bar_width

                    # Create bar
                    bar = ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                               hatch=hatch, edgecolor='lightgray', linewidth=0.3)

                    # Add value label on top of the bar
                    ax.text(bar_pos, value + 0.01, f'{value:.2f}',
                           ha='center', va='bottom', fontsize=9)

        # Customize the chart
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name.replace("_", " ").title()}',
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0, max_val * 1.1)  # Add 10% padding to the maximum value
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
        output_path = self.output_dir / dataset_name / "success_rate_aggregated.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Aggregated success rate chart saved to: {output_path}")

    def create_aggregated_efficiency_chart(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Create bar chart for efficiency metrics with aggregated precision levels.
        Each model shows only 2 bars (Zero-shot and Iterative).

        Args:
            df: DataFrame containing evaluation results
            dataset_name: Name of the dataset
        """
        # Model name mapping for clean display names
        name_mapping = {
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
        }

        # Focus only on efficiency metric
        metric = 'mean_efficiency'
        if metric not in df.columns:
            print(f"Warning: Efficiency metric not found in dataset '{dataset_name}'")
            return

        # Aggregate data across precision levels
        aggregated_df = df.groupby(['Model', 'Inference Mode'])[metric].mean().reset_index()

        # Apply model name mapping
        aggregated_df['Model'] = aggregated_df['Model'].apply(lambda x: name_mapping.get(x, x))

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Get unique models and inference modes
        models = sorted(aggregated_df['Model'].unique())
        inference_modes = ['Zero-shot', 'Iterative']

        # Prepare data for grouped bars
        bar_width = 0.35  # Width of individual bars
        model_spacing = 0.8  # Spacing between different models

        # Fixed colors
        zero_shot_color = '#BBBBBB'  # Light gray
        iterative_color = '#4169E1'  # Dark blue

        # Get max value for y-axis scaling
        max_val = aggregated_df[metric].max()

        # Calculate model positions
        x_positions = np.arange(len(models))

        for i, model in enumerate(models):
            model_data = aggregated_df[aggregated_df['Model'] == model]

            for j, mode in enumerate(inference_modes):
                mode_data = model_data[model_data['Inference Mode'] == mode]

                if len(mode_data) > 0:
                    value = mode_data[metric].iloc[0]

                    # Choose color and pattern based on inference mode
                    if mode == 'Zero-shot':
                        color = zero_shot_color
                        hatch = '///'  # Gray stripes
                    else:  # Iterative
                        color = iterative_color
                        hatch = None  # No pattern

                    # Calculate bar position
                    bar_pos = x_positions[i] + (j - 0.5) * bar_width

                    # Create bar
                    bar = ax.bar(bar_pos, value, bar_width, color=color, alpha=0.8,
                               hatch=hatch, edgecolor='lightgray', linewidth=0.3)

                    # Add value label on top of the bar
                    ax.text(bar_pos, value + max_val * 0.01, f'{value:.2f}',
                           ha='center', va='bottom', fontsize=9)

        # Customize the chart
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Efficiency', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name.replace("_", " ").title()}',
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0, max_val * 1.1)  # Add 10% padding to the maximum value
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
        output_path = self.output_dir / dataset_name / "efficiency_aggregated.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Aggregated efficiency chart saved to: {output_path}")
    
    def create_overview_chart_deprecated(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Create overview chart combining key metrics.
        
        Args:
            df: DataFrame containing evaluation results
            dataset_name: Name of the dataset
        """
        # Model names are already cleaned by upstream processing
        
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Success Rate by Model (aggregated across precision levels)
        success_by_model = df.groupby('Model')['success_rate'].mean().sort_values(ascending=False)
        success_by_model.plot(kind='bar', ax=ax1, color='skyblue', alpha=0.8)
        ax1.set_title('Average Success Rate by Model', fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        for i, v in enumerate(success_by_model.values):
            ax1.text(i, v + 0.01, f'{v:.2f}', ha='center', fontweight='bold')
        
        # 2. Success Rate by Inference Mode
        success_by_mode = df.groupby('Inference Mode')['success_rate'].mean()
        success_by_mode.plot(kind='bar', ax=ax2, color='lightcoral', alpha=0.8)
        ax2.set_title('Average Success Rate by Inference Mode', fontweight='bold')
        ax2.set_ylabel('Success Rate')
        ax2.tick_params(axis='x', rotation=0)
        ax2.grid(True, alpha=0.3)
        for i, v in enumerate(success_by_mode.values):
            ax2.text(i, v + 0.01, f'{v:.2f}', ha='center', fontweight='bold')
        
        # 3. Success Rate by Precision Level
        success_by_precision = df.groupby('Precision Level')['success_rate'].mean()
        success_by_precision.plot(kind='bar', ax=ax3, color='lightgreen', alpha=0.8)
        ax3.set_title('Average Success Rate by Precision Level', fontweight='bold')
        ax3.set_ylabel('Success Rate')
        ax3.tick_params(axis='x', rotation=0)
        ax3.grid(True, alpha=0.3)
        for i, v in enumerate(success_by_precision.values):
            ax3.text(i, v + 0.01, f'{v:.2f}', ha='center', fontweight='bold')
        
        # 4. Soft Success vs Hard Efficiency scatter plot (if available)
        if 'success_rate' in df.columns and 'mean_efficiency' in df.columns:
            scatter = ax4.scatter(df['success_rate'], df['mean_efficiency'], 
                                c=df['Precision Level'].astype('category').cat.codes, 
                                cmap='viridis', alpha=0.6, s=60)
            ax4.set_xlabel('Success Rate')
            ax4.set_ylabel('Mean Efficiency')
            ax4.set_title('Soft Success vs Hard Efficiency', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('Precision Level')
            cbar.set_ticks(range(len(df['Precision Level'].unique())))
            cbar.set_ticklabels(df['Precision Level'].unique())
        else:
            ax4.text(0.5, 0.5, 'Efficiency metrics\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Efficiency Analysis', fontweight='bold')
        
        plt.suptitle(f'Performance Overview - {dataset_name.replace("_", " ").title()}', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save chart
        output_path = self.output_dir / dataset_name / "overview.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Overview chart saved to: {output_path}")
    
    def generate_charts_for_dataset(self, dataset_name: str) -> bool:
        """
        Generate all charts for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if successful, False otherwise
        """
        print(f"\nProcessing dataset: {dataset_name}")

        # Load dataset
        df = self.load_dataset(dataset_name)
        if df is None:
            return False

        try:
            # Generate detailed charts: success rate and efficiency with precision levels
            self.create_success_rate_chart(df, dataset_name)
            self.create_efficiency_chart(df, dataset_name)

            # Generate aggregated charts: success rate and efficiency without precision levels
            self.create_aggregated_success_rate_chart(df, dataset_name)
            self.create_aggregated_efficiency_chart(df, dataset_name)

            print(f"✓ Successfully generated charts for dataset: {dataset_name}")
            return True

        except Exception as e:
            print(f"✗ Error generating charts for dataset '{dataset_name}': {e}")
            return False
    
    def generate_all_charts(self, datasets: Optional[List[str]] = None) -> None:
        """
        Generate charts for all specified datasets.
        
        Args:
            datasets: List of dataset names, or None to process all available
        """
        if datasets is None:
            datasets = self.find_available_datasets()
        
        if not datasets:
            print("No datasets found with summary CSV files.")
            return
        
        print(f"Found {len(datasets)} dataset(s): {', '.join(datasets)}")
        
        success_count = 0
        for dataset in datasets:
            if self.generate_charts_for_dataset(dataset):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Chart generation completed!")
        print(f"Successfully processed: {success_count}/{len(datasets)} datasets")
        print(f"Charts saved to: {self.output_dir}")
        print(f"{'='*60}")


def main():
    """Main function to run the bar chart generator."""
    parser = argparse.ArgumentParser(
        description="Generate professional bar charts from SimulCost-Bench evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluation/stats_utils/bar_chart.py -d burgers_1d
    python evaluation/stats_utils/bar_chart.py -d ns_2d
    python evaluation/stats_utils/bar_chart.py  # Process all datasets
        """
    )
    
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        help='Specific dataset name to process (e.g., burgers_1d, ns_2d)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='eval_results',
        help='Base directory containing evaluation results (default: eval_results)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='eval_results/stats/bar_chart',
        help='Output directory for charts (default: eval_results/stats/bar_chart)'
    )
    
    args = parser.parse_args()
    
    # Initialize chart generator
    generator = BarChartGenerator(base_dir=args.base_dir, output_dir=args.output_dir)
    
    # Generate charts
    if args.dataset:
        # Process specific dataset
        datasets = [args.dataset]
    else:
        # Process all available datasets
        datasets = None
    
    generator.generate_all_charts(datasets=datasets)


if __name__ == '__main__':
    main()