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
    
    def clean_model_names(self, model_name: str) -> str:
        """
        Clean and shorten model names for better display.
        Developers can easily add new model mappings here.
        
        Args:
            model_name: Original model name
            
        Returns:
            Cleaned model name
        """
        # Model name mapping - easily extensible for new models
        name_mapping = {
            'amazon.nova-premier-v1:0': 'Nova-Premier',
            'anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude-3.7-Sonnet',
            'mistral.mistral-large-2402-v1:0': 'Mistral-Large',
            'meta.llama3-70b-instruct-v1:0': 'Llama-3-70B-Instruct'
        }
        
        return name_mapping.get(model_name, model_name)
    
    def create_success_rate_chart(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Create bar chart for success rates across models and configurations.
        
        Args:
            df: DataFrame containing evaluation results
            dataset_name: Name of the dataset
        """
        # Clean model names
        df['Model_Clean'] = df['Model'].apply(self.clean_model_names)
        
        # Create figure with subplots for different precision levels
        precision_levels = df['Precision Level'].unique()
        fig, axes = plt.subplots(1, len(precision_levels), figsize=(4 * len(precision_levels), 5))
        
        if len(precision_levels) == 1:
            axes = [axes]
        
        # Get consistent y-axis limits for success rate (0 to 1)
        y_min, y_max = 0, 1.0
        
        for i, precision in enumerate(precision_levels):
            ax = axes[i]
            precision_data = df[df['Precision Level'] == precision]
            
            # Pivot data for grouped bar chart
            pivot_data = precision_data.pivot(index='Model_Clean', 
                                            columns='Inference Mode', 
                                            values='success_rate')
            
            # Reorder columns to put Zero-shot first, then Iterative
            if 'Zero-shot' in pivot_data.columns and 'Iterative' in pivot_data.columns:
                pivot_data = pivot_data[['Zero-shot', 'Iterative']]
            
            # Create grouped bar chart
            pivot_data.plot(kind='bar', ax=ax, width=0.6, alpha=0.8, legend=False)
            
            ax.set_title(f'Success Rate - {precision.title()} Precision', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Model', fontsize=10, fontweight='bold')
            ax.set_ylabel('Success Rate', fontsize=10, fontweight='bold')
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=7)
        
        # Add single figure-level legend next to title
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Inference Mode', title_fontsize=8, fontsize=7, 
                  loc='upper center', bbox_to_anchor=(0.68, 0.98), frameon=False)
        
        plt.suptitle(f'{dataset_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save chart
        output_path = self.output_dir / dataset_name / "success_rate.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Success rate chart saved to: {output_path}")
    
    def create_hard_efficiency_chart(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Create bar chart for hard efficiency metrics across models and configurations.
        
        Args:
            df: DataFrame containing evaluation results
            dataset_name: Name of the dataset
        """
        # Clean model names
        df['Model_Clean'] = df['Model'].apply(self.clean_model_names)
        
        # Focus only on hard efficiency metric
        metric = 'mean_hard_efficiency'
        if metric not in df.columns:
            print(f"Warning: Hard efficiency metric not found in dataset '{dataset_name}'")
            return
        
        precision_levels = df['Precision Level'].unique()
        fig, axes = plt.subplots(1, len(precision_levels), figsize=(4 * len(precision_levels), 5))
        
        if len(precision_levels) == 1:
            axes = [axes]
        
        # Get consistent y-axis limits for hard efficiency across all precision levels
        y_min = 0
        y_max = df[metric].max() * 1.1  # Add 10% padding to the maximum value
        
        for i, precision in enumerate(precision_levels):
            ax = axes[i]
            precision_data = df[df['Precision Level'] == precision]
            
            # Pivot data for grouped bar chart
            pivot_data = precision_data.pivot(index='Model_Clean', 
                                            columns='Inference Mode', 
                                            values=metric)
            
            # Reorder columns to put Zero-shot first, then Iterative
            if 'Zero-shot' in pivot_data.columns and 'Iterative' in pivot_data.columns:
                pivot_data = pivot_data[['Zero-shot', 'Iterative']]
            
            # Create grouped bar chart
            pivot_data.plot(kind='bar', ax=ax, width=0.6, alpha=0.8, legend=False)
            
            ax.set_title(f'Hard Efficiency - {precision.title()} Precision', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Model', fontsize=10, fontweight='bold')
            ax.set_ylabel('Hard Efficiency', fontsize=10, fontweight='bold')
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=7)
        
        # Add single figure-level legend next to title
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Inference Mode', title_fontsize=8, fontsize=7, 
                  loc='upper center', bbox_to_anchor=(0.68, 0.98), frameon=False)
        
        plt.suptitle(f'{dataset_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save chart
        output_path = self.output_dir / dataset_name / "hard_efficiency.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Hard efficiency chart saved to: {output_path}")
    
    def create_overview_chart_deprecated(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Create overview chart combining key metrics.
        
        Args:
            df: DataFrame containing evaluation results
            dataset_name: Name of the dataset
        """
        # Clean model names
        df['Model_Clean'] = df['Model'].apply(self.clean_model_names)
        
        # Create figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Success Rate by Model (aggregated across precision levels)
        success_by_model = df.groupby('Model_Clean')['success_rate'].mean().sort_values(ascending=False)
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
        if 'mean_soft_success' in df.columns and 'mean_hard_efficiency' in df.columns:
            scatter = ax4.scatter(df['mean_soft_success'], df['mean_hard_efficiency'], 
                                c=df['Precision Level'].astype('category').cat.codes, 
                                cmap='viridis', alpha=0.6, s=60)
            ax4.set_xlabel('Mean Soft Success')
            ax4.set_ylabel('Mean Hard Efficiency')
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
            # Generate only the key charts: success rate and hard efficiency
            self.create_success_rate_chart(df, dataset_name)
            self.create_hard_efficiency_chart(df, dataset_name)
            
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