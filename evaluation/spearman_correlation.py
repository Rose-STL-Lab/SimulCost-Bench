#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spearman Rank Correlation Analysis for Zero-shot vs Iterative Model Performance

This script computes Spearman rank correlation coefficients between zero-shot 
and iterative inference modes across different models and precision levels.
Analyzes correlation for multiple performance metrics to understand consistency
of model rankings across inference paradigms.

Usage
-----
python evaluation/spearman_correlation.py              # Analyze all available datasets
python evaluation/spearman_correlation.py -d euler_1d  # Analyze specific dataset
python evaluation/spearman_correlation.py -d heat_1d

Output: Creates analysis results with only essential files:
- correlation_summary.csv: Numerical correlation coefficients and p-values
- correlation_summary.xlsx: Excel version with formatting
- correlation_heatmap.png: Visual heatmap of correlations
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Setup path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration constants
METRICS_TO_ANALYZE = ['success_rate', 'mean_soft_success', 'mean_efficiency', 'mean_hard_efficiency']
PRECISION_LEVELS = ['low', 'medium', 'high', 'overall']
INFERENCE_MODES = ['Zero-shot', 'Iterative']

class SpearmanCorrelationAnalyzer:
    """
    Professional analyzer for computing Spearman correlations between 
    zero-shot and iterative model performance across multiple metrics.
    """
    
    def __init__(self, datasets=None):
        """
        Initialize the analyzer for one or multiple datasets.
        
        Parameters
        ----------
        datasets : list, str, or None
            Name(s) of dataset(s) (e.g., ['euler_1d', 'heat_1d'] or 'euler_1d')
            If None, auto-detect all available datasets
        """
        if datasets is None:
            self.datasets = self._discover_available_datasets()
        elif isinstance(datasets, str):
            self.datasets = [datasets]
        else:
            self.datasets = list(datasets)
        
        print(f"📊 Analyzing datasets: {', '.join(self.datasets)}")
        
        # Create output directory
        if len(self.datasets) == 1:
            self.output_path = Path(f"eval_results/spearman_correlation/{self.datasets[0]}")
        else:
            self.output_path = Path("eval_results/spearman_correlation/multi_dataset")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Storage for analysis results
        self.correlation_results = {}
        
    def _discover_available_datasets(self) -> List[str]:
        """
        Discover all datasets with complete zero-shot and iterative data.
        
        Returns
        -------
        List[str]
            List of available dataset names
        """
        eval_results_path = Path("eval_results")
        available_datasets = []
        
        for dataset_path in eval_results_path.iterdir():
            if dataset_path.is_dir() and dataset_path.name != "spearman_correlation":
                dataset_name = dataset_path.name
                zero_shot_path = dataset_path / "zero_shot" / f"{dataset_name}_sum.csv"
                iterative_path = dataset_path / "iterative" / f"{dataset_name}_sum.csv"
                
                if zero_shot_path.exists() and iterative_path.exists():
                    available_datasets.append(dataset_name)
        
        if not available_datasets:
            raise FileNotFoundError("No datasets with complete zero-shot and iterative data found")
        
        return sorted(available_datasets)
        
    def load_evaluation_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load zero-shot and iterative evaluation results from all datasets.
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Combined zero-shot and iterative dataframes from all datasets
            
        Raises
        ------
        FileNotFoundError
            If required CSV files are not found
        """
        all_zero_shot = []
        all_iterative = []
        
        for dataset in self.datasets:
            dataset_path = Path(f"eval_results/{dataset}")
            zero_shot_path = dataset_path / "zero_shot" / f"{dataset}_sum.csv"
            iterative_path = dataset_path / "iterative" / f"{dataset}_sum.csv"
            
            # Check file existence
            if not zero_shot_path.exists():
                print(f"⚠ Skipping {dataset}: Zero-shot results not found")
                continue
            if not iterative_path.exists():
                print(f"⚠ Skipping {dataset}: Iterative results not found")
                continue
            
            # Load data
            zero_shot_df = pd.read_csv(zero_shot_path)
            iterative_df = pd.read_csv(iterative_path)
            
            # Add dataset identifier
            zero_shot_df['Dataset'] = dataset
            iterative_df['Dataset'] = dataset
            
            all_zero_shot.append(zero_shot_df)
            all_iterative.append(iterative_df)
            
            print(f"✓ Loaded {dataset}: {len(zero_shot_df)} zero-shot, {len(iterative_df)} iterative records")
        
        if not all_zero_shot or not all_iterative:
            raise FileNotFoundError("No valid datasets found")
        
        # Combine all datasets
        combined_zero_shot = pd.concat(all_zero_shot, ignore_index=True)
        combined_iterative = pd.concat(all_iterative, ignore_index=True)
        
        print(f"✓ Total combined data: {len(combined_zero_shot)} zero-shot, {len(combined_iterative)} iterative records")
        
        return combined_zero_shot, combined_iterative
    
    def prepare_paired_data(self, zero_shot_df: pd.DataFrame, 
                           iterative_df: pd.DataFrame,
                           precision_level: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare paired data for correlation analysis by matching models and datasets.
        
        Parameters
        ----------
        zero_shot_df : pd.DataFrame
            Zero-shot evaluation results
        iterative_df : pd.DataFrame
            Iterative evaluation results
        precision_level : Optional[str]
            Specific precision level to filter ('low', 'medium', 'high'), 
            or None for all data
            
        Returns
        -------
        pd.DataFrame
            Paired data with zero-shot and iterative metrics side by side
        """
        # Filter by precision level if specified
        if precision_level and precision_level != 'overall':
            zero_shot_filtered = zero_shot_df[zero_shot_df['Precision Level'] == precision_level].copy()
            iterative_filtered = iterative_df[iterative_df['Precision Level'] == precision_level].copy()
        else:
            zero_shot_filtered = zero_shot_df.copy()
            iterative_filtered = iterative_df.copy()
        
        # Merge on model name and dataset for multi-dataset support
        merge_cols = ['Model']
        if 'Dataset' in zero_shot_filtered.columns:
            merge_cols.append('Dataset')
            
        merged = pd.merge(
            zero_shot_filtered[merge_cols + METRICS_TO_ANALYZE],
            iterative_filtered[merge_cols + METRICS_TO_ANALYZE],
            on=merge_cols,
            suffixes=('_zero_shot', '_iterative'),
            how='inner'
        )
        
        if len(merged) == 0:
            raise ValueError(f"No matching models found for precision level: {precision_level}")
            
        return merged
    
    def compute_spearman_correlation(self, paired_data: pd.DataFrame, 
                                   metric: str) -> Tuple[float, float, int]:
        """
        Compute Spearman correlation for a specific metric.
        
        Parameters
        ----------
        paired_data : pd.DataFrame
            Paired zero-shot and iterative data
        metric : str
            Metric name to analyze
            
        Returns
        -------
        Tuple[float, float, int]
            Correlation coefficient, p-value, and sample size
        """
        zero_shot_col = f"{metric}_zero_shot"
        iterative_col = f"{metric}_iterative"
        
        # Extract valid pairs (no NaN values)
        zero_shot_vals = paired_data[zero_shot_col].dropna()
        iterative_vals = paired_data[iterative_col].dropna()
        
        # Find common indices
        common_idx = zero_shot_vals.index.intersection(iterative_vals.index)
        
        if len(common_idx) < 3:
            return np.nan, np.nan, len(common_idx)
        
        # Compute correlation
        correlation, p_value = spearmanr(
            zero_shot_vals.loc[common_idx],
            iterative_vals.loc[common_idx]
        )
        
        return correlation, p_value, len(common_idx)
    
    def analyze_all_correlations(self, zero_shot_df: pd.DataFrame, 
                               iterative_df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive Spearman correlation analysis across all metrics and precision levels.
        
        Parameters
        ----------
        zero_shot_df : pd.DataFrame
            Zero-shot evaluation results
        iterative_df : pd.DataFrame
            Iterative evaluation results
            
        Returns
        -------
        Dict
            Comprehensive correlation results
        """
        results = {}
        
        print("\n🔍 Computing Spearman correlations...")
        
        for precision_level in PRECISION_LEVELS:
            print(f"\n📊 Analyzing precision level: {precision_level}")
            
            # Handle overall case by aggregating across precision levels
            if precision_level == 'overall':
                # Aggregate by Model (and Dataset if multi-dataset)
                group_cols = ['Model']
                if 'Dataset' in zero_shot_df.columns:
                    group_cols.append('Dataset')
                
                zero_shot_agg = zero_shot_df.groupby(group_cols)[METRICS_TO_ANALYZE].mean().reset_index()
                iterative_agg = iterative_df.groupby(group_cols)[METRICS_TO_ANALYZE].mean().reset_index()
                
                paired_data = pd.merge(
                    zero_shot_agg,
                    iterative_agg,
                    on=group_cols,
                    suffixes=('_zero_shot', '_iterative'),
                    how='inner'
                )
            else:
                paired_data = self.prepare_paired_data(zero_shot_df, iterative_df, precision_level)
            
            # Compute correlations for each metric
            level_results = {}
            for metric in METRICS_TO_ANALYZE:
                correlation, p_value, n_samples = self.compute_spearman_correlation(paired_data, metric)
                
                level_results[metric] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_samples': n_samples,
                    'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                }
                
                print(f"  {metric}: ρ = {correlation:.3f} (p = {p_value:.3f}, n = {n_samples})")
            
            results[precision_level] = level_results
        
        self.correlation_results = results
        return results
    
    def save_correlation_summary(self) -> None:
        """Save correlation summary to CSV and Excel files only."""
        summary_data = []
        
        for precision_level, metrics in self.correlation_results.items():
            for metric, stats in metrics.items():
                summary_data.append({
                    'Precision_Level': precision_level,
                    'Metric': metric,
                    'Spearman_rho': stats['correlation'],
                    'P_Value': stats['p_value'],
                    'N_Samples': stats['n_samples'],
                    'Significance': stats['significance']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_path = self.output_path / "correlation_summary.csv"
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"✓ Correlation summary saved: {summary_path}")
        
        # Save to Excel with formatting
        try:
            summary_excel_path = self.output_path / "correlation_summary.xlsx"
            with pd.ExcelWriter(summary_excel_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Correlation Summary', index=False, float_format='%.4f')
            print(f"✓ Correlation summary Excel saved: {summary_excel_path}")
        except ImportError:
            print("⚠ openpyxl not available, skipping Excel export")
        
        # Skip detailed data generation as requested
        print(f"ℹ Detailed paired data generation skipped (only essential files saved)")
    
    def create_correlation_heatmap(self) -> None:
        """Create and save correlation heatmap visualization."""
        # Prepare correlation matrix
        correlation_matrix = []
        precision_labels = []
        
        for precision_level in PRECISION_LEVELS:
            if precision_level in self.correlation_results:
                correlations = [
                    self.correlation_results[precision_level][metric]['correlation']
                    for metric in METRICS_TO_ANALYZE
                ]
                correlation_matrix.append(correlations)
                precision_labels.append(precision_level.capitalize())
        
        correlation_matrix = np.array(correlation_matrix)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create subplot with space for colorbar
        ax = plt.subplot(111)
        
        # Plot heatmap
        im = ax.imshow(correlation_matrix, cmap='RdYlBu_r', aspect='auto', 
                      vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(METRICS_TO_ANALYZE)))
        ax.set_yticks(range(len(precision_labels)))
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in METRICS_TO_ANALYZE])
        ax.set_yticklabels(precision_labels)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add correlation values as text
        for i in range(len(precision_labels)):
            for j in range(len(METRICS_TO_ANALYZE)):
                if not np.isnan(correlation_matrix[i, j]):
                    significance = self.correlation_results[PRECISION_LEVELS[i]][METRICS_TO_ANALYZE[j]]['significance']
                    text = f"{correlation_matrix[i, j]:.3f}\n{significance}"
                    ax.text(j, i, text, ha="center", va="center", 
                           color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black",
                           fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Spearman Correlation Coefficient', rotation=270, labelpad=20)
        
        # Set titles and labels
        dataset_title = ', '.join([d.upper() for d in self.datasets]) if len(self.datasets) > 1 else self.datasets[0].upper()
        plt.title(f'Zero-shot vs Iterative Performance Correlations\nDataset(s): {dataset_title}', 
                 fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision Levels', fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        heatmap_path = self.output_path / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Correlation heatmap saved: {heatmap_path}")
    
    def print_summary_report(self) -> None:
        """Print a comprehensive summary report to console."""
        dataset_title = ', '.join([d.upper() for d in self.datasets])
        print(f"\n{'='*70}")
        print(f"SPEARMAN CORRELATION ANALYSIS SUMMARY")
        print(f"Dataset(s): {dataset_title}")
        print(f"{'='*70}")
        
        for precision_level in PRECISION_LEVELS:
            if precision_level not in self.correlation_results:
                continue
                
            print(f"\n📈 {precision_level.upper()} PRECISION LEVEL:")
            print("-" * 50)
            
            for metric in METRICS_TO_ANALYZE:
                stats = self.correlation_results[precision_level][metric]
                correlation = stats['correlation']
                p_value = stats['p_value']
                n_samples = stats['n_samples']
                significance = stats['significance']
                
                # Interpretation
                if np.isnan(correlation):
                    interpretation = "Insufficient data"
                elif abs(correlation) >= 0.7:
                    interpretation = "Strong correlation"
                elif abs(correlation) >= 0.5:
                    interpretation = "Moderate correlation"
                elif abs(correlation) >= 0.3:
                    interpretation = "Weak correlation"
                else:
                    interpretation = "Very weak correlation"
                
                print(f"  {metric:20s}: ρ = {correlation:6.3f} {significance:3s} "
                      f"(p = {p_value:.3f}, n = {n_samples:2d}) - {interpretation}")
        
        print(f"\n{'='*70}")
        print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print(f"Results saved in: {self.output_path}")
        print(f"{'='*70}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute Spearman correlations between zero-shot and iterative model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/spearman_correlation.py              # Analyze all available datasets
  python evaluation/spearman_correlation.py -d euler_1d  # Analyze specific dataset
  python evaluation/spearman_correlation.py -d heat_1d

Output files (essential files only):
  - eval_results/spearman_correlation/{dataset}/correlation_summary.csv
  - eval_results/spearman_correlation/{dataset}/correlation_summary.xlsx
  - eval_results/spearman_correlation/{dataset}/correlation_heatmap.png
        """
    )
    
    parser.add_argument(
        '-d', '--dataset',
        required=False,
        help='Dataset name (e.g., euler_1d, heat_1d, burgers_1d). If not specified, analyzes all available datasets.'
    )
    
    args = parser.parse_args()
    
    try:
        if args.dataset:
            print(f"🚀 Starting Spearman correlation analysis for dataset: {args.dataset}")
            analyzer = SpearmanCorrelationAnalyzer(args.dataset)
        else:
            print(f"🚀 Starting Spearman correlation analysis for all available datasets")
            analyzer = SpearmanCorrelationAnalyzer()
        
        # Load data
        zero_shot_df, iterative_df = analyzer.load_evaluation_data()
        
        # Perform correlation analysis
        analyzer.analyze_all_correlations(zero_shot_df, iterative_df)
        
        # Save results (only essential files)
        analyzer.save_correlation_summary()
        analyzer.create_correlation_heatmap()
        
        # Print summary
        analyzer.print_summary_report()
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()