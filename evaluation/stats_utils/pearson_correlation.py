#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pearson Correlation Analysis for Zero-shot vs Iterative Model Performance

This script computes Pearson correlation coefficients between zero-shot 
and iterative inference modes across different models and precision levels.
Analyzes correlation for multiple performance metrics to understand linear
relationships between model performance across inference paradigms.

Usage
-----
python evaluation/stats_utils/pearson_correlation.py              # Analyze all available datasets
python evaluation/stats_utils/pearson_correlation.py -d euler_1d  # Analyze specific dataset
python evaluation/stats_utils/pearson_correlation.py -d heat_1d

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
from scipy.stats import pearsonr

# Setup path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration constants
METRICS_TO_ANALYZE = ['success_rate', 'mean_efficiency']
PRECISION_LEVELS = ['low', 'medium', 'high', 'overall']
INFERENCE_MODES = ['Zero-shot', 'Iterative']

class PearsonCorrelationAnalyzer:
    """
    Professional analyzer for computing Pearson correlations between 
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
            self.output_path = Path(f"eval_results/stats/pearson_correlation/{self.datasets[0]}")
        else:
            self.output_path = Path("eval_results/stats/pearson_correlation/multi_dataset")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Storage for analysis results
        self.correlation_results = {}
        self.individual_model_results = {}
        
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
            if dataset_path.is_dir() and dataset_path.name != "stats":
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
    
    def compute_pearson_correlation(self, paired_data: pd.DataFrame, 
                                   metric: str) -> Tuple[float, float, int]:
        """
        Compute Pearson correlation for a specific metric.
        
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
        correlation, p_value = pearsonr(
            zero_shot_vals.loc[common_idx],
            iterative_vals.loc[common_idx]
        )
        
        return correlation, p_value, len(common_idx)
    
    def analyze_all_correlations(self, zero_shot_df: pd.DataFrame, 
                               iterative_df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive Pearson correlation analysis across all metrics and precision levels.
        
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
        
        print("\n🔍 Computing Pearson correlations...")
        
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
                correlation, p_value, n_samples = self.compute_pearson_correlation(paired_data, metric)
                
                level_results[metric] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_samples': n_samples,
                    'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                }
                
                print(f"  {metric}: r = {correlation:.3f} (p = {p_value:.3f}, n = {n_samples})")
            
            results[precision_level] = level_results
        
        self.correlation_results = results
        return results
    
    def analyze_individual_model_correlations(self, zero_shot_df: pd.DataFrame, 
                                            iterative_df: pd.DataFrame) -> Dict:
        """
        Perform Pearson correlation analysis for each individual model.
        
        Parameters
        ----------
        zero_shot_df : pd.DataFrame
            Zero-shot evaluation results
        iterative_df : pd.DataFrame
            Iterative evaluation results
            
        Returns
        -------
        Dict
            Individual model correlation results
        """
        individual_results = {}
        
        print("\n🔍 Computing individual model correlations...")
        
        # Get unique models across all datasets
        zero_shot_models = set(zero_shot_df['Model'].unique())
        iterative_models = set(iterative_df['Model'].unique())
        common_models = zero_shot_models.intersection(iterative_models)
        
        print(f"Found {len(common_models)} models with both zero-shot and iterative data")
        
        for model_name in sorted(common_models):
            print(f"\n📊 Analyzing model: {model_name}")
            
            # Filter data for this specific model
            model_zero_shot = zero_shot_df[zero_shot_df['Model'] == model_name]
            model_iterative = iterative_df[iterative_df['Model'] == model_name]
            
            model_results = {}
            
            for precision_level in PRECISION_LEVELS:
                if precision_level == 'overall':
                    # For overall, use all precision levels for this model
                    paired_data = self.prepare_individual_model_data(
                        model_zero_shot, model_iterative, model_name, None
                    )
                else:
                    paired_data = self.prepare_individual_model_data(
                        model_zero_shot, model_iterative, model_name, precision_level
                    )
                
                if paired_data is None or len(paired_data) == 0:
                    print(f"  No data for {precision_level} precision level")
                    continue
                
                # Compute correlations for each metric
                level_results = {}
                for metric in METRICS_TO_ANALYZE:
                    correlation, p_value, n_samples = self.compute_pearson_correlation(paired_data, metric)
                    
                    level_results[metric] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'n_samples': n_samples,
                        'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                    }
                    
                    if not np.isnan(correlation):
                        print(f"  {precision_level}-{metric}: r = {correlation:.3f} (p = {p_value:.3f}, n = {n_samples})")
                
                model_results[precision_level] = level_results
            
            individual_results[model_name] = model_results
        
        self.individual_model_results = individual_results
        return individual_results
    
    def prepare_individual_model_data(self, model_zero_shot: pd.DataFrame, 
                                    model_iterative: pd.DataFrame,
                                    model_name: str,
                                    precision_level: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Prepare paired data for a specific model's correlation analysis.
        
        Parameters
        ----------
        model_zero_shot : pd.DataFrame
            Zero-shot results for specific model
        model_iterative : pd.DataFrame
            Iterative results for specific model
        model_name : str
            Name of the model being analyzed
        precision_level : Optional[str]
            Specific precision level to filter, or None for all data
            
        Returns
        -------
        Optional[pd.DataFrame]
            Paired data for the specific model, or None if insufficient data
        """
        # Filter by precision level if specified
        if precision_level and precision_level != 'overall':
            zero_shot_filtered = model_zero_shot[model_zero_shot['Precision Level'] == precision_level].copy()
            iterative_filtered = model_iterative[model_iterative['Precision Level'] == precision_level].copy()
        else:
            zero_shot_filtered = model_zero_shot.copy()
            iterative_filtered = model_iterative.copy()
        
        # For individual model analysis, we merge on Dataset (if present) and Precision Level
        merge_cols = []
        if 'Dataset' in zero_shot_filtered.columns:
            merge_cols.append('Dataset')
        if 'Precision Level' in zero_shot_filtered.columns and precision_level != 'overall':
            merge_cols.append('Precision Level')
        
        # If no merge columns, we aggregate all data for this model
        if not merge_cols:
            # Aggregate metrics across all data points for this model
            zero_shot_agg = zero_shot_filtered[METRICS_TO_ANALYZE].mean().to_frame().T
            iterative_agg = iterative_filtered[METRICS_TO_ANALYZE].mean().to_frame().T
            
            # Add suffixes for correlation calculation
            for metric in METRICS_TO_ANALYZE:
                zero_shot_agg[f"{metric}_zero_shot"] = zero_shot_agg[metric]
                iterative_agg[f"{metric}_iterative"] = iterative_agg[metric]
            
            # Combine data
            paired_data = pd.concat([
                zero_shot_agg[[f"{metric}_zero_shot" for metric in METRICS_TO_ANALYZE]],
                iterative_agg[[f"{metric}_iterative" for metric in METRICS_TO_ANALYZE]]
            ], axis=1)
        else:
            # Merge on available columns
            paired_data = pd.merge(
                zero_shot_filtered[merge_cols + METRICS_TO_ANALYZE],
                iterative_filtered[merge_cols + METRICS_TO_ANALYZE],
                on=merge_cols,
                suffixes=('_zero_shot', '_iterative'),
                how='inner'
            )
        
        if len(paired_data) == 0:
            return None
            
        return paired_data
    
    def save_correlation_summary(self) -> None:
        """Save correlation summary to CSV and Excel files only."""
        summary_data = []
        
        for precision_level, metrics in self.correlation_results.items():
            for metric, stats in metrics.items():
                summary_data.append({
                    'Precision_Level': precision_level,
                    'Metric': metric,
                    'Pearson_r': stats['correlation'],
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
    
    def save_individual_model_summary(self) -> None:
        """Save individual model correlation summary to CSV and Excel files."""
        if not self.individual_model_results:
            print("⚠ No individual model results available for summary generation")
            return
        
        summary_data = []
        
        for model_name, model_results in self.individual_model_results.items():
            for precision_level, metrics in model_results.items():
                for metric, stats in metrics.items():
                    summary_data.append({
                        'Model': model_name,
                        'Precision_Level': precision_level,
                        'Metric': metric,
                        'Pearson_r': stats['correlation'],
                        'P_Value': stats['p_value'],
                        'N_Samples': stats['n_samples'],
                        'Significance': stats['significance']
                    })
        
        if not summary_data:
            print("⚠ No individual model data to save")
            return
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        individual_summary_path = self.output_path / "individual_model_correlation_summary.csv"
        summary_df.to_csv(individual_summary_path, index=False, float_format='%.4f')
        print(f"✓ Individual model correlation summary saved: {individual_summary_path}")
        
        # Save to Excel with formatting
        try:
            individual_excel_path = self.output_path / "individual_model_correlation_summary.xlsx"
            with pd.ExcelWriter(individual_excel_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Individual Model Correlations', index=False, float_format='%.4f')
                
                # Also create a pivot table view for easier reading
                pivot_df = summary_df.pivot_table(
                    values='Pearson_r', 
                    index=['Model', 'Precision_Level'], 
                    columns='Metric',
                    fill_value=np.nan
                )
                pivot_df.to_excel(writer, sheet_name='Pivot View - Correlations')
                
                # Create significance pivot table
                significance_pivot = summary_df.pivot_table(
                    values='Significance', 
                    index=['Model', 'Precision_Level'], 
                    columns='Metric',
                    aggfunc='first',
                    fill_value='ns'
                )
                significance_pivot.to_excel(writer, sheet_name='Pivot View - Significance')
                
            print(f"✓ Individual model correlation summary Excel saved: {individual_excel_path}")
        except ImportError:
            print("⚠ openpyxl not available, skipping Excel export for individual models")
        
        # Also create a summary statistics table
        self._create_individual_model_summary_stats(summary_df)
    
    def _create_individual_model_summary_stats(self, summary_df: pd.DataFrame) -> None:
        """Create summary statistics for individual models."""
        print("\n📊 Creating individual model summary statistics...")
        
        # Calculate summary statistics per model
        model_stats = []
        
        for model in summary_df['Model'].unique():
            model_data = summary_df[summary_df['Model'] == model]
            
            # Remove NaN correlations for calculation
            valid_correlations = model_data['Pearson_r'].dropna()
            
            if len(valid_correlations) > 0:
                stats = {
                    'Model': model,
                    'Valid_Correlations': len(valid_correlations),
                    'Mean_Correlation': valid_correlations.mean(),
                    'Median_Correlation': valid_correlations.median(),
                    'Std_Correlation': valid_correlations.std(),
                    'Min_Correlation': valid_correlations.min(),
                    'Max_Correlation': valid_correlations.max(),
                    'Significant_Correlations': len(model_data[model_data['Significance'].isin(['*', '**', '***'])]),
                    'Strong_Correlations': len(model_data[abs(model_data['Pearson_r']) >= 0.7]),
                    'Moderate_Correlations': len(model_data[(abs(model_data['Pearson_r']) >= 0.5) & (abs(model_data['Pearson_r']) < 0.7)])
                }
                model_stats.append(stats)
        
        if model_stats:
            stats_df = pd.DataFrame(model_stats)
            
            # Save model statistics to CSV
            stats_path = self.output_path / "individual_model_statistics.csv"
            stats_df.to_csv(stats_path, index=False, float_format='%.4f')
            print(f"✓ Individual model statistics saved: {stats_path}")
            
            # Save to Excel if available
            try:
                stats_excel_path = self.output_path / "individual_model_statistics.xlsx"
                with pd.ExcelWriter(stats_excel_path, engine='openpyxl') as writer:
                    stats_df.to_excel(writer, sheet_name='Model Statistics', index=False, float_format='%.4f')
                print(f"✓ Individual model statistics Excel saved: {stats_excel_path}")
            except ImportError:
                pass
    
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
        
        # Keep x-axis labels horizontal for better readability
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
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
        cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=20)
        
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
    
    def create_individual_model_heatmaps(self) -> None:
        """Create and save correlation heatmaps for each individual model."""
        if not self.individual_model_results:
            print("⚠ No individual model results available for heatmap generation")
            return
        
        # Create subdirectory for individual model heatmaps
        individual_heatmaps_dir = self.output_path / "individual_model_heatmaps"
        individual_heatmaps_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 Creating individual model heatmaps...")
        
        for model_name, model_results in self.individual_model_results.items():
            # Prepare correlation matrix for this model
            correlation_matrix = []
            precision_labels = []
            
            for precision_level in PRECISION_LEVELS:
                if precision_level in model_results:
                    correlations = []
                    for metric in METRICS_TO_ANALYZE:
                        if metric in model_results[precision_level]:
                            corr_value = model_results[precision_level][metric]['correlation']
                            correlations.append(corr_value)
                        else:
                            correlations.append(np.nan)
                    correlation_matrix.append(correlations)
                    precision_labels.append(precision_level.capitalize())
            
            if not correlation_matrix:
                print(f"⚠ No correlation data available for model {model_name}")
                continue
            
            correlation_matrix = np.array(correlation_matrix)
            
            # Create heatmap for this specific model
            plt.figure(figsize=(12, 8))
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
                    if i < len(correlation_matrix) and j < len(correlation_matrix[i]):
                        corr_value = correlation_matrix[i, j]
                        if not np.isnan(corr_value):
                            precision_level = PRECISION_LEVELS[i]
                            metric = METRICS_TO_ANALYZE[j]
                            if (precision_level in model_results and 
                                metric in model_results[precision_level]):
                                significance = model_results[precision_level][metric]['significance']
                                text = f"{corr_value:.3f}\n{significance}"
                                ax.text(j, i, text, ha="center", va="center", 
                                       color="white" if abs(corr_value) > 0.5 else "black",
                                       fontsize=10, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=20)
            
            # Set titles and labels
            dataset_title = ', '.join([d.upper() for d in self.datasets]) if len(self.datasets) > 1 else self.datasets[0].upper()
            plt.title(f'Zero-shot vs Iterative Performance Correlations\nModel: {model_name} | Dataset(s): {dataset_title}', 
                     fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision Levels', fontsize=12, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot with model name in filename
            safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            heatmap_path = individual_heatmaps_dir / f"{safe_model_name}_correlation_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Individual heatmap saved for {model_name}: {heatmap_path}")
        
        print(f"✓ All individual model heatmaps saved in: {individual_heatmaps_dir}")
    
    def print_summary_report(self) -> None:
        """Print a comprehensive summary report to console."""
        dataset_title = ', '.join([d.upper() for d in self.datasets])
        print(f"\n{'='*70}")
        print(f"PEARSON CORRELATION ANALYSIS SUMMARY")
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
                
                print(f"  {metric:20s}: r = {correlation:6.3f} {significance:3s} "
                      f"(p = {p_value:.3f}, n = {n_samples:2d}) - {interpretation}")
        
        print(f"\n{'='*70}")
        print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print(f"Results saved in: {self.output_path}")
        print(f"{'='*70}")
    
    def print_individual_model_summary(self) -> None:
        """Print a summary report for individual model correlations."""
        if not self.individual_model_results:
            return
        
        dataset_title = ', '.join([d.upper() for d in self.datasets])
        print(f"\n{'='*70}")
        print(f"INDIVIDUAL MODEL CORRELATION ANALYSIS SUMMARY")
        print(f"Dataset(s): {dataset_title}")
        print(f"{'='*70}")
        
        for model_name, model_results in self.individual_model_results.items():
            print(f"\n🤖 MODEL: {model_name}")
            print("-" * 60)
            
            for precision_level in PRECISION_LEVELS:
                if precision_level not in model_results:
                    continue
                    
                print(f"\n📈 {precision_level.upper()} PRECISION LEVEL:")
                print("-" * 40)
                
                for metric in METRICS_TO_ANALYZE:
                    if metric not in model_results[precision_level]:
                        continue
                        
                    stats = model_results[precision_level][metric]
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
                    
                    print(f"  {metric:20s}: r = {correlation:6.3f} {significance:3s} "
                          f"(p = {p_value:.3f}, n = {n_samples:2d}) - {interpretation}")
        
        print(f"\n{'='*70}")
        print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print(f"Individual model results saved in: {self.output_path}")
        print(f"{'='*70}")
    
    def generate_correlation_report(self) -> None:
        """Generate a comprehensive correlation analysis report."""
        report_path = self.output_path / "pearson_correlation_report.txt"
        
        dataset_title = ', '.join([d.upper() for d in self.datasets]) if len(self.datasets) > 1 else self.datasets[0].upper()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PEARSON CORRELATION ANALYSIS REPORT\n")
            f.write(f"Dataset(s): {dataset_title}\n")
            f.write("="*80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*50 + "\n")
            f.write("This report analyzes the Pearson linear correlation between zero-shot and\n")
            f.write("iterative model performance across different precision levels and metrics.\n")
            f.write("Higher correlations indicate more consistent linear relationships between the\n")
            f.write("two inference approaches.\n\n")
            
            # Summary statistics
            strong_correlations = []
            moderate_correlations = []
            weak_correlations = []
            
            for precision_level, metrics in self.correlation_results.items():
                for metric, stats in metrics.items():
                    correlation = stats['correlation']
                    if not np.isnan(correlation):
                        abs_corr = abs(correlation)
                        corr_info = (precision_level, metric, correlation, stats['p_value'], stats['significance'])
                        
                        if abs_corr >= 0.7:
                            strong_correlations.append(corr_info)
                        elif abs_corr >= 0.4:
                            moderate_correlations.append(corr_info)
                        else:
                            weak_correlations.append(corr_info)
            
            f.write(f"• Strong correlations (|r| ≥ 0.7): {len(strong_correlations)} cases\n")
            f.write(f"• Moderate correlations (|r| ≥ 0.4): {len(moderate_correlations)} cases\n")
            f.write(f"• Weak correlations (|r| < 0.4): {len(weak_correlations)} cases\n\n")
            
            # Detailed findings by precision level
            f.write("DETAILED FINDINGS\n")
            f.write("-"*50 + "\n")
            
            for precision_level in PRECISION_LEVELS:
                if precision_level not in self.correlation_results:
                    continue
                    
                f.write(f"\n{precision_level.upper()} PRECISION LEVEL:\n")
                f.write("-"*40 + "\n")
                
                precision_results = self.correlation_results[precision_level]
                for metric, stats in precision_results.items():
                    correlation = stats['correlation']
                    p_value = stats['p_value']
                    n_samples = stats['n_samples']
                    significance = stats['significance']
                    
                    # Interpret correlation strength
                    if np.isnan(correlation):
                        interpretation = "Insufficient data"
                    else:
                        abs_corr = abs(correlation)
                        direction = "positive" if correlation > 0 else "negative"
                        
                        if abs_corr >= 0.7:
                            strength = "strong"
                        elif abs_corr >= 0.5:
                            strength = "moderate"
                        elif abs_corr >= 0.3:
                            strength = "weak"
                        else:
                            strength = "very weak"
                        
                        interpretation = f"{strength} {direction} correlation"
                    
                    f.write(f"  {metric}:\n")
                    f.write(f"    • Correlation: r = {correlation:.3f}\n")
                    f.write(f"    • Interpretation: {interpretation}\n")
                    f.write(f"    • Statistical significance: {significance} (p = {p_value:.4f})\n")
                    f.write(f"    • Sample size: n = {n_samples}\n")
                    f.write("\n")
            
            # Practical implications
            f.write("PRACTICAL IMPLICATIONS\n")
            f.write("-"*50 + "\n")
            
            if strong_correlations:
                strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                f.write("STRONG CORRELATIONS indicate consistent linear relationships:\n")
                for precision, metric, corr, p_val, sig in strong_correlations[:5]:
                    f.write(f"• {precision.capitalize()}/{metric}: r = {corr:.3f} {sig}\n")
                f.write("  → Zero-shot and iterative approaches show linear relationships\n")
                f.write("  → Model performance scales consistently across approaches\n\n")
            
            if weak_correlations:
                f.write("WEAK CORRELATIONS suggest different performance patterns:\n")
                weak_sorted = sorted(weak_correlations, key=lambda x: abs(x[2]))
                for precision, metric, corr, p_val, sig in weak_sorted[:3]:
                    f.write(f"• {precision.capitalize()}/{metric}: r = {corr:.3f} {sig}\n")
                f.write("  → Performance patterns differ between approaches\n")
                f.write("  → Consider approach-specific model selection\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*50 + "\n")
            
            # Analyze precision level patterns
            precision_avg_corrs = {}
            for precision_level in ['low', 'medium', 'high']:
                if precision_level in self.correlation_results:
                    corrs = [stats['correlation'] for stats in self.correlation_results[precision_level].values()
                            if not np.isnan(stats['correlation'])]
                    if corrs:
                        precision_avg_corrs[precision_level] = np.mean([abs(c) for c in corrs])
            
            if len(precision_avg_corrs) >= 2:
                best_precision = max(precision_avg_corrs.keys(), key=lambda x: precision_avg_corrs[x])
                worst_precision = min(precision_avg_corrs.keys(), key=lambda x: precision_avg_corrs[x])
                
                f.write("Based on correlation analysis:\n\n")
                f.write(f"1. Most consistent linear relationships: {best_precision.upper()} precision\n")
                f.write(f"   (Average |correlation| = {precision_avg_corrs[best_precision]:.3f})\n")
                f.write(f"   → Reliable to use either zero-shot or iterative for model selection\n\n")
                
                f.write(f"2. Least consistent linear relationships: {worst_precision.upper()} precision\n")
                f.write(f"   (Average |correlation| = {precision_avg_corrs[worst_precision]:.3f})\n")
                f.write(f"   → Choose evaluation approach carefully for model selection\n\n")
            
            # Find most reliable metrics
            metric_avg_corrs = {}
            for metric in METRICS_TO_ANALYZE:
                corrs = []
                for precision_level in self.correlation_results.values():
                    if metric in precision_level and not np.isnan(precision_level[metric]['correlation']):
                        corrs.append(abs(precision_level[metric]['correlation']))
                if corrs:
                    metric_avg_corrs[metric] = np.mean(corrs)
            
            if metric_avg_corrs:
                sorted_metrics = sorted(metric_avg_corrs.items(), key=lambda x: x[1], reverse=True)
                f.write("3. Most consistent metrics across precision levels:\n")
                for metric, avg_corr in sorted_metrics[:3]:
                    f.write(f"   • {metric}: average |correlation| = {avg_corr:.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Pearson correlation report saved: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute Pearson correlations between zero-shot and iterative model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/stats_utils/pearson_correlation.py              # Analyze all available datasets
  python evaluation/stats_utils/pearson_correlation.py -d euler_1d  # Analyze specific dataset
  python evaluation/stats_utils/pearson_correlation.py -d heat_1d
  python evaluation/stats_utils/pearson_correlation.py -i                   # Include individual model analysis

Output files (essential files only):
  - eval_results/stats/pearson_correlation/{dataset}/correlation_summary.csv
  - eval_results/stats/pearson_correlation/{dataset}/correlation_summary.xlsx
  - eval_results/stats/pearson_correlation/{dataset}/correlation_heatmap.png
  
Additional files when using -i (--individual-models):
  - eval_results/stats/pearson_correlation/{dataset}/individual_model_correlation_summary.csv
  - eval_results/stats/pearson_correlation/{dataset}/individual_model_correlation_summary.xlsx
  - eval_results/stats/pearson_correlation/{dataset}/individual_model_statistics.csv
  - eval_results/stats/pearson_correlation/{dataset}/individual_model_statistics.xlsx
  - eval_results/stats/pearson_correlation/{dataset}/individual_model_heatmaps/{model}_correlation_heatmap.png
        """
    )
    
    parser.add_argument(
        '-d', '--dataset',
        required=False,
        help='Dataset name (e.g., euler_1d, heat_1d, burgers_1d). If not specified, analyzes all available datasets.'
    )
    
    parser.add_argument(
        '-i', '--individual-models',
        action='store_true',
        help='Also perform individual model correlation analysis (in addition to overall analysis)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.dataset:
            print(f"🚀 Starting Pearson correlation analysis for dataset: {args.dataset}")
            analyzer = PearsonCorrelationAnalyzer(args.dataset)
        else:
            print(f"🚀 Starting Pearson correlation analysis for all available datasets")
            analyzer = PearsonCorrelationAnalyzer()
        
        # Load data
        zero_shot_df, iterative_df = analyzer.load_evaluation_data()
        
        # Perform correlation analysis
        analyzer.analyze_all_correlations(zero_shot_df, iterative_df)
        
        # Save results (only essential files)
        analyzer.save_correlation_summary()
        analyzer.create_correlation_heatmap()
        analyzer.generate_correlation_report()
        
        # Perform individual model analysis if requested
        if args.individual_models:
            print(f"\n🚀 Starting individual model correlation analysis...")
            analyzer.analyze_individual_model_correlations(zero_shot_df, iterative_df)
            analyzer.save_individual_model_summary()
            analyzer.create_individual_model_heatmaps()
        
        # Print summary
        analyzer.print_summary_report()
        
        # Print individual model summary if available
        if hasattr(analyzer, 'individual_model_results') and analyzer.individual_model_results:
            analyzer.print_individual_model_summary()
        
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()