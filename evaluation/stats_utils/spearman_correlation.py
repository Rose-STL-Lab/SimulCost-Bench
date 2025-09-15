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
python evaluation/stats_utils/spearman_correlation.py              # Analyze all available datasets
python evaluation/stats_utils/spearman_correlation.py -d euler_1d  # Analyze specific dataset
python evaluation/stats_utils/spearman_correlation.py -d heat_1d

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
METRICS_TO_ANALYZE = ['success_rate', 'mean_efficiency']
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
            self.output_path = Path(f"eval_results/stats/spearman_correlation/{self.datasets[0]}")
        else:
            self.output_path = Path("eval_results/stats/spearman_correlation/multi_dataset")
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
    
    def analyze_individual_model_correlations(self, zero_shot_df: pd.DataFrame, 
                                            iterative_df: pd.DataFrame) -> Dict:
        """
        Perform Spearman correlation analysis for each individual model.
        
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
                    correlation, p_value, n_samples = self.compute_spearman_correlation(paired_data, metric)
                    
                    level_results[metric] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'n_samples': n_samples,
                        'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                    }
                    
                    if not np.isnan(correlation):
                        print(f"  {precision_level}-{metric}: ρ = {correlation:.3f} (p = {p_value:.3f}, n = {n_samples})")
                
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
                        'Spearman_rho': stats['correlation'],
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
                    values='Spearman_rho', 
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
            valid_correlations = model_data['Spearman_rho'].dropna()
            
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
                    'Strong_Correlations': len(model_data[abs(model_data['Spearman_rho']) >= 0.7]),
                    'Moderate_Correlations': len(model_data[(abs(model_data['Spearman_rho']) >= 0.5) & (abs(model_data['Spearman_rho']) < 0.7)])
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
                    text = f"{correlation_matrix[i, j]:.3f}"
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
                                text = f"{corr_value:.3f}"
                                ax.text(j, i, text, ha="center", va="center",
                                       color="white" if abs(corr_value) > 0.5 else "black",
                                       fontsize=10, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Spearman Correlation Coefficient', rotation=270, labelpad=20)
            
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
                    
                    print(f"  {metric:20s}: ρ = {correlation:6.3f} {significance:3s} "
                          f"(p = {p_value:.3f}, n = {n_samples:2d}) - {interpretation}")
        
        print(f"\n{'='*70}")
        print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        print(f"Individual model results saved in: {self.output_path}")
        print(f"{'='*70}")
    
    def generate_correlation_report(self) -> None:
        """Generate a comprehensive correlation analysis report."""
        report_path = self.output_path / "spearman_correlation_report.txt"
        
        dataset_title = ', '.join([d.upper() for d in self.datasets]) if len(self.datasets) > 1 else self.datasets[0].upper()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SPEARMAN RANK CORRELATION ANALYSIS REPORT\n")
            f.write(f"Dataset(s): {dataset_title}\n")
            f.write("="*80 + "\n\n")

            # Statistical Principles
            f.write("STATISTICAL PRINCIPLES AND METHODOLOGY\n")
            f.write("-"*50 + "\n")
            f.write("This analysis employs Spearman rank correlation to measure the monotonic\n")
            f.write("relationship between zero-shot and iterative model performance rankings.\n")
            f.write("Unlike Pearson correlation, Spearman correlation assesses whether the\n")
            f.write("relative ordering of models is consistent across inference modes.\n\n")

            f.write("Mathematical Formulation:\n\n")
            f.write("Spearman correlation coefficient: ρ = 1 - (6∑di²) / (n(n²-1))\n")
            f.write("Where:\n")
            f.write("  • di = difference between ranks of model i in zero-shot vs iterative\n")
            f.write("  • n = number of models\n")
            f.write("  • ρ ranges from -1 to +1\n\n")

            f.write("Alternative formulation (equivalent):\n")
            f.write("ρ = Pearson correlation between rank(Xi) and rank(Yi)\n")
            f.write("Where:\n")
            f.write("  • rank(Xi) = rank of model i in zero-shot performance\n")
            f.write("  • rank(Yi) = rank of model i in iterative performance\n\n")

            f.write("Correlation Interpretation:\n")
            f.write("• |ρ| ≥ 0.7:  Strong monotonic relationship (consistent rankings)\n")
            f.write("• 0.5 ≤ |ρ| < 0.7:  Moderate monotonic relationship\n")
            f.write("• 0.3 ≤ |ρ| < 0.5:  Weak monotonic relationship\n")
            f.write("• |ρ| < 0.3:  Very weak/negligible monotonic relationship\n\n")

            f.write("Positive correlation (ρ > 0): Models ranked higher in zero-shot tend to\n")
            f.write("be ranked higher in iterative mode (and vice versa).\n")
            f.write("Negative correlation (ρ < 0): Models ranked higher in zero-shot tend to\n")
            f.write("be ranked lower in iterative mode (inverse ranking relationship).\n\n")

            f.write("Key Advantages of Spearman over Pearson:\n")
            f.write("• Robust to outliers and non-linear relationships\n")
            f.write("• Focuses on ranking consistency rather than linear scaling\n")
            f.write("• No assumptions about data distribution\n")
            f.write("• More appropriate for ordinal data and model comparisons\n\n")

            f.write("Statistical Significance:\n")
            f.write("• p < 0.001: *** (highly significant)\n")
            f.write("• p < 0.01:  ** (very significant)\n")
            f.write("• p < 0.05:  * (significant)\n")
            f.write("• p ≥ 0.05:  ns (not significant)\n\n")

            # Model Ranking Consistency Analysis
            f.write("MODEL RANKING CONSISTENCY ANALYSIS\n")
            f.write("-"*50 + "\n")
            f.write("This analysis addresses the critical question: Do models maintain consistent\n")
            f.write("relative rankings between zero-shot and iterative inference modes? High\n")
            f.write("positive correlations indicate that the top-performing models in zero-shot\n")
            f.write("evaluation are also likely to be top performers in iterative evaluation.\n\n")
            
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
            
            f.write(f"• Strong correlations (|ρ| ≥ 0.7): {len(strong_correlations)} cases\n")
            f.write(f"• Moderate correlations (|ρ| ≥ 0.4): {len(moderate_correlations)} cases\n")
            f.write(f"• Weak correlations (|ρ| < 0.4): {len(weak_correlations)} cases\n\n")
            
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
                    f.write(f"    • Correlation: ρ = {correlation:.3f}\n")
                    f.write(f"    • Interpretation: {interpretation}\n")
                    f.write(f"    • Statistical significance: {significance} (p = {p_value:.4f})\n")
                    f.write(f"    • Sample size: n = {n_samples}\n")
                    f.write("\n")
            
            # Model Ranking Consistency Findings
            f.write("MODEL RANKING CONSISTENCY FINDINGS\n")
            f.write("-"*50 + "\n")

            # Analyze ranking consistency
            highly_consistent = []
            moderately_consistent = []
            poorly_consistent = []

            for precision_level, metrics in self.correlation_results.items():
                for metric, stats in metrics.items():
                    correlation = stats['correlation']
                    if not np.isnan(correlation):
                        abs_corr = abs(correlation)
                        # Calculate ranking agreement percentage (approximate)
                        rank_agreement = ((correlation + 1) / 2 * 100) if correlation > 0 else 0
                        corr_info = (precision_level, metric, correlation, rank_agreement, stats['p_value'], stats['significance'])

                        if abs_corr >= 0.7:
                            highly_consistent.append(corr_info)
                        elif abs_corr >= 0.4:
                            moderately_consistent.append(corr_info)
                        else:
                            poorly_consistent.append(corr_info)

            if highly_consistent:
                highly_consistent.sort(key=lambda x: abs(x[2]), reverse=True)
                f.write("HIGH RANKING CONSISTENCY (Strong correlations |ρ| ≥ 0.7):\n")
                f.write("Zero-shot rankings strongly predict iterative rankings\n\n")
                for precision, metric, corr, rank_agree, p_val, sig in highly_consistent:
                    f.write(f"• {precision.capitalize()} precision - {metric}:\n")
                    f.write(f"  - Spearman correlation: ρ = {corr:.3f} {sig}\n")
                    if corr > 0:
                        f.write(f"  - Ranking agreement: ~{rank_agree:.0f}%\n")
                        f.write(f"  - Interpretation: Top zero-shot models are also top iterative models\n")
                        f.write(f"  - Implication: Zero-shot evaluation reliably identifies best models for iterative use\n")
                    else:
                        f.write(f"  - Interpretation: Inverse ranking relationship (rare)\n")
                        f.write(f"  - Implication: Top zero-shot models may be poor iterative models\n")
                    f.write("\n")

            if moderately_consistent:
                moderately_consistent.sort(key=lambda x: abs(x[2]), reverse=True)
                f.write("MODERATE RANKING CONSISTENCY (Moderate correlations 0.4 ≤ |ρ| < 0.7):\n")
                f.write("Zero-shot rankings moderately predict iterative rankings\n\n")
                for precision, metric, corr, rank_agree, p_val, sig in moderately_consistent:
                    f.write(f"• {precision.capitalize()} precision - {metric}:\n")
                    f.write(f"  - Spearman correlation: ρ = {corr:.3f} {sig}\n")
                    if corr > 0:
                        f.write(f"  - Ranking agreement: ~{rank_agree:.0f}%\n")
                        f.write(f"  - Interpretation: Moderate consistency in model rankings\n")
                    f.write("\n")

            if poorly_consistent:
                f.write("LOW RANKING CONSISTENCY (Weak correlations |ρ| < 0.4):\n")
                f.write("Zero-shot rankings poorly predict iterative rankings\n\n")
                poorly_consistent_sorted = sorted(poorly_consistent, key=lambda x: abs(x[2]), reverse=True)
                for precision, metric, corr, rank_agree, p_val, sig in poorly_consistent_sorted:
                    f.write(f"• {precision.capitalize()} precision - {metric}:\n")
                    f.write(f"  - Spearman correlation: ρ = {corr:.3f} {sig}\n")
                    if corr > 0:
                        f.write(f"  - Ranking agreement: ~{rank_agree:.0f}%\n")
                    else:
                        f.write(f"  - Ranking agreement: Very low (inconsistent orderings)\n")
                    f.write(f"  - Implication: Zero-shot rankings provide limited guidance for iterative model selection\n")
                    f.write("\n")
            
            # Model Selection Strategy Based on Ranking Consistency
            f.write("MODEL SELECTION STRATEGY BASED ON RANKING CONSISTENCY\n")
            f.write("-"*50 + "\n")

            # Calculate overall ranking consistency statistics
            all_correlations = []
            highly_consistent_cases = []
            for precision_level, metrics in self.correlation_results.items():
                for metric, stats in metrics.items():
                    if not np.isnan(stats['correlation']):
                        all_correlations.append(abs(stats['correlation']))
                        if abs(stats['correlation']) >= 0.7 and stats['correlation'] > 0:
                            highly_consistent_cases.append((precision_level, metric, stats['correlation']))

            if all_correlations:
                avg_consistency = np.mean(all_correlations)
                f.write("OVERALL RANKING CONSISTENCY ASSESSMENT:\n")
                f.write(f"• Average ranking consistency: |ρ| = {avg_consistency:.3f}\n")
                f.write(f"• Strong ranking consistency (|ρ| ≥ 0.7): {len([c for c in all_correlations if c >= 0.7])}/{len(all_correlations)} cases\n")
                f.write(f"• Moderate+ ranking consistency (|ρ| ≥ 0.4): {len([c for c in all_correlations if c >= 0.4])}/{len(all_correlations)} cases\n\n")

            # Precision level ranking consistency analysis
            precision_consistency = {}
            for precision_level in ['low', 'medium', 'high']:
                if precision_level in self.correlation_results:
                    corrs = [stats['correlation'] for stats in self.correlation_results[precision_level].values()
                            if not np.isnan(stats['correlation'])]
                    if corrs:
                        abs_corrs = [abs(c) for c in corrs]
                        positive_corrs = [c for c in corrs if c > 0]
                        precision_consistency[precision_level] = {
                            'avg_consistency': np.mean(abs_corrs),
                            'strong_consistency_count': len([c for c in abs_corrs if c >= 0.7]),
                            'total_count': len(abs_corrs),
                            'positive_correlations': len(positive_corrs),
                            'avg_positive_correlation': np.mean(positive_corrs) if positive_corrs else 0
                        }

            if precision_consistency:
                f.write("PRECISION-LEVEL RANKING CONSISTENCY:\n")
                for precision, stats in precision_consistency.items():
                    f.write(f"• {precision.capitalize()} precision tasks:\n")
                    f.write(f"  - Average ranking consistency: |ρ| = {stats['avg_consistency']:.3f}\n")
                    f.write(f"  - Strong consistency: {stats['strong_consistency_count']}/{stats['total_count']} metrics\n")
                    f.write(f"  - Positive correlations: {stats['positive_correlations']}/{stats['total_count']} metrics\n")

                    if stats['avg_consistency'] >= 0.7:
                        f.write(f"  - Model selection strategy: Zero-shot rankings highly reliable\n")
                        f.write(f"  - Recommendation: Use zero-shot evaluation for efficient model ranking\n")
                    elif stats['avg_consistency'] >= 0.4:
                        f.write(f"  - Model selection strategy: Zero-shot rankings moderately reliable\n")
                        f.write(f"  - Recommendation: Zero-shot provides reasonable initial model ranking\n")
                    else:
                        f.write(f"  - Model selection strategy: Rankings inconsistent between modes\n")
                        f.write(f"  - Recommendation: Evaluate models in both zero-shot and iterative modes\n")
                    f.write("\n")

                # Find most reliable precision level for ranking consistency
                best_precision = max(precision_consistency.keys(),
                                   key=lambda x: precision_consistency[x]['avg_consistency'])
                f.write(f"MOST RELIABLE PRECISION LEVEL FOR ZERO-SHOT RANKING: {best_precision.upper()}\n")
                f.write(f"(Average consistency: |ρ| = {precision_consistency[best_precision]['avg_consistency']:.3f})\n\n")

            # Metric-specific ranking consistency
            metric_consistency = {}
            for metric in METRICS_TO_ANALYZE:
                corrs = []
                for precision_level in self.correlation_results.values():
                    if metric in precision_level and not np.isnan(precision_level[metric]['correlation']):
                        corrs.append(precision_level[metric]['correlation'])
                if corrs:
                    abs_corrs = [abs(c) for c in corrs]
                    positive_corrs = [c for c in corrs if c > 0]
                    metric_consistency[metric] = {
                        'avg_consistency': np.mean(abs_corrs),
                        'avg_positive_correlation': np.mean(positive_corrs) if positive_corrs else 0,
                        'consistency_level': 'High' if np.mean(abs_corrs) >= 0.7 else 'Moderate' if np.mean(abs_corrs) >= 0.4 else 'Low',
                        'positive_cases': len(positive_corrs),
                        'total_cases': len(corrs)
                    }

            if metric_consistency:
                f.write("METRIC-SPECIFIC RANKING CONSISTENCY:\n")
                sorted_metrics = sorted(metric_consistency.items(),
                                      key=lambda x: x[1]['avg_consistency'], reverse=True)
                for metric, stats in sorted_metrics:
                    f.write(f"• {metric.replace('_', ' ').title()}:\n")
                    f.write(f"  - Average ranking consistency: |ρ| = {stats['avg_consistency']:.3f}\n")
                    f.write(f"  - Consistency level: {stats['consistency_level']}\n")
                    f.write(f"  - Positive correlations: {stats['positive_cases']}/{stats['total_cases']} precision levels\n")

                    if stats['consistency_level'] == 'High':
                        f.write(f"  - Strategy: Zero-shot {metric} rankings are highly reliable for model selection\n")
                    elif stats['consistency_level'] == 'Moderate':
                        f.write(f"  - Strategy: Zero-shot {metric} rankings provide reasonable model guidance\n")
                    else:
                        f.write(f"  - Strategy: Evaluate {metric} in both modes for accurate model comparison\n")
                    f.write("\n")

            f.write("PRACTICAL MODEL SELECTION RECOMMENDATIONS:\n")
            f.write("1. FOR EFFICIENT MODEL SCREENING:\n")
            if highly_consistent_cases:
                f.write("   Use zero-shot rankings to identify top models for:\n")
                for precision, metric, corr in highly_consistent_cases:
                    f.write(f"   • {precision.capitalize()} precision {metric} tasks (ρ = {corr:.3f})\n")
                f.write("   These show strong ranking agreement - top zero-shot models will likely\n")
                f.write("   remain top performers in iterative mode.\n\n")
            else:
                f.write("   • No cases with strong ranking consistency found\n")
                f.write("   • Consider evaluating both modes for reliable model selection\n\n")

            f.write("2. FOR COMPREHENSIVE MODEL EVALUATION:\n")
            f.write("   • Always validate final model selection with both evaluation modes\n")
            f.write("   • Pay special attention to metrics with low ranking consistency\n")
            f.write("   • Consider that rankings may shift significantly between modes\n")
            f.write("   • Use zero-shot for initial screening, iterative for final validation\n\n")

            f.write("3. UNDERSTANDING RANKING SHIFTS:\n")
            f.write("   • Strong positive correlations: Consistent relative model performance\n")
            f.write("   • Weak correlations: Different models excel in different inference modes\n")
            f.write("   • Consider task requirements when choosing evaluation approach\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Spearman correlation report saved: {report_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute Spearman correlations between zero-shot and iterative model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/stats_utils/spearman_correlation.py              # Analyze all available datasets
  python evaluation/stats_utils/spearman_correlation.py -d euler_1d  # Analyze specific dataset
  python evaluation/stats_utils/spearman_correlation.py -d heat_1d
  python evaluation/stats_utils/spearman_correlation.py -i                   # Include individual model analysis

Output files (essential files only):
  - eval_results/stats/spearman_correlation/{dataset}/correlation_summary.csv
  - eval_results/stats/spearman_correlation/{dataset}/correlation_summary.xlsx
  - eval_results/stats/spearman_correlation/{dataset}/correlation_heatmap.png
  
Additional files when using -i (--individual-models):
  - eval_results/stats/spearman_correlation/{dataset}/individual_model_correlation_summary.csv
  - eval_results/stats/spearman_correlation/{dataset}/individual_model_correlation_summary.xlsx
  - eval_results/stats/spearman_correlation/{dataset}/individual_model_statistics.csv
  - eval_results/stats/spearman_correlation/{dataset}/individual_model_statistics.xlsx
  - eval_results/stats/spearman_correlation/{dataset}/individual_model_heatmaps/{model}_correlation_heatmap.png
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