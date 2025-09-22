#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect Size Analysis for Zero-shot vs Iterative Model Performance

This script computes effect sizes (Cohen's d, Glass's Δ, Hedges' g) to quantify 
the magnitude of performance differences between zero-shot and iterative inference 
modes across different models and precision levels.

Usage
-----
python evaluation/effect_size_analysis.py              # Analyze all available datasets
python evaluation/effect_size_analysis.py -d euler_1d  # Analyze specific dataset
python evaluation/effect_size_analysis.py -d heat_1d

Output: Creates analysis results with:
- effect_size_summary.csv: Numerical effect size coefficients and interpretations
- effect_size_summary.xlsx: Excel version with formatting
- effect_size_heatmap.png: Heatmap visualization of effect sizes across metrics
- effect_size_bar_chart.png: Bar chart comparing effect sizes by precision level
- effect_size_report.txt: Detailed interpretation report
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
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon

# Setup path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration constants
METRICS_TO_ANALYZE = ['success_rate', 'mean_efficiency']
PRECISION_LEVELS = ['low', 'medium', 'high', 'overall']
INFERENCE_MODES = ['Zero-shot', 'Iterative']

class EffectSizeAnalyzer:
    """
    Professional analyzer for computing effect sizes between zero-shot and 
    iterative model performance across multiple metrics and precision levels.
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
            self.output_path = Path(f"eval_results/stats/effect_size_analysis/{self.datasets[0]}")
        else:
            self.output_path = Path("eval_results/stats/effect_size_analysis/multi_dataset")
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Storage for analysis results
        self.effect_size_results = {}
        self.statistical_test_results = {}
        
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
            if dataset_path.is_dir() and dataset_path.name not in ["spearman_correlation", "effect_size_analysis"]:
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
        Prepare paired data for effect size analysis by matching models and datasets.
        
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
        if precision_level != 'overall' and 'Precision Level' in zero_shot_filtered.columns:
            merge_cols.append('Precision Level')
            
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
    
    def compute_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size.
        
        Cohen's d = (mean1 - mean2) / pooled_standard_deviation
        
        Parameters
        ----------
        group1, group2 : np.ndarray
            Two groups to compare
            
        Returns
        -------
        float
            Cohen's d effect size
        """
        if len(group1) == 0 or len(group2) == 0:
            return np.nan
            
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return np.nan
            
        return (mean1 - mean2) / pooled_std
    
    def compute_hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Hedges' g effect size (bias-corrected Cohen's d).
        
        Parameters
        ----------
        group1, group2 : np.ndarray
            Two groups to compare
            
        Returns
        -------
        float
            Hedges' g effect size
        """
        cohens_d = self.compute_cohens_d(group1, group2)
        if np.isnan(cohens_d):
            return np.nan
            
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        
        # Bias correction factor
        j = 1 - (3 / (4 * df - 1))
        
        return cohens_d * j
    
    def compute_glass_delta(self, group1: np.ndarray, group2: np.ndarray, control_is_group2: bool = True) -> float:
        """
        Compute Glass's Δ effect size.
        
        Glass's Δ = (mean1 - mean2) / std_control
        
        Parameters
        ----------
        group1, group2 : np.ndarray
            Two groups to compare
        control_is_group2 : bool
            If True, group2 is the control group; otherwise group1
            
        Returns
        -------
        float
            Glass's Δ effect size
        """
        if len(group1) == 0 or len(group2) == 0:
            return np.nan
            
        mean1, mean2 = np.mean(group1), np.mean(group2)
        
        if control_is_group2:
            control_std = np.std(group2, ddof=1)
        else:
            control_std = np.std(group1, ddof=1)
            
        if control_std == 0:
            return np.nan
            
        return (mean1 - mean2) / control_std
    
    def interpret_effect_size(self, effect_size: float, metric_name: str = "Cohen's d") -> str:
        """
        Interpret effect size magnitude according to Cohen's conventions.
        
        Parameters
        ----------
        effect_size : float
            The computed effect size
        metric_name : str
            Name of the effect size metric
            
        Returns
        -------
        str
            Interpretation of the effect size
        """
        if np.isnan(effect_size):
            return "Cannot determine (insufficient data)"
            
        abs_effect = abs(effect_size)
        direction = "favoring iterative" if effect_size > 0 else "favoring zero-shot"
        
        if abs_effect < 0.2:
            magnitude = "negligible"
        elif abs_effect < 0.5:
            magnitude = "small"
        elif abs_effect < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
            
        return f"{magnitude} effect {direction}"
    
    def compute_statistical_tests(self, iterative_vals: np.ndarray, zero_shot_vals: np.ndarray) -> Dict:
        """
        Compute statistical significance tests for the difference.
        
        Parameters
        ----------
        iterative_vals, zero_shot_vals : np.ndarray
            Paired values for statistical testing
            
        Returns
        -------
        Dict
            Dictionary with test statistics and p-values
        """
        results = {}
        
        if len(iterative_vals) != len(zero_shot_vals) or len(iterative_vals) < 3:
            return {'paired_t_test': {'statistic': np.nan, 'p_value': np.nan, 'note': 'Insufficient data'},
                   'wilcoxon': {'statistic': np.nan, 'p_value': np.nan, 'note': 'Insufficient data'}}
        
        # Paired t-test
        try:
            t_stat, t_pval = ttest_rel(iterative_vals, zero_shot_vals)
            results['paired_t_test'] = {
                'statistic': t_stat,
                'p_value': t_pval,
                'significant': t_pval < 0.05,
                'note': 'Parametric test assuming normal distribution'
            }
        except Exception as e:
            results['paired_t_test'] = {'statistic': np.nan, 'p_value': np.nan, 'note': f'Error: {str(e)}'}
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            # Only test if there are differences
            differences = iterative_vals - zero_shot_vals
            if np.any(differences != 0):
                w_stat, w_pval = wilcoxon(iterative_vals, zero_shot_vals)
                results['wilcoxon'] = {
                    'statistic': w_stat,
                    'p_value': w_pval,
                    'significant': w_pval < 0.05,
                    'note': 'Non-parametric test (robust to outliers)'
                }
            else:
                results['wilcoxon'] = {'statistic': 0, 'p_value': 1.0, 'significant': False, 'note': 'No differences found'}
        except Exception as e:
            results['wilcoxon'] = {'statistic': np.nan, 'p_value': np.nan, 'note': f'Error: {str(e)}'}
            
        return results
    
    def analyze_effect_sizes(self, zero_shot_df: pd.DataFrame, 
                           iterative_df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive effect size analysis across all metrics and precision levels.
        
        Parameters
        ----------
        zero_shot_df : pd.DataFrame
            Zero-shot evaluation results
        iterative_df : pd.DataFrame
            Iterative evaluation results
            
        Returns
        -------
        Dict
            Comprehensive effect size results
        """
        results = {}
        
        print("\n🔍 Computing effect sizes...")
        
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
            
            # Compute effect sizes for each metric
            level_results = {}
            for metric in METRICS_TO_ANALYZE:
                iterative_col = f"{metric}_iterative"
                zero_shot_col = f"{metric}_zero_shot"
                
                # Extract valid pairs
                mask = (~paired_data[iterative_col].isna()) & (~paired_data[zero_shot_col].isna())
                iterative_vals = paired_data[iterative_col][mask].values
                zero_shot_vals = paired_data[zero_shot_col][mask].values
                
                if len(iterative_vals) < 3:
                    level_results[metric] = {
                        'n_samples': len(iterative_vals),
                        'cohens_d': np.nan,
                        'hedges_g': np.nan,
                        'glass_delta': np.nan,
                        'interpretation': "Insufficient data for analysis",
                        'mean_difference': np.nan,
                        'mean_iterative': np.nan,
                        'mean_zero_shot': np.nan,
                        'statistical_tests': {}
                    }
                    continue
                
                # Compute effect sizes
                cohens_d = self.compute_cohens_d(iterative_vals, zero_shot_vals)
                hedges_g = self.compute_hedges_g(iterative_vals, zero_shot_vals)
                glass_delta = self.compute_glass_delta(iterative_vals, zero_shot_vals, control_is_group2=True)
                
                # Compute statistical tests
                statistical_tests = self.compute_statistical_tests(iterative_vals, zero_shot_vals)
                
                level_results[metric] = {
                    'n_samples': len(iterative_vals),
                    'cohens_d': cohens_d,
                    'hedges_g': hedges_g,
                    'glass_delta': glass_delta,
                    'interpretation': self.interpret_effect_size(cohens_d),
                    'mean_difference': np.mean(iterative_vals) - np.mean(zero_shot_vals),
                    'mean_iterative': np.mean(iterative_vals),
                    'mean_zero_shot': np.mean(zero_shot_vals),
                    'statistical_tests': statistical_tests
                }
                
                print(f"  {metric}: Cohen's d = {cohens_d:.3f} ({self.interpret_effect_size(cohens_d)}, n = {len(iterative_vals)})")
            
            results[precision_level] = level_results
        
        self.effect_size_results = results
        self.statistical_test_results = {level: {metric: data['statistical_tests'] 
                                               for metric, data in metrics.items()} 
                                       for level, metrics in results.items()}
        return results
    
    def save_effect_size_summary(self) -> None:
        """Save effect size summary to CSV and Excel files."""
        summary_data = []
        
        for precision_level, metrics in self.effect_size_results.items():
            for metric, stats in metrics.items():
                # Statistical test results
                t_test_p = stats['statistical_tests'].get('paired_t_test', {}).get('p_value', np.nan)
                wilcoxon_p = stats['statistical_tests'].get('wilcoxon', {}).get('p_value', np.nan)
                
                summary_data.append({
                    'Precision_Level': precision_level,
                    'Metric': metric,
                    'N_Samples': stats['n_samples'],
                    'Mean_Iterative': stats['mean_iterative'],
                    'Mean_Zero_Shot': stats['mean_zero_shot'],
                    'Mean_Difference': stats['mean_difference'],
                    'Cohens_d': stats['cohens_d'],
                    'Hedges_g': stats['hedges_g'],
                    'Glass_Delta': stats['glass_delta'],
                    'Interpretation': stats['interpretation'],
                    'T_Test_P_Value': t_test_p,
                    'Wilcoxon_P_Value': wilcoxon_p
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_path = self.output_path / "effect_size_summary.csv"
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"✓ Effect size summary saved: {summary_path}")
        
        # Save to Excel with formatting
        try:
            summary_excel_path = self.output_path / "effect_size_summary.xlsx"
            with pd.ExcelWriter(summary_excel_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Effect Size Summary', index=False, float_format='%.4f')
            print(f"✓ Effect size summary Excel saved: {summary_excel_path}")
        except ImportError:
            print("⚠ openpyxl not available, skipping Excel export")
    
    def create_effect_size_heatmap(self) -> None:
        """Create and save effect size heatmap visualization."""
        # Prepare effect size matrix for Cohen's d
        cohens_d_matrix = []
        precision_labels = []
        
        for precision_level in PRECISION_LEVELS:
            if precision_level in self.effect_size_results:
                cohens_d_values = [
                    self.effect_size_results[precision_level][metric]['cohens_d']
                    for metric in METRICS_TO_ANALYZE
                ]
                cohens_d_matrix.append(cohens_d_values)
                precision_labels.append(precision_level.capitalize())
        
        if not cohens_d_matrix:
            print("⚠ No data available for heatmap generation")
            return
            
        cohens_d_matrix = np.array(cohens_d_matrix)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Create subplot with space for colorbar
        ax = plt.subplot(111)
        
        # Plot heatmap with diverging colormap
        im = ax.imshow(cohens_d_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-2, vmax=2)
        
        # Set ticks and labels
        ax.set_xticks(range(len(METRICS_TO_ANALYZE)))
        ax.set_yticks(range(len(precision_labels)))
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in METRICS_TO_ANALYZE])
        ax.set_yticklabels(precision_labels)
        
        # Set x-axis labels to horizontal for better readability
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add Cohen's d values as text with interpretations
        for i in range(len(precision_labels)):
            for j in range(len(METRICS_TO_ANALYZE)):
                if i < len(cohens_d_matrix) and j < len(cohens_d_matrix[i]):
                    cohens_d = cohens_d_matrix[i, j]
                    if not np.isnan(cohens_d):
                        text = f"{cohens_d:.3f}"
                        ax.text(j, i, text, ha="center", va="center",
                               color="white" if abs(cohens_d) > 1.0 else "black",
                               fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Cohen's d Effect Size", rotation=270, labelpad=20)
        
        # Add reference lines for effect size thresholds
        cbar.ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.7)
        cbar.ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        cbar.ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
        cbar.ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.7)
        cbar.ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.7)
        cbar.ax.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.7)
        
        # Set titles and labels
        ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision Levels', fontsize=12, fontweight='bold')

        # Adjust layout to leave space for interpretation guide
        plt.subplots_adjust(top=0.89)

        # Add interpretation guide at the top, aligned with the heatmap center
        # Get the position of the axes to align the text with the heatmap center
        ax_pos = ax.get_position()
        heatmap_center_x = (ax_pos.x0 + ax_pos.x1) / 2
        plt.figtext(heatmap_center_x, 0.92, '|d| ≥ 0.8: Large effect  |  0.5 ≤ |d| < 0.8: Medium  |  0.2 ≤ |d| < 0.5: Small  |  |d| < 0.2: Negligible  |  Positive: Iterative > Zero-shot',
                   ha='center', va='center', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # Save plot
        heatmap_path = self.output_path / "effect_size_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Effect size heatmap saved: {heatmap_path}")
    
    def create_effect_size_violin_plot(self) -> None:
        """Create and save violin plot showing distribution of effect sizes."""
        # Prepare data for violin plot
        plot_data = []
        
        for precision_level, metrics in self.effect_size_results.items():
            for metric, stats in metrics.items():
                if not np.isnan(stats['cohens_d']):
                    plot_data.append({
                        'Precision Level': precision_level.capitalize(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Cohen\'s d': stats['cohens_d'],
                        'Sample Size': stats['n_samples']
                    })
        
        if not plot_data:
            print("⚠ No data available for violin plot generation")
            return
            
        plot_df = pd.DataFrame(plot_data)
        
        # Create violin plot
        plt.figure(figsize=(14, 10))
        
        # Create subplots: one for each precision level
        precision_levels_available = plot_df['Precision Level'].unique()
        n_precision = len(precision_levels_available)
        
        fig, axes = plt.subplots(n_precision, 1, figsize=(14, 4 * n_precision), sharex=True)
        if n_precision == 1:
            axes = [axes]
        
        for i, precision in enumerate(precision_levels_available):
            precision_data = plot_df[plot_df['Precision Level'] == precision]
            
            # Create violin plot
            parts = axes[i].violinplot([precision_data[precision_data['Metric'] == metric]['Cohen\'s d'].values 
                                       for metric in precision_data['Metric'].unique()],
                                     positions=range(len(precision_data['Metric'].unique())),
                                     widths=0.6, showmeans=True, showextrema=True)
            
            # Customize violin plot colors
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)
            
            # Add reference lines for effect size interpretations
            axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            axes[i].axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
            axes[i].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
            axes[i].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
            axes[i].axhline(y=-0.2, color='green', linestyle='--', alpha=0.5)
            axes[i].axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5)
            axes[i].axhline(y=-0.8, color='red', linestyle='--', alpha=0.5)
            
            # Set labels and title
            axes[i].set_title(f'{precision} Precision Level', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Cohen\'s d Effect Size', fontsize=12)
            axes[i].set_xticks(range(len(precision_data['Metric'].unique())))
            axes[i].set_xticklabels(precision_data['Metric'].unique(), rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            
            # Add sample size annotations
            for j, metric in enumerate(precision_data['Metric'].unique()):
                metric_data = precision_data[precision_data['Metric'] == metric]
                if not metric_data.empty:
                    sample_size = metric_data['Sample Size'].iloc[0]
                    cohens_d = metric_data['Cohen\'s d'].iloc[0]
                    axes[i].text(j, cohens_d + 0.1, f'n={sample_size}', 
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add legend to the last subplot
        axes[-1].legend(loc='upper right')
        
        # Set overall title and x-label
        axes[-1].set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')

        # Adjust layout to leave space for interpretation guide
        plt.subplots_adjust(top=0.9)

        # Add interpretation guide at the top, aligned with the plot center
        # For multi-subplot figure, center the text over the main plotting area
        fig.text(0.5, 0.95, '|d| ≥ 0.8: Large effect  |  0.5 ≤ |d| < 0.8: Medium  |  0.2 ≤ |d| < 0.5: Small  |  |d| < 0.2: Negligible  |  Positive: Iterative > Zero-shot',
                ha='center', va='center', fontsize=10, style='italic', transform=fig.transFigure,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # Save plot
        violin_path = self.output_path / "effect_size_violin_plot.png"
        plt.savefig(violin_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Effect size violin plot saved: {violin_path}")
    
    def create_effect_size_bar_chart(self) -> None:
        """Create and save bar chart showing effect sizes across metrics and precision levels."""
        # Prepare data for bar chart
        plot_data = []
        
        for precision_level, metrics in self.effect_size_results.items():
            for metric, stats in metrics.items():
                if not np.isnan(stats['cohens_d']):
                    plot_data.append({
                        'Precision Level': precision_level.capitalize(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Cohen\'s d': stats['cohens_d'],
                        'Error': 0.1,  # Placeholder for error bars (could be confidence intervals)
                        'Interpretation': stats['interpretation']
                    })
        
        if not plot_data:
            print("⚠ No data available for bar chart generation")
            return
            
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        plt.figure(figsize=(16, 10))
        
        # Pivot data for grouped bar chart
        pivot_df = plot_df.pivot(index='Metric', columns='Precision Level', values='Cohen\'s d')
        
        # Reorder columns to desired precision level order
        desired_column_order = ['Low', 'Medium', 'High', 'Overall']
        available_columns = [col for col in desired_column_order if col in pivot_df.columns]
        pivot_df = pivot_df[available_columns]
        
        # Reorder rows (metrics) to desired order: success_rate first, then efficiency
        desired_metric_order = ['Success Rate', 'Mean Hard Efficiency']
        available_metrics = [metric for metric in desired_metric_order if metric in pivot_df.index]
        pivot_df = pivot_df.reindex(available_metrics)
        
        # Create bar chart with colors matching the order
        color_map = {'Low': 'lightblue', 'Medium': 'lightgreen', 'High': 'salmon', 'Overall': 'gold'}
        colors = [color_map[col] for col in pivot_df.columns]
        ax = pivot_df.plot(kind='bar', figsize=(16, 10), width=0.8, color=colors)
        
        # Customize the chart
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small effect')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
        ax.axhline(y=-0.2, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=-0.8, color='red', linestyle='--', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cohen\'s d Effect Size', fontsize=12, fontweight='bold')
        ax.legend(title='Precision Level', title_fontsize=12, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Keep x-axis labels horizontal
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

        # Add value labels on bars (horizontal)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=0, fontsize=8)

        # Adjust layout to leave space for interpretation guide
        plt.subplots_adjust(top=0.85)

        # Add interpretation guide at the top, aligned with the bar chart center
        # Get the position of the axes to align the text with the chart center
        ax_pos = ax.get_position()
        chart_center_x = (ax_pos.x0 + ax_pos.x1) / 2
        plt.figtext(chart_center_x, 0.92, '|d| ≥ 0.8: Large effect  |  0.5 ≤ |d| < 0.8: Medium  |  0.2 ≤ |d| < 0.5: Small  |  |d| < 0.2: Negligible  |  Positive: Iterative > Zero-shot',
                   ha='center', va='center', fontsize=10, style='italic', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # Save plot
        bar_chart_path = self.output_path / "effect_size_bar_chart.png"
        plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Effect size bar chart saved: {bar_chart_path}")
    
    def generate_interpretation_report(self) -> None:
        """Generate a comprehensive interpretation report of effect size results."""
        report_path = self.output_path / "effect_size_report.txt"
        
        dataset_title = ', '.join([d.upper() for d in self.datasets]) if len(self.datasets) > 1 else self.datasets[0].upper()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EFFECT SIZE ANALYSIS REPORT\n")
            f.write(f"Dataset(s): {dataset_title}\n")
            f.write("="*80 + "\n\n")

            # Statistical Principles
            f.write("STATISTICAL PRINCIPLES AND METHODOLOGY\n")
            f.write("-"*50 + "\n")
            f.write("This analysis employs effect size metrics to quantify the magnitude of performance\n")
            f.write("differences between iterative and zero-shot inference approaches. Effect sizes\n")
            f.write("provide standardized measures independent of sample size, enabling meaningful\n")
            f.write("comparison across different conditions and datasets.\n\n")

            f.write("Mathematical Formulations:\n\n")
            f.write("1. Cohen's d = (μ_iterative - μ_zero_shot) / σ_pooled\n")
            f.write("   Where σ_pooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]\n")
            f.write("   - Measures standardized mean difference using pooled standard deviation\n")
            f.write("   - Positive values indicate iterative > zero-shot performance\n\n")

            f.write("2. Hedges' g = Cohen's d × J\n")
            f.write("   Where J = 1 - 3/(4(n₁+n₂-2)-1) (bias correction factor)\n")
            f.write("   - Bias-corrected version of Cohen's d for small samples\n\n")

            f.write("3. Glass's Δ = (μ_iterative - μ_zero_shot) / σ_control\n")
            f.write("   - Uses only control group (zero-shot) standard deviation\n")
            f.write("   - Appropriate when control group represents baseline variability\n\n")

            f.write("Effect Size Interpretation (Cohen's conventions):\n")
            f.write("• |d| < 0.2:  Negligible effect\n")
            f.write("• 0.2 ≤ |d| < 0.5:  Small effect\n")
            f.write("• 0.5 ≤ |d| < 0.8:  Medium effect  \n")
            f.write("• |d| ≥ 0.8:  Large effect\n\n")

            f.write("Statistical Significance Testing:\n")
            f.write("• Paired t-test: Parametric test for mean differences (assumes normality)\n")
            f.write("• Wilcoxon signed-rank test: Non-parametric alternative (robust to outliers)\n")
            f.write("• p < 0.05: Statistically significant difference\n\n")

            # Performance Improvement Analysis
            f.write("ITERATIVE vs ZERO-SHOT PERFORMANCE ANALYSIS\n")
            f.write("-"*50 + "\n")
            f.write("This analysis focuses on quantifying the performance gains achieved when\n")
            f.write("switching from zero-shot to iterative inference approaches. Positive effect\n")
            f.write("sizes indicate iterative inference outperforms zero-shot, while negative\n")
            f.write("values suggest zero-shot may be preferable.\n\n")
            
            # Find overall trends
            large_effects = []
            medium_effects = []
            small_effects = []
            negligible_effects = []
            
            for precision_level, metrics in self.effect_size_results.items():
                for metric, stats in metrics.items():
                    if not np.isnan(stats['cohens_d']):
                        abs_effect = abs(stats['cohens_d'])
                        effect_info = (precision_level, metric, stats['cohens_d'], stats['interpretation'])
                        
                        if abs_effect >= 0.8:
                            large_effects.append(effect_info)
                        elif abs_effect >= 0.5:
                            medium_effects.append(effect_info)
                        elif abs_effect >= 0.2:
                            small_effects.append(effect_info)
                        else:
                            negligible_effects.append(effect_info)
            
            f.write(f"• Large effects found: {len(large_effects)} cases\n")
            f.write(f"• Medium effects found: {len(medium_effects)} cases\n")
            f.write(f"• Small effects found: {len(small_effects)} cases\n")
            f.write(f"• Negligible effects found: {len(negligible_effects)} cases\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*50 + "\n")
            
            # Analyze patterns by precision level
            for precision_level in PRECISION_LEVELS:
                if precision_level not in self.effect_size_results:
                    continue
                    
                f.write(f"\n{precision_level.upper()} PRECISION LEVEL:\n")
                
                precision_results = self.effect_size_results[precision_level]
                significant_improvements = []
                significant_degradations = []
                
                for metric, stats in precision_results.items():
                    if np.isnan(stats['cohens_d']):
                        f.write(f"  {metric}: Insufficient data for analysis\n")
                        continue
                        
                    cohens_d = stats['cohens_d']
                    mean_diff = stats['mean_difference']
                    interpretation = stats['interpretation']
                    
                    # Statistical significance
                    t_test_result = stats['statistical_tests'].get('paired_t_test', {})
                    t_significant = t_test_result.get('significant', False)
                    t_p = t_test_result.get('p_value', np.nan)
                    
                    significance_note = f"(p = {t_p:.3f})" if not np.isnan(t_p) else ""
                    
                    f.write(f"  {metric}:\n")
                    f.write(f"    • Cohen's d = {cohens_d:.3f} ({interpretation})\n")
                    f.write(f"    • Mean difference = {mean_diff:.3f}\n")
                    f.write(f"    • Statistical significance: {'Yes' if t_significant else 'No'} {significance_note}\n")
                    f.write(f"    • Iterative mean: {stats['mean_iterative']:.3f}\n")
                    f.write(f"    • Zero-shot mean: {stats['mean_zero_shot']:.3f}\n")
                    
                    if abs(cohens_d) >= 0.5:  # Medium or large effect
                        if cohens_d > 0:
                            significant_improvements.append((metric, cohens_d, mean_diff))
                        else:
                            significant_degradations.append((metric, cohens_d, mean_diff))
                    
                    f.write("\n")
                
                # Summary for this precision level
                if significant_improvements:
                    f.write(f"  SIGNIFICANT IMPROVEMENTS in {precision_level} precision:\n")
                    for metric, d, diff in significant_improvements:
                        f.write(f"    - {metric}: {diff:+.3f} improvement (Cohen's d = {d:.3f})\n")
                    f.write("\n")
                
                if significant_degradations:
                    f.write(f"  SIGNIFICANT DEGRADATIONS in {precision_level} precision:\n")
                    for metric, d, diff in significant_degradations:
                        f.write(f"    - {metric}: {diff:+.3f} degradation (Cohen's d = {d:.3f})\n")
                    f.write("\n")
            
            # Performance Improvement Quantification
            f.write("PERFORMANCE IMPROVEMENT QUANTIFICATION\n")
            f.write("-"*50 + "\n")

            # Analyze all improvements (positive effect sizes)
            all_improvements = []
            all_degradations = []
            for precision_level, metrics in self.effect_size_results.items():
                for metric, stats in metrics.items():
                    if not np.isnan(stats['cohens_d']):
                        if stats['cohens_d'] > 0:
                            all_improvements.append((precision_level, metric, stats['cohens_d'], stats['mean_difference'], stats['mean_iterative'], stats['mean_zero_shot']))
                        elif stats['cohens_d'] < 0:
                            all_degradations.append((precision_level, metric, stats['cohens_d'], stats['mean_difference'], stats['mean_iterative'], stats['mean_zero_shot']))

            if all_improvements:
                all_improvements.sort(key=lambda x: x[2], reverse=True)  # Sort by Cohen's d
                f.write("ITERATIVE INFERENCE PERFORMANCE GAINS:\n")
                f.write("(Ranked by effect size magnitude)\n\n")
                for i, (precision, metric, d, diff, iter_mean, zero_mean) in enumerate(all_improvements, 1):
                    # Calculate relative improvement percentage
                    rel_improvement = (diff / zero_mean * 100) if zero_mean != 0 else 0
                    f.write(f"{i}. {precision.capitalize()} precision - {metric}:\n")
                    f.write(f"   • Effect size (Cohen's d): {d:.3f} ({self.interpret_effect_size(d)})\n")
                    f.write(f"   • Absolute improvement: {diff:+.3f}\n")
                    f.write(f"   • Relative improvement: {rel_improvement:+.1f}%\n")
                    f.write(f"   • Iterative performance: {iter_mean:.3f}\n")
                    f.write(f"   • Zero-shot performance: {zero_mean:.3f}\n\n")
                f.write("\n")
            
            if all_degradations:
                all_degradations.sort(key=lambda x: x[2])  # Sort by Cohen's d (most negative first)
                f.write("CASES WHERE ZERO-SHOT OUTPERFORMS ITERATIVE:\n")
                f.write("(Areas where iterative inference shows performance decline)\n\n")
                for i, (precision, metric, d, diff, iter_mean, zero_mean) in enumerate(all_degradations, 1):
                    # Calculate relative degradation percentage
                    rel_degradation = (diff / zero_mean * 100) if zero_mean != 0 else 0
                    f.write(f"{i}. {precision.capitalize()} precision - {metric}:\n")
                    f.write(f"   • Effect size (Cohen's d): {d:.3f} ({self.interpret_effect_size(d)})\n")
                    f.write(f"   • Absolute degradation: {diff:.3f}\n")
                    f.write(f"   • Relative degradation: {rel_degradation:.1f}%\n")
                    f.write(f"   • Iterative performance: {iter_mean:.3f}\n")
                    f.write(f"   • Zero-shot performance: {zero_mean:.3f}\n\n")
            else:
                f.write("CASES WHERE ZERO-SHOT OUTPERFORMS ITERATIVE:\n")
                f.write("No cases found where zero-shot approach shows meaningful advantages.\n\n")
            
            # Strategic Recommendations
            f.write("STRATEGIC RECOMMENDATIONS FOR ITERATIVE INFERENCE ADOPTION\n")
            f.write("-"*50 + "\n")

            # Calculate overall performance improvement statistics
            if all_improvements:
                avg_effect_size = np.mean([x[2] for x in all_improvements])
                avg_abs_improvement = np.mean([x[3] for x in all_improvements])
                max_improvement = max(all_improvements, key=lambda x: x[2])

                f.write("OVERALL ITERATIVE INFERENCE BENEFITS:\n")
                f.write(f"• Average effect size across all improvements: {avg_effect_size:.3f}\n")
                f.write(f"• Average absolute performance gain: {avg_abs_improvement:+.3f}\n")
                f.write(f"• Maximum improvement observed: {max_improvement[3]:+.3f} ")
                f.write(f"({max_improvement[1]} at {max_improvement[0]} precision, d = {max_improvement[2]:.3f})\n\n")

            # Analyze patterns by precision level
            precision_analysis = {}
            for precision_level in ['low', 'medium', 'high']:
                if precision_level in self.effect_size_results:
                    effects = [stats['cohens_d'] for stats in self.effect_size_results[precision_level].values()
                             if not np.isnan(stats['cohens_d'])]
                    improvements = [stats['mean_difference'] for stats in self.effect_size_results[precision_level].values()
                                  if not np.isnan(stats['cohens_d']) and stats['cohens_d'] > 0]

                    if effects:
                        precision_analysis[precision_level] = {
                            'avg_effect': np.mean(effects),
                            'avg_improvement': np.mean(improvements) if improvements else 0,
                            'improvement_count': len(improvements),
                            'total_metrics': len(effects)
                        }

            if precision_analysis:
                f.write("PRECISION-LEVEL ANALYSIS:\n")
                for precision, stats in precision_analysis.items():
                    f.write(f"• {precision.capitalize()} precision:\n")
                    f.write(f"  - Average effect size: {stats['avg_effect']:+.3f}\n")
                    f.write(f"  - Average performance improvement: {stats['avg_improvement']:+.3f}\n")
                    f.write(f"  - Metrics showing improvement: {stats['improvement_count']}/{stats['total_metrics']}\n")

                # Determine best precision level for iterative inference
                best_precision = max(precision_analysis.keys(), key=lambda x: precision_analysis[x]['avg_effect'])
                f.write(f"\n• MOST EFFECTIVE PRECISION LEVEL: {best_precision.upper()}\n")
                f.write(f"  (Average effect size: {precision_analysis[best_precision]['avg_effect']:+.3f})\n\n")
                
                # Find metrics with most consistent benefits across precision levels
                metric_consistency = {}
                for metric in METRICS_TO_ANALYZE:
                    metric_improvements = []
                    metric_effects = []
                    for precision_level in ['low', 'medium', 'high']:
                        if (precision_level in self.effect_size_results and
                            metric in self.effect_size_results[precision_level] and
                            not np.isnan(self.effect_size_results[precision_level][metric]['cohens_d'])):
                            effect = self.effect_size_results[precision_level][metric]['cohens_d']
                            improvement = self.effect_size_results[precision_level][metric]['mean_difference']
                            metric_effects.append(effect)
                            if effect > 0:
                                metric_improvements.append(improvement)

                    if len(metric_effects) >= 2:
                        metric_consistency[metric] = {
                            'avg_effect': np.mean(metric_effects),
                            'avg_improvement': np.mean(metric_improvements) if metric_improvements else 0,
                            'consistent_improvements': len(metric_improvements),
                            'total_measurements': len(metric_effects)
                        }

                if metric_consistency:
                    f.write("METRIC-SPECIFIC RECOMMENDATIONS:\n")
                    sorted_metrics = sorted(metric_consistency.items(), key=lambda x: x[1]['avg_effect'], reverse=True)
                    for metric, stats in sorted_metrics:
                        f.write(f"• {metric.replace('_', ' ').title()}:\n")
                        f.write(f"  - Average effect size: {stats['avg_effect']:+.3f}\n")
                        f.write(f"  - Average improvement when positive: {stats['avg_improvement']:+.3f}\n")
                        f.write(f"  - Consistent improvement rate: {stats['consistent_improvements']}/{stats['total_measurements']} precision levels\n")

            f.write(f"\nPRACTICAL IMPLEMENTATION GUIDANCE:\n")
            f.write("1. WHEN TO USE ITERATIVE INFERENCE:\n")
            if all_improvements:
                # Find conditions with largest improvements
                large_improvements = [x for x in all_improvements if x[2] >= 0.8]  # Large effect sizes
                if large_improvements:
                    f.write("   • Strongly recommended for:\n")
                    for precision, metric, d, diff, _, _ in large_improvements:
                        f.write(f"     - {precision.capitalize()} precision {metric} tasks (d = {d:.3f}, improvement = {diff:+.3f})\n")

            if all_degradations:
                f.write("\n2. WHEN TO PREFER ZERO-SHOT:\n")
                f.write("   • Consider zero-shot for:\n")
                for precision, metric, d, diff, _, _ in all_degradations:
                    f.write(f"     - {precision.capitalize()} precision {metric} tasks (performance may decline by {abs(diff):.3f})\n")

            f.write(f"\n3. COST-BENEFIT CONSIDERATIONS:\n")
            f.write("   • Iterative inference requires additional computational resources\n")
            f.write("   • Weigh performance gains against increased inference time and costs\n")
            if all_improvements:
                f.write(f"   • Consider the {avg_abs_improvement:+.3f} average performance improvement\n")
                f.write(f"     against the computational overhead of iterative processing\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Effect size interpretation report saved: {report_path}")
    
    def print_summary_report(self) -> None:
        """Print a brief summary report to console."""
        dataset_title = ', '.join([d.upper() for d in self.datasets])
        print(f"\n{'='*70}")
        print(f"EFFECT SIZE ANALYSIS SUMMARY")
        print(f"Dataset(s): {dataset_title}")
        print(f"{'='*70}")
        
        # Count effects by magnitude
        large_count = medium_count = small_count = negligible_count = 0
        
        for precision_level, metrics in self.effect_size_results.items():
            for metric, stats in metrics.items():
                if not np.isnan(stats['cohens_d']):
                    abs_effect = abs(stats['cohens_d'])
                    if abs_effect >= 0.8:
                        large_count += 1
                    elif abs_effect >= 0.5:
                        medium_count += 1
                    elif abs_effect >= 0.2:
                        small_count += 1
                    else:
                        negligible_count += 1
        
        print(f"\n📊 EFFECT SIZE DISTRIBUTION:")
        print(f"   Large effects (|d| ≥ 0.8):    {large_count}")
        print(f"   Medium effects (|d| ≥ 0.5):   {medium_count}")
        print(f"   Small effects (|d| ≥ 0.2):    {small_count}")
        print(f"   Negligible effects (|d| < 0.2): {negligible_count}")
        
        # Show top improvements
        improvements = []
        for precision_level, metrics in self.effect_size_results.items():
            for metric, stats in metrics.items():
                if not np.isnan(stats['cohens_d']) and stats['cohens_d'] > 0:
                    improvements.append((precision_level, metric, stats['cohens_d'], stats['mean_difference']))
        
        if improvements:
            improvements.sort(key=lambda x: x[2], reverse=True)
            print(f"\n🚀 TOP ITERATIVE IMPROVEMENTS:")
            for i, (precision, metric, d, diff) in enumerate(improvements[:3], 1):
                print(f"   {i}. {precision.capitalize()}/{metric}: +{diff:.3f} (Cohen's d = {d:.3f})")
        
        print(f"\n📁 Results saved in: {self.output_path}")
        print(f"{'='*70}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute effect sizes between zero-shot and iterative model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/effect_size_analysis.py              # Analyze all available datasets
  python evaluation/effect_size_analysis.py -d euler_1d  # Analyze specific dataset
  python evaluation/effect_size_analysis.py -d heat_1d

Output files:
  - eval_results/stats/effect_size_analysis/{dataset}/effect_size_summary.csv
  - eval_results/stats/effect_size_analysis/{dataset}/effect_size_summary.xlsx
  - eval_results/stats/effect_size_analysis/{dataset}/effect_size_heatmap.png
  - eval_results/stats/effect_size_analysis/{dataset}/effect_size_bar_chart.png
  - eval_results/stats/effect_size_analysis/{dataset}/effect_size_report.txt
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
            print(f"🚀 Starting effect size analysis for dataset: {args.dataset}")
            analyzer = EffectSizeAnalyzer(args.dataset)
        else:
            print(f"🚀 Starting effect size analysis for all available datasets")
            analyzer = EffectSizeAnalyzer()
        
        # Load data
        zero_shot_df, iterative_df = analyzer.load_evaluation_data()
        
        # Perform effect size analysis
        analyzer.analyze_effect_sizes(zero_shot_df, iterative_df)
        
        # Save results
        analyzer.save_effect_size_summary()
        analyzer.create_effect_size_heatmap()
        analyzer.create_effect_size_bar_chart()
        analyzer.generate_interpretation_report()
        
        # Print summary
        analyzer.print_summary_report()
        
        print(f"\n✅ Effect size analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()