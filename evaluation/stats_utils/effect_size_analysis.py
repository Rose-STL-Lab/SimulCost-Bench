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
METRICS_TO_ANALYZE = ['success_rate', 'mean_hard_efficiency']
# Temporarily commenting out soft metrics for focused analysis
# METRICS_TO_ANALYZE = ['success_rate', 'mean_soft_success', 'mean_soft_efficiency', 'mean_hard_efficiency']
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
                        interpretation = self.interpret_effect_size(cohens_d).split()[0]  # Get magnitude only
                        text = f"{cohens_d:.3f}\n({interpretation})"
                        ax.text(j, i, text, ha="center", va="center", 
                               color="white" if abs(cohens_d) > 1.0 else "black",
                               fontsize=9, fontweight='bold')
        
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
        dataset_title = ', '.join([d.upper() for d in self.datasets]) if len(self.datasets) > 1 else self.datasets[0].upper()
        plt.title(f'Effect Sizes: Iterative vs Zero-shot Performance\nDataset(s): {dataset_title}', 
                 fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision Levels', fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
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
        dataset_title = ', '.join([d.upper() for d in self.datasets]) if len(self.datasets) > 1 else self.datasets[0].upper()
        fig.suptitle(f'Effect Size Distribution: Iterative vs Zero-shot\nDataset(s): {dataset_title}', 
                    fontsize=16, fontweight='bold')
        axes[-1].set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
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
        dataset_title = ', '.join([d.upper() for d in self.datasets]) if len(self.datasets) > 1 else self.datasets[0].upper()
        ax.set_title(f'Effect Sizes by Metric and Precision Level\nDataset(s): {dataset_title}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cohen\'s d Effect Size', fontsize=12, fontweight='bold')
        ax.legend(title='Precision Level', title_fontsize=12, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Keep x-axis labels horizontal
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        
        # Add value labels on bars (horizontal)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=0, fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
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
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*50 + "\n")
            
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
            
            # Key findings
            f.write("KEY FINDINGS\n")
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
            
            # Practical implications
            f.write("PRACTICAL IMPLICATIONS\n")
            f.write("-"*50 + "\n")
            
            # Find the best performing combinations
            best_combinations = []
            for precision_level, metrics in self.effect_size_results.items():
                for metric, stats in metrics.items():
                    if not np.isnan(stats['cohens_d']) and stats['cohens_d'] > 0.5:
                        best_combinations.append((precision_level, metric, stats['cohens_d'], stats['mean_difference']))
            
            if best_combinations:
                best_combinations.sort(key=lambda x: x[2], reverse=True)  # Sort by Cohen's d
                f.write("Most effective use cases for ITERATIVE approach:\n")
                for i, (precision, metric, d, diff) in enumerate(best_combinations[:5], 1):
                    f.write(f"{i}. {precision.capitalize()} precision, {metric}: {diff:+.3f} improvement (d = {d:.3f})\n")
                f.write("\n")
            
            # Find cases where zero-shot might be better
            zero_shot_better = []
            for precision_level, metrics in self.effect_size_results.items():
                for metric, stats in metrics.items():
                    if not np.isnan(stats['cohens_d']) and stats['cohens_d'] < -0.2:
                        zero_shot_better.append((precision_level, metric, stats['cohens_d'], stats['mean_difference']))
            
            if zero_shot_better:
                zero_shot_better.sort(key=lambda x: x[2])  # Sort by Cohen's d (most negative first)
                f.write("Cases where ZERO-SHOT approach might be preferred:\n")
                for precision, metric, d, diff in zero_shot_better:
                    f.write(f"• {precision.capitalize()} precision, {metric}: {diff:+.3f} difference (d = {d:.3f})\n")
                f.write("\n")
            else:
                f.write("No cases found where zero-shot approach shows meaningful advantages.\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*50 + "\n")
            
            # Analyze patterns
            high_precision_effects = [stats['cohens_d'] for stats in self.effect_size_results.get('high', {}).values() 
                                    if not np.isnan(stats['cohens_d'])]
            medium_precision_effects = [stats['cohens_d'] for stats in self.effect_size_results.get('medium', {}).values() 
                                      if not np.isnan(stats['cohens_d'])]
            low_precision_effects = [stats['cohens_d'] for stats in self.effect_size_results.get('low', {}).values() 
                                   if not np.isnan(stats['cohens_d'])]
            
            if high_precision_effects and medium_precision_effects and low_precision_effects:
                avg_high = np.mean(high_precision_effects)
                avg_medium = np.mean(medium_precision_effects)
                avg_low = np.mean(low_precision_effects)
                
                f.write("Based on the effect size analysis:\n\n")
                
                if avg_low > avg_medium > avg_high:
                    f.write("1. Iterative approach shows DECREASING benefits as precision requirements increase\n")
                    f.write("   → Consider iterative for low-precision tasks, zero-shot for high-precision\n")
                elif avg_high > avg_medium > avg_low:
                    f.write("1. Iterative approach shows INCREASING benefits as precision requirements increase\n")
                    f.write("   → Strongly recommend iterative for high-precision tasks\n")
                else:
                    f.write("1. Effect sizes vary across precision levels without clear pattern\n")
                    f.write("   → Evaluate iterative vs zero-shot based on specific use case requirements\n")
                
                # Find metrics with most consistent benefits
                consistent_metrics = []
                for metric in METRICS_TO_ANALYZE:
                    metric_effects = []
                    for precision_level in ['low', 'medium', 'high']:
                        if (precision_level in self.effect_size_results and 
                            metric in self.effect_size_results[precision_level] and
                            not np.isnan(self.effect_size_results[precision_level][metric]['cohens_d'])):
                            metric_effects.append(self.effect_size_results[precision_level][metric]['cohens_d'])
                    
                    if len(metric_effects) >= 2 and all(d > 0.3 for d in metric_effects):
                        consistent_metrics.append((metric, np.mean(metric_effects)))
                
                if consistent_metrics:
                    consistent_metrics.sort(key=lambda x: x[1], reverse=True)
                    f.write(f"\n2. Metrics showing most consistent iterative benefits:\n")
                    for metric, avg_effect in consistent_metrics:
                        f.write(f"   • {metric}: average Cohen's d = {avg_effect:.3f}\n")
            
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