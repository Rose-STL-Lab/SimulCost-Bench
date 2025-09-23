#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect Size Analysis for Zero-shot vs Iterative Model Performance

This script computes appropriate effect sizes to quantify the magnitude of
performance differences between zero-shot and iterative inference modes:
- Cohen's h for success rates (proportion-based metrics)
- Cohen's d for efficiency measures (continuous metrics)

Usage
-----
python evaluation/effect_size_analysis.py              # Analyze all available datasets
python evaluation/effect_size_analysis.py -d euler_1d  # Analyze specific dataset
python evaluation/effect_size_analysis.py -d heat_1d

Output: Creates analysis results with:
- effect_size_summary.csv: Numerical effect size coefficients and interpretations
- effect_size_summary.xlsx: Excel version with formatting
- effect_size_heatmap.png: Heatmap visualization of effect sizes across metrics
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
        Return only the specific datasets to analyze (consistent with task_correlation_analysis.py).

        Returns
        -------
        List[str]
            List of target dataset names
        """
        target_datasets = ['burgers_1d', 'epoch_1d', 'euler_1d', 'heat_1d', 'heat_2d']
        available_datasets = []

        for dataset_name in target_datasets:
            dataset_path = Path(f"eval_results/{dataset_name}")
            zero_shot_path = dataset_path / "zero_shot" / f"{dataset_name}_sum.csv"
            iterative_path = dataset_path / "iterative" / f"{dataset_name}_sum.csv"

            if zero_shot_path.exists() and iterative_path.exists():
                available_datasets.append(dataset_name)
            else:
                print(f"⚠ Skipping {dataset_name}: Required files not found")

        if not available_datasets:
            raise FileNotFoundError("No target datasets with complete zero-shot and iterative data found")

        return available_datasets
        
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

        # Filter for target models (consistent with task_correlation_analysis.py)
        target_models = ['Claude-3.7-Sonnet', 'GPT-5', 'Llama-3-70B-Instruct', 'Qwen3-32B']

        # Apply model name standardization and filtering
        combined_zero_shot = self._standardize_and_filter_models(combined_zero_shot, target_models)
        combined_iterative = self._standardize_and_filter_models(combined_iterative, target_models)

        print(f"✓ Total combined data: {len(combined_zero_shot)} zero-shot, {len(combined_iterative)} iterative records")
        print(f"✓ Filtered for target models: {target_models}")

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

    def _standardize_and_filter_models(self, df: pd.DataFrame, target_models: List[str]) -> pd.DataFrame:
        """
        Standardize model names and filter for target models (consistent with task_correlation_analysis.py).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with model data
        target_models : List[str]
            List of target model names to keep

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with standardized model names
        """
        model_mapping = {
            # Claude variants
            'Claude-3.7-Sonnet': 'Claude-3.7-Sonnet',
            'anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude-3.7-Sonnet',

            # GPT variants
            'GPT-5': 'GPT-5',
            'gpt-5-2025-08-07': 'GPT-5',

            # Llama variants
            'Llama-3-70B-Instruct': 'Llama-3-70B-Instruct',
            'meta.llama3-70b-instruct-v1:0': 'Llama-3-70B-Instruct',

            # Qwen variants
            'Qwen3-32B': 'Qwen3-32B',
            'qwen3_32b': 'Qwen3-32B',
        }

        # Standardize model names
        df['Model_Standardized'] = df['Model'].map(model_mapping).fillna(df['Model'])

        # Filter for target models only
        df_filtered = df[df['Model_Standardized'].isin(target_models)].copy()

        # Update the Model column with standardized names
        df_filtered['Model'] = df_filtered['Model_Standardized']
        df_filtered = df_filtered.drop('Model_Standardized', axis=1)

        print(f"   - Before filtering: {len(df)} entries")
        print(f"   - After filtering: {len(df_filtered)} entries for models: {sorted(df_filtered['Model'].unique())}")

        return df_filtered
    
    def compute_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size for continuous metrics (e.g., efficiency).

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

    def compute_cohens_h(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute Cohen's h effect size for proportions (e.g., success rates).

        Cohen's h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))

        Parameters
        ----------
        p1, p2 : np.ndarray
            Two proportion arrays to compare

        Returns
        -------
        float
            Cohen's h effect size
        """
        if len(p1) == 0 or len(p2) == 0:
            return np.nan

        mean_p1, mean_p2 = np.mean(p1), np.mean(p2)

        # Ensure proportions are in valid range [0, 1]
        mean_p1 = np.clip(mean_p1, 0, 1)
        mean_p2 = np.clip(mean_p2, 0, 1)

        # Cohen's h formula
        h = 2 * (np.arcsin(np.sqrt(mean_p1)) - np.arcsin(np.sqrt(mean_p2)))

        return h
    
    
    
    def interpret_effect_size(self, effect_size: float, metric_name: str = "Cohen's d") -> str:
        """
        Interpret effect size magnitude according to Cohen's conventions.

        Parameters
        ----------
        effect_size : float
            The computed effect size
        metric_name : str
            Name of the effect size metric ("Cohen's d" or "Cohen's h")

        Returns
        -------
        str
            Interpretation of the effect size
        """
        if np.isnan(effect_size):
            return "Cannot determine (insufficient data)"

        abs_effect = abs(effect_size)
        direction = "favoring iterative" if effect_size > 0 else "favoring zero-shot"

        # Use same thresholds for both Cohen's d and Cohen's h
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
                        'primary_effect_size': np.nan,
                        'effect_size_name': "Cohen's h" if 'success_rate' in metric else "Cohen's d",
                        'cohens_d': np.nan,
                        'interpretation': "Insufficient data for analysis",
                        'mean_difference': np.nan,
                        'mean_iterative': np.nan,
                        'mean_zero_shot': np.nan,
                        'statistical_tests': {}
                    }
                    continue
                
                # Choose appropriate effect size measure based on metric
                if 'success_rate' in metric:
                    # Use Cohen's h for proportions/success rates
                    primary_effect_size = self.compute_cohens_h(iterative_vals, zero_shot_vals)
                    effect_size_name = "Cohen's h"
                    # Also compute Cohen's d for completeness
                    cohens_d = self.compute_cohens_d(iterative_vals, zero_shot_vals)
                else:
                    # Use Cohen's d for continuous metrics (efficiency)
                    primary_effect_size = self.compute_cohens_d(iterative_vals, zero_shot_vals)
                    effect_size_name = "Cohen's d"
                    cohens_d = primary_effect_size


                # Compute statistical tests
                statistical_tests = self.compute_statistical_tests(iterative_vals, zero_shot_vals)

                level_results[metric] = {
                    'n_samples': len(iterative_vals),
                    'primary_effect_size': primary_effect_size,
                    'effect_size_name': effect_size_name,
                    'cohens_d': cohens_d,
                    'interpretation': self.interpret_effect_size(primary_effect_size, effect_size_name),
                    'mean_difference': np.mean(iterative_vals) - np.mean(zero_shot_vals),
                    'mean_iterative': np.mean(iterative_vals),
                    'mean_zero_shot': np.mean(zero_shot_vals),
                    'statistical_tests': statistical_tests
                }

                print(f"  {metric}: {effect_size_name} = {primary_effect_size:.3f} ({self.interpret_effect_size(primary_effect_size, effect_size_name)}, n = {len(iterative_vals)})")
            
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
                    'Effect_Size_Type': stats['effect_size_name'],
                    'Primary_Effect_Size': stats['primary_effect_size'],
                    'Cohens_d': stats['cohens_d'],
                    'Interpretation': stats['interpretation'],
                    'T_Test_P_Value': t_test_p,
                    'T_Test_Note': 'Paired t-test (parametric)',
                    'Wilcoxon_P_Value': wilcoxon_p,
                    'Wilcoxon_Note': 'Wilcoxon signed-rank test (non-parametric)'
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
        # Prepare effect size matrix using primary effect sizes (h for success_rate, d for efficiency)
        effect_size_matrix = []
        precision_labels = []
        metric_labels = []
        effect_size_types = []

        for precision_level in PRECISION_LEVELS:
            if precision_level in self.effect_size_results:
                effect_size_values = []
                for metric in METRICS_TO_ANALYZE:
                    if metric in self.effect_size_results[precision_level]:
                        effect_size_values.append(self.effect_size_results[precision_level][metric]['primary_effect_size'])
                    else:
                        effect_size_values.append(np.nan)
                effect_size_matrix.append(effect_size_values)
                precision_labels.append(precision_level.capitalize())

        # Prepare metric labels and effect size types
        for metric in METRICS_TO_ANALYZE:
            metric_labels.append(metric.replace('_', ' ').title())
            if 'success_rate' in metric:
                effect_size_types.append('h')
            else:
                effect_size_types.append('d')

        if not effect_size_matrix:
            print("⚠ No data available for heatmap generation")
            return

        effect_size_matrix = np.array(effect_size_matrix)

        # Create a single figure for combined heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Create custom colormap: red for positive values (iterative better), green for negative values (zero-shot better)
        from matplotlib.colors import LinearSegmentedColormap, Normalize

        # Define custom colors
        green_color = '#2E8B57'  # SeaGreen (for negative values - zero-shot better)
        red_color = '#B22222'    # FireBrick (for positive values - iterative better)

        # Create a diverging colormap with specified colors
        colors = [green_color, 'white', red_color]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)

        # Plot the combined heatmap
        norm = Normalize(vmin=-2, vmax=2)
        im = ax.imshow(effect_size_matrix, cmap=cmap, aspect='auto', norm=norm)

        # Set title
        ax.set_title(r'$\mathbf{Effect\ Size\ Analysis:\ Zero\text{-}shot\ vs\ Iterative}$',
                    fontsize=16, fontweight='bold', pad=20)

        # Set ticks and labels
        ax.set_xticks(range(len(metric_labels)))
        ax.set_yticks(range(len(precision_labels)))
        ax.set_xticklabels([f"{label}\n(Cohen's {effect_size_types[i]})" for i, label in enumerate(metric_labels)],
                          fontsize=12, fontweight='bold')
        ax.set_yticklabels(precision_labels, fontsize=12, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision Levels', fontsize=14, fontweight='bold')

        # Add text annotations
        for i in range(len(precision_labels)):
            for j in range(len(metric_labels)):
                if not np.isnan(effect_size_matrix[i, j]):
                    effect_size = effect_size_matrix[i, j]
                    text = f"{effect_size:.3f}"
                    # Use white text for dark colors, black for light colors
                    text_color = "white" if abs(effect_size) > 1.0 else "black"
                    ax.text(j, i, text, ha="center", va="center",
                           color=text_color, fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Effect Size", rotation=270, labelpad=20, fontsize=14, fontweight='bold')

        # Add reference lines for effect size thresholds on colorbar
        cbar.ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        cbar.ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        cbar.ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        cbar.ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        cbar.ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        cbar.ax.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.7, linewidth=1)

        # Add grid for better readability
        ax.set_xticks(np.arange(len(metric_labels)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(precision_labels)+1)-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)

        # Adjust layout
        plt.tight_layout()

        # Save plot
        heatmap_path = self.output_path / "effect_size_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Effect size heatmap saved: {heatmap_path}")
    
    
    
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
            f.write("This analysis employs appropriate effect size metrics to quantify the magnitude of\n")
            f.write("performance differences between iterative and zero-shot inference approaches:\n")
            f.write("- Cohen's h for success rates (proportion-based metrics)\n")
            f.write("- Cohen's d for efficiency measures (continuous metrics)\n")
            f.write("Effect sizes provide standardized measures independent of sample size, enabling\n")
            f.write("meaningful comparison across different conditions and datasets.\n\n")

            f.write("Mathematical Formulations:\n\n")
            f.write("1. Cohen's d = (μ_iterative - μ_zero_shot) / σ_pooled\n")
            f.write("   Where σ_pooled = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]\n")
            f.write("   - Measures standardized mean difference using pooled standard deviation\n")
            f.write("   - Positive values indicate iterative > zero-shot performance\n\n")

            f.write("2. Cohen's h = 2 * (arcsin(√p₁) - arcsin(√p₂))\n")
            f.write("   Where p₁ and p₂ are the two proportions being compared\n")
            f.write("   - Measures standardized difference between proportions (success rates)\n")
            f.write("   - Uses arcsine transformation for proper variance stabilization\n")
            f.write("   - Positive values indicate iterative > zero-shot success rate\n\n")


            f.write("Effect Size Interpretation (Cohen's conventions):\n")
            f.write("Both Cohen's d and Cohen's h use the same interpretation thresholds:\n")
            f.write("• |effect| < 0.2:  Negligible effect\n")
            f.write("• 0.2 ≤ |effect| < 0.5:  Small effect\n")
            f.write("• 0.5 ≤ |effect| < 0.8:  Medium effect  \n")
            f.write("• |effect| ≥ 0.8:  Large effect\n\n")

            f.write("Statistical Significance Testing:\n")
            f.write("• Paired t-test: Parametric test for mean differences (assumes normality)\n")
            f.write("  - Tests null hypothesis that mean difference = 0\n")
            f.write("  - Requires normally distributed differences\n")
            f.write("• Wilcoxon signed-rank test: Non-parametric alternative (robust to outliers)\n")
            f.write("  - Tests null hypothesis that median difference = 0\n")
            f.write("  - Does not assume normal distribution\n")
            f.write("• p < 0.05: Statistically significant difference\n")
            f.write("• Both tests performed on paired samples (same models compared across conditions)\n\n")

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
                    if not np.isnan(stats['primary_effect_size']):
                        abs_effect = abs(stats['primary_effect_size'])
                        effect_info = (precision_level, metric, stats['primary_effect_size'], stats['effect_size_name'], stats['interpretation'])
                        
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
                    if np.isnan(stats['primary_effect_size']):
                        f.write(f"  {metric}: Insufficient data for analysis\n")
                        continue

                    primary_effect = stats['primary_effect_size']
                    effect_type = stats['effect_size_name']
                    mean_diff = stats['mean_difference']
                    interpretation = stats['interpretation']

                    # Statistical significance
                    t_test_result = stats['statistical_tests'].get('paired_t_test', {})
                    t_significant = t_test_result.get('significant', False)
                    t_p = t_test_result.get('p_value', np.nan)

                    wilcoxon_result = stats['statistical_tests'].get('wilcoxon', {})
                    w_significant = wilcoxon_result.get('significant', False)
                    w_p = wilcoxon_result.get('p_value', np.nan)

                    t_significance_note = f"(p = {t_p:.3f})" if not np.isnan(t_p) else ""
                    w_significance_note = f"(p = {w_p:.3f})" if not np.isnan(w_p) else ""

                    f.write(f"  {metric}:\n")
                    f.write(f"    • {effect_type} = {primary_effect:.3f} ({interpretation})\n")
                    f.write(f"    • Mean difference = {mean_diff:.3f}\n")
                    f.write(f"    • Paired t-test: {'Significant' if t_significant else 'Not significant'} {t_significance_note}\n")
                    f.write(f"    • Wilcoxon test: {'Significant' if w_significant else 'Not significant'} {w_significance_note}\n")
                    f.write(f"    • Iterative mean: {stats['mean_iterative']:.3f}\n")
                    f.write(f"    • Zero-shot mean: {stats['mean_zero_shot']:.3f}\n")

                    if abs(primary_effect) >= 0.5:  # Medium or large effect
                        if primary_effect > 0:
                            significant_improvements.append((metric, primary_effect, mean_diff))
                        else:
                            significant_degradations.append((metric, primary_effect, mean_diff))
                    
                    f.write("\n")
                
                # Summary for this precision level
                if significant_improvements:
                    f.write(f"  SIGNIFICANT IMPROVEMENTS in {precision_level} precision:\n")
                    for metric, effect, diff in significant_improvements:
                        effect_type = self.effect_size_results[precision_level][metric]['effect_size_name']
                        f.write(f"    - {metric}: {diff:+.3f} improvement ({effect_type} = {effect:.3f})\n")
                    f.write("\n")

                if significant_degradations:
                    f.write(f"  SIGNIFICANT DEGRADATIONS in {precision_level} precision:\n")
                    for metric, effect, diff in significant_degradations:
                        effect_type = self.effect_size_results[precision_level][metric]['effect_size_name']
                        f.write(f"    - {metric}: {diff:+.3f} degradation ({effect_type} = {effect:.3f})\n")
                    f.write("\n")
            
            # Performance Improvement Quantification
            f.write("PERFORMANCE IMPROVEMENT QUANTIFICATION\n")
            f.write("-"*50 + "\n")

            # Analyze all improvements (positive effect sizes)
            all_improvements = []
            all_degradations = []
            for precision_level, metrics in self.effect_size_results.items():
                for metric, stats in metrics.items():
                    if not np.isnan(stats['primary_effect_size']):
                        if stats['primary_effect_size'] > 0:
                            all_improvements.append((precision_level, metric, stats['primary_effect_size'], stats['effect_size_name'], stats['mean_difference'], stats['mean_iterative'], stats['mean_zero_shot']))
                        elif stats['primary_effect_size'] < 0:
                            all_degradations.append((precision_level, metric, stats['primary_effect_size'], stats['effect_size_name'], stats['mean_difference'], stats['mean_iterative'], stats['mean_zero_shot']))

            if all_improvements:
                all_improvements.sort(key=lambda x: x[2], reverse=True)  # Sort by effect size
                f.write("ITERATIVE INFERENCE PERFORMANCE GAINS:\n")
                f.write("(Ranked by effect size magnitude)\n\n")
                for i, (precision, metric, effect_size, effect_type, diff, iter_mean, zero_mean) in enumerate(all_improvements, 1):
                    # Calculate relative improvement percentage
                    rel_improvement = (diff / zero_mean * 100) if zero_mean != 0 else 0
                    f.write(f"{i}. {precision.capitalize()} precision - {metric}:\n")
                    f.write(f"   • Effect size ({effect_type}): {effect_size:.3f} ({self.interpret_effect_size(effect_size, effect_type)})\n")
                    f.write(f"   • Absolute improvement: {diff:+.3f}\n")
                    f.write(f"   • Relative improvement: {rel_improvement:+.1f}%\n")
                    f.write(f"   • Iterative performance: {iter_mean:.3f}\n")
                    f.write(f"   • Zero-shot performance: {zero_mean:.3f}\n\n")
                f.write("\n")
            
            if all_degradations:
                all_degradations.sort(key=lambda x: x[2])  # Sort by effect size (most negative first)
                f.write("CASES WHERE ZERO-SHOT OUTPERFORMS ITERATIVE:\n")
                f.write("(Areas where iterative inference shows performance decline)\n\n")
                for i, (precision, metric, effect_size, effect_type, diff, iter_mean, zero_mean) in enumerate(all_degradations, 1):
                    # Calculate relative degradation percentage
                    rel_degradation = (diff / zero_mean * 100) if zero_mean != 0 else 0
                    f.write(f"{i}. {precision.capitalize()} precision - {metric}:\n")
                    f.write(f"   • Effect size ({effect_type}): {effect_size:.3f} ({self.interpret_effect_size(effect_size, effect_type)})\n")
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
                avg_abs_improvement = np.mean([x[4] for x in all_improvements])  # mean_difference is now at index 4
                max_improvement = max(all_improvements, key=lambda x: x[2])

                f.write("OVERALL ITERATIVE INFERENCE BENEFITS:\n")
                f.write(f"• Average effect size across all improvements: {avg_effect_size:.3f}\n")
                f.write(f"• Average absolute performance gain: {avg_abs_improvement:+.3f}\n")
                f.write(f"• Maximum improvement observed: {max_improvement[4]:+.3f} ")
                f.write(f"({max_improvement[1]} at {max_improvement[0]} precision, {max_improvement[3]} = {max_improvement[2]:.3f})\n\n")

            # Analyze patterns by precision level
            precision_analysis = {}
            for precision_level in ['low', 'medium', 'high']:
                if precision_level in self.effect_size_results:
                    effects = [stats['primary_effect_size'] for stats in self.effect_size_results[precision_level].values()
                             if not np.isnan(stats['primary_effect_size'])]
                    improvements = [stats['mean_difference'] for stats in self.effect_size_results[precision_level].values()
                                  if not np.isnan(stats['primary_effect_size']) and stats['primary_effect_size'] > 0]

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
                            not np.isnan(self.effect_size_results[precision_level][metric]['primary_effect_size'])):
                            effect = self.effect_size_results[precision_level][metric]['primary_effect_size']
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
                    for precision, metric, effect_size, effect_type, diff, _, _ in large_improvements:
                        f.write(f"     - {precision.capitalize()} precision {metric} tasks ({effect_type} = {effect_size:.3f}, improvement = {diff:+.3f})\n")

            if all_degradations:
                f.write("\n2. WHEN TO PREFER ZERO-SHOT:\n")
                f.write("   • Consider zero-shot for:\n")
                for precision, metric, effect_size, effect_type, diff, _, _ in all_degradations:
                    f.write(f"     - {precision.capitalize()} precision {metric} tasks (performance may decline by {abs(diff):.3f}, {effect_type} = {effect_size:.3f})\n")

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
                if not np.isnan(stats['primary_effect_size']):
                    abs_effect = abs(stats['primary_effect_size'])
                    if abs_effect >= 0.8:
                        large_count += 1
                    elif abs_effect >= 0.5:
                        medium_count += 1
                    elif abs_effect >= 0.2:
                        small_count += 1
                    else:
                        negligible_count += 1
        
        print(f"\n📊 EFFECT SIZE DISTRIBUTION:")
        print(f"   Large effects (|effect| ≥ 0.8):    {large_count}")
        print(f"   Medium effects (|effect| ≥ 0.5):   {medium_count}")
        print(f"   Small effects (|effect| ≥ 0.2):    {small_count}")
        print(f"   Negligible effects (|effect| < 0.2): {negligible_count}")
        
        # Show top improvements
        improvements = []
        for precision_level, metrics in self.effect_size_results.items():
            for metric, stats in metrics.items():
                if not np.isnan(stats['primary_effect_size']) and stats['primary_effect_size'] > 0:
                    improvements.append((precision_level, metric, stats['primary_effect_size'], stats['effect_size_name'], stats['mean_difference']))
        
        if improvements:
            improvements.sort(key=lambda x: x[2], reverse=True)
            print(f"\n🚀 TOP ITERATIVE IMPROVEMENTS:")
            for i, (precision, metric, effect_size, effect_type, diff) in enumerate(improvements[:3], 1):
                print(f"   {i}. {precision.capitalize()}/{metric}: +{diff:.3f} ({effect_type} = {effect_size:.3f})")
        
        print(f"\n📁 Results saved in: {self.output_path}")
        print(f"{'='*70}\n")
        print(f"Statistical Tests Used:")
        print(f"  • Paired t-test: Parametric test for mean differences (assumes normality)")
        print(f"  • Wilcoxon signed-rank test: Non-parametric alternative (robust to outliers)")
        print(f"  • Effect Sizes: Cohen's h for success rates, Cohen's d for efficiency metrics")
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
        analyzer.generate_interpretation_report()
        
        # Print summary
        analyzer.print_summary_report()
        
        print(f"\n✅ Effect size analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()