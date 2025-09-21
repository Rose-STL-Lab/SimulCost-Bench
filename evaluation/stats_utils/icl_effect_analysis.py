#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In-Context Learning (ICL) Effect Analysis for Euler 1D Dataset

This script compares the performance between euler_1d (standard) and euler_1d_icl
(with in-context learning examples) to analyze the effectiveness of ICL on model performance.

Usage:
    python evaluation/stats_utils/icl_effect_analysis.py

Input files:
    - eval_results/euler_1d/euler_1d_sum_aggregated.csv
    - eval_results/euler_1d/euler_1d_sum.csv
    - eval_results/euler_1d_icl/euler_1d_icl_sum_aggregated.csv
    - eval_results/euler_1d_icl/euler_1d_icl_sum.csv

Output:
    - eval_results/stats/icl_effect/icl_effect_aggregated.csv (aggregated comparison table)
    - eval_results/stats/icl_effect/icl_effect_detailed.csv (detailed comparison table)
    - eval_results/stats/icl_effect/icl_effect_analysis.xlsx (formatted comparison table)
    - eval_results/stats/icl_effect/success_rate_zero_shot.png (zero-shot success rate)
    - eval_results/stats/icl_effect/success_rate_iterative.png (iterative success rate)
    - eval_results/stats/icl_effect/efficiency_zero_shot.png (zero-shot efficiency)
    - eval_results/stats/icl_effect/efficiency_iterative.png (iterative efficiency)
    - eval_results/stats/icl_effect/success_rate_detailed_zero_shot.png (detailed success rate by precision - zero-shot)
    - eval_results/stats/icl_effect/success_rate_detailed_iterative.png (detailed success rate by precision - iterative)
    - eval_results/stats/icl_effect/efficiency_detailed_zero_shot.png (detailed efficiency by precision - zero-shot)
    - eval_results/stats/icl_effect/efficiency_detailed_iterative.png (detailed efficiency by precision - iterative)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import csv


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the required CSV files for comparison.

    Returns:
        Tuple of (euler_1d_agg, euler_1d_detailed, euler_1d_icl_agg, euler_1d_icl_detailed, euler_1d_icl_no_cost_agg, euler_1d_icl_no_cost_detailed)
    """
    base_path = Path("eval_results")

    # Load aggregated data
    euler_1d_agg = pd.read_csv(base_path / "euler_1d" / "euler_1d_sum_aggregated.csv")
    euler_1d_icl_agg = pd.read_csv(base_path / "euler_1d_icl" / "euler_1d_icl_sum_aggregated.csv")
    euler_1d_icl_no_cost_agg = pd.read_csv(base_path / "euler_1d_icl_no_cost" / "euler_1d_icl_sum_aggregated.csv")

    # Load detailed data
    euler_1d_detailed = pd.read_csv(base_path / "euler_1d" / "euler_1d_sum.csv")
    euler_1d_icl_detailed = pd.read_csv(base_path / "euler_1d_icl" / "euler_1d_icl_sum.csv")
    euler_1d_icl_no_cost_detailed = pd.read_csv(base_path / "euler_1d_icl_no_cost" / "euler_1d_icl_sum.csv")

    return euler_1d_agg, euler_1d_detailed, euler_1d_icl_agg, euler_1d_icl_detailed, euler_1d_icl_no_cost_agg, euler_1d_icl_no_cost_detailed


def is_number(v) -> bool:
    """Check if a value can be converted to float."""
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def calculate_improvement(baseline: float, icl: float) -> float:
    """
    Calculate percentage improvement from baseline to ICL.

    Args:
        baseline: Baseline performance (euler_1d)
        icl: ICL performance (euler_1d_icl)

    Returns:
        Percentage improvement (positive means ICL is better)
    """
    if not is_number(baseline) or not is_number(icl) or baseline == 0:
        return 0.0

    return ((float(icl) - float(baseline)) / float(baseline)) * 100


def analyze_aggregated_data(euler_1d_agg: pd.DataFrame, euler_1d_icl_agg: pd.DataFrame, euler_1d_icl_no_cost_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze aggregated data to compare overall performance.

    Args:
        euler_1d_agg: Aggregated euler_1d results
        euler_1d_icl_agg: Aggregated euler_1d_icl results
        euler_1d_icl_no_cost_agg: Aggregated euler_1d_icl_no_cost results

    Returns:
        DataFrame with comparison results
    """
    comparison_results = []

    # Merge datasets by Model and Inference Mode
    merged = pd.merge(euler_1d_agg, euler_1d_icl_agg, on=['Model', 'Inference Mode'], suffixes=('_baseline', '_icl'))
    merged = pd.merge(merged, euler_1d_icl_no_cost_agg, on=['Model', 'Inference Mode'], suffixes=('', '_icl_no_cost'))

    for _, row in merged.iterrows():
        model = row['Model']
        mode = row['Inference Mode']

        # Calculate improvements for key metrics
        success_rate_baseline = row.get('success_rate_baseline', 0)
        success_rate_icl = row.get('success_rate_icl', 0)
        success_rate_icl_no_cost = row.get('success_rate', 0)  # No suffix for the third dataset
        efficiency_baseline = row.get('mean_efficiency_baseline', 0)
        efficiency_icl = row.get('mean_efficiency_icl', 0)
        efficiency_icl_no_cost = row.get('mean_efficiency', 0)  # No suffix for the third dataset

        success_improvement = calculate_improvement(success_rate_baseline, success_rate_icl)
        success_improvement_no_cost = calculate_improvement(success_rate_baseline, success_rate_icl_no_cost)
        efficiency_improvement = calculate_improvement(efficiency_baseline, efficiency_icl)
        efficiency_improvement_no_cost = calculate_improvement(efficiency_baseline, efficiency_icl_no_cost)

        comparison_results.append({
            'Model': model,
            'Inference Mode': mode,
            'Success Rate (Baseline)': f"{float(success_rate_baseline):.2f}" if is_number(success_rate_baseline) else "N/A",
            'Success Rate (ICL)': f"{float(success_rate_icl):.2f}" if is_number(success_rate_icl) else "N/A",
            'Success Rate (ICL no cost)': f"{float(success_rate_icl_no_cost):.2f}" if is_number(success_rate_icl_no_cost) else "N/A",
            'Success Rate Improvement (%)': f"{success_improvement:.2f}" if success_improvement != 0 else "N/A",
            'Success Rate Improvement no cost (%)': f"{success_improvement_no_cost:.2f}" if success_improvement_no_cost != 0 else "N/A",
            'Efficiency (Baseline)': f"{float(efficiency_baseline):.2f}" if is_number(efficiency_baseline) else "N/A",
            'Efficiency (ICL)': f"{float(efficiency_icl):.2f}" if is_number(efficiency_icl) else "N/A",
            'Efficiency (ICL no cost)': f"{float(efficiency_icl_no_cost):.2f}" if is_number(efficiency_icl_no_cost) else "N/A",
            'Efficiency Improvement (%)': f"{efficiency_improvement:.2f}" if efficiency_improvement != 0 else "N/A",
            'Efficiency Improvement no cost (%)': f"{efficiency_improvement_no_cost:.2f}" if efficiency_improvement_no_cost != 0 else "N/A",
            'Samples (Baseline)': row.get('Number of Samples_baseline', 0),
            'Samples (ICL)': row.get('Number of Samples_icl', 0),
            'Samples (ICL no cost)': row.get('Number of Samples', 0)  # No suffix for the third dataset
        })

    return pd.DataFrame(comparison_results)


def analyze_detailed_data(euler_1d_detailed: pd.DataFrame, euler_1d_icl_detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze detailed data to compare performance by precision level.

    Args:
        euler_1d_detailed: Detailed euler_1d results
        euler_1d_icl_detailed: Detailed euler_1d_icl results

    Returns:
        DataFrame with detailed comparison results
    """
    comparison_results = []

    # Merge datasets by Model, Inference Mode, and Precision Level
    merged = pd.merge(
        euler_1d_detailed,
        euler_1d_icl_detailed,
        on=['Model', 'Inference Mode', 'Precision Level'],
        suffixes=('_baseline', '_icl')
    )

    for _, row in merged.iterrows():
        model = row['Model']
        mode = row['Inference Mode']
        precision = row['Precision Level']

        # Calculate improvements for key metrics
        success_rate_baseline = row.get('success_rate_baseline', 0)
        success_rate_icl = row.get('success_rate_icl', 0)
        efficiency_baseline = row.get('mean_efficiency_baseline', 0)
        efficiency_icl = row.get('mean_efficiency_icl', 0)

        success_improvement = calculate_improvement(success_rate_baseline, success_rate_icl)
        efficiency_improvement = calculate_improvement(efficiency_baseline, efficiency_icl)

        comparison_results.append({
            'Model': model,
            'Inference Mode': mode,
            'Precision Level': precision,
            'Success Rate (Baseline)': f"{float(success_rate_baseline):.2f}" if is_number(success_rate_baseline) else "N/A",
            'Success Rate (ICL)': f"{float(success_rate_icl):.2f}" if is_number(success_rate_icl) else "N/A",
            'Success Rate Improvement (%)': f"{success_improvement:.2f}" if success_improvement != 0 else "N/A",
            'Efficiency (Baseline)': f"{float(efficiency_baseline):.2f}" if is_number(efficiency_baseline) else "N/A",
            'Efficiency (ICL)': f"{float(efficiency_icl):.2f}" if is_number(efficiency_icl) else "N/A",
            'Efficiency Improvement (%)': f"{efficiency_improvement:.2f}" if efficiency_improvement != 0 else "N/A",
            'Samples (Baseline)': row.get('Number of Samples_baseline', 0),
            'Samples (ICL)': row.get('Number of Samples_icl', 0)
        })

    return pd.DataFrame(comparison_results)


def save_comparison_tables(agg_comparison: pd.DataFrame, detailed_comparison: pd.DataFrame) -> None:
    """
    Save comparison tables to CSV and Excel files.

    Args:
        agg_comparison: Aggregated comparison results
        detailed_comparison: Detailed comparison results
    """
    output_dir = Path("eval_results/stats/icl_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated comparison to CSV
    agg_csv_path = output_dir / "icl_effect_aggregated.csv"
    agg_comparison.to_csv(agg_csv_path, index=False)

    # Save detailed comparison to CSV
    detailed_csv_path = output_dir / "icl_effect_detailed.csv"
    detailed_comparison.to_csv(detailed_csv_path, index=False)

    # Save to Excel with multiple sheets
    excel_path = output_dir / "icl_effect_analysis.xlsx"
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        wb = writer.book

        # Format styles
        header_fmt = wb.add_format({
            'bold': True,
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#F0F0F0'
        })

        positive_fmt = wb.add_format({
            'border': 1,
            'bg_color': '#E8F5E8',  # Light green for positive improvements
            'num_format': '0.00'
        })

        negative_fmt = wb.add_format({
            'border': 1,
            'bg_color': '#FFE8E8',  # Light red for negative improvements
            'num_format': '0.00'
        })

        neutral_fmt = wb.add_format({
            'border': 1,
            'num_format': '0.00'
        })

        # Write aggregated results
        agg_comparison.to_excel(writer, sheet_name='Aggregated Comparison', index=False)
        ws1 = writer.sheets['Aggregated Comparison']

        # Format aggregated sheet
        for i, col in enumerate(agg_comparison.columns):
            ws1.write(0, i, col, header_fmt)

            # Auto-adjust column width
            max_len = max(agg_comparison[col].astype(str).map(len).max(), len(col)) + 2
            ws1.set_column(i, i, min(max_len, 25))

        # Write detailed results
        detailed_comparison.to_excel(writer, sheet_name='Detailed Comparison', index=False)
        ws2 = writer.sheets['Detailed Comparison']

        # Format detailed sheet
        for i, col in enumerate(detailed_comparison.columns):
            ws2.write(0, i, col, header_fmt)

            # Auto-adjust column width
            max_len = max(detailed_comparison[col].astype(str).map(len).max(), len(col)) + 2
            ws2.set_column(i, i, min(max_len, 25))

        # Apply conditional formatting for improvement columns
        for sheet, df in [('Aggregated Comparison', agg_comparison), ('Detailed Comparison', detailed_comparison)]:
            ws = writer.sheets[sheet]
            improvement_cols = [col for col in df.columns if 'Improvement (%)' in col]

            for col_name in improvement_cols:
                col_idx = df.columns.get_loc(col_name)

                # Apply conditional formatting based on improvement values
                for row_idx in range(1, len(df) + 1):
                    cell_value = df.iloc[row_idx - 1][col_name]
                    try:
                        if cell_value != "N/A" and is_number(cell_value):
                            value = float(cell_value)
                            if value > 0:
                                ws.write(row_idx, col_idx, value, positive_fmt)
                            elif value < 0:
                                ws.write(row_idx, col_idx, value, negative_fmt)
                            else:
                                ws.write(row_idx, col_idx, value, neutral_fmt)
                    except:
                        pass

    print(f"✅ Comparison tables saved:")
    print(f"   📊 Aggregated CSV: {agg_csv_path}")
    print(f"   📊 Detailed CSV: {detailed_csv_path}")
    print(f"   📈 Excel: {excel_path}")


def create_aggregated_visualization(euler_1d_agg: pd.DataFrame, euler_1d_icl_agg: pd.DataFrame, euler_1d_icl_no_cost_agg: pd.DataFrame, metric: str, mode: str) -> None:
    """
    Create visualization for aggregated ICL effect analysis with three datasets.

    Args:
        euler_1d_agg: Baseline aggregated results
        euler_1d_icl_agg: ICL aggregated results
        euler_1d_icl_no_cost_agg: ICL no cost aggregated results
        metric: Metric type ('success_rate' or 'efficiency')
        mode: Inference mode ('Zero-shot' or 'Iterative')
    """
    output_dir = Path("eval_results/stats/icl_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter data by mode
    baseline_data = euler_1d_agg[euler_1d_agg['Inference Mode'] == mode].copy()
    icl_data = euler_1d_icl_agg[euler_1d_icl_agg['Inference Mode'] == mode].copy()
    icl_no_cost_data = euler_1d_icl_no_cost_agg[euler_1d_icl_no_cost_agg['Inference Mode'] == mode].copy()

    if len(baseline_data) == 0:
        print(f"⚠️  No data available for {mode} mode {metric} visualization")
        return

    # Merge datasets by Model
    merged = pd.merge(baseline_data, icl_data, on=['Model'], suffixes=('_baseline', '_icl'))
    merged = pd.merge(merged, icl_no_cost_data, on=['Model'], suffixes=('', '_no_cost'))

    models = []
    baseline_values = []
    icl_values = []
    icl_no_cost_values = []

    # Get metric column names
    if metric == 'success_rate':
        baseline_col = 'success_rate_baseline'
        icl_col = 'success_rate_icl'
        no_cost_col = 'success_rate'
    else:  # efficiency
        baseline_col = 'mean_efficiency_baseline'
        icl_col = 'mean_efficiency_icl'
        no_cost_col = 'mean_efficiency'

    for _, row in merged.iterrows():
        model = row['Model']
        baseline = row.get(baseline_col, 'N/A')
        icl = row.get(icl_col, 'N/A')
        icl_no_cost = row.get(no_cost_col, 'N/A')

        if baseline != 'N/A' and icl != 'N/A' and icl_no_cost != 'N/A':
            try:
                baseline_val = float(baseline)
                icl_val = float(icl)
                icl_no_cost_val = float(icl_no_cost)
                models.append(model)
                baseline_values.append(baseline_val)
                icl_values.append(icl_val)
                icl_no_cost_values.append(icl_no_cost_val)
            except:
                continue

    if not models:
        print(f"⚠️  No valid data for {mode} mode {metric} visualization")
        return

    # Set up the plot style
    plt.style.use('default')

    # Create figure with square dimensions (width = height)
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))

    # Set fixed subplot parameters to ensure consistent layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

    # Create horizontal bar plot for comparison
    y_pos = np.arange(len(models))
    bar_width = 0.25

    # Create bars for three datasets
    bars1 = ax.barh(y_pos + bar_width, baseline_values, bar_width, label='Baseline',
                    color='gray', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.barh(y_pos, icl_values, bar_width, label='ICL',
                    color='orange', alpha=0.8, edgecolor='black', linewidth=1,
                    hatch='/')
    bars3 = ax.barh(y_pos - bar_width, icl_no_cost_values, bar_width, label='ICL (no cost)',
                    color='white', alpha=1.0, edgecolor='blue', linewidth=2)

    # Customize plot
    metric_title = metric.replace('_', ' ').title()
    ax.set_xlabel(f'{metric_title}', fontweight='bold', fontsize=12)
    ax.set_ylabel('')  # Remove "Models" label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='best', fontsize=11)

    # Set x-axis limits to provide space for text labels
    max_value = max(baseline_values + icl_values + icl_no_cost_values)
    ax.set_xlim(0, max_value * 1.2)  # Add 10% extra space on the right

    # Add value labels on bars
    for bar, value in zip(bars1, baseline_values):
        width = bar.get_width()
        ax.text(width + max_value * 0.01, bar.get_y() + bar.get_height()/2.,
                f'{value:.2f}', ha='left', va='center', fontsize=9)

    for bar, value in zip(bars2, icl_values):
        width = bar.get_width()
        ax.text(width + max_value * 0.01, bar.get_y() + bar.get_height()/2.,
                f'{value:.2f}', ha='left', va='center', fontsize=9)

    for bar, value in zip(bars3, icl_no_cost_values):
        width = bar.get_width()
        ax.text(width + max_value * 0.01, bar.get_y() + bar.get_height()/2.,
                f'{value:.2f}', ha='left', va='center', fontsize=9)

    # Save with fixed dimensions instead of tight bbox to ensure consistency
    output_file = output_dir / f"{metric}_{mode.lower().replace('-', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches=None, facecolor='white')
    plt.close()

    print(f"✅ {mode} mode {metric} visualization saved: {output_file}")


def create_detailed_visualization(detailed_comparison: pd.DataFrame, metric: str, mode: str) -> None:
    """
    Create detailed visualization showing ICL effects across precision levels for a specific mode.

    Args:
        detailed_comparison: Detailed comparison DataFrame
        metric: Metric type ('success_rate' or 'efficiency')
        mode: Inference mode ('Zero-shot' or 'Iterative')
    """
    output_dir = Path("eval_results/stats/icl_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_col = f"{metric.replace('_', ' ').title()} (Baseline)"
    icl_col = f"{metric.replace('_', ' ').title()} (ICL)"

    # Prepare data for visualization
    baseline_data = []
    icl_data = []

    for _, row in detailed_comparison.iterrows():
        if row['Inference Mode'] != mode:
            continue

        baseline = row.get(baseline_col, 'N/A')
        icl = row.get(icl_col, 'N/A')

        if baseline != 'N/A' and icl != 'N/A':
            try:
                baseline_val = float(baseline)
                icl_val = float(icl)
                baseline_data.append({
                    'Model': row['Model'],
                    'Precision Level': row['Precision Level'],
                    'Value': baseline_val,
                    'Type': 'Baseline'
                })
                icl_data.append({
                    'Model': row['Model'],
                    'Precision Level': row['Precision Level'],
                    'Value': icl_val,
                    'Type': 'ICL'
                })
            except:
                continue

    if not baseline_data or not icl_data:
        print(f"⚠️  No valid data for detailed {metric} {mode} visualization")
        return

    # Combine data
    plot_data = baseline_data + icl_data
    df_plot = pd.DataFrame(plot_data)

    # Set up the plot style
    plt.style.use('default')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Create grouped bar chart
    unique_models = df_plot['Model'].unique()
    unique_precisions = sorted(df_plot['Precision Level'].unique())  # Sort for consistent order

    # Define colors and patterns - updated color scheme
    precision_colors = {'low': '#2E8B57', 'medium': '#4169E1', 'high': '#FF69B4'}

    # Create horizontal grouped bar chart
    y_models = np.arange(len(unique_models))
    n_precisions = len(unique_precisions)

    # Calculate positions for bars with gaps between precision groups
    group_height = 0.15  # Height of each bar
    group_gap = 0.05     # Gap between baseline and ICL within same precision
    precision_gap = 0.1  # Gap between different precision levels

    total_group_height = 2 * group_height + group_gap  # Height for one precision level (baseline + ICL)
    total_height_per_model = n_precisions * total_group_height + (n_precisions - 1) * precision_gap

    # Offset to center all bars for each model
    offset_start = -total_height_per_model / 2 + group_height / 2

    # Create bars for each precision level and type combination
    for i, precision in enumerate(unique_precisions):
        baseline_values = []
        icl_values = []

        for model in unique_models:
            baseline_val = df_plot[(df_plot['Model'] == model) &
                                 (df_plot['Precision Level'] == precision) &
                                 (df_plot['Type'] == 'Baseline')]['Value']
            icl_val = df_plot[(df_plot['Model'] == model) &
                            (df_plot['Precision Level'] == precision) &
                            (df_plot['Type'] == 'ICL')]['Value']

            baseline_values.append(baseline_val.iloc[0] if len(baseline_val) > 0 else 0)
            icl_values.append(icl_val.iloc[0] if len(icl_val) > 0 else 0)

        # Calculate y positions for this precision level
        baseline_y = y_models + offset_start + i * (total_group_height + precision_gap)
        icl_y = baseline_y + group_height + group_gap

        # Plot baseline bars (first)
        ax.barh(baseline_y, baseline_values, group_height,
               label=f'{precision.title()} - Baseline' if i < len(unique_precisions) else '',
               color=precision_colors.get(precision, 'gray'), alpha=0.8,
               edgecolor='black', linewidth=0.5)

        # Plot ICL bars with hatching (second)
        ax.barh(icl_y, icl_values, group_height,
               label=f'{precision.title()} - ICL' if i < len(unique_precisions) else '',
               color=precision_colors.get(precision, 'gray'), alpha=0.8,
               edgecolor='black', linewidth=0.5, hatch='/')

    # Customize plot
    metric_title = metric.replace('_', ' ').title()
    ax.set_xlabel(f'{metric_title}', fontweight='bold', fontsize=12)
    ax.set_ylabel('Models', fontweight='bold', fontsize=12)
    ax.set_yticks(y_models)
    ax.set_yticklabels(unique_models, fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # Adjust layout and save
    plt.tight_layout()

    output_file = output_dir / f"{metric}_detailed_{mode.lower().replace('-', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Detailed {metric} {mode} visualization saved: {output_file}")


def print_summary_statistics(agg_comparison: pd.DataFrame) -> None:
    """
    Print summary statistics of ICL effects.

    Args:
        agg_comparison: Aggregated comparison results
    """
    print("\n" + "="*80)
    print("📊 ICL EFFECT ANALYSIS SUMMARY")
    print("="*80)

    for mode in ['Zero-shot', 'Iterative']:
        mode_data = agg_comparison[agg_comparison['Inference Mode'] == mode]

        if len(mode_data) == 0:
            continue

        print(f"\n🔍 {mode} Mode Results:")
        print("-" * 40)

        # Success rate improvements
        success_improvements = []
        success_improvements_no_cost = []
        efficiency_improvements = []
        efficiency_improvements_no_cost = []

        for _, row in mode_data.iterrows():
            success_imp = row.get('Success Rate Improvement (%)', 'N/A')
            success_imp_no_cost = row.get('Success Rate Improvement no cost (%)', 'N/A')
            efficiency_imp = row.get('Efficiency Improvement (%)', 'N/A')
            efficiency_imp_no_cost = row.get('Efficiency Improvement no cost (%)', 'N/A')

            if success_imp != 'N/A' and is_number(success_imp):
                success_improvements.append(float(success_imp))
            if success_imp_no_cost != 'N/A' and is_number(success_imp_no_cost):
                success_improvements_no_cost.append(float(success_imp_no_cost))
            if efficiency_imp != 'N/A' and is_number(efficiency_imp):
                efficiency_improvements.append(float(efficiency_imp))
            if efficiency_imp_no_cost != 'N/A' and is_number(efficiency_imp_no_cost):
                efficiency_improvements_no_cost.append(float(efficiency_imp_no_cost))

        print(f"   📈 ICL vs Baseline:")
        if success_improvements:
            avg_success = np.mean(success_improvements)
            positive_success = sum(1 for x in success_improvements if x > 0)
            print(f"      Success Rate: {avg_success:.2f}% average improvement")
            print(f"      Success Rate: {positive_success}/{len(success_improvements)} models improved")

        if efficiency_improvements:
            avg_efficiency = np.mean(efficiency_improvements)
            positive_efficiency = sum(1 for x in efficiency_improvements if x > 0)
            print(f"      Efficiency: {avg_efficiency:.2f}% average improvement")
            print(f"      Efficiency: {positive_efficiency}/{len(efficiency_improvements)} models improved")

        print(f"   📈 ICL (no cost) vs Baseline:")
        if success_improvements_no_cost:
            avg_success_no_cost = np.mean(success_improvements_no_cost)
            positive_success_no_cost = sum(1 for x in success_improvements_no_cost if x > 0)
            print(f"      Success Rate: {avg_success_no_cost:.2f}% average improvement")
            print(f"      Success Rate: {positive_success_no_cost}/{len(success_improvements_no_cost)} models improved")

        if efficiency_improvements_no_cost:
            avg_efficiency_no_cost = np.mean(efficiency_improvements_no_cost)
            positive_efficiency_no_cost = sum(1 for x in efficiency_improvements_no_cost if x > 0)
            print(f"      Efficiency: {avg_efficiency_no_cost:.2f}% average improvement")
            print(f"      Efficiency: {positive_efficiency_no_cost}/{len(efficiency_improvements_no_cost)} models improved")

    print("\n" + "="*80)


def main():
    """Main function to run ICL effect analysis."""
    print("🔍 Starting ICL Effect Analysis...")
    print("Loading data files...")

    try:
        # Load data
        euler_1d_agg, euler_1d_detailed, euler_1d_icl_agg, euler_1d_icl_detailed, euler_1d_icl_no_cost_agg, euler_1d_icl_no_cost_detailed = load_data()
        print("✅ Data loaded successfully")

        # Analyze aggregated data
        print("\n📊 Analyzing aggregated results...")
        agg_comparison = analyze_aggregated_data(euler_1d_agg, euler_1d_icl_agg, euler_1d_icl_no_cost_agg)

        # Analyze detailed data
        print("📊 Analyzing detailed results...")
        detailed_comparison = analyze_detailed_data(euler_1d_detailed, euler_1d_icl_detailed)

        # Save comparison tables
        print("\n💾 Saving comparison tables...")
        save_comparison_tables(agg_comparison, detailed_comparison)

        # Create visualizations
        print("\n📈 Creating aggregated visualizations...")
        for mode in ['Zero-shot', 'Iterative']:
            for metric in ['success_rate', 'efficiency']:
                create_aggregated_visualization(euler_1d_agg, euler_1d_icl_agg, euler_1d_icl_no_cost_agg, metric, mode)

        # Create detailed visualizations (excluding success_rate_detailed_zero_shot.png and efficiency_detailed_zero_shot.png)
        print("\n📈 Creating detailed visualizations...")
        for metric in ['success_rate', 'efficiency']:
            # Only create iterative mode detailed visualizations
            create_detailed_visualization(detailed_comparison, metric, 'Iterative')

        # Print summary statistics
        print_summary_statistics(agg_comparison)

        print("\n🎉 ICL Effect Analysis completed successfully!")
        print("Check eval_results/stats/icl_effect/ for all output files.")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()