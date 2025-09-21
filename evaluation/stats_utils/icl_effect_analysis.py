用中文解释并完全理解以下脚本的所有代码逻辑：
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
    - eval_results/euler_1d/euler_1d_high_summary.csv
    - eval_results/euler_1d_icl/euler_1d_icl_sum_aggregated.csv
    - eval_results/euler_1d_icl/euler_1d_icl_sum.csv
    - eval_results/euler_1d_icl/euler_1d_icl_high_summary.csv

Output:
    - eval_results/stats/icl_effect/icl_effect_aggregated.csv (aggregated comparison table)
    - eval_results/stats/icl_effect/icl_effect_detailed.csv (detailed comparison table)
    - eval_results/stats/icl_effect/icl_effect_analysis.xlsx (formatted comparison table)
    - eval_results/stats/icl_effect/common_tasks_comparison.csv (common tasks analysis)
    - eval_results/stats/icl_effect/uncommon_tasks_comparison.csv (uncommon tasks analysis)
    - eval_results/stats/icl_effect/task_category_analysis.xlsx (task category analysis)
    - eval_results/stats/icl_effect/success_rate_zero_shot.png (zero-shot success rate)
    - eval_results/stats/icl_effect/success_rate_iterative.png (iterative success rate)
    - eval_results/stats/icl_effect/efficiency_zero_shot.png (zero-shot efficiency)
    - eval_results/stats/icl_effect/efficiency_iterative.png (iterative efficiency)
    - eval_results/stats/icl_effect/success_rate_detailed_zero_shot.png (detailed success rate by precision - zero-shot)
    - eval_results/stats/icl_effect/success_rate_detailed_iterative.png (detailed success rate by precision - iterative)
    - eval_results/stats/icl_effect/efficiency_detailed_zero_shot.png (detailed efficiency by precision - zero-shot)
    - eval_results/stats/icl_effect/efficiency_detailed_iterative.png (detailed efficiency by precision - iterative)
    - eval_results/stats/icl_effect/common_success_rate_zero_shot.png (common tasks success rate - zero-shot)
    - eval_results/stats/icl_effect/common_success_rate_iterative.png (common tasks success rate - iterative)
    - eval_results/stats/icl_effect/common_efficiency_zero_shot.png (common tasks efficiency - zero-shot)
    - eval_results/stats/icl_effect/common_efficiency_iterative.png (common tasks efficiency - iterative)
    - eval_results/stats/icl_effect/uncommon_success_rate_zero_shot.png (uncommon tasks success rate - zero-shot)
    - eval_results/stats/icl_effect/uncommon_success_rate_iterative.png (uncommon tasks success rate - iterative)
    - eval_results/stats/icl_effect/uncommon_efficiency_zero_shot.png (uncommon tasks efficiency - zero-shot)
    - eval_results/stats/icl_effect/uncommon_efficiency_iterative.png (uncommon tasks efficiency - iterative)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import csv

# Task categorization
COMMON_TASKS = ['cfl', 'n_space']
UNCOMMON_TASKS = ['beta', 'k']


def load_task_specific_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load task-specific high-level summary data for baseline and all ICL variants.

    Returns:
        Tuple of (baseline_task_data, icl_task_data, icl_no_cost_task_data, icl_uniform_task_data)
    """
    base_path = Path("eval_results")

    # Load high-level summary data for task-specific analysis
    baseline_task_data = pd.read_csv(base_path / "euler_1d" / "euler_1d_high_summary.csv")
    icl_task_data = pd.read_csv(base_path / "euler_1d_icl" / "euler_1d_icl_high_summary.csv")
    icl_no_cost_task_data = pd.read_csv(base_path / "euler_1d_icl_no_cost" / "euler_1d_icl_high_summary.csv")
    icl_uniform_task_data = pd.read_csv(base_path / "euler_1d_icl_uniform" / "euler_1d_icl_high_summary.csv")

    return baseline_task_data, icl_task_data, icl_no_cost_task_data, icl_uniform_task_data


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the required CSV files for comparison.

    Returns:
        Tuple of (euler_1d_agg, euler_1d_detailed, euler_1d_icl_agg, euler_1d_icl_detailed,
                 euler_1d_icl_no_cost_agg, euler_1d_icl_no_cost_detailed,
                 euler_1d_icl_uniform_agg, euler_1d_icl_uniform_detailed)
    """
    base_path = Path("eval_results")

    # Load aggregated data
    euler_1d_agg = pd.read_csv(base_path / "euler_1d" / "euler_1d_sum_aggregated.csv")
    euler_1d_icl_agg = pd.read_csv(base_path / "euler_1d_icl" / "euler_1d_icl_sum_aggregated.csv")
    euler_1d_icl_no_cost_agg = pd.read_csv(base_path / "euler_1d_icl_no_cost" / "euler_1d_icl_sum_aggregated.csv")
    euler_1d_icl_uniform_agg = pd.read_csv(base_path / "euler_1d_icl_uniform" / "euler_1d_icl_sum_aggregated.csv")

    # Load detailed data
    euler_1d_detailed = pd.read_csv(base_path / "euler_1d" / "euler_1d_sum.csv")
    euler_1d_icl_detailed = pd.read_csv(base_path / "euler_1d_icl" / "euler_1d_icl_sum.csv")
    euler_1d_icl_no_cost_detailed = pd.read_csv(base_path / "euler_1d_icl_no_cost" / "euler_1d_icl_sum.csv")
    euler_1d_icl_uniform_detailed = pd.read_csv(base_path / "euler_1d_icl_uniform" / "euler_1d_icl_sum.csv")

    return (euler_1d_agg, euler_1d_detailed, euler_1d_icl_agg, euler_1d_icl_detailed,
            euler_1d_icl_no_cost_agg, euler_1d_icl_no_cost_detailed,
            euler_1d_icl_uniform_agg, euler_1d_icl_uniform_detailed)


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


def analyze_aggregated_data(euler_1d_agg: pd.DataFrame, euler_1d_icl_agg: pd.DataFrame,
                           euler_1d_icl_no_cost_agg: pd.DataFrame, euler_1d_icl_uniform_agg: pd.DataFrame) -> pd.DataFrame:
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
    merged = pd.merge(merged, euler_1d_icl_uniform_agg, on=['Model', 'Inference Mode'], suffixes=('', '_icl_uniform'))

    for _, row in merged.iterrows():
        model = row['Model']
        mode = row['Inference Mode']

        # Calculate improvements for key metrics
        success_rate_baseline = row.get('success_rate_baseline', 0)
        success_rate_icl = row.get('success_rate_icl', 0)
        success_rate_icl_no_cost = row.get('success_rate', 0)  # No suffix for the third dataset
        success_rate_icl_uniform = row.get('success_rate_icl_uniform', 0)
        efficiency_baseline = row.get('mean_efficiency_baseline', 0)
        efficiency_icl = row.get('mean_efficiency_icl', 0)
        efficiency_icl_no_cost = row.get('mean_efficiency', 0)  # No suffix for the third dataset
        efficiency_icl_uniform = row.get('mean_efficiency_icl_uniform', 0)

        success_improvement = calculate_improvement(success_rate_baseline, success_rate_icl)
        success_improvement_no_cost = calculate_improvement(success_rate_baseline, success_rate_icl_no_cost)
        success_improvement_uniform = calculate_improvement(success_rate_baseline, success_rate_icl_uniform)
        efficiency_improvement = calculate_improvement(efficiency_baseline, efficiency_icl)
        efficiency_improvement_no_cost = calculate_improvement(efficiency_baseline, efficiency_icl_no_cost)
        efficiency_improvement_uniform = calculate_improvement(efficiency_baseline, efficiency_icl_uniform)

        comparison_results.append({
            'Model': model,
            'Inference Mode': mode,
            'Success Rate (Baseline)': f"{float(success_rate_baseline):.2f}" if is_number(success_rate_baseline) else "N/A",
            'Success Rate (ICL)': f"{float(success_rate_icl):.2f}" if is_number(success_rate_icl) else "N/A",
            'Success Rate (ICL no cost)': f"{float(success_rate_icl_no_cost):.2f}" if is_number(success_rate_icl_no_cost) else "N/A",
            'Success Rate (ICL uniform)': f"{float(success_rate_icl_uniform):.2f}" if is_number(success_rate_icl_uniform) else "N/A",
            'Success Rate Improvement (%)': f"{success_improvement:.2f}" if success_improvement != 0 else "N/A",
            'Success Rate Improvement no cost (%)': f"{success_improvement_no_cost:.2f}" if success_improvement_no_cost != 0 else "N/A",
            'Success Rate Improvement uniform (%)': f"{success_improvement_uniform:.2f}" if success_improvement_uniform != 0 else "N/A",
            'Efficiency (Baseline)': f"{float(efficiency_baseline):.2f}" if is_number(efficiency_baseline) else "N/A",
            'Efficiency (ICL)': f"{float(efficiency_icl):.2f}" if is_number(efficiency_icl) else "N/A",
            'Efficiency (ICL no cost)': f"{float(efficiency_icl_no_cost):.2f}" if is_number(efficiency_icl_no_cost) else "N/A",
            'Efficiency (ICL uniform)': f"{float(efficiency_icl_uniform):.2f}" if is_number(efficiency_icl_uniform) else "N/A",
            'Efficiency Improvement (%)': f"{efficiency_improvement:.2f}" if efficiency_improvement != 0 else "N/A",
            'Efficiency Improvement no cost (%)': f"{efficiency_improvement_no_cost:.2f}" if efficiency_improvement_no_cost != 0 else "N/A",
            'Efficiency Improvement uniform (%)': f"{efficiency_improvement_uniform:.2f}" if efficiency_improvement_uniform != 0 else "N/A",
            'Samples (Baseline)': row.get('Number of Samples_baseline', 0),
            'Samples (ICL)': row.get('Number of Samples_icl', 0),
            'Samples (ICL no cost)': row.get('Number of Samples', 0),  # No suffix for the third dataset
            'Samples (ICL uniform)': row.get('Number of Samples_icl_uniform', 0)
        })

    return pd.DataFrame(comparison_results)


def analyze_detailed_data(euler_1d_detailed: pd.DataFrame, euler_1d_icl_detailed: pd.DataFrame,
                         euler_1d_icl_no_cost_detailed: pd.DataFrame, euler_1d_icl_uniform_detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze detailed data to compare performance by precision level.

    Args:
        euler_1d_detailed: Detailed euler_1d results
        euler_1d_icl_detailed: Detailed euler_1d_icl results
        euler_1d_icl_no_cost_detailed: Detailed euler_1d_icl_no_cost results
        euler_1d_icl_uniform_detailed: Detailed euler_1d_icl_uniform results

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
    merged = pd.merge(merged, euler_1d_icl_no_cost_detailed, on=['Model', 'Inference Mode', 'Precision Level'], suffixes=('', '_icl_no_cost'))
    merged = pd.merge(merged, euler_1d_icl_uniform_detailed, on=['Model', 'Inference Mode', 'Precision Level'], suffixes=('', '_icl_uniform'))

    for _, row in merged.iterrows():
        model = row['Model']
        mode = row['Inference Mode']
        precision = row['Precision Level']

        # Calculate improvements for key metrics
        success_rate_baseline = row.get('success_rate_baseline', 0)
        success_rate_icl = row.get('success_rate_icl', 0)
        success_rate_icl_no_cost = row.get('success_rate', 0)  # No suffix for the third dataset
        success_rate_icl_uniform = row.get('success_rate_icl_uniform', 0)
        efficiency_baseline = row.get('mean_efficiency_baseline', 0)
        efficiency_icl = row.get('mean_efficiency_icl', 0)
        efficiency_icl_no_cost = row.get('mean_efficiency', 0)  # No suffix for the third dataset
        efficiency_icl_uniform = row.get('mean_efficiency_icl_uniform', 0)

        success_improvement = calculate_improvement(success_rate_baseline, success_rate_icl)
        success_improvement_no_cost = calculate_improvement(success_rate_baseline, success_rate_icl_no_cost)
        success_improvement_uniform = calculate_improvement(success_rate_baseline, success_rate_icl_uniform)
        efficiency_improvement = calculate_improvement(efficiency_baseline, efficiency_icl)
        efficiency_improvement_no_cost = calculate_improvement(efficiency_baseline, efficiency_icl_no_cost)
        efficiency_improvement_uniform = calculate_improvement(efficiency_baseline, efficiency_icl_uniform)

        comparison_results.append({
            'Model': model,
            'Inference Mode': mode,
            'Precision Level': precision,
            'Success Rate (Baseline)': f"{float(success_rate_baseline):.2f}" if is_number(success_rate_baseline) else "N/A",
            'Success Rate (ICL)': f"{float(success_rate_icl):.2f}" if is_number(success_rate_icl) else "N/A",
            'Success Rate (ICL no cost)': f"{float(success_rate_icl_no_cost):.2f}" if is_number(success_rate_icl_no_cost) else "N/A",
            'Success Rate (ICL uniform)': f"{float(success_rate_icl_uniform):.2f}" if is_number(success_rate_icl_uniform) else "N/A",
            'Success Rate Improvement (%)': f"{success_improvement:.2f}" if success_improvement != 0 else "N/A",
            'Success Rate Improvement no cost (%)': f"{success_improvement_no_cost:.2f}" if success_improvement_no_cost != 0 else "N/A",
            'Success Rate Improvement uniform (%)': f"{success_improvement_uniform:.2f}" if success_improvement_uniform != 0 else "N/A",
            'Efficiency (Baseline)': f"{float(efficiency_baseline):.2f}" if is_number(efficiency_baseline) else "N/A",
            'Efficiency (ICL)': f"{float(efficiency_icl):.2f}" if is_number(efficiency_icl) else "N/A",
            'Efficiency (ICL no cost)': f"{float(efficiency_icl_no_cost):.2f}" if is_number(efficiency_icl_no_cost) else "N/A",
            'Efficiency (ICL uniform)': f"{float(efficiency_icl_uniform):.2f}" if is_number(efficiency_icl_uniform) else "N/A",
            'Efficiency Improvement (%)': f"{efficiency_improvement:.2f}" if efficiency_improvement != 0 else "N/A",
            'Efficiency Improvement no cost (%)': f"{efficiency_improvement_no_cost:.2f}" if efficiency_improvement_no_cost != 0 else "N/A",
            'Efficiency Improvement uniform (%)': f"{efficiency_improvement_uniform:.2f}" if efficiency_improvement_uniform != 0 else "N/A",
            'Samples (Baseline)': row.get('Number of Samples_baseline', 0),
            'Samples (ICL)': row.get('Number of Samples_icl', 0),
            'Samples (ICL no cost)': row.get('Number of Samples', 0),  # No suffix for the third dataset
            'Samples (ICL uniform)': row.get('Number of Samples_icl_uniform', 0)
        })

    return pd.DataFrame(comparison_results)


def analyze_task_category_data(baseline_task_data: pd.DataFrame, icl_task_data: pd.DataFrame,
                              icl_no_cost_task_data: pd.DataFrame, icl_uniform_task_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze task performance breakdown by common vs uncommon tasks.

    Args:
        baseline_task_data: Baseline task-specific data
        icl_task_data: ICL task-specific data
        icl_no_cost_task_data: ICL no cost task-specific data
        icl_uniform_task_data: ICL uniform task-specific data

    Returns:
        Tuple of (common_task_comparison, uncommon_task_comparison)
    """
    def create_category_comparison(tasks: List[str], category_name: str) -> pd.DataFrame:
        comparison_results = []

        # Filter data for the specific tasks
        baseline_filtered = baseline_task_data[baseline_task_data['Task'].isin(tasks)]
        icl_filtered = icl_task_data[icl_task_data['Task'].isin(tasks)]
        icl_no_cost_filtered = icl_no_cost_task_data[icl_no_cost_task_data['Task'].isin(tasks)]
        icl_uniform_filtered = icl_uniform_task_data[icl_uniform_task_data['Task'].isin(tasks)]

        # Group by Model and Inference Mode, sum the metrics across tasks
        baseline_grouped = baseline_filtered.groupby(['Model', 'Inference Mode']).agg({
            'Number of Samples': 'sum',
            'success_rate': 'mean',
            'mean_efficiency': 'mean'
        }).reset_index()

        icl_grouped = icl_filtered.groupby(['Model', 'Inference Mode']).agg({
            'Number of Samples': 'sum',
            'success_rate': 'mean',
            'mean_efficiency': 'mean'
        }).reset_index()

        icl_no_cost_grouped = icl_no_cost_filtered.groupby(['Model', 'Inference Mode']).agg({
            'Number of Samples': 'sum',
            'success_rate': 'mean',
            'mean_efficiency': 'mean'
        }).reset_index()

        icl_uniform_grouped = icl_uniform_filtered.groupby(['Model', 'Inference Mode']).agg({
            'Number of Samples': 'sum',
            'success_rate': 'mean',
            'mean_efficiency': 'mean'
        }).reset_index()

        # Merge all datasets
        merged = pd.merge(baseline_grouped, icl_grouped, on=['Model', 'Inference Mode'], suffixes=('_baseline', '_icl'))
        merged = pd.merge(merged, icl_no_cost_grouped, on=['Model', 'Inference Mode'], suffixes=('', '_no_cost'))
        merged = pd.merge(merged, icl_uniform_grouped, on=['Model', 'Inference Mode'], suffixes=('', '_uniform'))

        for _, row in merged.iterrows():
            model = row['Model']
            mode = row['Inference Mode']

            success_rate_baseline = row.get('success_rate_baseline', 0)
            success_rate_icl = row.get('success_rate_icl', 0)
            success_rate_no_cost = row.get('success_rate', 0)  # no suffix for no_cost
            success_rate_uniform = row.get('success_rate_uniform', 0)

            efficiency_baseline = row.get('mean_efficiency_baseline', 0)
            efficiency_icl = row.get('mean_efficiency_icl', 0)
            efficiency_no_cost = row.get('mean_efficiency', 0)  # no suffix for no_cost
            efficiency_uniform = row.get('mean_efficiency_uniform', 0)

            success_improvement = calculate_improvement(success_rate_baseline, success_rate_icl)
            success_improvement_no_cost = calculate_improvement(success_rate_baseline, success_rate_no_cost)
            success_improvement_uniform = calculate_improvement(success_rate_baseline, success_rate_uniform)

            efficiency_improvement = calculate_improvement(efficiency_baseline, efficiency_icl)
            efficiency_improvement_no_cost = calculate_improvement(efficiency_baseline, efficiency_no_cost)
            efficiency_improvement_uniform = calculate_improvement(efficiency_baseline, efficiency_uniform)

            comparison_results.append({
                'Model': model,
                'Inference Mode': mode,
                'Task Category': category_name,
                'Success Rate (Baseline)': f"{float(success_rate_baseline):.2f}" if is_number(success_rate_baseline) else "N/A",
                'Success Rate (ICL)': f"{float(success_rate_icl):.2f}" if is_number(success_rate_icl) else "N/A",
                'Success Rate (ICL no cost)': f"{float(success_rate_no_cost):.2f}" if is_number(success_rate_no_cost) else "N/A",
                'Success Rate (ICL uniform)': f"{float(success_rate_uniform):.2f}" if is_number(success_rate_uniform) else "N/A",
                'Success Rate Improvement (%)': f"{success_improvement:.2f}" if success_improvement != 0 else "N/A",
                'Success Rate Improvement no cost (%)': f"{success_improvement_no_cost:.2f}" if success_improvement_no_cost != 0 else "N/A",
                'Success Rate Improvement uniform (%)': f"{success_improvement_uniform:.2f}" if success_improvement_uniform != 0 else "N/A",
                'Efficiency (Baseline)': f"{float(efficiency_baseline):.2f}" if is_number(efficiency_baseline) else "N/A",
                'Efficiency (ICL)': f"{float(efficiency_icl):.2f}" if is_number(efficiency_icl) else "N/A",
                'Efficiency (ICL no cost)': f"{float(efficiency_no_cost):.2f}" if is_number(efficiency_no_cost) else "N/A",
                'Efficiency (ICL uniform)': f"{float(efficiency_uniform):.2f}" if is_number(efficiency_uniform) else "N/A",
                'Efficiency Improvement (%)': f"{efficiency_improvement:.2f}" if efficiency_improvement != 0 else "N/A",
                'Efficiency Improvement no cost (%)': f"{efficiency_improvement_no_cost:.2f}" if efficiency_improvement_no_cost != 0 else "N/A",
                'Efficiency Improvement uniform (%)': f"{efficiency_improvement_uniform:.2f}" if efficiency_improvement_uniform != 0 else "N/A",
                'Samples (Baseline)': row.get('Number of Samples_baseline', 0),
                'Samples (ICL)': row.get('Number of Samples_icl', 0),
                'Samples (ICL no cost)': row.get('Number of Samples', 0),  # no suffix for no_cost
                'Samples (ICL uniform)': row.get('Number of Samples_uniform', 0)
            })

        return pd.DataFrame(comparison_results)

    common_comparison = create_category_comparison(COMMON_TASKS, 'Common')
    uncommon_comparison = create_category_comparison(UNCOMMON_TASKS, 'Uncommon')

    return common_comparison, uncommon_comparison


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


def create_aggregated_visualization(euler_1d_agg: pd.DataFrame, euler_1d_icl_agg: pd.DataFrame,
                                   euler_1d_icl_no_cost_agg: pd.DataFrame, euler_1d_icl_uniform_agg: pd.DataFrame,
                                   metric: str, mode: str) -> None:
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
    icl_uniform_data = euler_1d_icl_uniform_agg[euler_1d_icl_uniform_agg['Inference Mode'] == mode].copy()

    if len(baseline_data) == 0:
        print(f"⚠️  No data available for {mode} mode {metric} visualization")
        return

    # Merge datasets by Model
    merged = pd.merge(baseline_data, icl_data, on=['Model'], suffixes=('_baseline', '_icl'))
    merged = pd.merge(merged, icl_no_cost_data, on=['Model'], suffixes=('', '_no_cost'))
    merged = pd.merge(merged, icl_uniform_data, on=['Model'], suffixes=('', '_uniform'))

    models = []
    baseline_values = []
    icl_values = []
    icl_no_cost_values = []
    icl_uniform_values = []

    # Get metric column names
    if metric == 'success_rate':
        baseline_col = 'success_rate_baseline'
        icl_col = 'success_rate_icl'
        no_cost_col = 'success_rate'
        uniform_col = 'success_rate_uniform'
    else:  # efficiency
        baseline_col = 'mean_efficiency_baseline'
        icl_col = 'mean_efficiency_icl'
        no_cost_col = 'mean_efficiency'
        uniform_col = 'mean_efficiency_uniform'

    for _, row in merged.iterrows():
        model = row['Model']
        baseline = row.get(baseline_col, 'N/A')
        icl = row.get(icl_col, 'N/A')
        icl_no_cost = row.get(no_cost_col, 'N/A')
        icl_uniform = row.get(uniform_col, 'N/A')

        if baseline != 'N/A' and icl != 'N/A' and icl_no_cost != 'N/A' and icl_uniform != 'N/A':
            try:
                baseline_val = float(baseline)
                icl_val = float(icl)
                icl_no_cost_val = float(icl_no_cost)
                icl_uniform_val = float(icl_uniform)
                models.append(model)
                baseline_values.append(baseline_val)
                icl_values.append(icl_val)
                icl_no_cost_values.append(icl_no_cost_val)
                icl_uniform_values.append(icl_uniform_val)
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
    y_pos = np.arange(len(models)) * 1.2  # Increase spacing between models
    bar_width = 0.15
    bar_gap = 0.1

    # Create bars for four datasets with gaps
    bars1 = ax.barh(y_pos + 1.5*(bar_width + bar_gap), baseline_values, bar_width, label='Baseline',
                    color='gray', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.barh(y_pos + 0.5*(bar_width + bar_gap), icl_values, bar_width, label='ICL',
                    color='orange', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.barh(y_pos - 0.5*(bar_width + bar_gap), icl_no_cost_values, bar_width, label='ICL (no cost)',
                    color='white', alpha=1.0, edgecolor='orange', linewidth=2)
    bars4 = ax.barh(y_pos - 1.5*(bar_width + bar_gap), icl_uniform_values, bar_width, label='ICL (uniform)',
                    color='orange', alpha=0.8, edgecolor='black', linewidth=1,
                    hatch='///')

    # Customize plot
    metric_title = metric.replace('_', ' ').title()
    ax.set_xlabel(f'{metric_title}', fontweight='bold', fontsize=12)
    ax.set_ylabel('')  # Remove "Models" label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=4, fontsize=11, frameon=False, columnspacing=2.5)

    # Set x-axis limits
    max_value = max(baseline_values + icl_values + icl_no_cost_values + icl_uniform_values)
    ax.set_xlim(0, max_value * 1.05)  # Add 5% extra space on the right


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
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=6, fontsize=9, frameon=False, columnspacing=2.0)

    # Adjust layout and save
    plt.tight_layout()

    output_file = output_dir / f"{metric}_detailed_{mode.lower().replace('-', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✅ Detailed {metric} {mode} visualization saved: {output_file}")


def create_task_category_visualization(category_data: pd.DataFrame, category_name: str, metric: str, mode: str) -> None:
    """
    Create visualization for task category ICL effect analysis with 4 bars matching the main visualization style.

    Args:
        category_data: Category-specific comparison data
        category_name: Category name ('Common' or 'Uncommon')
        metric: Metric type ('success_rate' or 'efficiency')
        mode: Inference mode ('Zero-shot' or 'Iterative')
    """
    output_dir = Path("eval_results/stats/icl_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter data by mode
    mode_data = category_data[category_data['Inference Mode'] == mode].copy()

    if len(mode_data) == 0:
        print(f"⚠️  No data available for {category_name} tasks {mode} mode {metric} visualization")
        return

    models = []
    baseline_values = []
    icl_values = []
    icl_no_cost_values = []
    icl_uniform_values = []

    # Get metric column names
    if metric == 'success_rate':
        baseline_col = 'Success Rate (Baseline)'
        icl_col = 'Success Rate (ICL)'
        no_cost_col = 'Success Rate (ICL no cost)'
        uniform_col = 'Success Rate (ICL uniform)'
    else:  # efficiency
        baseline_col = 'Efficiency (Baseline)'
        icl_col = 'Efficiency (ICL)'
        no_cost_col = 'Efficiency (ICL no cost)'
        uniform_col = 'Efficiency (ICL uniform)'

    for _, row in mode_data.iterrows():
        model = row['Model']
        baseline = row.get(baseline_col, 'N/A')
        icl = row.get(icl_col, 'N/A')
        icl_no_cost = row.get(no_cost_col, 'N/A')
        icl_uniform = row.get(uniform_col, 'N/A')

        if baseline != 'N/A' and icl != 'N/A' and icl_no_cost != 'N/A' and icl_uniform != 'N/A':
            try:
                baseline_val = float(baseline)
                icl_val = float(icl)
                icl_no_cost_val = float(icl_no_cost)
                icl_uniform_val = float(icl_uniform)
                models.append(model)
                baseline_values.append(baseline_val)
                icl_values.append(icl_val)
                icl_no_cost_values.append(icl_no_cost_val)
                icl_uniform_values.append(icl_uniform_val)
            except:
                continue

    if not models:
        print(f"⚠️  No valid data for {category_name} tasks {mode} mode {metric} visualization")
        return

    # Set up the plot style
    plt.style.use('default')

    # Create figure with same dimensions as main visualization
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))

    # Set fixed subplot parameters to ensure consistent layout
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

    # Create horizontal bar plot for comparison - match exact style from create_aggregated_visualization
    y_pos = np.arange(len(models)) * 1.2  # Increase spacing between models (same as original)
    bar_width = 0.15
    bar_gap = 0.1

    # Use category-specific colors for ICL bars while keeping the exact same style
    if category_name.lower() == 'common':
        icl_color = 'lightgreen'
    else:  # uncommon
        icl_color = 'steelblue'

    # Create bars for four datasets with gaps - exact same positioning as original
    bars1 = ax.barh(y_pos + 1.5*(bar_width + bar_gap), baseline_values, bar_width, label='Baseline',
                    color='gray', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.barh(y_pos + 0.5*(bar_width + bar_gap), icl_values, bar_width, label='ICL',
                    color=icl_color, alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.barh(y_pos - 0.5*(bar_width + bar_gap), icl_no_cost_values, bar_width, label='ICL (no cost)',
                    color='white', alpha=1.0, edgecolor=icl_color, linewidth=2)
    bars4 = ax.barh(y_pos - 1.5*(bar_width + bar_gap), icl_uniform_values, bar_width, label='ICL (uniform)',
                    color=icl_color, alpha=0.8, edgecolor='black', linewidth=1,
                    hatch='///')

    # Customize plot - match exact style from original
    metric_title = metric.replace('_', ' ').title()
    ax.set_xlabel(f'{metric_title}', fontweight='bold', fontsize=12)
    ax.set_ylabel('')  # Remove "Models" label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=4, fontsize=11, frameon=False, columnspacing=2.5)

    # Set x-axis limits
    max_value = max(baseline_values + icl_values + icl_no_cost_values + icl_uniform_values)
    ax.set_xlim(0, max_value * 1.05)  # Add 5% extra space on the right

    # Save with fixed dimensions instead of tight bbox to ensure consistency
    output_file = output_dir / f"{category_name.lower()}_{metric}_{mode.lower().replace('-', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches=None, facecolor='white')
    plt.close()

    print(f"✅ {category_name} tasks {mode} mode {metric} visualization saved: {output_file}")


def save_improvement_summary(agg_comparison: pd.DataFrame) -> None:
    """
    Save detailed improvement statistics to a text file.

    Args:
        agg_comparison: Aggregated comparison results
    """
    output_dir = Path("eval_results/stats/icl_effect")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "improvement_summary.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ICL EFFECT ANALYSIS - DETAILED IMPROVEMENT SUMMARY\n")
        f.write("="*80 + "\n\n")

        for mode in ['Zero-shot', 'Iterative']:
            mode_data = agg_comparison[agg_comparison['Inference Mode'] == mode]

            if len(mode_data) == 0:
                continue

            f.write(f"📊 {mode} Mode Results:\n")
            f.write("-" * 50 + "\n\n")

            # Process each model
            for _, row in mode_data.iterrows():
                model = row['Model']
                f.write(f"🔍 Model: {model}\n")

                # Success Rate Results
                f.write(f"   Success Rate Results:\n")
                baseline_sr = row.get('Success Rate (Baseline)', 'N/A')
                icl_sr = row.get('Success Rate (ICL)', 'N/A')
                icl_no_cost_sr = row.get('Success Rate (ICL no cost)', 'N/A')
                icl_uniform_sr = row.get('Success Rate (ICL uniform)', 'N/A')

                f.write(f"     Baseline: {baseline_sr}\n")
                f.write(f"     ICL: {icl_sr}\n")
                f.write(f"     ICL (no cost): {icl_no_cost_sr}\n")
                f.write(f"     ICL (uniform): {icl_uniform_sr}\n")

                # Success Rate Improvements
                icl_imp = row.get('Success Rate Improvement (%)', 'N/A')
                icl_no_cost_imp = row.get('Success Rate Improvement no cost (%)', 'N/A')
                icl_uniform_imp = row.get('Success Rate Improvement uniform (%)', 'N/A')

                f.write(f"     Improvements vs Baseline:\n")
                f.write(f"       ICL: {icl_imp}%\n")
                f.write(f"       ICL (no cost): {icl_no_cost_imp}%\n")
                f.write(f"       ICL (uniform): {icl_uniform_imp}%\n")

                # Efficiency Results
                f.write(f"   Efficiency Results:\n")
                baseline_eff = row.get('Efficiency (Baseline)', 'N/A')
                icl_eff = row.get('Efficiency (ICL)', 'N/A')
                icl_no_cost_eff = row.get('Efficiency (ICL no cost)', 'N/A')
                icl_uniform_eff = row.get('Efficiency (ICL uniform)', 'N/A')

                f.write(f"     Baseline: {baseline_eff}\n")
                f.write(f"     ICL: {icl_eff}\n")
                f.write(f"     ICL (no cost): {icl_no_cost_eff}\n")
                f.write(f"     ICL (uniform): {icl_uniform_eff}\n")

                # Efficiency Improvements
                eff_icl_imp = row.get('Efficiency Improvement (%)', 'N/A')
                eff_icl_no_cost_imp = row.get('Efficiency Improvement no cost (%)', 'N/A')
                eff_icl_uniform_imp = row.get('Efficiency Improvement uniform (%)', 'N/A')

                f.write(f"     Improvements vs Baseline:\n")
                f.write(f"       ICL: {eff_icl_imp}%\n")
                f.write(f"       ICL (no cost): {eff_icl_no_cost_imp}%\n")
                f.write(f"       ICL (uniform): {eff_icl_uniform_imp}%\n")

                f.write("\n")

            # Calculate summary statistics for this mode
            f.write(f"📈 {mode} Mode Summary Statistics:\n")
            f.write("-" * 30 + "\n")

            # Collect all improvements for averaging
            success_improvements = []
            success_improvements_no_cost = []
            success_improvements_uniform = []
            efficiency_improvements = []
            efficiency_improvements_no_cost = []
            efficiency_improvements_uniform = []

            for _, row in mode_data.iterrows():
                for improvements, col_name in [
                    (success_improvements, 'Success Rate Improvement (%)'),
                    (success_improvements_no_cost, 'Success Rate Improvement no cost (%)'),
                    (success_improvements_uniform, 'Success Rate Improvement uniform (%)'),
                    (efficiency_improvements, 'Efficiency Improvement (%)'),
                    (efficiency_improvements_no_cost, 'Efficiency Improvement no cost (%)'),
                    (efficiency_improvements_uniform, 'Efficiency Improvement uniform (%)')
                ]:
                    imp = row.get(col_name, 'N/A')
                    if imp != 'N/A' and is_number(imp):
                        improvements.append(float(imp))

            # Write averages and positive counts
            if success_improvements:
                avg_success = np.mean(success_improvements)
                positive_success = sum(1 for x in success_improvements if x > 0)
                f.write(f"   Success Rate - ICL vs Baseline:\n")
                f.write(f"     Average improvement: {avg_success:.2f}%\n")
                f.write(f"     Models improved: {positive_success}/{len(success_improvements)}\n")

            if success_improvements_no_cost:
                avg_success_no_cost = np.mean(success_improvements_no_cost)
                positive_success_no_cost = sum(1 for x in success_improvements_no_cost if x > 0)
                f.write(f"   Success Rate - ICL (no cost) vs Baseline:\n")
                f.write(f"     Average improvement: {avg_success_no_cost:.2f}%\n")
                f.write(f"     Models improved: {positive_success_no_cost}/{len(success_improvements_no_cost)}\n")

            if success_improvements_uniform:
                avg_success_uniform = np.mean(success_improvements_uniform)
                positive_success_uniform = sum(1 for x in success_improvements_uniform if x > 0)
                f.write(f"   Success Rate - ICL (uniform) vs Baseline:\n")
                f.write(f"     Average improvement: {avg_success_uniform:.2f}%\n")
                f.write(f"     Models improved: {positive_success_uniform}/{len(success_improvements_uniform)}\n")

            if efficiency_improvements:
                avg_efficiency = np.mean(efficiency_improvements)
                positive_efficiency = sum(1 for x in efficiency_improvements if x > 0)
                f.write(f"   Efficiency - ICL vs Baseline:\n")
                f.write(f"     Average improvement: {avg_efficiency:.2f}%\n")
                f.write(f"     Models improved: {positive_efficiency}/{len(efficiency_improvements)}\n")

            if efficiency_improvements_no_cost:
                avg_efficiency_no_cost = np.mean(efficiency_improvements_no_cost)
                positive_efficiency_no_cost = sum(1 for x in efficiency_improvements_no_cost if x > 0)
                f.write(f"   Efficiency - ICL (no cost) vs Baseline:\n")
                f.write(f"     Average improvement: {avg_efficiency_no_cost:.2f}%\n")
                f.write(f"     Models improved: {positive_efficiency_no_cost}/{len(efficiency_improvements_no_cost)}\n")

            if efficiency_improvements_uniform:
                avg_efficiency_uniform = np.mean(efficiency_improvements_uniform)
                positive_efficiency_uniform = sum(1 for x in efficiency_improvements_uniform if x > 0)
                f.write(f"   Efficiency - ICL (uniform) vs Baseline:\n")
                f.write(f"     Average improvement: {avg_efficiency_uniform:.2f}%\n")
                f.write(f"     Models improved: {positive_efficiency_uniform}/{len(efficiency_improvements_uniform)}\n")

            f.write("\n")

        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")

    print(f"✅ Improvement summary saved: {output_file}")


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
        success_improvements_uniform = []
        efficiency_improvements = []
        efficiency_improvements_no_cost = []
        efficiency_improvements_uniform = []

        for _, row in mode_data.iterrows():
            success_imp = row.get('Success Rate Improvement (%)', 'N/A')
            success_imp_no_cost = row.get('Success Rate Improvement no cost (%)', 'N/A')
            success_imp_uniform = row.get('Success Rate Improvement uniform (%)', 'N/A')
            efficiency_imp = row.get('Efficiency Improvement (%)', 'N/A')
            efficiency_imp_no_cost = row.get('Efficiency Improvement no cost (%)', 'N/A')
            efficiency_imp_uniform = row.get('Efficiency Improvement uniform (%)', 'N/A')

            if success_imp != 'N/A' and is_number(success_imp):
                success_improvements.append(float(success_imp))
            if success_imp_no_cost != 'N/A' and is_number(success_imp_no_cost):
                success_improvements_no_cost.append(float(success_imp_no_cost))
            if success_imp_uniform != 'N/A' and is_number(success_imp_uniform):
                success_improvements_uniform.append(float(success_imp_uniform))
            if efficiency_imp != 'N/A' and is_number(efficiency_imp):
                efficiency_improvements.append(float(efficiency_imp))
            if efficiency_imp_no_cost != 'N/A' and is_number(efficiency_imp_no_cost):
                efficiency_improvements_no_cost.append(float(efficiency_imp_no_cost))
            if efficiency_imp_uniform != 'N/A' and is_number(efficiency_imp_uniform):
                efficiency_improvements_uniform.append(float(efficiency_imp_uniform))

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

        print(f"   📈 ICL (uniform) vs Baseline:")
        if success_improvements_uniform:
            avg_success_uniform = np.mean(success_improvements_uniform)
            positive_success_uniform = sum(1 for x in success_improvements_uniform if x > 0)
            print(f"      Success Rate: {avg_success_uniform:.2f}% average improvement")
            print(f"      Success Rate: {positive_success_uniform}/{len(success_improvements_uniform)} models improved")

        if efficiency_improvements_uniform:
            avg_efficiency_uniform = np.mean(efficiency_improvements_uniform)
            positive_efficiency_uniform = sum(1 for x in efficiency_improvements_uniform if x > 0)
            print(f"      Efficiency: {avg_efficiency_uniform:.2f}% average improvement")
            print(f"      Efficiency: {positive_efficiency_uniform}/{len(efficiency_improvements_uniform)} models improved")

    print("\n" + "="*80)


def main():
    """Main function to run ICL effect analysis."""
    print("🔍 Starting ICL Effect Analysis...")
    print("Loading data files...")

    try:
        # Load data
        (euler_1d_agg, euler_1d_detailed, euler_1d_icl_agg, euler_1d_icl_detailed,
         euler_1d_icl_no_cost_agg, euler_1d_icl_no_cost_detailed,
         euler_1d_icl_uniform_agg, euler_1d_icl_uniform_detailed) = load_data()
        print("✅ Data loaded successfully")

        # Load task-specific data
        print("Loading task-specific data...")
        baseline_task_data, icl_task_data, icl_no_cost_task_data, icl_uniform_task_data = load_task_specific_data()
        print("✅ Task-specific data loaded successfully")

        # Analyze aggregated data
        print("\n📊 Analyzing aggregated results...")
        agg_comparison = analyze_aggregated_data(euler_1d_agg, euler_1d_icl_agg,
                                                euler_1d_icl_no_cost_agg, euler_1d_icl_uniform_agg)

        # Analyze detailed data
        print("📊 Analyzing detailed results...")
        detailed_comparison = analyze_detailed_data(euler_1d_detailed, euler_1d_icl_detailed,
                                                   euler_1d_icl_no_cost_detailed, euler_1d_icl_uniform_detailed)

        # Analyze task category data
        print("📊 Analyzing task category results...")
        common_comparison, uncommon_comparison = analyze_task_category_data(baseline_task_data, icl_task_data,
                                                                           icl_no_cost_task_data, icl_uniform_task_data)

        # Save comparison tables
        print("\n💾 Saving comparison tables...")
        save_comparison_tables(agg_comparison, detailed_comparison)

        # Save task category tables
        print("💾 Saving task category tables...")
        output_dir = Path("eval_results/stats/icl_effect")
        output_dir.mkdir(parents=True, exist_ok=True)

        common_comparison.to_csv(output_dir / "common_tasks_comparison.csv", index=False)
        uncommon_comparison.to_csv(output_dir / "uncommon_tasks_comparison.csv", index=False)

        # Combine and save task category data to Excel
        with pd.ExcelWriter(output_dir / "task_category_analysis.xlsx", engine='xlsxwriter') as writer:
            common_comparison.to_excel(writer, sheet_name='Common Tasks', index=False)
            uncommon_comparison.to_excel(writer, sheet_name='Uncommon Tasks', index=False)

        print(f"✅ Task category tables saved:")
        print(f"   📊 Common tasks CSV: {output_dir / 'common_tasks_comparison.csv'}")
        print(f"   📊 Uncommon tasks CSV: {output_dir / 'uncommon_tasks_comparison.csv'}")
        print(f"   📈 Task category Excel: {output_dir / 'task_category_analysis.xlsx'}")

        # Save improvement summary to txt file
        print("💾 Saving improvement summary...")
        save_improvement_summary(agg_comparison)

        # Create visualizations
        print("\n📈 Creating aggregated visualizations...")
        for mode in ['Zero-shot', 'Iterative']:
            for metric in ['success_rate', 'efficiency']:
                create_aggregated_visualization(euler_1d_agg, euler_1d_icl_agg,
                                              euler_1d_icl_no_cost_agg, euler_1d_icl_uniform_agg,
                                              metric, mode)

        # Create detailed visualizations (excluding success_rate_detailed_zero_shot.png and efficiency_detailed_zero_shot.png)
        print("\n📈 Creating detailed visualizations...")
        for metric in ['success_rate', 'efficiency']:
            # Only create iterative mode detailed visualizations
            create_detailed_visualization(detailed_comparison, metric, 'Iterative')

        # Create task category visualizations
        print("\n📈 Creating task category visualizations...")
        for mode in ['Zero-shot', 'Iterative']:
            for metric in ['success_rate', 'efficiency']:
                create_task_category_visualization(common_comparison, 'Common', metric, mode)
                create_task_category_visualization(uncommon_comparison, 'Uncommon', metric, mode)

        # Print summary statistics
        print_summary_statistics(agg_comparison)

        print("\n🎉 ICL Effect Analysis completed successfully!")
        print("Check eval_results/stats/icl_effect/ for all output files.")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()