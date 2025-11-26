#!/usr/bin/env python3
"""
LaTeX Table Generator for SimulCost-Bench Per-Dataset Results

This script generates LaTeX tables from the merged evaluation results,
creating separate tables for each dataset (simulator).

Usage:
    python evaluation/stats_utils/latex/simulator_wise.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_process_data(parquet_path: str):
    """Load and process the evaluation results."""
    df = pd.read_parquet(parquet_path)

    # Filter only the target datasets (exclude icl variants)
    target_datasets = ['burgers_1d', 'diff_react_1d', 'epoch_1d', 'euler_1d', 'heat_1d', 'heat_2d', 'mpm_2d', 'ns_transient_2d', 'euler_2d', 'hasegawa_mima_linear', 'hasegawa_mima_nonlinear', 'fem_2d']
    df_filtered = df[df['dataset'].isin(target_datasets)].copy()

    # Standardize precision level names for sorting
    precision_order = {'low': 0, 'medium': 1, 'high': 2}
    df_filtered['precision_order'] = df_filtered['precision_level'].map(precision_order)

    return df_filtered


def calculate_metrics_per_dataset(df, dataset_name):
    """
    Calculate success rate and efficiency metrics using two-level aggregation
    for a specific dataset.

    Level 1: Average within each task
    Level 2: Average across tasks for each (model, precision, mode)
    """
    # Filter for this dataset only
    df_dataset = df[df['dataset'] == dataset_name].copy()

    # Level 1: Calculate metrics per task
    task_level = df_dataset.groupby(['model_name', 'precision_level', 'precision_order', 'inference_mode', 'task']).agg({
        'is_successful': 'mean',  # Success rate per task (0-1)
        'efficiency': 'mean'      # Mean efficiency per task
    }).reset_index()

    task_level.columns = ['model_name', 'precision_level', 'precision_order', 'inference_mode', 'task', 'success_rate_task', 'mean_efficiency_task']

    # Level 2: Average across tasks for each (model, precision, mode)
    final_metrics = task_level.groupby(['model_name', 'precision_level', 'precision_order', 'inference_mode']).agg({
        'success_rate_task': 'mean',      # Simple average across tasks
        'mean_efficiency_task': 'mean'    # Simple average across tasks
    }).reset_index()

    # Round first (to match overall_stats.py methodology), then convert to percentage
    final_metrics['success_rate_rounded'] = final_metrics['success_rate_task'].round(2)
    final_metrics['success_rate'] = (final_metrics['success_rate_rounded'] * 100).round(1)
    final_metrics['mean_efficiency'] = final_metrics['mean_efficiency_task'].round(2)

    # Keep only needed columns
    final_metrics = final_metrics[['model_name', 'precision_level', 'precision_order', 'inference_mode', 'success_rate', 'mean_efficiency']]

    # Sort by model name and precision level
    final_metrics = final_metrics.sort_values(['model_name', 'precision_order'])

    return final_metrics


def format_dataset_name(dataset_name):
    """Format dataset name for display in table caption."""
    name_mapping = {
        'burgers_1d': 'Burgers 1D',
        'diff_react_1d': 'Diffusion-Reaction 1D',
        'epoch_1d': 'Epoch 1D',
        'euler_1d': 'Euler 1D',
        'heat_1d': 'Heat 1D',
        'heat_2d': 'Heat 2D',
        'mpm_2d': 'MPM 2D',
        'ns_transient_2d': 'NS Transient 2D',
        'euler_2d': 'Euler 2D',
        'hasegawa_mima_linear': 'Hasegawa-Mima Linear',
        'hasegawa_mima_nonlinear': 'Hasegawa-Mima Nonlinear',
        'fem_2d': 'FEM 2D'
    }
    return name_mapping.get(dataset_name, dataset_name)


def generate_zero_shot_table(df, dataset_name, model_order):
    """Generate Single-round table for a specific dataset."""
    df_zero_shot = df[df['inference_mode'] == 'zero_shot'].copy()
    formatted_name = format_dataset_name(dataset_name)

    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(f"    \\caption{{Single-round results on \\textbf{{{formatted_name}}}. Abbreviations: S - Success Rate (\\%), E - Efficiency. L/M/H - Low/Medium/High accuracy levels.}}")
    lines.append(r"    \begin{tabular}{lcccccccr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model/Acc level & \multicolumn{2}{c}{L} & \multicolumn{2}{c}{M} & \multicolumn{2}{c}{H} & \multicolumn{2}{c}{\textbf{Ave}} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"        Metrics & S & E & S & E & S & E & S & E \\ \midrule")

    for model in model_order:
        model_data = df_zero_shot[df_zero_shot['model_name'] == model]

        if len(model_data) == 0:
            continue

        # Get values for each precision level
        low = model_data[model_data['precision_level'] == 'low']
        medium = model_data[model_data['precision_level'] == 'medium']
        high = model_data[model_data['precision_level'] == 'high']

        s_low = low['success_rate'].values[0] if len(low) > 0 else 0
        e_low = low['mean_efficiency'].values[0] if len(low) > 0 else 0
        s_medium = medium['success_rate'].values[0] if len(medium) > 0 else 0
        e_medium = medium['mean_efficiency'].values[0] if len(medium) > 0 else 0
        s_high = high['success_rate'].values[0] if len(high) > 0 else 0
        e_high = high['mean_efficiency'].values[0] if len(high) > 0 else 0

        # Calculate averages
        s_avg = (s_low + s_medium + s_high) / 3
        e_avg = (e_low + e_medium + e_high) / 3

        lines.append(f"        {model} & {s_low:.1f} & {e_low:.2f} & {s_medium:.1f} & {e_medium:.2f} & {s_high:.1f} & {e_high:.2f} & {s_avg:.1f} & {e_avg:.2f} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


def generate_iterative_table(df, dataset_name, model_order):
    """Generate Multi-round table for a specific dataset."""
    df_iterative = df[df['inference_mode'] == 'iterative'].copy()
    formatted_name = format_dataset_name(dataset_name)

    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(f"    \\caption{{Multi-round results on \\textbf{{{formatted_name}}}. Abbreviations: S - Success Rate (\\%), E - Efficiency. L/M/H - Low/Medium/High accuracy levels.}}")
    lines.append(r"    \begin{tabular}{lcccccccr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model/Acc level & \multicolumn{2}{c}{L} & \multicolumn{2}{c}{M} & \multicolumn{2}{c}{H} & \multicolumn{2}{c}{\textbf{Ave}} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"        Metrics & S & E & S & E & S & E & S & E \\ \midrule")

    for model in model_order:
        model_data = df_iterative[df_iterative['model_name'] == model]

        if len(model_data) == 0:
            continue

        # Get values for each precision level
        low = model_data[model_data['precision_level'] == 'low']
        medium = model_data[model_data['precision_level'] == 'medium']
        high = model_data[model_data['precision_level'] == 'high']

        s_low = low['success_rate'].values[0] if len(low) > 0 else 0
        e_low = low['mean_efficiency'].values[0] if len(low) > 0 else 0
        s_medium = medium['success_rate'].values[0] if len(medium) > 0 else 0
        e_medium = medium['mean_efficiency'].values[0] if len(medium) > 0 else 0
        s_high = high['success_rate'].values[0] if len(high) > 0 else 0
        e_high = high['mean_efficiency'].values[0] if len(high) > 0 else 0

        # Calculate averages
        s_avg = (s_low + s_medium + s_high) / 3
        e_avg = (e_low + e_medium + e_high) / 3

        lines.append(f"        {model} & {s_low:.1f} & {e_low:.2f} & {s_medium:.1f} & {e_medium:.2f} & {s_high:.1f} & {e_high:.2f} & {s_avg:.1f} & {e_avg:.2f} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


def main():
    """Main function to generate LaTeX tables for each dataset."""
    # Load and process data
    parquet_path = "eval_results/merged_results.parquet"
    df = load_and_process_data(parquet_path)

    # Define datasets and model order
    datasets = sorted(['burgers_1d', 'diff_react_1d', 'epoch_1d', 'euler_1d', 'heat_1d', 'heat_2d', 'mpm_2d', 'ns_transient_2d', 'euler_2d', 'hasegawa_mima_linear', 'hasegawa_mima_nonlinear', 'fem_2d'])
    model_order = sorted(df['model_name'].unique())

    # Output directory
    output_dir = Path("eval_results/stats_utils/latex")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables for each dataset
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Generating tables for: {format_dataset_name(dataset)}")
        print('='*80)

        # Calculate metrics for this dataset
        metrics_df = calculate_metrics_per_dataset(df, dataset)

        # Generate tables
        zero_shot_table = generate_zero_shot_table(metrics_df, dataset, model_order)
        iterative_table = generate_iterative_table(metrics_df, dataset, model_order)

        # Combine tables
        combined_tables = f"{zero_shot_table}\n\n{iterative_table}"

        # Print to console
        print("\nLATEX TABLES - COPY THE CONTENT BELOW")
        print("-" * 80)
        print(combined_tables)
        print("-" * 80)

        # Save to file
        output_file = output_dir / f"{dataset}_tables.txt"
        with open(output_file, 'w') as f:
            f.write(combined_tables)

        print(f"\nSaved to: {output_file}")

    print(f"\n{'='*80}")
    print("All tables generated successfully!")
    print(f"Output directory: {output_dir}")
    print('='*80)


if __name__ == '__main__':
    main()
