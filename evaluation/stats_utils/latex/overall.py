#!/usr/bin/env python3
"""
LaTeX Table Generator for SimulCost-Bench Overall Results

This script generates LaTeX tables from the merged evaluation results.
It creates 4 tables comparing model performance across precision levels and inference modes.

Usage:
    python evaluation/stats_utils/latex/overall.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_process_data(parquet_path: str):
    """Load and process the evaluation results."""
    df = pd.read_parquet(parquet_path)

    # Filter only the target datasets
    target_datasets = ['burgers_1d', 'euler_1d', 'heat_1d', 'heat_2d', 'ns_transient_2d', 'mpm_2d']
    df_filtered = df[df['dataset'].isin(target_datasets)].copy()

    # Standardize model names
    model_mapping = {
        'Claude-3.7-Sonnet': 'Claude-3.7-Sonnet',
        'GPT-5': 'GPT-5',
        'Llama-3-70B-Instruct': 'Llama-3-70B-Instruct',
        'Qwen3-32B': 'Qwen3-32B'
    }
    df_filtered['model_name'] = df_filtered['model_name'].map(model_mapping)

    # Standardize inference mode names
    df_filtered['inference_mode'] = df_filtered['inference_mode'].replace({
        'zero_shot': '0-shot',
        'iterative': 'multi-round'
    })

    # Standardize precision level names for sorting
    precision_order = {'low': 0, 'medium': 1, 'high': 2}
    df_filtered['precision_order'] = df_filtered['precision_level'].map(precision_order)

    return df_filtered


def calculate_metrics(df):
    """
    Calculate success rate and efficiency metrics using three-level aggregation.
    This matches the methodology in evaluation/overall_stats.py.

    Level 1: Average within each task
    Level 2: Average across tasks for each (dataset, model, precision, mode)
    Level 3: Average across datasets
    """
    # Level 1: Calculate metrics per task
    task_level = df.groupby(['dataset', 'model_name', 'precision_level', 'precision_order', 'inference_mode', 'task']).agg({
        'is_successful': 'mean',  # Success rate per task (0-1)
        'efficiency': 'mean'      # Mean efficiency per task
    }).reset_index()

    task_level.columns = ['dataset', 'model_name', 'precision_level', 'precision_order', 'inference_mode', 'task', 'success_rate_task', 'mean_efficiency_task']

    # Level 2: Average across tasks for each dataset
    dataset_level = task_level.groupby(['dataset', 'model_name', 'precision_level', 'precision_order', 'inference_mode']).agg({
        'success_rate_task': 'mean',      # Simple average across tasks
        'mean_efficiency_task': 'mean'    # Simple average across tasks
    }).reset_index()

    dataset_level.columns = ['dataset', 'model_name', 'precision_level', 'precision_order', 'inference_mode', 'success_rate_dataset', 'mean_efficiency_dataset']

    # Level 3: Average across datasets
    final_metrics = dataset_level.groupby(['model_name', 'precision_level', 'precision_order', 'inference_mode']).agg({
        'success_rate_dataset': 'mean',      # Simple average across datasets
        'mean_efficiency_dataset': 'mean'    # Simple average across datasets
    }).reset_index()

    # Round first (to match overall_stats.py methodology), then convert to percentage
    final_metrics['success_rate_rounded'] = final_metrics['success_rate_dataset'].round(2)
    final_metrics['success_rate'] = (final_metrics['success_rate_rounded'] * 100).round(1)
    final_metrics['mean_efficiency'] = final_metrics['mean_efficiency_dataset'].round(2)

    # Keep only needed columns
    final_metrics = final_metrics[['model_name', 'precision_level', 'precision_order', 'inference_mode', 'success_rate', 'mean_efficiency']]

    # Sort by model name and precision level
    final_metrics = final_metrics.sort_values(['model_name', 'precision_order'])

    return final_metrics


def generate_table1(df, model_order):
    """Generate Table 1: Single-round inference results."""
    df_zero_shot = df[df['inference_mode'] == '0-shot'].copy()

    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \caption{The overall results on the full dataset. Abbreviations: S - Success Rate (\%), E - Efficiency. L/M/H - Low/Medium/High accuracy levels. \textbf{Measurements reported for the single-round inference mode.}}")
    lines.append(r"    \label{tab:overall:0-shot}")
    lines.append(r"    \begin{tabular}{lcccccccr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model/Acc level & \multicolumn{2}{c}{L} & \multicolumn{2}{c}{M} & \multicolumn{2}{c}{H} & \multicolumn{2}{c}{\textbf{Ave}}                             \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"        Metrics         & S                     & E                     & S                     & E                                & S    & E    & S    & E    \\ \midrule")

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


def generate_table2(df, model_order):
    """Generate Table 2: Multi-round inference results."""
    df_multi = df[df['inference_mode'] == 'multi-round'].copy()

    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \caption{The overall results on the full dataset. Abbreviations: S - Success Rate (\%), E - Efficiency. L/M/H - Low/Medium/High accuracy levels. \textbf{Measurements reported are for multi-round tunable parameters only.}}")
    lines.append(r"    \label{tab:overall:multi-round}")
    lines.append(r"    \begin{tabular}{lcccccccr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model/Acc level & \multicolumn{2}{c}{L} & \multicolumn{2}{c}{M} & \multicolumn{2}{c}{H} & \multicolumn{2}{c}{\textbf{Ave}}                             \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"        Metrics         & S                     & E                     & S                     & E                                & S    & E    & S    & E    \\ \midrule")

    for model in model_order:
        model_data = df_multi[df_multi['model_name'] == model]

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


def generate_table3(df, model_order):
    """Generate Table 3: Comparison of success rates between inference modes."""
    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \caption{Comparison between single-round and multi-round inference modes: \textbf{Success Rate}. Abbreviations: L/M/H - Low/Medium/High accuracy levels, 0/i - single-round and multi-round modes.}")
    lines.append(r"    \label{tab:mode_compare:success}")
    lines.append(r"    \begin{tabular}{lcccccccr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model / Acc level & \multicolumn{2}{c}{L} & \multicolumn{2}{c}{M} & \multicolumn{2}{c}{H} & \multicolumn{2}{c}{\textbf{Ave}}                             \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"        Inference Modes   & 0                     & i                     & 0                     & i                                & 0    & i    & 0    & i    \\ \midrule")

    for model in model_order:
        model_data = df[df['model_name'] == model]

        if len(model_data) == 0:
            continue

        # Get 0-shot and multi-round data
        zero_shot = model_data[model_data['inference_mode'] == '0-shot']
        multi_round = model_data[model_data['inference_mode'] == 'multi-round']

        # Low precision
        s0_low = zero_shot[zero_shot['precision_level'] == 'low']['success_rate'].values[0] if len(zero_shot[zero_shot['precision_level'] == 'low']) > 0 else 0
        si_low = multi_round[multi_round['precision_level'] == 'low']['success_rate'].values[0] if len(multi_round[multi_round['precision_level'] == 'low']) > 0 else 0

        # Medium precision
        s0_medium = zero_shot[zero_shot['precision_level'] == 'medium']['success_rate'].values[0] if len(zero_shot[zero_shot['precision_level'] == 'medium']) > 0 else 0
        si_medium = multi_round[multi_round['precision_level'] == 'medium']['success_rate'].values[0] if len(multi_round[multi_round['precision_level'] == 'medium']) > 0 else 0

        # High precision
        s0_high = zero_shot[zero_shot['precision_level'] == 'high']['success_rate'].values[0] if len(zero_shot[zero_shot['precision_level'] == 'high']) > 0 else 0
        si_high = multi_round[multi_round['precision_level'] == 'high']['success_rate'].values[0] if len(multi_round[multi_round['precision_level'] == 'high']) > 0 else 0

        # Averages
        s0_avg = (s0_low + s0_medium + s0_high) / 3
        si_avg = (si_low + si_medium + si_high) / 3

        lines.append(f"        {model} & {s0_low:.1f} & {si_low:.1f} & {s0_medium:.1f} & {si_medium:.1f} & {s0_high:.1f} & {si_high:.1f} & {s0_avg:.1f} & {si_avg:.1f} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


def generate_table4(df, model_order):
    """Generate Table 4: Comparison of efficiency between inference modes."""
    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \caption{Comparison between single-round and multi-round inference modes: \textbf{Efficiency}. Abbreviations: L/M/H - Low/Medium/High accuracy levels, 0/i - single-round and multi-round modes.}")
    lines.append(r"    \label{tab:mode_compare:efficiency}")
    lines.append(r"    \begin{tabular}{lcccccccr}")
    lines.append(r"        \toprule")
    lines.append(r"        Model / Acc level & \multicolumn{2}{c}{L} & \multicolumn{2}{c}{M} & \multicolumn{2}{c}{H} & \multicolumn{2}{c}{\textbf{Ave}}                             \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    lines.append(r"        Inference Modes   & 0                     & i                     & 0                     & i                                & 0    & i    & 0    & i    \\ \midrule")

    for model in model_order:
        model_data = df[df['model_name'] == model]

        if len(model_data) == 0:
            continue

        # Get 0-shot and multi-round data
        zero_shot = model_data[model_data['inference_mode'] == '0-shot']
        multi_round = model_data[model_data['inference_mode'] == 'multi-round']

        # Low precision
        e0_low = zero_shot[zero_shot['precision_level'] == 'low']['mean_efficiency'].values[0] if len(zero_shot[zero_shot['precision_level'] == 'low']) > 0 else 0
        ei_low = multi_round[multi_round['precision_level'] == 'low']['mean_efficiency'].values[0] if len(multi_round[multi_round['precision_level'] == 'low']) > 0 else 0

        # Medium precision
        e0_medium = zero_shot[zero_shot['precision_level'] == 'medium']['mean_efficiency'].values[0] if len(zero_shot[zero_shot['precision_level'] == 'medium']) > 0 else 0
        ei_medium = multi_round[multi_round['precision_level'] == 'medium']['mean_efficiency'].values[0] if len(multi_round[multi_round['precision_level'] == 'medium']) > 0 else 0

        # High precision
        e0_high = zero_shot[zero_shot['precision_level'] == 'high']['mean_efficiency'].values[0] if len(zero_shot[zero_shot['precision_level'] == 'high']) > 0 else 0
        ei_high = multi_round[multi_round['precision_level'] == 'high']['mean_efficiency'].values[0] if len(multi_round[multi_round['precision_level'] == 'high']) > 0 else 0

        # Averages
        e0_avg = (e0_low + e0_medium + e0_high) / 3
        ei_avg = (ei_low + ei_medium + ei_high) / 3

        lines.append(f"        {model} & {e0_low:.2f} & {ei_low:.2f} & {e0_medium:.2f} & {ei_medium:.2f} & {e0_high:.2f} & {ei_high:.2f} & {e0_avg:.2f} & {ei_avg:.2f} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table*}")

    return '\n'.join(lines)


def main():
    """Main function to generate LaTeX tables."""
    # Load and process data
    parquet_path = "eval_results/merged_results.parquet"
    df = load_and_process_data(parquet_path)

    # Calculate metrics
    metrics_df = calculate_metrics(df)

    # Define model order (alphabetical)
    model_order = ['Claude-3.7-Sonnet', 'GPT-5', 'Llama-3-70B-Instruct', 'Qwen3-32B']

    # Generate all tables
    table1 = generate_table1(metrics_df, model_order)
    table2 = generate_table2(metrics_df, model_order)
    table3 = generate_table3(metrics_df, model_order)
    table4 = generate_table4(metrics_df, model_order)

    # Combine all tables
    all_tables = f"{table1}\n\n{table2}\n\n{table3}\n\n{table4}"

    # Print to console (for copy-paste)
    print("=" * 80)
    print("LATEX TABLES - COPY THE CONTENT BELOW")
    print("=" * 80)
    print()
    print(all_tables)
    print()
    print("=" * 80)

    # Save to file
    output_path = Path("eval_results/stats_utils/latex/overall_tables.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(all_tables)

    print(f"\nTables also saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
