#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_zero_shot_data():
    """Load all zero-shot CSV summary files and filter for zero-shot inference mode only"""

    datasets = ['heat_1d', 'heat_2d', 'euler_1d', 'burgers_1d', 'epoch_1d', 'ns_transient_2d']
    accuracy_levels = ['low', 'medium', 'high']

    all_data = []

    for dataset in datasets:
        for accuracy in accuracy_levels:
            csv_path = f"eval_results/{dataset}/{dataset}_{accuracy}_summary.csv"

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                # Filter for zero-shot inference mode only
                if 'Inference Mode' in df.columns:
                    df = df[df['Inference Mode'] == 'Zero-shot'].copy()

                # Skip if no zero-shot data found
                if len(df) == 0:
                    print(f"Warning: No zero-shot data found in {csv_path}")
                    continue

                df['dataset'] = dataset
                df['accuracy_level'] = accuracy
                all_data.append(df)
            else:
                print(f"Warning: File not found: {csv_path}")

    if not all_data:
        raise ValueError("No CSV files found!")

    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"Loaded {len(combined_df)} zero-shot entries from {len(all_data)} files")

    return combined_df

def standardize_model_names(df):
    """Standardize model names to be consistent across datasets"""

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

        # Mistral variants
        'Mistral-Large': 'Mistral-Large',
        'mistral.mistral-large-2402-v1:0': 'Mistral-Large',

        # Nova variants
        'Nova-Premier': 'Nova-Premier',
        'amazon.nova-premier-v1:0': 'Nova-Premier',

        # Qwen variants
        'Qwen3-32B': 'Qwen3-32B',
        'qwen3_32b': 'Qwen3-32B',

        # Add more mappings as needed
    }

    df['Model_Standardized'] = df['Model'].map(model_mapping).fillna(df['Model'])

    print(f"Model standardization mapping:")
    for original, standardized in model_mapping.items():
        if original in df['Model'].values:
            print(f"  {original} -> {standardized}")

    # Check if we have any unmapped models
    unmapped = df[df['Model_Standardized'] == df['Model']]['Model'].unique()
    if len(unmapped) > 0:
        print(f"Warning: Unmapped models found: {list(unmapped)}")

    return df

def prepare_task_performance_matrix(df, metric='mean_efficiency'):
    """Prepare task performance matrix with model-accuracy combinations as rows and tasks as columns"""

    # Use standardized model names
    df = standardize_model_names(df)

    # Create a combined identifier for model-accuracy level using standardized names
    df['Model_Accuracy'] = df['Model_Standardized'] + '_' + df['accuracy_level']

    # For same tasks appearing in different datasets, we want to average their performance
    # Group by Model_Accuracy, Task, and dataset, then take mean across datasets
    print(f"Preparing performance matrix from {len(df)} entries...")

    # First aggregate within each dataset (in case there are multiple entries for same model-task-dataset combination)
    df_dataset_agg = df.groupby(['Model_Accuracy', 'Task', 'dataset'])[metric].mean().reset_index()

    # Then aggregate across datasets for the same model-task combination
    df_final_agg = df_dataset_agg.groupby(['Model_Accuracy', 'Task'])[metric].mean().reset_index()

    print(f"After aggregation: {len(df_final_agg)} unique Model_Accuracy-Task combinations")

    # Create a pivot table with model-accuracy as index, tasks as columns, and metric as values
    perf_matrix = df_final_agg.pivot_table(index='Model_Accuracy', columns='Task', values=metric, aggfunc='mean')

    # Show some statistics about data availability
    print(f"Performance matrix shape: {perf_matrix.shape}")
    print(f"Tasks with data: {perf_matrix.columns.tolist()}")

    # Show how many model-accuracy combinations have data for each task
    task_coverage = perf_matrix.notna().sum()
    print(f"Data coverage per task:")
    for task, count in task_coverage.items():
        print(f"  {task}: {count}/{len(perf_matrix)} model-accuracy combinations")

    return perf_matrix

def compute_correlation_matrix(perf_matrix):
    """Compute Pearson correlation coefficients between all tasks"""

    tasks = perf_matrix.columns.tolist()
    n_tasks = len(tasks)

    correlation_matrix = np.eye(n_tasks)
    p_values = np.eye(n_tasks)

    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            if i != j:
                # Get performance values for both tasks across all models
                task1_values = perf_matrix[task1].dropna()
                task2_values = perf_matrix[task2].dropna()

                # Find common model-accuracy combinations that have data for both tasks
                common_entries = task1_values.index.intersection(task2_values.index)

                if len(common_entries) >= 2:  # Need at least 2 data points
                    perf1 = task1_values[common_entries]
                    perf2 = task2_values[common_entries]

                    corr, p_val = pearsonr(perf1, perf2)
                    correlation_matrix[i, j] = corr
                    p_values[i, j] = p_val
                else:
                    correlation_matrix[i, j] = np.nan
                    p_values[i, j] = np.nan
            else:
                correlation_matrix[i, j] = 1.0
                p_values[i, j] = 0.0

    return correlation_matrix, p_values, tasks

def create_correlation_table(correlation_matrix, tasks):
    """Create formatted correlation table with common tasks on left/bottom, uncommon on right/top"""

    common_tasks = ['cfl', 'dx', 'n_space', 'resolution', 'mesh_x', 'mesh_y', 'dt_multiplier', 'nx', 'npart']

    tasks_list = list(tasks)

    common_present = [task for task in common_tasks if task in tasks_list]
    uncommon_present = [task for task in tasks_list if task not in common_tasks]

    # For the desired layout:
    # - Columns: common tasks (left) + uncommon tasks (right)
    # - Rows: reversed order - uncommon tasks (top) + common tasks (bottom) so cfl->t_init goes bottom to top
    ordered_tasks_cols = common_present + uncommon_present  # columns: common left, uncommon right
    ordered_tasks_rows = list(reversed(common_present + uncommon_present))  # rows: reversed order for bottom-to-top cfl->t_init

    task_to_idx = {task: i for i, task in enumerate(tasks_list)}
    col_indices = [task_to_idx[task] for task in ordered_tasks_cols]
    row_indices = [task_to_idx[task] for task in ordered_tasks_rows]

    reordered_matrix = correlation_matrix[np.ix_(row_indices, col_indices)]

    df_corr = pd.DataFrame(reordered_matrix,
                          index=ordered_tasks_rows,
                          columns=ordered_tasks_cols)

    return df_corr, common_present, uncommon_present

def analyze_data_availability(perf_matrix):
    """Analyze which tasks have data for which model-accuracy combinations"""

    print("\nData Availability Analysis:")
    print("-" * 50)

    for task in perf_matrix.columns:
        available_models = perf_matrix[task].dropna().index.tolist()
        print(f"{task}: {len(available_models)} model-accuracy combinations")
        if len(available_models) <= 5:  # Show details for tasks with limited data
            print(f"  Available in: {', '.join(available_models)}")

    # Find task pairs with no common data
    tasks = perf_matrix.columns.tolist()
    missing_pairs = []

    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            if i < j:  # Only check upper triangle
                task1_models = set(perf_matrix[task1].dropna().index)
                task2_models = set(perf_matrix[task2].dropna().index)
                common_models = task1_models.intersection(task2_models)

                if len(common_models) < 2:  # Less than 2 common data points
                    missing_pairs.append((task1, task2, len(common_models)))

    if missing_pairs:
        print(f"\nTask pairs with insufficient data (< 2 common models):")
        for task1, task2, count in missing_pairs:
            print(f"  {task1} ↔ {task2}: {count} common models")

    return missing_pairs

def generate_correlation_heatmap(df_corr, common_tasks, uncommon_tasks, output_dir):
    """Generate correlation matrix heatmap with common tasks on left/bottom, uncommon on right/top"""

    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a custom colormap for better visualization
    # Use a colormap that clearly shows NaN values
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors

    # Create a diverging colormap with gray for NaN
    colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#ffffff',
              '#fdbf6f', '#ff7f00', '#d94701', '#8b0000']
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
    cmap.set_bad(color='lightgray', alpha=0.8)  # Set NaN color to light gray

    # Create a mask for NaN values to add special annotation
    mask_nan = df_corr.isna()

    # Generate the heatmap
    im = ax.imshow(df_corr.values, cmap=cmap, aspect='equal',
                   vmin=-1, vmax=1, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson Correlation Coefficient', rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(range(len(df_corr.columns)))
    ax.set_yticks(range(len(df_corr.index)))
    ax.set_xticklabels(df_corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(df_corr.index, rotation=0)

    # Add correlation values as text annotations
    for i in range(len(df_corr.index)):
        for j in range(len(df_corr.columns)):
            value = df_corr.iloc[i, j]
            if pd.isna(value):
                text = 'N/A'
                color = 'black'
                fontsize = 8
            else:
                text = f'{value:.1f}'
                # Choose text color based on background
                color = 'white' if abs(value) > 0.5 else 'black'
                fontsize = 9

            ax.text(j, i, text, ha='center', va='center',
                   color=color, fontsize=fontsize, fontweight='bold')

    # Add grid lines
    ax.set_xticks(np.arange(len(df_corr.columns)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(df_corr.index)) + 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1, alpha=0.7)

    # Customize the plot - no title

    # Add dividing lines to separate common and uncommon tasks
    n_common_cols = len(common_tasks)  # common tasks on left side of columns
    n_uncommon_rows = len(uncommon_tasks)  # uncommon tasks on top side of rows (due to reversed order)

    if n_common_cols > 0 and len(uncommon_tasks) > 0:
        # Vertical line to separate common (left) from uncommon (right) columns
        ax.axvline(x=n_common_cols-0.5, color='black', linewidth=3, alpha=0.8)
        # Horizontal line to separate uncommon (top) from common (bottom) rows (reversed order)
        ax.axhline(y=n_uncommon_rows-0.5, color='black', linewidth=3, alpha=0.8)

    # No legend needed

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the heatmap
    heatmap_path = os.path.join(output_dir, "task_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.savefig(heatmap_path.replace('.png', '.pdf'), bbox_inches='tight')

    print(f"Heatmap saved to: {heatmap_path}")
    print(f"PDF version saved to: {heatmap_path.replace('.png', '.pdf')}")

    plt.close()  # Close the figure to free memory

    return heatmap_path

def generate_md_report(df_corr, common_tasks, uncommon_tasks, output_dir, missing_pairs=None):
    """Generate markdown report with representative task correlations"""

    # Find high correlations (> 0.7) and negative correlations (< -0.3)
    high_correlations = []
    negative_correlations = []

    for i, task1 in enumerate(df_corr.index):
        for j, task2 in enumerate(df_corr.columns):
            if i < j:  # Only upper triangle to avoid duplicates
                corr_val = df_corr.iloc[i, j]
                if not np.isnan(corr_val):
                    if corr_val > 0.7:
                        high_correlations.append((task1, task2, corr_val))
                    elif corr_val < -0.3:
                        negative_correlations.append((task1, task2, corr_val))

    # Sort by correlation strength
    high_correlations.sort(key=lambda x: x[2], reverse=True)
    negative_correlations.sort(key=lambda x: x[2])

    # Generate the markdown report
    report_content = f"""# Task Correlation Analysis Report

## Overview

This report analyzes the correlations between different simulation tasks based on model performance efficiency scores.

## Task Categories

### Common Tasks ({len(common_tasks)} tasks)
These are fundamental simulation parameters that appear across multiple domains:
{', '.join(f'`{task}`' for task in common_tasks)}

### Uncommon Tasks ({len(uncommon_tasks)} tasks)
These are specialized parameters specific to certain simulation types:
{', '.join(f'`{task}`' for task in uncommon_tasks)}

## Data Availability

### Missing Data Explanation
The correlation matrix contains gray cells (N/A values) where insufficient data exists to calculate reliable correlations. This occurs when fewer than 2 model-accuracy combinations have data for both tasks being compared."""

    # Add missing pairs information if available
    if missing_pairs:
        report_content += f"\n\n### Task Pairs with Insufficient Data ({len(missing_pairs)} pairs)\n"
        for task1, task2, count in missing_pairs:
            report_content += f"- **{task1}** ↔ **{task2}**: {count} common model(s)\n"

    report_content += f"""

## Key Findings

### Strong Positive Correlations (r > 0.7)
"""

    if high_correlations:
        for task1, task2, corr in high_correlations[:10]:  # Top 10
            report_content += f"- **{task1}** ↔ **{task2}**: r = {corr:.3f}\n"
    else:
        report_content += "No strong positive correlations found (threshold: r > 0.7)\n"

    report_content += f"""
### Notable Negative Correlations (r < -0.3)
"""

    if negative_correlations:
        for task1, task2, corr in negative_correlations[:10]:  # Top 10
            report_content += f"- **{task1}** ↔ **{task2}**: r = {corr:.3f}\n"
    else:
        report_content += "No notable negative correlations found (threshold: r < -0.3)\n"

    # Calculate summary statistics
    upper_triangle = np.triu(df_corr.values, k=1)
    valid_correlations = upper_triangle[upper_triangle != 0]
    valid_correlations = valid_correlations[~np.isnan(valid_correlations)]

    if len(valid_correlations) > 0:
        mean_corr = np.mean(valid_correlations)
        max_corr = np.max(valid_correlations)
        min_corr = np.min(valid_correlations)
        std_corr = np.std(valid_correlations)

        report_content += f"""
## Statistical Summary

- **Total task pairs analyzed**: {len(valid_correlations)}
- **Mean correlation**: {mean_corr:.3f}
- **Standard deviation**: {std_corr:.3f}
- **Maximum correlation**: {max_corr:.3f}
- **Minimum correlation**: {min_corr:.3f}

## Interpretation

### Task Clustering Patterns
"""

        # Analyze common vs uncommon task correlations
        common_indices = [i for i, task in enumerate(df_corr.index) if task in common_tasks]
        uncommon_indices = [i for i, task in enumerate(df_corr.index) if task not in common_tasks]

        if len(common_indices) > 1:
            # Common-common correlations
            common_corr_values = []
            for i in range(len(common_indices)):
                for j in range(i+1, len(common_indices)):
                    val = df_corr.iloc[common_indices[i], common_indices[j]]
                    if not np.isnan(val):
                        common_corr_values.append(val)

            if common_corr_values:
                report_content += f"- **Common-Common task correlations**: Mean = {np.mean(common_corr_values):.3f}, N = {len(common_corr_values)}\n"

        if len(uncommon_indices) > 1:
            # Uncommon-uncommon correlations
            uncommon_corr_values = []
            for i in range(len(uncommon_indices)):
                for j in range(i+1, len(uncommon_indices)):
                    val = df_corr.iloc[uncommon_indices[i], uncommon_indices[j]]
                    if not np.isnan(val):
                        uncommon_corr_values.append(val)

            if uncommon_corr_values:
                report_content += f"- **Uncommon-Uncommon task correlations**: Mean = {np.mean(uncommon_corr_values):.3f}, N = {len(uncommon_corr_values)}\n"

        if len(common_indices) > 0 and len(uncommon_indices) > 0:
            # Common-uncommon correlations
            cross_corr_values = []
            for i in common_indices:
                for j in uncommon_indices:
                    val = df_corr.iloc[i, j]
                    if not np.isnan(val):
                        cross_corr_values.append(val)

            if cross_corr_values:
                report_content += f"- **Common-Uncommon task correlations**: Mean = {np.mean(cross_corr_values):.3f}, N = {len(cross_corr_values)}\n"

    report_content += f"""
## Methodology

- **Correlation method**: Pearson correlation coefficient
- **Performance metric**: Mean efficiency scores across model-accuracy combinations
- **Data source**: Zero-shot evaluation results across multiple simulation domains
- **Minimum data points**: At least 2 common model-accuracy combinations required for correlation calculation

## Files Generated

- `task_correlation_matrix.csv`: Full correlation matrix in CSV format
- `task_correlation_heatmap.png`: Visual heatmap representation
- `task_correlation_heatmap.pdf`: PDF version of the heatmap
- `task_correlation_report.md`: This analysis report

---
*Report generated automatically from simulation evaluation data*
"""

    # Save the report
    report_path = os.path.join(output_dir, "task_correlation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"MD report saved to: {report_path}")

    return report_path

def main():
    print("Loading zero-shot evaluation data...")
    df = load_zero_shot_data()

    print(f"Loaded data for {len(df)} entries across {df['Task'].nunique()} tasks")
    print(f"Tasks found: {sorted(df['Task'].unique())}")

    print("\nPreparing task performance matrix...")
    perf_matrix = prepare_task_performance_matrix(df, metric='mean_efficiency')

    print(f"Performance matrix shape: {perf_matrix.shape}")
    print(f"Model-Accuracy combinations: {len(perf_matrix.index)}")
    print(f"Tasks: {list(perf_matrix.columns)}")
    print(f"Sample entries: {list(perf_matrix.index)[:5]}")

    # Analyze data availability first
    missing_pairs = analyze_data_availability(perf_matrix)

    print("\nComputing correlation matrix...")
    correlation_matrix, p_values, tasks = compute_correlation_matrix(perf_matrix)

    print("\nCreating formatted correlation table...")
    df_corr, common_tasks, uncommon_tasks = create_correlation_table(correlation_matrix, tasks)

    print("\n" + "="*80)
    print("TASK CORRELATION MATRIX (Pearson Correlation Coefficients)")
    print("="*80)
    print(f"\nCommon tasks ({len(common_tasks)}): {', '.join(common_tasks)}")
    print(f"Uncommon tasks ({len(uncommon_tasks)}): {', '.join(uncommon_tasks)}")
    print("\nCorrelation Matrix:")
    print("-" * 80)

    print(df_corr.round(3).to_string())

    # Create output directory if it doesn't exist
    output_dir = "eval_results/stats/task_correlation"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "task_correlation_matrix.csv")
    df_corr.to_csv(output_file)
    print(f"\nCorrelation matrix saved to: {output_file}")

    print(f"\nSummary Statistics:")
    print(f"- Total tasks: {len(df_corr)}")
    print(f"- Common tasks: {len(common_tasks)}")
    print(f"- Uncommon tasks: {len(uncommon_tasks)}")

    upper_triangle = np.triu(df_corr.values, k=1)
    valid_correlations = upper_triangle[upper_triangle != 0]
    valid_correlations = valid_correlations[~np.isnan(valid_correlations)]

    if len(valid_correlations) > 0:
        print(f"- Mean correlation (excluding diagonal): {np.mean(valid_correlations):.3f}")
        print(f"- Max correlation: {np.max(valid_correlations):.3f}")
        print(f"- Min correlation: {np.min(valid_correlations):.3f}")

    # Generate heatmap visualization
    print(f"\nGenerating correlation heatmap...")
    heatmap_path = generate_correlation_heatmap(df_corr, common_tasks, uncommon_tasks, output_dir)

    # Generate markdown report
    print(f"\nGenerating analysis report...")
    report_path = generate_md_report(df_corr, common_tasks, uncommon_tasks, output_dir, missing_pairs)

    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Files generated:")
    print(f"- CSV matrix: {output_file}")
    print(f"- Heatmap PNG: {heatmap_path}")
    print(f"- Heatmap PDF: {heatmap_path.replace('.png', '.pdf')}")
    print(f"- Analysis report: {report_path}")
    print("="*80)

if __name__ == "__main__":
    main()