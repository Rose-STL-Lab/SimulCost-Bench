#!/usr/bin/env python3
"""
Script to convert overall_summary.csv to LaTeX table format.
Generates four tables for different aspects of the results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_and_process_data(csv_path):
    """Load and process the CSV data."""
    df = pd.read_csv(csv_path)

    # Convert success rate from proportion to percentage
    df['Success Rate (%)'] = (df['Success Rate'] * 100).round(1)
    df['Efficiency'] = df['Efficiency'].round(2)

    return df

def calculate_averages(df):
    """Calculate averages across precision levels for each model and mode."""
    result = []

    for model in df['Model'].unique():
        for mode in df['Inference Mode'].unique():
            model_mode_data = df[(df['Model'] == model) & (df['Inference Mode'] == mode)]
            if len(model_mode_data) > 0:
                avg_success = model_mode_data['Success Rate (%)'].mean()
                avg_efficiency = model_mode_data['Efficiency'].mean()

                result.append({
                    'Model': model,
                    'Inference Mode': mode,
                    'Precision Level': 'average',
                    'Success Rate (%)': round(avg_success, 1),
                    'Efficiency': round(avg_efficiency, 2)
                })

    return pd.DataFrame(result)

def format_model_name(model_name):
    """Format model names for LaTeX display."""
    # Remove common suffixes and make more readable
    name = model_name.replace('-Instruct', '').replace('Claude-3.7-', 'Claude-3.7-')
    return name

def generate_zero_shot_table(df):
    """Generate zero-shot results table."""
    zero_shot_df = df[df['Inference Mode'] == 'Zero-shot'].copy()
    avg_df = calculate_averages(zero_shot_df)

    # Pivot data for table format
    success_pivot = zero_shot_df.pivot(index='Model', columns='Precision Level', values='Success Rate (%)')
    efficiency_pivot = zero_shot_df.pivot(index='Model', columns='Precision Level', values='Efficiency')

    # Add averages
    avg_pivot_success = avg_df.pivot(index='Model', columns='Precision Level', values='Success Rate (%)')
    avg_pivot_efficiency = avg_df.pivot(index='Model', columns='Precision Level', values='Efficiency')

    latex_content = """\\begin{table*}[]
    \\centering
    \\small
    \\caption{The overall results on the full dataset. Abbreviations: S - Success Rate (\\%), E - Efficiency. L/M/H - Low/Medium/High accuracy levels. \\textbf{Measurements reported for the zero-shot inference mode.}}
    \\label{tab:overall:0-shot}
    \\begin{tabular}{lcccccccr}
        \\toprule
        Model/Acc level & \\multicolumn{2}{c}{L} & \\multicolumn{2}{c}{M} & \\multicolumn{2}{c}{H} & \\multicolumn{2}{c}{\\textbf{Ave}}                             \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}
        Metrics         & S                     & E                     & S                     & E                                & S    & E    & S    & E    \\\\ \\midrule\n"""

    for model in success_pivot.index:
        formatted_name = format_model_name(model)
        low_s = success_pivot.loc[model, 'low'] if 'low' in success_pivot.columns else 0
        med_s = success_pivot.loc[model, 'medium'] if 'medium' in success_pivot.columns else 0
        high_s = success_pivot.loc[model, 'high'] if 'high' in success_pivot.columns else 0

        low_e = efficiency_pivot.loc[model, 'low'] if 'low' in efficiency_pivot.columns else 0
        med_e = efficiency_pivot.loc[model, 'medium'] if 'medium' in efficiency_pivot.columns else 0
        high_e = efficiency_pivot.loc[model, 'high'] if 'high' in efficiency_pivot.columns else 0

        avg_s = avg_pivot_success.loc[model, 'average'] if model in avg_pivot_success.index else 0
        avg_e = avg_pivot_efficiency.loc[model, 'average'] if model in avg_pivot_efficiency.index else 0

        latex_content += f"        {formatted_name} & {low_s:.1f} & {low_e:.2f} & {med_s:.1f} & {med_e:.2f} & {high_s:.1f} & {high_e:.2f} & {avg_s:.1f} & {avg_e:.2f} \\\\\n"

    latex_content += """        \\bottomrule
    \\end{tabular}
\\end{table*}"""

    return latex_content

def generate_iterative_table(df):
    """Generate iterative results table."""
    iterative_df = df[df['Inference Mode'] == 'Iterative'].copy()
    avg_df = calculate_averages(iterative_df)

    # Pivot data for table format
    success_pivot = iterative_df.pivot(index='Model', columns='Precision Level', values='Success Rate (%)')
    efficiency_pivot = iterative_df.pivot(index='Model', columns='Precision Level', values='Efficiency')

    # Add averages
    avg_pivot_success = avg_df.pivot(index='Model', columns='Precision Level', values='Success Rate (%)')
    avg_pivot_efficiency = avg_df.pivot(index='Model', columns='Precision Level', values='Efficiency')

    latex_content = """\\begin{table*}[]
    \\centering
    \\small
    \\caption{The overall results on the full dataset. Abbreviations: S - Success Rate (\\%), E - Efficiency. L/M/H - Low/Medium/High accuracy levels. \\textbf{Measurements reported are for iterative tunable parameters only.}}
    \\label{tab:overall:iterative}
    \\begin{tabular}{lcccccccr}
        \\toprule
        Model/Acc level & \\multicolumn{2}{c}{L} & \\multicolumn{2}{c}{M} & \\multicolumn{2}{c}{H} & \\multicolumn{2}{c}{\\textbf{Ave}}                             \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}
        Metrics         & S                     & E                     & S                     & E                                & S    & E    & S    & E    \\\\ \\midrule\n"""

    for model in success_pivot.index:
        formatted_name = format_model_name(model)
        low_s = success_pivot.loc[model, 'low'] if 'low' in success_pivot.columns else 0
        med_s = success_pivot.loc[model, 'medium'] if 'medium' in success_pivot.columns else 0
        high_s = success_pivot.loc[model, 'high'] if 'high' in success_pivot.columns else 0

        low_e = efficiency_pivot.loc[model, 'low'] if 'low' in efficiency_pivot.columns else 0
        med_e = efficiency_pivot.loc[model, 'medium'] if 'medium' in efficiency_pivot.columns else 0
        high_e = efficiency_pivot.loc[model, 'high'] if 'high' in efficiency_pivot.columns else 0

        avg_s = avg_pivot_success.loc[model, 'average'] if model in avg_pivot_success.index else 0
        avg_e = avg_pivot_efficiency.loc[model, 'average'] if model in avg_pivot_efficiency.index else 0

        latex_content += f"        {formatted_name} & {low_s:.1f} & {low_e:.2f} & {med_s:.1f} & {med_e:.2f} & {high_s:.1f} & {high_e:.2f} & {avg_s:.1f} & {avg_e:.2f} \\\\\n"

    latex_content += """        \\bottomrule
    \\end{tabular}
\\end{table*}"""

    return latex_content

def generate_mode_comparison_success(df):
    """Generate mode comparison table for success rates."""
    # Pivot to get both modes side by side
    success_pivot = df.pivot_table(
        index=['Model'],
        columns=['Precision Level', 'Inference Mode'],
        values='Success Rate (%)',
        aggfunc='first'
    )

    # Calculate averages
    avg_df = calculate_averages(df)
    avg_pivot = avg_df.pivot_table(
        index=['Model'],
        columns=['Inference Mode'],
        values='Success Rate (%)',
        aggfunc='first'
    )

    latex_content = """\\begin{table*}[]
    \\centering
    \\small
    \\caption{Comparison between zero-shot and iterative inference modes: Success Rate. Abbreviations: L/M/H - Low/Medium/High accuracy levels, 0/i - zero-shot and iterative modes. \\textbf{Measurements reported are for iterative tunable parameters only.}}
    \\label{tab:mode_compare:success}
    \\begin{tabular}{lcccccccr}
        \\toprule
        Model / Acc level & \\multicolumn{2}{c}{L} & \\multicolumn{2}{c}{M} & \\multicolumn{2}{c}{H} & \\multicolumn{2}{c}{\\textbf{Ave}}                             \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}
        Inference Modes   & 0                     & i                     & 0                     & i                                & 0    & i    & 0    & i    \\\\ \\midrule\n"""

    for model in success_pivot.index:
        formatted_name = format_model_name(model)

        low_0 = success_pivot.loc[model, ('low', 'Zero-shot')] if ('low', 'Zero-shot') in success_pivot.columns else 0
        low_i = success_pivot.loc[model, ('low', 'Iterative')] if ('low', 'Iterative') in success_pivot.columns else 0
        med_0 = success_pivot.loc[model, ('medium', 'Zero-shot')] if ('medium', 'Zero-shot') in success_pivot.columns else 0
        med_i = success_pivot.loc[model, ('medium', 'Iterative')] if ('medium', 'Iterative') in success_pivot.columns else 0
        high_0 = success_pivot.loc[model, ('high', 'Zero-shot')] if ('high', 'Zero-shot') in success_pivot.columns else 0
        high_i = success_pivot.loc[model, ('high', 'Iterative')] if ('high', 'Iterative') in success_pivot.columns else 0

        avg_0 = avg_pivot.loc[model, 'Zero-shot'] if model in avg_pivot.index and 'Zero-shot' in avg_pivot.columns else 0
        avg_i = avg_pivot.loc[model, 'Iterative'] if model in avg_pivot.index and 'Iterative' in avg_pivot.columns else 0

        # Handle NaN values
        low_0 = low_0 if not pd.isna(low_0) else 0
        low_i = low_i if not pd.isna(low_i) else 0
        med_0 = med_0 if not pd.isna(med_0) else 0
        med_i = med_i if not pd.isna(med_i) else 0
        high_0 = high_0 if not pd.isna(high_0) else 0
        high_i = high_i if not pd.isna(high_i) else 0
        avg_0 = avg_0 if not pd.isna(avg_0) else 0
        avg_i = avg_i if not pd.isna(avg_i) else 0

        latex_content += f"        {formatted_name} & {low_0:.1f} & {low_i:.1f} & {med_0:.1f} & {med_i:.1f} & {high_0:.1f} & {high_i:.1f} & {avg_0:.1f} & {avg_i:.1f} \\\\\n"

    latex_content += """        \\bottomrule
    \\end{tabular}
\\end{table*}"""

    return latex_content

def generate_mode_comparison_efficiency(df):
    """Generate mode comparison table for efficiency."""
    # Pivot to get both modes side by side
    efficiency_pivot = df.pivot_table(
        index=['Model'],
        columns=['Precision Level', 'Inference Mode'],
        values='Efficiency',
        aggfunc='first'
    )

    # Calculate averages
    avg_df = calculate_averages(df)
    avg_pivot = avg_df.pivot_table(
        index=['Model'],
        columns=['Inference Mode'],
        values='Efficiency',
        aggfunc='first'
    )

    latex_content = """\\begin{table*}[]
    \\centering
    \\small
    \\caption{Comparison between zero-shot and iterative inference modes: Efficiency. Abbreviations: L/M/H - Low/Medium/High accuracy levels, 0/i - zero-shot and iterative modes. \\textbf{Measurements reported for iterative tunable parameters only.}}
    \\label{tab:mode_compare:efficiency}
    \\begin{tabular}{lcccccccr}
        \\toprule
        Model / Acc level & \\multicolumn{2}{c}{L} & \\multicolumn{2}{c}{M} & \\multicolumn{2}{c}{H} & \\multicolumn{2}{c}{\\textbf{Ave}}                             \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}
        Inference Modes   & 0                     & i                     & 0                     & i                                & 0    & i    & 0    & i    \\\\ \\midrule\n"""

    for model in efficiency_pivot.index:
        formatted_name = format_model_name(model)

        low_0 = efficiency_pivot.loc[model, ('low', 'Zero-shot')] if ('low', 'Zero-shot') in efficiency_pivot.columns else 0
        low_i = efficiency_pivot.loc[model, ('low', 'Iterative')] if ('low', 'Iterative') in efficiency_pivot.columns else 0
        med_0 = efficiency_pivot.loc[model, ('medium', 'Zero-shot')] if ('medium', 'Zero-shot') in efficiency_pivot.columns else 0
        med_i = efficiency_pivot.loc[model, ('medium', 'Iterative')] if ('medium', 'Iterative') in efficiency_pivot.columns else 0
        high_0 = efficiency_pivot.loc[model, ('high', 'Zero-shot')] if ('high', 'Zero-shot') in efficiency_pivot.columns else 0
        high_i = efficiency_pivot.loc[model, ('high', 'Iterative')] if ('high', 'Iterative') in efficiency_pivot.columns else 0

        avg_0 = avg_pivot.loc[model, 'Zero-shot'] if model in avg_pivot.index and 'Zero-shot' in avg_pivot.columns else 0
        avg_i = avg_pivot.loc[model, 'Iterative'] if model in avg_pivot.index and 'Iterative' in avg_pivot.columns else 0

        # Handle NaN values
        low_0 = low_0 if not pd.isna(low_0) else 0
        low_i = low_i if not pd.isna(low_i) else 0
        med_0 = med_0 if not pd.isna(med_0) else 0
        med_i = med_i if not pd.isna(med_i) else 0
        high_0 = high_0 if not pd.isna(high_0) else 0
        high_i = high_i if not pd.isna(high_i) else 0
        avg_0 = avg_0 if not pd.isna(avg_0) else 0
        avg_i = avg_i if not pd.isna(avg_i) else 0

        latex_content += f"        {formatted_name} & {low_0:.2f} & {low_i:.2f} & {med_0:.2f} & {med_i:.2f} & {high_0:.2f} & {high_i:.2f} & {avg_0:.2f} & {avg_i:.2f} \\\\\n"

    latex_content += """        \\bottomrule
    \\end{tabular}
\\end{table*}"""

    return latex_content


def save_to_file(content, filename):
    """Save content to a file in the latex output directory."""
    # Create the output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent.parent / "eval_results" / "stats" / "latex"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to file
    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Saved to: {output_path}")

def load_detailed_task_stats(csv_path):
    """Load and process the detailed task stats CSV data."""
    df = pd.read_csv(csv_path)

    # Convert success rate from proportion to percentage and round efficiency
    df['Success Rate (%)'] = (df['success_rate_mean'] * 100).round(1)
    df['Efficiency'] = df['mean_efficiency_mean'].round(2)

    return df

def generate_commonality_comparison_table(df):
    """Generate commonality comparison table with colored backgrounds."""

    # Separate data by method type and task category
    zero_shot_df = df[df['MethodType'] == 'zero_shot'].copy()
    iterative_df = df[df['MethodType'] == 'iterative'].copy()

    # Get all unique tasks
    all_tasks = sorted(df['Task'].unique())

    # Separate common and uncommon tasks
    common_tasks = sorted(df[df['TaskCategory'] == 'Common']['Task'].unique())
    uncommon_tasks = sorted(df[df['TaskCategory'] == 'Uncommon']['Task'].unique())

    latex_content = """\\begin{table*}[h]
    \\centering
    \\small
    \\caption{Comparison based on the commonality of the tunable parameters. Abbreviations: S - Success Rate (\\%), E - Efficiency.
    }
    \\label{tab:commonb_compare:zeroshot}
    \\begin{tabular}{lcccccc}
        \\toprule
        Task & \\multicolumn{2}{c}{Zero-shot} & \\multicolumn{2}{c}{Iterative} & \\multicolumn{2}{c}{\\textbf{Ave}}                             \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}
        Metrics           & S                     & E                     & S                     & E                                & S    & E    \\\\ \\midrule\n"""

    # Add common tasks with light green background
    if common_tasks:
        latex_content += "        \\rowcolor{green!10}\n"
        latex_content += "        \\multicolumn{7}{c}{\\textbf{Common Parameters}} \\\\\n"

        for task in common_tasks:
            # Get data for this task from both modes
            zero_shot_task = zero_shot_df[zero_shot_df['Task'] == task]
            iterative_task = iterative_df[iterative_df['Task'] == task]

            # Zero-shot values (use -- if not found)
            if len(zero_shot_task) > 0:
                zero_s = f"{zero_shot_task['Success Rate (%)'].iloc[0]:.1f}"
                zero_e = f"{zero_shot_task['Efficiency'].iloc[0]:.2f}"
            else:
                zero_s = "--"
                zero_e = "--"

            # Iterative values (use -- if not found)
            if len(iterative_task) > 0:
                iter_s = f"{iterative_task['Success Rate (%)'].iloc[0]:.1f}"
                iter_e = f"{iterative_task['Efficiency'].iloc[0]:.2f}"
            else:
                iter_s = "--"
                iter_e = "--"

            # Calculate averages (only if both values exist)
            if zero_s != "--" and iter_s != "--":
                avg_s = f"{(float(zero_s) + float(iter_s)) / 2:.1f}"
            elif zero_s != "--":
                avg_s = zero_s
            elif iter_s != "--":
                avg_s = iter_s
            else:
                avg_s = "--"

            if zero_e != "--" and iter_e != "--":
                avg_e = f"{(float(zero_e) + float(iter_e)) / 2:.2f}"
            elif zero_e != "--":
                avg_e = zero_e
            elif iter_e != "--":
                avg_e = iter_e
            else:
                avg_e = "--"

            latex_content += f"        \\rowcolor{{green!10}}\n"
            latex_content += f"        {task} & {zero_s} & {zero_e} & {iter_s} & {iter_e} & {avg_s} & {avg_e} \\\\\n"

    # Add uncommon tasks with light blue background
    if uncommon_tasks:
        latex_content += "        \\rowcolor{blue!10}\n"
        latex_content += "        \\multicolumn{7}{c}{\\textbf{Uncommon Parameters}} \\\\\n"

        for task in uncommon_tasks:
            # Get data for this task from both modes
            zero_shot_task = zero_shot_df[zero_shot_df['Task'] == task]
            iterative_task = iterative_df[iterative_df['Task'] == task]

            # Zero-shot values (use -- if not found)
            if len(zero_shot_task) > 0:
                zero_s = f"{zero_shot_task['Success Rate (%)'].iloc[0]:.1f}"
                zero_e = f"{zero_shot_task['Efficiency'].iloc[0]:.2f}"
            else:
                zero_s = "--"
                zero_e = "--"

            # Iterative values (use -- if not found)
            if len(iterative_task) > 0:
                iter_s = f"{iterative_task['Success Rate (%)'].iloc[0]:.1f}"
                iter_e = f"{iterative_task['Efficiency'].iloc[0]:.2f}"
            else:
                iter_s = "--"
                iter_e = "--"

            # Calculate averages (only if both values exist)
            if zero_s != "--" and iter_s != "--":
                avg_s = f"{(float(zero_s) + float(iter_s)) / 2:.1f}"
            elif zero_s != "--":
                avg_s = zero_s
            elif iter_s != "--":
                avg_s = iter_s
            else:
                avg_s = "--"

            if zero_e != "--" and iter_e != "--":
                avg_e = f"{(float(zero_e) + float(iter_e)) / 2:.2f}"
            elif zero_e != "--":
                avg_e = zero_e
            elif iter_e != "--":
                avg_e = iter_e
            else:
                avg_e = "--"

            latex_content += f"        \\rowcolor{{blue!10}}\n"
            latex_content += f"        {task} & {zero_s} & {zero_e} & {iter_s} & {iter_e} & {avg_s} & {avg_e} \\\\\n"

    latex_content += """        \\bottomrule
    \\end{tabular}
\\end{table*}"""

    return latex_content

def main():
    """Main function to generate all LaTeX tables."""

    # Load overall summary data
    overall_csv_path = Path(__file__).parent.parent.parent / "eval_results" / "overall" / "overall_summary.csv"

    # Check if overall summary exists and process it
    if overall_csv_path.exists():
        overall_df = load_and_process_data(overall_csv_path)

        # Generate all tables
        all_content = ""

        # Table 1: Zero-shot Results
        table1 = generate_zero_shot_table(overall_df)
        print("=" * 80)
        print("TABLE 1: Zero-shot Results")
        print("=" * 80)
        print(table1)
        print("\n\n")
        all_content += "TABLE 1: Zero-shot Results\n" + "=" * 80 + "\n" + table1 + "\n\n\n"

        # Table 2: Iterative Results
        table2 = generate_iterative_table(overall_df)
        print("=" * 80)
        print("TABLE 2: Iterative Results")
        print("=" * 80)
        print(table2)
        print("\n\n")
        all_content += "TABLE 2: Iterative Results\n" + "=" * 80 + "\n" + table2 + "\n\n\n"

        # Table 3: Mode Comparison - Success Rate
        table3 = generate_mode_comparison_success(overall_df)
        print("=" * 80)
        print("TABLE 3: Mode Comparison - Success Rate")
        print("=" * 80)
        print(table3)
        print("\n\n")
        all_content += "TABLE 3: Mode Comparison - Success Rate\n" + "=" * 80 + "\n" + table3 + "\n\n\n"

        # Table 4: Mode Comparison - Efficiency
        table4 = generate_mode_comparison_efficiency(overall_df)
        print("=" * 80)
        print("TABLE 4: Mode Comparison - Efficiency")
        print("=" * 80)
        print(table4)
        print("\n\n")
        all_content += "TABLE 4: Mode Comparison - Efficiency\n" + "=" * 80 + "\n" + table4 + "\n\n\n"

        # Save all content to file
        save_to_file(all_content, "all_latex_tables.txt")

        # Save individual tables
        save_to_file(table1, "table1_overall_zeroshot.txt")
        save_to_file(table2, "table2_overall_iterative.txt")
        save_to_file(table3, "table3_overall_mode_comparison_success.txt")
        save_to_file(table4, "table4_overall_mode_comparison_efficiency.txt")

    # Load detailed task stats data
    detailed_csv_path = Path(__file__).parent.parent.parent / "eval_results" / "stats" / "task_difficulty" / "detailed_task_stats.csv"

    # Check if detailed task stats file exists and process it
    if detailed_csv_path.exists():
        detailed_df = load_detailed_task_stats(detailed_csv_path)

        # Table 5: Commonality Comparison
        table5 = generate_commonality_comparison_table(detailed_df)
        print("=" * 80)
        print("TABLE 5: Commonality Comparison")
        print("=" * 80)
        print(table5)
        print("\n\n")

        # Save commonality comparison table
        save_to_file(table5, "table5_detailed_commonality_comparison.txt")
    else:
        print("Detailed task stats CSV file not found. Skipping commonality comparison table generation.")

if __name__ == "__main__":
    main()