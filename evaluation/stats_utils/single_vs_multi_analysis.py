#!/usr/bin/env python3
"""
Analyze ABNORMAL cases where single-round performance exceeds multi-round performance.
This should NOT happen - multi-round should always perform better than single-round.
These cases indicate performance degradation that needs investigation.

Analyzes Success Rate (S) metric only across different datasets,
models, and precision levels.
"""

import re
import os
import glob
import csv
from typing import Dict, List, Tuple


def parse_latex_table(table_text: str) -> Tuple[str, str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Parse a LaTeX table to extract dataset, round type, and results.

    Returns:
        (dataset_name, round_type, results_dict)
        where results_dict = {model: {precision: {metric: value}}}
    """
    # Extract dataset name and round type from caption
    caption_match = re.search(r'\\caption\{(Single-round|Multi-round) results on \\textbf\{([^}]+)\}', table_text)
    if not caption_match:
        return None, None, {}

    round_type = caption_match.group(1)
    dataset_name = caption_match.group(2)

    # Extract data rows (between \midrule and \bottomrule)
    lines = table_text.split('\n')
    in_data_section = False
    results = {}

    for line in lines:
        if '\\midrule' in line:
            in_data_section = True
            continue
        if '\\bottomrule' in line:
            break
        if not in_data_section or not line.strip():
            continue

        # Skip header lines that contain "&" but are part of header
        if '\\multicolumn' in line or 'Model' in line or 'Metrics' in line:
            continue

        # Parse data row: ModelName & S_L & E_L & S_M & E_M & S_H & E_H & S_Ave & E_Ave \\
        if '&' in line and '\\\\' in line:
            parts = [p.strip() for p in line.split('&')]
            if len(parts) >= 7:  # At least model + 6 values (3 precision levels x 2 metrics)
                model_name = parts[0].strip()
                try:
                    # Extract values (handling potential missing data with '-' or empty)
                    values = []
                    for i in range(1, min(9, len(parts))):  # Get up to 8 values
                        val_str = parts[i].replace('\\\\', '').strip()
                        if val_str and val_str != '-':
                            values.append(float(val_str))
                        else:
                            values.append(None)

                    if len(values) >= 6:  # Need at least L, M, H metrics
                        results[model_name] = {
                            'L': {'S': values[0], 'E': values[1]},
                            'M': {'S': values[2], 'E': values[3]},
                            'H': {'S': values[4], 'E': values[5]},
                        }
                        if len(values) >= 8:
                            results[model_name]['Ave'] = {'S': values[6], 'E': values[7]}
                except (ValueError, IndexError) as e:
                    # Skip rows that don't parse correctly
                    continue

    return dataset_name, round_type, results


def parse_dataset_file(filepath: str) -> Tuple[Dict, Dict]:
    """
    Parse a dataset file containing both single-round and multi-round tables.

    Returns:
        (single_round_results, multi_round_results)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into tables using \begin{table*}
    tables = re.split(r'\\begin\{table\*\}', content)

    single_round_data = {}
    multi_round_data = {}

    for table in tables[1:]:  # Skip first split (before first table)
        dataset, round_type, results = parse_latex_table(table)
        if dataset and results:
            if round_type == 'Single-round':
                single_round_data = {'dataset': dataset, 'results': results}
            elif round_type == 'Multi-round':
                multi_round_data = {'dataset': dataset, 'results': results}

    return single_round_data, multi_round_data


def compare_performances(single_data: Dict, multi_data: Dict) -> List[Dict]:
    """
    Compare single-round vs multi-round performance.

    Returns list of ABNORMAL cases where single-round > multi-round.
    These represent performance degradations in the multi-round system.
    """
    comparisons = []

    if not single_data or not multi_data:
        return comparisons

    dataset = single_data.get('dataset', 'Unknown')
    single_results = single_data.get('results', {})
    multi_results = multi_data.get('results', {})

    # Compare each model
    for model in single_results:
        if model not in multi_results:
            continue

        # Compare each precision level
        for precision in ['L', 'M', 'H']:
            if precision not in single_results[model] or precision not in multi_results[model]:
                continue

            # Compare only Success Rate (S), not Efficiency (E)
            for metric in ['S']:
                single_val = single_results[model][precision].get(metric)
                multi_val = multi_results[model][precision].get(metric)

                if single_val is not None and multi_val is not None:
                    # Check if single-round is better (ABNORMAL - should not happen!)
                    if single_val > multi_val:
                        precision_name = {'L': 'Low', 'M': 'Medium', 'H': 'High'}[precision]
                        metric_name = 'Success Rate'  # Only analyzing Success Rate

                        difference = single_val - multi_val

                        comparisons.append({
                            'dataset': dataset,
                            'model': model,
                            'precision': precision_name,
                            'metric': metric_name,
                            'single_value': single_val,
                            'multi_value': multi_val,
                            'difference': difference
                        })

    return comparisons


def generate_human_readable_report(all_comparisons: List[Dict], output_path: str):
    """Generate a comprehensive human-readable report of the degradations."""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("MULTI-ROUND PERFORMANCE DEGRADATION ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write("This report identifies ABNORMAL cases where single-round performance exceeds\n")
        f.write("multi-round performance. This should NOT happen under normal circumstances, as\n")
        f.write("multi-round systems have multiple attempts to correct errors and should always\n")
        f.write("achieve equal or better performance than single-round systems.\n\n")

        f.write(f"Total abnormal cases found: {len(all_comparisons)}\n\n")

        # Detailed findings sorted by degradation magnitude
        f.write("=" * 100 + "\n")
        f.write("DETAILED FINDINGS (Sorted by Degradation - Worst First)\n")
        f.write("=" * 100 + "\n\n")

        # Sort by absolute difference (largest degradation first)
        sorted_comps = sorted(all_comparisons, key=lambda x: -x['difference'])

        for i, comp in enumerate(sorted_comps, 1):
            f.write(f"Case #{i}\n")
            f.write(f"  Dataset:        {comp['dataset']}\n")
            f.write(f"  Model:          {comp['model']}\n")
            f.write(f"  Precision:      {comp['precision']}\n")
            f.write(f"  Metric:         {comp['metric']}\n")
            f.write(f"  Single-round:   {comp['single_value']:.2f}\n")
            f.write(f"  Multi-round:    {comp['multi_value']:.2f}\n")
            f.write(f"  Degradation:    {comp['difference']:.2f}\n")
            f.write("\n")

        # Statistics section
        f.write("\n" + "=" * 100 + "\n")
        f.write("STATISTICAL BREAKDOWN\n")
        f.write("=" * 100 + "\n\n")

        # Note: Only analyzing Success Rate, not Efficiency
        f.write("Metric Analyzed:\n")
        f.write(f"  Success Rate (S) only - {len(all_comparisons)} degradation cases\n")
        f.write(f"  (Efficiency metric excluded from this analysis)\n\n")

        # By dataset
        dataset_counts = {}
        for comp in all_comparisons:
            dataset_counts[comp['dataset']] = dataset_counts.get(comp['dataset'], 0) + 1

        f.write("By Dataset (Most affected first):\n")
        for dataset, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {dataset:30s} {count} cases\n")
        f.write("\n")

        # By model
        model_counts = {}
        for comp in all_comparisons:
            model_counts[comp['model']] = model_counts.get(comp['model'], 0) + 1

        f.write("By Model (Most affected first):\n")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {model:30s} {count} cases\n")
        f.write("\n")

        # By precision
        precision_counts = {}
        for comp in all_comparisons:
            precision_counts[comp['precision']] = precision_counts.get(comp['precision'], 0) + 1

        f.write("By Precision Level:\n")
        for precision in ['Low', 'Medium', 'High']:
            count = precision_counts.get(precision, 0)
            if count > 0:
                f.write(f"  {precision:10s} {count} cases\n")
        f.write("\n")

        # Recommendations
        f.write("=" * 100 + "\n")
        f.write("RECOMMENDATIONS FOR INVESTIGATION\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. Priority:\n")
        f.write("   - Investigate cases with largest absolute degradation first\n")
        f.write("   - Check for bugs in multi-round feedback loops\n")
        f.write("   - Review error accumulation in iterative processes\n\n")

        f.write("2. Model-Specific Issues:\n")
        if 'Claude-3.7-Sonnet' in model_counts:
            f.write(f"   - Claude-3.7-Sonnet shows {model_counts['Claude-3.7-Sonnet']} degradation cases\n")
        if 'Llama-3-70B-Instruct' in model_counts:
            f.write(f"   - Llama-3-70B-Instruct shows {model_counts['Llama-3-70B-Instruct']} degradation cases\n")
        f.write("   - Review model-specific multi-round prompting strategies\n\n")

        f.write("3. Dataset-Specific Patterns:\n")
        top_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for dataset, count in top_datasets:
            f.write(f"   - {dataset}: {count} cases - requires detailed analysis\n")
        f.write("\n")

        f.write("4. Precision Level Analysis:\n")
        if precision_counts.get('Low', 0) > precision_counts.get('High', 0):
            f.write("   - Low precision shows more degradation - may indicate parameter tuning issues\n")
        f.write("   - Compare multi-round strategies across precision levels\n\n")

        f.write("5. General Recommendations:\n")
        f.write("   - Review multi-round termination conditions\n")
        f.write("   - Check for overfitting to training examples in multi-round\n")
        f.write("   - Validate multi-round feedback quality\n")
        f.write("   - Consider if multi-round is introducing unnecessary complexity\n")
        f.write("   - Investigate cases with largest absolute degradation first\n\n")


def main():
    # Find all dataset table files
    input_dir = 'eval_results/stats_utils/latex'
    output_dir = 'eval_results/stats_utils'
    pattern = os.path.join(input_dir, '*_tables.txt')
    files = glob.glob(pattern)

    # Exclude overall_tables.txt
    files = [f for f in files if 'overall_tables.txt' not in f]

    print(f"Found {len(files)} dataset files to analyze\n")
    print("=" * 100)
    print("⚠️  ABNORMAL CASES: SINGLE-ROUND PERFORMANCE EXCEEDS MULTI-ROUND PERFORMANCE")
    print("=" * 100)
    print("\n⚠️  WARNING: These are ABNORMAL cases indicating multi-round degradation!")
    print("Multi-round should ALWAYS perform >= single-round.\n")

    all_comparisons = []

    for filepath in sorted(files):
        single_data, multi_data = parse_dataset_file(filepath)
        comparisons = compare_performances(single_data, multi_data)
        all_comparisons.extend(comparisons)

    if not all_comparisons:
        print("\n✓ Good news! No abnormal cases found.")
        print("Multi-round performance meets or exceeds single-round in all cases.")
        return

    # Sort by absolute difference (largest degradation first)
    all_comparisons.sort(key=lambda x: -x['difference'])

    # Print results
    print(f"\n⚠️  Total abnormal cases found: {len(all_comparisons)}\n")
    print("(Sorted by degradation magnitude - worst first)\n")

    for i, comp in enumerate(all_comparisons, 1):
        print(f"\nCase #{i}")
        print(f"  Dataset: {comp['dataset']}")
        print(f"  Model: {comp['model']}")
        print(f"  Precision Level: {comp['precision']}")
        print(f"  Metric: {comp['metric']}")
        print(f"  Single-round value: {comp['single_value']:.2f}")
        print(f"  Multi-round value:  {comp['multi_value']:.2f}")
        print(f"  Multi-round DROP: -{comp['difference']:.2f}")

    # Save to CSV
    output_csv = os.path.join(output_dir, 'single_vs_multi_comparison.csv')
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['dataset', 'model', 'precision', 'metric', 'single_value',
                     'multi_value', 'difference']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_comparisons)

    # Generate human-readable report
    output_txt = os.path.join(output_dir, 'single_vs_multi_degradation_report.txt')
    generate_human_readable_report(all_comparisons, output_txt)

    print(f"\n\n{'=' * 100}")
    print(f"REPORTS GENERATED")
    print('=' * 100)
    print(f"CSV report:  {output_csv}")
    print(f"Text report: {output_txt}")
    print('=' * 100)

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    # Note: Only analyzing Success Rate
    print(f"\nMetric Analyzed:")
    print(f"  Success Rate (S) only: {len(all_comparisons)} degradation cases")
    print(f"  (Efficiency metric excluded from this analysis)")

    # Count by dataset
    dataset_counts = {}
    for comp in all_comparisons:
        dataset_counts[comp['dataset']] = dataset_counts.get(comp['dataset'], 0) + 1

    print("\nBy Dataset (Most affected first):")
    for dataset, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dataset}: {count} cases")

    # Count by model
    model_counts = {}
    for comp in all_comparisons:
        model_counts[comp['model']] = model_counts.get(comp['model'], 0) + 1

    print("\nBy Model (Most affected first):")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model}: {count} cases")

    # Count by precision
    precision_counts = {}
    for comp in all_comparisons:
        precision_counts[comp['precision']] = precision_counts.get(comp['precision'], 0) + 1

    print("\nBy Precision Level:")
    for precision in ['Low', 'Medium', 'High']:
        count = precision_counts.get(precision, 0)
        if count > 0:
            print(f"  {precision}: {count} cases")


if __name__ == '__main__':
    main()
