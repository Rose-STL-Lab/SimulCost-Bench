#!/usr/bin/env python3
"""
Task Statistics Script for SimulCost-Bench

This script counts all tasks in the benchmark by parsing JSON question files
and generates a detailed statistical report.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def find_question_files(data_dir: str = "data") -> List[Path]:
    """
    Find all question JSON files, excluding icl and human_write directories.

    Args:
        data_dir: Root data directory

    Returns:
        List of Path objects for question files
    """
    question_files = []
    data_path = Path(data_dir)

    for json_file in data_path.rglob("*_questions.json"):
        # Skip icl directory
        if "icl" in json_file.parts:
            continue
        # Skip human_write directories
        if "human_write" in json_file.parts:
            continue
        question_files.append(json_file)

    return sorted(question_files)


def parse_file_path(file_path: Path) -> Tuple[str, str, str, str]:
    """
    Extract metadata from file path.

    Expected format: data/{dataset}/{parameter}/{precision_level}/{mode}_questions.json

    Args:
        file_path: Path to question file

    Returns:
        Tuple of (dataset, parameter, precision_level, mode)
    """
    parts = file_path.parts
    dataset = parts[1]  # data/{dataset}/...
    parameter = parts[2]  # data/{dataset}/{parameter}/...
    precision_level = parts[3]  # data/{dataset}/{parameter}/{precision_level}/...

    # Extract mode from filename (iterative_questions.json or zero_shot_questions.json)
    filename = file_path.stem  # Remove .json extension
    mode = filename.replace("_questions", "")  # Get "iterative" or "zero_shot"

    return dataset, parameter, precision_level, mode


def count_tasks_in_file(file_path: Path) -> int:
    """
    Count the number of tasks in a JSON file.

    Args:
        file_path: Path to question file

    Returns:
        Number of tasks in the file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return len(data)


def collect_statistics(question_files: List[Path]) -> Dict:
    """
    Collect task statistics from all question files.

    Args:
        question_files: List of question file paths

    Returns:
        Nested dictionary with statistics
    """
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

    for file_path in question_files:
        dataset, parameter, precision_level, mode = parse_file_path(file_path)
        task_count = count_tasks_in_file(file_path)

        stats[dataset][parameter][precision_level][mode] = task_count

    return dict(stats)


def generate_report(stats: Dict, output_file: str = "eval_results/stats_utils/task_statistics_report.md"):
    """
    Generate a detailed markdown report.

    Args:
        stats: Statistics dictionary
        output_file: Path to output markdown file
    """
    lines = []

    # Header
    lines.append("# SimulCost-Bench Task Statistics Report\n")
    lines.append("This report provides detailed statistics on the number of tasks in the benchmark.\n")
    lines.append("---\n")

    # Fine-grained statistics
    lines.append("## Fine-Grained Task Statistics\n")
    lines.append("Detailed breakdown by dataset, parameter, precision level, and mode.\n")

    total_tasks_global = 0
    total_zero_shot_global = 0
    total_iterative_global = 0

    dataset_summaries = []

    for dataset in sorted(stats.keys()):
        lines.append(f"\n### Dataset: `{dataset}`\n")

        total_tasks_dataset = 0
        total_zero_shot_dataset = 0
        total_iterative_dataset = 0

        for parameter in sorted(stats[dataset].keys()):
            lines.append(f"\n#### Parameter: `{parameter}`\n")
            lines.append("| Precision Level | Mode | Task Count |")
            lines.append("|----------------|------|------------|")

            for precision_level in sorted(stats[dataset][parameter].keys()):
                for mode in sorted(stats[dataset][parameter][precision_level].keys()):
                    count = stats[dataset][parameter][precision_level][mode]
                    lines.append(f"| {precision_level} | {mode} | {count} |")

                    total_tasks_dataset += count
                    total_tasks_global += count

                    if mode == "zero_shot":
                        total_zero_shot_dataset += count
                        total_zero_shot_global += count
                    elif mode == "iterative":
                        total_iterative_dataset += count
                        total_iterative_global += count

        dataset_summaries.append({
            "dataset": dataset,
            "total": total_tasks_dataset,
            "zero_shot": total_zero_shot_dataset,
            "iterative": total_iterative_dataset
        })

    # Dataset-level summary
    lines.append("\n---\n")
    lines.append("## Summary Statistics by Dataset\n")
    lines.append("| Dataset | Total Tasks | Zero-Shot Tasks | Iterative Tasks |")
    lines.append("|---------|-------------|-----------------|-----------------|")

    for summary in dataset_summaries:
        lines.append(f"| `{summary['dataset']}` | {summary['total']} | {summary['zero_shot']} | {summary['iterative']} |")

    # Global summary
    lines.append("\n---\n")
    lines.append("## Global Summary Statistics\n")
    lines.append(f"- **Total Number of Datasets**: {len(stats)}")
    lines.append(f"- **Total Number of Tasks**: {total_tasks_global}")
    lines.append(f"- **Total Zero-Shot Tasks**: {total_zero_shot_global}")
    lines.append(f"- **Total Iterative Tasks**: {total_iterative_global}")

    # Count unique parameters
    all_parameters = set()
    for dataset_data in stats.values():
        all_parameters.update(dataset_data.keys())
    lines.append(f"- **Total Number of Parameters**: {len(all_parameters)}")

    # Precision level distribution
    precision_counts = defaultdict(int)
    for dataset_data in stats.values():
        for param_data in dataset_data.values():
            for precision_level in param_data.keys():
                precision_counts[precision_level] += 1

    lines.append("\n### Distribution by Precision Level")
    for precision_level in sorted(precision_counts.keys()):
        lines.append(f"- **{precision_level}**: {precision_counts[precision_level]} configurations")

    lines.append("\n---\n")
    lines.append(f"*Report generated automatically by count_tasks.py*\n")

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report generated successfully: {output_file}")
    print(f"\nQuick Summary:")
    print(f"  Total Datasets: {len(stats)}")
    print(f"  Total Tasks: {total_tasks_global}")
    print(f"  - Zero-Shot: {total_zero_shot_global}")
    print(f"  - Iterative: {total_iterative_global}")


def main():
    """Main execution function."""
    print("SimulCost-Bench Task Statistics Generator")
    print("=" * 50)

    # Find all question files
    print("\nFinding question files...")
    question_files = find_question_files()
    print(f"Found {len(question_files)} question files")

    # Collect statistics
    print("\nCollecting statistics...")
    stats = collect_statistics(question_files)

    # Generate report
    print("\nGenerating report...")
    generate_report(stats)

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
