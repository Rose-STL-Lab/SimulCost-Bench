"""
Merge evaluation results from multiple datasets into a single parquet file.

This script automatically detects and merges parquet files from:
- eval_results/epoch_1d/dataframes/
- eval_results/euler_1d/dataframes/
- eval_results/euler_2d/dataframes/
- eval_results/ns_transient_2d/dataframes/
- eval_results/burgers_1d/dataframes/
- eval_results/heat_1d/dataframes/
- eval_results/heat_2d/dataframes/
- eval_results/fem_2d/dataframes/
- eval_results/mpm_2d/dataframes/
- eval_results/diff_react_1d/dataframes/
- eval_results/hasegawa_mima_linear/dataframes/
- eval_results/hasegawa_mima_nonlinear/dataframes/
- eval_results/euler_1d_icl_accuracy_focused/dataframes/
- eval_results/euler_1d_icl_cost_excluded/dataframes/
- eval_results/euler_1d_icl_full/dataframes/
- eval_results/heat_1d_icl_accuracy_focused/dataframes/
- eval_results/heat_1d_icl_cost_excluded/dataframes/
- eval_results/heat_1d_icl_full/dataframes/
- eval_results/ns_transient_2d_icl_accuracy_focused/dataframes/
- eval_results/ns_transient_2d_icl_cost_excluded/dataframes/
- eval_results/ns_transient_2d_icl_full/dataframes/
- eval_results/mpm_2d_icl_accuracy_focused/dataframes/
- eval_results/mpm_2d_icl_cost_excluded/dataframes/
- eval_results/mpm_2d_icl_full/dataframes/

The merged result is saved to: eval_results/merged_results.parquet

Model names are automatically mapped to standardized names for consistency.
"""

import os
from pathlib import Path
from typing import List, Dict
import pandas as pd


# Model name mapping dictionary
MODEL_NAME_MAPPING = {
    # Support both underscore and colon formats
    'amazon.nova-premier-v1:0': 'Nova-Premier',
    'amazon.nova-premier-v1_0': 'Nova-Premier',
    'anthropic.claude-3-7-sonnet-20250219-v1:0': 'Claude-3.7-Sonnet',
    'anthropic.claude-3-7-sonnet-20250219-v1_0': 'Claude-3.7-Sonnet',
    'mistral.mistral-large-2402-v1:0': 'Mistral-Large',
    'mistral.mistral-large-2402-v1_0': 'Mistral-Large',
    'meta.llama3-70b-instruct-v1:0': 'Llama-3-70B-Instruct',
    'meta.llama3-70b-instruct-v1_0': 'Llama-3-70B-Instruct',
    'gpt-5-2025-08-07': 'GPT-5',
    'gpt-5-2025-08-07-re-minimal': 'GPT-5-RE-Minimal',
    'gpt-5-2025-08-07-re-high': 'GPT-5-RE-High',
    'qwen3_32b': 'Qwen3-32B',
    'qwen3_0_6b': 'Qwen3-0.6B',
    'qwen3_8b': 'Qwen3-8B',
    'anthropic.claude-3-5-haiku-20241022-v1:0': 'Claude-3.5-Haiku',
    'anthropic.claude-3-5-haiku-20241022-v1_0': 'Claude-3.5-Haiku',
    'anthropic.claude-3-5-sonnet-20240620-v1:0': 'Claude-3.5-Sonnet',
    'anthropic.claude-3-5-sonnet-20240620-v1_0': 'Claude-3.5-Sonnet',
    'openai.gpt-oss-120b-1:0': 'GPT-OSS-120B',
}


def validate_model_names(df: pd.DataFrame) -> None:
    """
    Validate that all model names in the dataframe are in the mapping dictionary.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If any model name is not in the mapping dictionary
    """
    if 'model_name' not in df.columns:
        return

    unique_models = df['model_name'].unique()
    unmapped_models = [model for model in unique_models if model not in MODEL_NAME_MAPPING]

    if unmapped_models:
        raise ValueError(
            f"Found unmapped model names: {unmapped_models}\n"
            f"Please add these models to MODEL_NAME_MAPPING in merge_results.py"
        )


def apply_model_name_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply model name mapping to standardize model names.

    Args:
        df: DataFrame with model_name column

    Returns:
        DataFrame with mapped model names
    """
    if 'model_name' not in df.columns:
        return df

    # Validate all model names are mapped
    validate_model_names(df)

    # Apply mapping
    df['model_name'] = df['model_name'].map(MODEL_NAME_MAPPING)

    return df


def find_parquet_files(base_dir: str, datasets: List[str]) -> Dict[str, List[Path]]:
    """
    Find all parquet files for specified datasets.

    Args:
        base_dir: Base directory for evaluation results
        datasets: List of dataset names

    Returns:
        Dictionary mapping dataset names to list of parquet file paths
    """
    files_by_dataset = {}

    for dataset in datasets:
        dataset_dir = Path(base_dir) / dataset / "dataframes"
        if dataset_dir.exists():
            parquet_files = list(dataset_dir.glob("*.parquet"))
            if parquet_files:
                files_by_dataset[dataset] = parquet_files
                print(f"Found {len(parquet_files)} parquet files in {dataset}")
            else:
                print(f"Warning: No parquet files found in {dataset_dir}")
        else:
            print(f"Warning: Directory {dataset_dir} does not exist")

    return files_by_dataset


def merge_parquet_files(files_by_dataset: Dict[str, List[Path]]) -> pd.DataFrame:
    """
    Merge parquet files from multiple datasets.

    Args:
        files_by_dataset: Dictionary mapping dataset names to parquet file paths

    Returns:
        Merged DataFrame with all data
    """
    all_dataframes = []

    for dataset, files in files_by_dataset.items():
        print(f"\nProcessing {dataset}...")

        for file in files:
            try:
                df = pd.read_parquet(file)

                # Ensure dataset column exists and is set correctly
                if 'dataset' not in df.columns:
                    df['dataset'] = dataset
                else:
                    # Verify or override dataset column
                    if df['dataset'].iloc[0] != dataset:
                        print(f"  Note: Overriding dataset column in {file.name}")
                        df['dataset'] = dataset

                # Apply model name mapping
                df = apply_model_name_mapping(df)

                all_dataframes.append(df)
                print(f"  ✓ Loaded {file.name}: {len(df)} rows, {len(df.columns)} columns")

            except Exception as e:
                print(f"  ✗ Error loading {file.name}: {e}")
                raise  # Re-raise to ensure validation errors are not silently ignored

    if not all_dataframes:
        raise ValueError("No dataframes were successfully loaded")

    # Get all unique columns across all dataframes
    all_columns = set()
    for df in all_dataframes:
        all_columns.update(df.columns)

    print(f"\nTotal unique columns across all datasets: {len(all_columns)}")

    # Check for column differences
    common_columns = set(all_dataframes[0].columns)
    for df in all_dataframes[1:]:
        common_columns = common_columns.intersection(df.columns)

    print(f"Common columns across all files: {len(common_columns)}")

    if len(common_columns) < len(all_columns):
        diff_columns = all_columns - common_columns
        print(f"Columns that differ: {sorted(diff_columns)}")

    # Merge all dataframes with outer join to preserve all columns
    print("\nMerging dataframes...")
    merged_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

    return merged_df


def generate_summary_report(df: pd.DataFrame) -> None:
    """
    Generate and print a summary report of the merged data.

    Args:
        df: Merged DataFrame
    """
    print("\n" + "="*60)
    print("MERGE SUMMARY REPORT")
    print("="*60)

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nRows per dataset:")
    dataset_counts = df['dataset'].value_counts().sort_index()
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count:,} rows")

    if 'model_name' in df.columns:
        print(f"\nUnique models: {df['model_name'].nunique()}")

    if 'inference_mode' in df.columns:
        print(f"\nInference modes: {df['inference_mode'].unique().tolist()}")

    if 'precision_level' in df.columns:
        print(f"Precision levels: {df['precision_level'].unique().tolist()}")

    # Check for missing values
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    if len(columns_with_missing) > 0:
        print(f"\nColumns with missing values:")
        for col, count in columns_with_missing.items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count:,} ({pct:.1f}%)")
    else:
        print("\n✓ No missing values in common columns")

    print("\n" + "="*60)


def main():
    """Main execution function."""
    # Configuration
    base_dir = "eval_results"
    datasets = [
        "epoch_1d",
        "euler_1d",
        "euler_2d",
        "ns_transient_2d",
        "burgers_1d",
        "heat_1d",
        "heat_2d",
        "fem_2d",
        "mpm_2d",
        "diff_react_1d",
        "hasegawa_mima_linear",
        "hasegawa_mima_nonlinear",
        "euler_1d_icl_accuracy_focused",
        "euler_1d_icl_cost_excluded",
        "euler_1d_icl_full",
        "heat_1d_icl_accuracy_focused",
        "heat_1d_icl_cost_excluded",
        "heat_1d_icl_full",
        "ns_transient_2d_icl_accuracy_focused",
        "ns_transient_2d_icl_cost_excluded",
        "ns_transient_2d_icl_full",
        "mpm_2d_icl_accuracy_focused",
        "mpm_2d_icl_cost_excluded",
        "mpm_2d_icl_full"
    ]
    output_file = Path(base_dir) / "merged_results.parquet"

    print("="*60)
    print("MERGING EVALUATION RESULTS")
    print("="*60)

    # Find all parquet files
    files_by_dataset = find_parquet_files(base_dir, datasets)

    if not files_by_dataset:
        print("\nError: No parquet files found in any dataset directory")
        return

    # Merge files
    merged_df = merge_parquet_files(files_by_dataset)

    # Generate summary report
    generate_summary_report(merged_df)

    # Save merged results
    print(f"\nSaving merged results to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_file, index=False, compression='snappy')

    # Verify saved file
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Successfully saved: {file_size_mb:.2f} MB")

    print("\nDone!")


if __name__ == "__main__":
    main()
