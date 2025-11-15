#!/usr/bin/env python3
"""
Debug script to investigate why MPM + Llama has zero efficiency.
"""

import pandas as pd

# Load the full dataset
parquet_path = "../eval_results/merged_results.parquet"
df = pd.read_parquet(parquet_path)

# Filter for MPM dataset and Llama model
mpm_llama = df[
    (df["dataset"].str.contains("mpm", case=False)) &
    (df["model_name"].str.contains("llama", case=False))
].copy()

print("=" * 80)
print(f"MPM + LLAMA ENTRIES")
print("=" * 80)
print(f"Total entries: {len(mpm_llama)}")
print(f"\nDatasets: {mpm_llama['dataset'].unique()}")
print(f"Models: {mpm_llama['model_name'].unique()}")
print(f"Inference modes: {mpm_llama['inference_mode'].unique()}")
print(f"\nAvailable columns: {list(mpm_llama.columns)}")

# Check success rate
print("\n" + "=" * 80)
print("SUCCESS RATE")
print("=" * 80)
success_by_mode = mpm_llama.groupby(["inference_mode", "precision_level"])["is_successful"].agg(["mean", "count"])
print(success_by_mode)

# Check efficiency distribution
print("\n" + "=" * 80)
print("EFFICIENCY STATISTICS")
print("=" * 80)
eff_stats = mpm_llama.groupby(["inference_mode", "precision_level"])["efficiency"].describe()
print(eff_stats)

# Sample some entries to examine
print("\n" + "=" * 80)
print("SAMPLE ENTRIES (first 5 from each mode/precision)")
print("=" * 80)

for mode in mpm_llama["inference_mode"].unique():
    for prec in ["low", "medium", "high"]:
        subset = mpm_llama[(mpm_llama["inference_mode"] == mode) & (mpm_llama["precision_level"] == prec)]
        if len(subset) > 0:
            print(f"\n{'='*80}")
            print(f"{mode.upper()} - {prec.upper()} PRECISION (showing first 5)")
            print('='*80)

            # For each row, dynamically determine which columns to show based on the task
            for idx, row in subset.head(5).iterrows():
                task = row["task"]
                # Determine model and dummy columns for this task
                model_col = f"model_{task}"
                dummy_col = f"dummy_{task}"

                # Base columns
                base_cols = [
                    "dataset", "task", "model_name", "inference_mode", "precision_level",
                    "is_successful", "efficiency", "model_cost", "dummy_cost",
                    "profile", "target_parameters", "non_target_parameters"
                ]

                # Add task-specific columns if they exist
                cols_to_show = base_cols.copy()
                if model_col in subset.columns:
                    cols_to_show.append(model_col)
                if dummy_col in subset.columns:
                    cols_to_show.append(dummy_col)

                if idx == subset.head(5).index[0]:
                    # Print header
                    print(" | ".join(cols_to_show))
                    print("-" * 80)

                # Print row values
                values = []
                for col in cols_to_show:
                    val = row[col]
                    if isinstance(val, float):
                        if col == "efficiency":
                            values.append(f"{val:.6f}")
                        elif "cost" in col:
                            values.append(f"{val:.2e}")
                        else:
                            values.append(f"{val:.4f}")
                    else:
                        values.append(str(val))
                print(" | ".join(values))

# Check successful zero-shot entries
print("\n" + "=" * 80)
print("SUCCESSFUL ZERO-SHOT ENTRIES (showing first 3)")
print("=" * 80)
successful_zs = mpm_llama[(mpm_llama["is_successful"] == 1) & (mpm_llama["inference_mode"] == "zero_shot")]
print(f"Total successful zero-shot: {len(successful_zs)}")
if len(successful_zs) > 0:
    for i, (idx, row) in enumerate(successful_zs.head(3).iterrows()):
        print("\n" + "-" * 80)
        print(f"Entry {i+1}:")
        task = row["task"]
        model_col = f"model_{task}"
        dummy_col = f"dummy_{task}"

        print(f"  Task: {row['task']} | Precision: {row['precision_level']} | Profile: {row['profile']}")
        print(f"  Is successful: {row['is_successful']} | Efficiency: {row['efficiency']:.6f}")
        print(f"  Model cost: {row['model_cost']:.2e} | Dummy cost: {row['dummy_cost']:.2e}")
        if model_col in successful_zs.columns and dummy_col in successful_zs.columns:
            print(f"  Model {task}: {row[model_col]:.1f} | Dummy {task}: {row[dummy_col]:.1f}")

# Check if efficiency is zero for failed entries
print("\n" + "=" * 80)
print("FAILED ENTRIES")
print("=" * 80)
failed = mpm_llama[mpm_llama["is_successful"] == 0]
print(f"Count: {len(failed)}")
print(f"Efficiency range: [{failed['efficiency'].min()}, {failed['efficiency'].max()}]")
print(f"Mean efficiency: {failed['efficiency'].mean()}")

# Check zero-shot vs iterative separately
print("\n" + "=" * 80)
print("ZERO-SHOT EFFICIENCY BREAKDOWN")
print("=" * 80)
zs = mpm_llama[mpm_llama["inference_mode"] == "zero_shot"]
print(f"Total zero-shot entries: {len(zs)}")
print(f"Successful: {zs['is_successful'].sum()}")
print(f"Mean efficiency (all): {zs['efficiency'].mean():.4f}")
print(f"Mean efficiency (successful only): {zs[zs['is_successful']==1]['efficiency'].mean():.4f}")

print("\n" + "=" * 80)
print("ITERATIVE EFFICIENCY BREAKDOWN")
print("=" * 80)
iter_mode = mpm_llama[mpm_llama["inference_mode"] == "iterative"]
print(f"Total iterative entries: {len(iter_mode)}")
print(f"Successful: {iter_mode['is_successful'].sum()}")
print(f"Mean efficiency (all): {iter_mode['efficiency'].mean():.4f}")
print(f"Mean efficiency (successful only): {iter_mode[iter_mode['is_successful']==1]['efficiency'].mean():.4f}")
