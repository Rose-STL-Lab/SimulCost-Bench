#!/usr/bin/env python3
"""
Debug script to find the prompt and full details for a specific entry.
"""

import pandas as pd
import json
from pprint import pprint

# Load the full dataset
parquet_path = "./eval_results/merged_results.parquet"
df = pd.read_parquet(parquet_path)

# Find the specific entry
target = df[
    (df["dataset"] == "mpm_2d") &
    (df["task"] == "nx") &
    (df["model_name"] == "Llama-3-70B-Instruct") &
    (df["inference_mode"] == "iterative") &
    (df["precision_level"] == "low") &
    (df["profile"] == "p1") &
    (df["model_nx"] == 40.0) &
    (df["dummy_nx"] == 44.0)
].copy()

print("=" * 100)
print("FOUND ENTRIES")
print("=" * 100)
print(f"Number of matching entries: {len(target)}")

if len(target) > 0:
    # Print all columns for this entry
    entry = target.iloc[0]

    print("\n" + "=" * 100)
    print("BASIC INFO")
    print("=" * 100)
    print(f"Dataset: {entry['dataset']}")
    print(f"Task: {entry['task']}")
    print(f"Model: {entry['model_name']}")
    print(f"Inference mode: {entry['inference_mode']}")
    print(f"Precision: {entry['precision_level']}")
    print(f"Profile: {entry['profile']}")
    print(f"QID: {entry['qid']}")

    print("\n" + "=" * 100)
    print("PARAMETERS")
    print("=" * 100)
    print(f"Target parameter: {entry['target_parameters']}")
    print(f"Non-target parameters: {entry['non_target_parameters']}")
    print(f"Model nx: {entry['model_nx']}")
    print(f"Dummy nx: {entry['dummy_nx']}")
    print(f"Model npart: {entry['model_npart']}")
    print(f"Dummy npart: {entry['dummy_npart']}")
    print(f"Model cfl: {entry.get('model_cfl', 'N/A')}")
    print(f"Dummy cfl: {entry.get('dummy_cfl', 'N/A')}")

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"Is successful: {entry['is_successful']}")
    print(f"Is converged: {entry['is_converged']}")
    print(f"Model cost: {entry['model_cost']:.2e}")
    print(f"Dummy cost: {entry['dummy_cost']:.2e}")
    print(f"Efficiency: {entry['efficiency']:.6f}")
    print(f"RMSE: {entry.get('rmse', 'N/A')}")
    print(f"Tolerance: {entry.get('tolerance', 'N/A')}")

    print("\n" + "=" * 100)
    print("ATTEMPT HISTORY")
    print("=" * 100)
    if 'attempt_history' in entry and entry['attempt_history']:
        history = entry['attempt_history']
        # for h in history:
        #     pprint(h)
        pprint(history)
    else:
        print("No attempt history available")

    # print("\n" + "=" * 100)
    # print("ALL COLUMNS")
    # print("=" * 100)
    # for col in df.columns:
    #     val = entry[col]
    #     if pd.notna(val):
    #         if isinstance(val, float):
    #             print(f"{col}: {val:.6e}")
    #         else:
    #             print(f"{col}: {val}")

# # Now let's check the cost calculation
# print("\n\n" + "=" * 100)
# print("COST CALCULATION ANALYSIS")
# print("=" * 100)

# if len(target) > 0:
#     entry = target.iloc[0]

#     # For MPM, cost should be proportional to nx^2 * npart
#     model_nx = entry['model_nx']
#     dummy_nx = entry['dummy_nx']
#     model_npart = entry['model_npart']
#     dummy_npart = entry['dummy_npart']

#     print(f"\nModel parameters: nx={model_nx}, npart={model_npart}")
#     print(f"Dummy parameters: nx={dummy_nx}, npart={dummy_npart}")

#     model_complexity = model_nx**2 * model_npart
#     dummy_complexity = dummy_nx**2 * dummy_npart

#     print(f"\nExpected complexity ratio (model/dummy): {model_complexity / dummy_complexity:.4f}")
#     print(f"Actual cost ratio (model/dummy): {entry['model_cost'] / entry['dummy_cost']:.4f}")

#     print(f"\nModel complexity: nx^2 * npart = {model_nx}^2 * {model_npart} = {model_complexity:.2e}")
#     print(f"Dummy complexity: nx^2 * npart = {dummy_nx}^2 * {dummy_npart} = {dummy_complexity:.2e}")
