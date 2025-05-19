#!/usr/bin/env python3
"""
compute_cost_efficiency.py

Reads two JSON files:
- ITERATIVE_QUESTIONS_JSON: baseline dummy data (assumed perfect convergence)
- ITERATIVE_RESULTS_JSON: model run results

Calculates and prints:
- Converged rate (num_converged / num_samples)
- Model Cost Efficiency (MCE)   = num_converged / total_model_cost
- Dummy Cost Efficiency (DCE)   = num_samples / total_dummy_cost
- Relative Cost Efficiency (RCE)= MCE / DCE

The same information is saved to `cost_efficiency_results.txt` in the current
working directory.
"""

import json
from pathlib import Path
from typing import List, Dict

# ---- File locations ---------------------------------------------------------
ITERATIVE_QUESTIONS_JSON = (
    "/home/leo/workspace/coastbench/data/1D_heat_transfer/n_space/zero_shot_question.json"
)

ITERATIVE_RESULTS_JSON = (
    "/home/leo/workspace/coastbench/results/1D_heat_transfer/n_space/zero_shot_anthropic.claude-3-7-sonnet-20250219-v1:0.json"
)

# ---- Helpers ----------------------------------------------------------------

def load_json(path: str | Path) -> List[Dict]:
    """Load a JSON file and return its top‑level list."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)

# ---- Main computation -------------------------------------------------------

def main() -> None:
    # Load data
    questions = load_json(ITERATIVE_QUESTIONS_JSON)
    results = load_json(ITERATIVE_RESULTS_JSON)

    num_samples = len(results)
    if num_samples == 0:
        raise ValueError("No records found in results JSON — nothing to compute.")

    # Convergence statistics
    num_converged = sum(1 for item in results if item.get("converged"))
    converged_rate = num_converged / num_samples

    # Cost calculations (field is named `dummy_cost` in both JSON files)
    total_model_cost = sum(item.get("accumulated_cost") for item in results)
    total_dummy_cost = sum(item.get("dummy_cost") for item in questions)

    # Efficiency metrics
    model_cost_efficiency = (
        num_converged / total_model_cost if total_model_cost else 0.0
    )
    dummy_cost_efficiency = (
        num_samples / total_dummy_cost if total_dummy_cost else 0.0
    )
    relative_cost_efficiency = (
        model_cost_efficiency / dummy_cost_efficiency
        if dummy_cost_efficiency else float("inf")
    )

    # Prepare nicely‑formatted summary lines
    summary_lines = [
        f"Model's Converged rate: {converged_rate:.3f}",
        f"Model cost efficiency: {model_cost_efficiency:.2e}",
        f"Dummy cost efficiency: {dummy_cost_efficiency:.2e}",
        f"Relative cost efficiency (model vs dummy): {relative_cost_efficiency:.3f}",
    ]

    # Print to stdout
    print("\n".join(summary_lines))

    # Persist to file for later reference
    output_path = Path("cost_efficiency_results.txt")
    output_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nResults saved to '{output_path.resolve()}'")


if __name__ == "__main__":
    main()
