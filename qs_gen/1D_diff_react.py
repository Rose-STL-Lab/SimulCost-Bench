# -*- coding: utf-8 -*-
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from qs_gen.utils import *

# -----------------------------------------------------------------------------#
#  Global settings                                                             #
# -----------------------------------------------------------------------------#
DIFF_REACT_1D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "diff_react_1d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "diff_react_1d"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "low": {
        "tolerance_rmse": {
            "n_space": 0.05,
            "cfl": 0.05,
            "tol": 1e-5
        },
        "description": "Relaxed convergence criteria (wave position based)"
    },
    "medium": {
        "tolerance_rmse": {
            "n_space": 0.01,
            "cfl": 0.005,
            "tol": 1e-6
        },
        "description": "Moderate convergence criteria (wave position based)"
    },
    "high": {
        "tolerance_rmse": {
            "n_space": 0.001,
            "cfl": 0.0001,
            "tol": 1e-7
        },
        "description": "Most stringent convergence criteria (wave position based)"
    },
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class OneDDiffReactQuestionGenerator:
    """Generate 1D Diffusion-Reaction training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()

    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results with strict validation (fail-fast)"""
        if not TASKS_JSON_PATH.exists():
            raise FileNotFoundError(f"Tasks file not found: {TASKS_JSON_PATH}")

        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Strict validation - fail fast
        if "tasks" not in data:
            raise KeyError(f"Missing required key 'tasks' in {TASKS_JSON_PATH}")

        return data

    def _infer_precision_level(self, precision_config: Dict[str, Any]) -> str:
        """Infer precision level from precision_config values

        Since tasks.json has nested tolerance_rmse structure, we need to infer it
        from the tolerance_rmse dict values.
        """
        tolerance_rmse = precision_config.get("tolerance_rmse")

        if not isinstance(tolerance_rmse, dict):
            raise ValueError(f"Expected tolerance_rmse to be a dict, got {type(tolerance_rmse)}")

        # Match against known precision levels
        for level_name, level_config in PRECISION_LEVELS.items():
            expected_tol = level_config["tolerance_rmse"]
            if tolerance_rmse == expected_tol:
                return level_name

        # Fallback: if exact match not found, raise error for clarity
        raise ValueError(
            f"Unknown precision level: tolerance_rmse={tolerance_rmse}"
        )

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["cfl", "n_space", "tol"]
        # All three tasks support both iterative and zero-shot modes

        total_files = 0
        total_questions = 0

        print("=" * 80)
        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)

            for precision_level in PRECISION_LEVELS.keys():
                precision_config = PRECISION_LEVELS[precision_level]
                tolerance_rmse = precision_config["tolerance_rmse"][task]

                print(f"  {precision_level.upper()} precision:")
                print(f"    Tolerance RMSE: {tolerance_rmse}")

                # Generate iterative questions (all tasks support iterative)
                dataset_iter = self.generate_question_dataset(
                    task=task,
                    precision_level=precision_level,
                    zero_shot=False
                )
                if dataset_iter:  # Only save if there are questions
                    self._save_dataset(
                        dataset_iter,
                        task=task,
                        precision_level=precision_level,
                        zero_shot=False
                    )
                    total_files += 1
                    total_questions += len(dataset_iter)
                    print(f"      Iterative: {len(dataset_iter):2d} questions")
                else:
                    print(f"      Iterative: 0 questions (no data)")

                # Generate zero-shot questions (all tasks support zero-shot)
                dataset_zero = self.generate_question_dataset(
                    task=task,
                    precision_level=precision_level,
                    zero_shot=True
                )
                if dataset_zero:  # Only save if there are questions
                    self._save_dataset(
                        dataset_zero,
                        task=task,
                        precision_level=precision_level,
                        zero_shot=True
                    )
                    total_files += 1
                    total_questions += len(dataset_zero)
                    print(f"      Zero-shot: {len(dataset_zero):2d} questions")
                else:
                    print(f"      Zero-shot: 0 questions (no data)")

        print("\n" + "=" * 80)
        print(f"SUMMARY: Generated {total_questions} questions across {total_files} files")
        print("=" * 80)

    def generate_question_dataset(self, task: str, precision_level: str, zero_shot: bool) -> List[Dict]:
        """Generate question dataset for specific task and precision level"""
        dataset: List[Dict] = []
        precision_config = PRECISION_LEVELS[precision_level]

        # Filter tasks by target parameter and inferred precision level
        matching_tasks = []
        for t in self.tasks_data["tasks"]:
            if t["target_parameter"] == task:
                # Infer precision level from precision_config
                inferred_level = self._infer_precision_level(t["precision_config"])
                if inferred_level == precision_level:
                    matching_tasks.append(t)

        qid = 1
        for task_data in matching_tasks:
            # Strict validation - fail fast
            if "profile" not in task_data:
                raise KeyError(f"Missing required key 'profile' in task data")
            if "results" not in task_data:
                raise KeyError(f"Missing required key 'results' in task data for profile {task_data.get('profile', 'unknown')}")

            profile = task_data["profile"]

            # Load profile configuration
            profile_path = DIFF_REACT_1D_CFG_DIR / f"{profile}.yaml"
            if not profile_path.exists():
                raise FileNotFoundError(f"Profile config not found: {profile_path}")

            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            # Strict validation of params
            if params is None:
                raise ValueError(f"Empty configuration file: {profile_path}")

            # Extract results with strict validation
            results = task_data["results"]

            if "cost_history" not in results:
                raise KeyError(f"Missing 'cost_history' in results for profile {profile}")
            if "parameter_history" not in results:
                raise KeyError(f"Missing 'parameter_history' in results for profile {profile}")
            if "converged" not in results:
                raise KeyError(f"Missing 'converged' in results for profile {profile}")
            if "optimal_parameter_value" not in results:
                raise KeyError(f"Missing 'optimal_parameter_value' in results for profile {profile}")

            cost_history = results["cost_history"]
            param_history = results["parameter_history"]
            is_converged = results["converged"]

            # Get optimal value first (needed for dummy_cost calculation)
            optimal_value = results["optimal_parameter_value"]

            # Calculate dummy cost according to requirements
            # For zero-shot: use the second-to-last cost
            # For iterative: sum all costs except the last one
            if zero_shot and len(cost_history) > 1:
                dummy_cost = cost_history[-2]
            elif len(cost_history) > 1:
                dummy_cost = sum(cost_history[:-1])
            else:
                dummy_cost = cost_history[0]

            # Strict validation for non_target_parameters
            if "non_target_parameters" not in task_data:
                raise KeyError(f"Missing 'non_target_parameters' in task data for profile {profile}")

            non_target_params = task_data["non_target_parameters"]
            tunable_params = ["cfl", "n_space", "tol"]

            # Find the best_params from parameter_history based on optimal_parameter_value
            best_params = None
            for param_set in param_history:
                if param_set.get(task) == optimal_value:
                    best_params = param_set.copy()
                    break

            # If not found, this is an error in the data
            if best_params is None:
                raise ValueError(
                    f"Could not find optimal parameter {task}={optimal_value} "
                    f"in parameter_history"
                )

            # Filter best_params to only include the 3 tunable parameters
            filtered_best_params = {
                key: best_params[key] for key in tunable_params
                if key in best_params
            }

            # Filter param_history to only include the 3 tunable parameters
            filtered_param_history = []
            for param_set in param_history:
                filtered_set = {
                    key: param_set[key] for key in tunable_params
                    if key in param_set
                }
                filtered_param_history.append(filtered_set)

            # Build question entry
            question_entry = {
                "QID": qid,
                "profile": profile,
                "reaction_type": params.get("reaction_type"),
                "zero_shot": zero_shot,
                "target_parameter": task,
                "precision_level": precision_level,
                "precision_config": precision_config,
                "is_converged": is_converged,
                "dummy_cost": dummy_cost,
                "cost_history": cost_history,
                "best_params": filtered_best_params,
                "param_history": filtered_param_history,
                "question": self._build_question_text(params, task, precision_level, precision_config),
            }

            dataset.append(question_entry)
            qid += 1

        return dataset

    @staticmethod
    def _build_question_text(params: dict, task: str, precision_level: str, precision_config: dict) -> str:
        """Build question text from diff_react_1d.md documentation"""

        # Strict validation - fail fast
        required_keys = ["reaction_type", "L", "record_dt", "end_frame", "max_iter"]
        for key in required_keys:
            if key not in params:
                raise KeyError(f"Missing required parameter '{key}' in profile configuration")

        reaction_type = params["reaction_type"]

        question_lines = [
            "# 1D Diffusion-Reaction Equations with Fully Implicit Newton Method",
            "",
            "## Introduction",
            "",
            "This simulation solves the 1D diffusion-reaction equation using a fully implicit Newton method with adaptive line search. The solver supports multiple reaction terms including Fisher-KPP, Allee effect, and Allen-Cahn (cubic) reactions.",
            "",
            "**PDE form:**",
            r"$$\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} + f(u)$$",
            "",
            "Where the reaction term $f(u)$ can be:",
            "",
            "**Fisher-KPP reaction:**",
            r"$$f(u) = u(1-u)$$",
            "",
            "**Allee effect reaction:**",
            r"$$f(u) = u(1-u)(u-a)$$",
            "",
            "where $a$ is the Allee threshold parameter ($0 < a < 1$).",
            "",
            "**Allen-Cahn (cubic) reaction:**",
            r"$$f(u) = u(1-u^2)$$",
            "",
            "### Boundary and Initial Conditions",
            "",
            "- **Boundary conditions:** Dirichlet conditions $u(0,t) = 1$ and $u(L,t) = 0$",
            "- **Initial condition:** Step function $u(x,0) = 1$ for $0 \\leq x \\leq 2$, $u(x,0) = 0$ elsewhere",
            f"- **Domain:** $[0, L]$ where $L = {params['L']}$",
            "",
            "### Spatial Discretization",
            "",
            "The spatial discretization uses second-order central differences for the Laplacian:",
            "",
            r"$$\frac{\partial^2 u}{\partial x^2}\bigg|_i \approx \frac{u_{i-1} - 2u_i + u_{i+1}}{(\Delta x)^2}$$",
            "",
            "where $\\Delta x = L/n_{space}$ is the spatial step size.",
            "",
            "### Temporal Discretization",
            "",
            "The temporal discretization uses a fully implicit scheme:",
            "",
            r"$$\frac{u^{n+1} - u^n}{\Delta t} = \frac{\partial^2 u^{n+1}}{\partial x^2} + f(u^{n+1})$$",
            "",
            "This results in a nonlinear system of equations that is solved using Newton's method.",
            "",
            "### Newton Method Implementation",
            "",
            "The Newton solver assembles the residual vector and Jacobian matrix:",
            "",
            "**Residual (scaled by $\\Delta x^2$ for stability):**",
            r"$$R_i = \frac{u_i^{n+1} - u_i^n}{\Delta t} - \frac{u_{i-1}^{n+1} - 2u_i^{n+1} + u_{i+1}^{n+1}}{(\Delta x)^2} - f(u_i^{n+1})$$",
            "",
            "**Jacobian matrix elements:**",
            r"- Diagonal: $J_{i,i} = \frac{1}{\Delta t} + \frac{2}{(\Delta x)^2} - \frac{df}{du}(u_i^{n+1})$",
            r"- Off-diagonal: $J_{i,i\pm1} = -\frac{1}{(\Delta x)^2}$",
            "",
            "### Line Search",
            "",
            "The Newton method uses a greedy line search strategy:",
            "",
            "1. Start with initial step size $\\alpha = \\alpha_0$",
            "2. Try step: $u_{trial} = u + \\alpha \\cdot \\Delta u$",
            "3. If residual norm decreases, accept step; otherwise backtrack with $\\alpha \\leftarrow \\alpha/2$",
            "4. Continue until residual norm falls below tolerance or minimum step size is reached",
            "",
            "## Test Cases",
            "",
            "### Current Configuration",
            "",
        ]

        # Add reaction type specific description
        if reaction_type == "fisher":
            question_lines.extend([
                "**Fisher-KPP reaction:** $f(u) = u(1-u)$",
                "- Classic logistic growth with diffusion",
                "- Generates traveling wave solutions"
            ])
        elif reaction_type == "allee":
            allee_threshold = params.get("allee_threshold", 0.3)
            question_lines.extend([
                f"**Allee effect reaction:** $f(u) = u(1-u)(u-a)$ with $a = {allee_threshold}$",
                "- Includes critical population threshold",
                "- Can lead to population extinction below threshold"
            ])
        elif reaction_type == "cubic":
            question_lines.extend([
                "**Allen-Cahn (cubic) reaction:** $f(u) = u(1-u^2)$",
                "- Phase field model with bistable potential",
                "- Generates interface dynamics"
            ])
        else:
            question_lines.extend([
                f"**Custom reaction type:** {reaction_type}"
            ])

        question_lines.extend([
            "",
            "### Physical Parameters",
            f"- **Reaction Type**: {reaction_type}",
            f"- **Domain Length (L)**: {params['L']}",
            f"- **Record Interval (record_dt)**: {params['record_dt']}",
            f"- **Simulation End Frame**: {params['end_frame']}",
            f"- **Maximum Newton Iterations**: {params['max_iter']}",
        ])

        # Add allee_threshold if present
        if "allee_threshold" in params:
            question_lines.append(f"- **Allee Threshold**: {params['allee_threshold']}")

        # Get task-specific tolerance values
        tolerance_rmse_dict = precision_config["tolerance_rmse"]

        question_lines.extend([
            "",
            "## Convergence Metrics",
            "",
            "The simulated results are considered correct if they meet the precision-dependent tolerance criteria and satisfy convergence criteria:",
            "",
            "**Convergence Criteria:**",
            "1. **Wave position convergence:** Relative wave position error falls below specified tolerance",
            "2. **Physical constraints:** Solution satisfies appropriate physical metrics based on reaction type",
            "3. **Wave propagation:** For Fisher-KPP, traveling waves propagate at expected speeds",
            "4. **Specialized metrics:** For Allee effect (p2), wave propagation quality is checked (max gradient > 0.04)",
            "5. **Reaction balance:** Temporal variation measure for solution stability",
            "",
            "### Physical Metrics",
            "",
            "The solver computes several physical metrics to assess solution quality:",
            "",
            "**For Fisher-KPP and Allen-Cahn reactions:**",
            "- **Maximum principle:** Solution should be bounded by initial min/max values",
            "- **Boundary conditions:** Dirichlet conditions u(0,t)=1, u(L,t)=0 must be satisfied",
            "- **Reaction balance:** Temporal variation measure `np.mean(np.abs(np.diff(u, axis=0)), axis=1)`",
            "",
            "**For Allee effect reactions:**",
            "- **Wave propagation quality:** Checks for sharp wave fronts (max gradient > 0.04)",
            "- **Reaction balance:** Same temporal variation measure as other reactions",
            "- **Maximum principle:** Bypassed (hardcoded as satisfied due to expected threshold dynamics)",
            "- **Boundary conditions:** Bypassed (hardcoded as satisfied due to complex wave behavior)",
            "",
            "**Reaction Balance Calculation:**",
            "The reaction balance measures temporal variation in the solution:",
            "```python",
            "reaction_balance = np.mean(np.abs(np.diff(u, axis=0)), axis=1)",
            "```",
            "This calculates the average absolute change between consecutive time steps across all spatial points, providing a measure of solution stability and temporal evolution.",
            "",
            "## Parameter Tuning Tasks",
            "",
            "### Tasks",
            "",
            "1. **CFL Convergence Search**",
            "   - CFL number controls time step size: $\\Delta t = \\text{CFL} \\cdot (\\Delta x)^2$ for diffusion stability",
            "",
            "2. **n_space Convergence Search**",
            "   - n_space determines spatial resolution: $\\Delta x = L / n_{space}$",
            "",
            "3. **Tolerance Convergence Search**",
            "   - Newton solver tolerance controls convergence criteria",
            "",
            "### Cost Calculation",
            "",
            "The computational cost is tracked as:",
            "",
            "- **Newton cost:** $3 \\times \\text{total\\_newton\\_iters} \\times n_{space}$",
            "- **Line search cost:** $\\text{total\\_line\\_search\\_iters} \\times n_{space}$",
            "- **Total cost:** $\\text{newton\\_cost} + \\text{line\\_search\\_cost}$",
            "",
            "The factor of 3 in Newton cost accounts for:",
            "1. Residual calculation",
            "2. Jacobian assembly",
            "3. Linear system solve",
            "",
            f"**Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Tolerance RMSE**: ≤ {tolerance_rmse_dict[task]}",
        ])

        return "\n".join(question_lines)

    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        """Save dataset to the new path format: data/diff_react_1d/{task}/{precision_level}/"""
        base_dir = Path("data") / "diff_react_1d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename

        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all 1D Diffusion-Reaction questions for all tasks and precision levels"""
    print("1D DIFFUSION-REACTION QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/diff_react_1d/{{task}}/{{precision_level}}/")
    print(f"Tasks (all support iterative + zero-shot): cfl, n_space, tol")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types:")
    print(f"  - iterative_questions.json (for all tasks)")
    print(f"  - zero_shot_questions.json (for all tasks)")

    gen = OneDDiffReactQuestionGenerator()
    gen.generate_all_questions()

    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()
