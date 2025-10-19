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
        "tolerance_rmse": 0.15,
        "description": "Relaxed convergence criteria"
    },
    "medium": {
        "tolerance_rmse": 0.001,
        "description": "Moderate convergence criteria"
    },
    "high": {
        "tolerance_rmse": 0.0001,
        "description": "Most stringent convergence criteria"
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

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        # cfl, n_space, tol, min_step support both iterative and zero-shot
        iterative_supported_tasks = ["cfl", "n_space", "tol", "min_step"]
        # initial_step_guess only supports zero-shot
        zero_shot_only_tasks = ["initial_step_guess"]

        all_tasks = iterative_supported_tasks + zero_shot_only_tasks

        total_files = 0
        total_questions = 0

        print("=" * 80)
        for task in all_tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)

            for precision_level in PRECISION_LEVELS.keys():
                precision_config = PRECISION_LEVELS[precision_level]
                tolerance_rmse = precision_config["tolerance_rmse"]

                print(f"  {precision_level.upper()} precision:")
                print(f"    Tolerance RMSE: {tolerance_rmse}")

                # Generate iterative questions only for supported tasks
                if task in iterative_supported_tasks:
                    dataset_iter = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=False)
                    if dataset_iter:  # Only save if there are questions
                        self._save_dataset(dataset_iter, task=task, precision_level=precision_level, zero_shot=False)
                        total_files += 1
                        total_questions += len(dataset_iter)
                        print(f"      Iterative: {len(dataset_iter):2d} questions")
                    else:
                        print(f"      Iterative: 0 questions (no data)")
                else:
                    print(f"      Iterative: Not supported for {task}")

                # Generate zero-shot questions for all tasks
                dataset_zero = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=True)
                if dataset_zero:  # Only save if there are questions
                    self._save_dataset(dataset_zero, task=task, precision_level=precision_level, zero_shot=True)
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

        # Filter tasks by target parameter and precision level
        matching_tasks = [
            t for t in self.tasks_data["tasks"]
            if (t["target_parameter"] == task and
                t["precision_level"] == precision_level)
        ]

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
            # Special handling for initial_step_guess (grid search): use cost corresponding to optimal_parameter_value
            if task == "initial_step_guess" and zero_shot:
                # Find the index of optimal_parameter_value in param_history
                try:
                    optimal_index = param_history.index(optimal_value)
                    dummy_cost = cost_history[optimal_index]
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Cannot find optimal value {optimal_value} in parameter_history for profile {profile}: {e}")
            else:
                # For other tasks: use standard logic
                # zero_shot: use second-to-last cost (cost before final convergence check)
                # iterative: use sum of all costs except the last one
                dummy_cost = cost_history[-2] if zero_shot and len(cost_history) > 1 else sum(cost_history[:-1]) if len(cost_history) > 1 else cost_history[0]

            # Strict validation for non_target_parameters
            if "non_target_parameters" not in task_data:
                raise KeyError(f"Missing 'non_target_parameters' in task data for profile {profile}")

            non_target_params = task_data["non_target_parameters"]
            tunable_params = ["cfl", "n_space", "tol", "min_step", "initial_step_guess"]

            # Handle two different formats of parameter_history:
            # 1. List of dicts (for cfl, n_space, tol): contains all tunable parameters
            # 2. List of scalars (for min_step, initial_step_guess): contains only target parameter values

            if param_history and isinstance(param_history[0], dict):
                # Format 1: parameter_history is a list of dicts
                # Find the best_params from parameter_history based on optimal_parameter_value
                best_params = None
                for param_set in param_history:
                    if param_set.get(task) == optimal_value:
                        best_params = param_set.copy()
                        break

                if best_params is None:
                    raise ValueError(f"Could not find optimal parameter {task}={optimal_value} in parameter_history for profile {profile}")

                # Filter to only include tunable parameters
                filtered_best_params = {
                    key: best_params[key] for key in tunable_params
                    if key in best_params
                }

                # Filter param_history to only include tunable parameters
                filtered_param_history = []
                for param_set in param_history:
                    filtered_set = {
                        key: param_set[key] for key in tunable_params
                        if key in param_set
                    }
                    filtered_param_history.append(filtered_set)

            else:
                # Format 2: parameter_history is a list of scalar values
                # Reconstruct full parameter sets by combining with non_target_parameters

                # Build best_params by combining non-target params with optimal value
                # For initial_step_guess (grid search): use optimal_value
                # For min_step (iterative search): use param_history[-1] (last iteration)
                filtered_best_params = {}
                for key in tunable_params:
                    if key == task:
                        if task == "initial_step_guess":
                            # Grid search: optimal_value is the best among all trials
                            filtered_best_params[key] = optimal_value
                        else:
                            # Iterative search: last iteration is the best
                            filtered_best_params[key] = param_history[-1]
                    elif key in non_target_params:
                        filtered_best_params[key] = non_target_params[key]

                # Reconstruct param_history as list of dicts
                filtered_param_history = []
                for param_value in param_history:
                    param_dict = {}
                    for key in tunable_params:
                        if key == task:
                            param_dict[key] = param_value
                        elif key in non_target_params:
                            param_dict[key] = non_target_params[key]
                    filtered_param_history.append(param_dict)

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
                "question": self._build_question_text(params, precision_level, precision_config),
            }

            dataset.append(question_entry)
            qid += 1

        return dataset

    @staticmethod
    def _build_question_text(params: dict, precision_level: str, precision_config: dict) -> str:
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

        question_lines.extend([
            "",
            "## Parameter Tuning Tasks",
            "",
            "### Tasks",
            "",
            "1. **CFL Convergence Search**",
            "   - **cfl**: CFL number controls time step size: $\\Delta t = \\text{CFL} \\cdot (\\Delta x)^2$ for diffusion stability",
            "",
            "2. **n_space Convergence Search**",
            "   - **n_space**: n_space determines spatial resolution: $\\Delta x = L / n_{space}$",
            "",
            "3. **Tolerance Convergence Search**",
            "   - **tol**: Newton solver tolerance controls convergence criteria",
            "",
            "4. **Min Step Optimization**",
            "   - **min_step**: Minimum step size for line search controls robustness",
            "",
            "5. **Initial Step Guess Optimization**",
            "   - **initial_step_guess**: Initial step size for line search controls aggressiveness",
            "",
            "### Convergence Criteria",
            "",
            "The solution is considered converged when:",
            "",
            "1. **Newton convergence:** Residual norm falls below specified tolerance",
            "2. **Physical bounds:** Solution remains bounded and physically reasonable",
            "3. **Wave propagation:** For Fisher-KPP, traveling waves propagate at expected speeds",
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
            f"- **Tolerance RMSE**: ≤ {precision_config['tolerance_rmse']}",
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
    print(f"Iterative tasks: cfl, n_space, tol, min_step")
    print(f"Zero-shot only tasks: initial_step_guess")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types:")
    print(f"  - iterative_questions.json (for cfl, n_space, tol, min_step)")
    print(f"  - zero_shot_questions.json (for all tasks)")

    gen = OneDDiffReactQuestionGenerator()
    gen.generate_all_questions()

    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()
