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
HASEGAWA_MIMA_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "hasegawa_mima_linear"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "hasegawa_mima_linear"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "high": {
        "tolerance_rmse": 0.0005,
        "description": "High precision convergence - very challenging"
    },
    "medium": {
        "tolerance_rmse": 0.001,
        "description": "Medium precision convergence - challenging"
    },
    "low": {
        "tolerance_rmse": 0.01,
        "description": "Low precision convergence - achievable"
    },
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class HasegawaMimaLinearQuestionGenerator:
    """Generate Hasegawa-Mima linear training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()

    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        if not TASKS_JSON_PATH.exists():
            raise FileNotFoundError(
                f"Tasks file not found at {TASKS_JSON_PATH}. "
                "Please ensure the tasks.json file exists."
            )

        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _infer_precision_level(self, tolerance_rmse: float) -> str:
        """Infer precision level from tolerance_rmse value

        Since we need to map tolerance_rmse back to precision level names.
        """
        # Match against known precision levels
        for level_name, level_config in PRECISION_LEVELS.items():
            if abs(level_config["tolerance_rmse"] - tolerance_rmse) < 1e-10:
                return level_name

        # Fallback: if exact match not found, raise error for clarity (fail-fast)
        raise ValueError(
            f"Unknown precision level: tolerance_rmse={tolerance_rmse}. "
            f"Expected one of: {[cfg['tolerance_rmse'] for cfg in PRECISION_LEVELS.values()]}"
        )

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["N", "dt", "cg_atol"]
        # All three tasks support iterative and zero-shot modes

        total_files = 0
        total_questions = 0

        print("=" * 80)
        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)

            for precision_level in PRECISION_LEVELS.keys():
                precision_config = PRECISION_LEVELS[precision_level]
                tolerance_rmse = precision_config["tolerance_rmse"]

                print(f"  {precision_level.upper()} precision:")
                print(f"    Tolerance RMSE: {tolerance_rmse}")

                # Generate iterative questions
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

                # Generate zero-shot questions
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

    def generate_question_dataset(
        self,
        task: str,
        precision_level: str,
        zero_shot: bool
    ) -> List[Dict]:
        """Generate question dataset for specific task and precision level"""
        dataset: List[Dict] = []
        precision_config = PRECISION_LEVELS[precision_level]

        # Filter tasks by target parameter and inferred precision level
        matching_tasks = []
        for t in self.tasks_data["tasks"]:
            if t["target_parameter"] == task:
                # Check if tolerance_rmse field exists (fail-fast)
                if "precision_config" not in t or "tolerance_rmse" not in t["precision_config"]:
                    raise KeyError(
                        f"Required field 'precision_config.tolerance_rmse' not found in task {t.get('task_id', 'unknown')}. "
                        f"Task data: {t}"
                    )

                # Infer precision level from tolerance_rmse
                tolerance_rmse = t["precision_config"]["tolerance_rmse"]
                inferred_level = self._infer_precision_level(tolerance_rmse)
                if inferred_level == precision_level:
                    matching_tasks.append(t)

        qid = 1
        for task_data in matching_tasks:
            profile = task_data["profile"]

            # Load profile configuration
            profile_path = HASEGAWA_MIMA_CFG_DIR / f"{profile}.yaml"
            if not profile_path.exists():
                raise FileNotFoundError(
                    f"Profile configuration file not found at {profile_path}. "
                    f"Required for profile '{profile}'."
                )

            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            # Validate required fields in params (fail-fast)
            required_fields = ["case", "L", "v_star", "Dx", "N", "dt", "cg_atol"]
            for field in required_fields:
                if field not in params:
                    raise KeyError(
                        f"Required configuration field '{field}' not found in {profile_path}. "
                        f"Please check the configuration file."
                    )

            # Extract results
            results = task_data["results"]

            # Validate results structure (fail-fast)
            if "cost_history" not in results:
                raise KeyError(f"'cost_history' not found in results for task {task_data.get('task_id')}")
            if "parameter_history" not in results:
                raise KeyError(f"'parameter_history' not found in results for task {task_data.get('task_id')}")
            if "converged" not in results:
                raise KeyError(f"'converged' not found in results for task {task_data.get('task_id')}")

            cost_history = results["cost_history"]
            param_history = results["parameter_history"]
            is_converged = results["converged"]

            # Calculate dummy cost according to requirements for hasegawa_mima_linear
            # For zero-shot: use the last cost (cost_history[-1])
            # For iterative: sum all costs (sum(cost_history))
            # Note: This is different from nonlinear!
            if zero_shot:
                dummy_cost = cost_history[-1]
            else:
                dummy_cost = sum(cost_history)

            # Find the best_params from parameter_history based on optimal_parameter_value
            if "optimal_parameter_value" not in results:
                raise KeyError(f"'optimal_parameter_value' not found in results for task {task_data.get('task_id')}")

            optimal_value = results["optimal_parameter_value"]
            best_params = None

            # Search through parameter_history to find the matching parameter combination
            for param_set in param_history:
                if param_set.get(task) == optimal_value:
                    best_params = param_set.copy()
                    break

            # If not found, this is an error in the data (fail-fast)
            if best_params is None:
                raise ValueError(
                    f"Could not find optimal parameter {task}={optimal_value} "
                    f"in parameter_history for task {task_data.get('task_id')}"
                )

            # Filter best_params to only include the 3 tunable parameters
            filtered_best_params = {
                key: best_params[key] for key in ["N", "dt", "cg_atol"]
                if key in best_params
            }

            # Filter param_history to only include the 3 tunable parameters
            filtered_param_history = []
            for param_set in param_history:
                filtered_set = {
                    key: param_set[key] for key in ["N", "dt", "cg_atol"]
                    if key in param_set
                }
                filtered_param_history.append(filtered_set)

            # Extract case from params for storage (flat structure, not nested)
            case = params["case"]

            # Build question entry
            question_entry = {
                "QID": qid,
                "profile": profile,
                "case": case,
                "zero_shot": zero_shot,
                "target_parameter": task,
                "precision_level": precision_level,
                "precision_config": precision_config,
                "is_converged": is_converged,
                "dummy_cost": dummy_cost,
                "cost_history": cost_history,
                "best_params": filtered_best_params,
                "param_history": filtered_param_history,
                "question": self._build_question_text(
                    params,
                    profile,
                    precision_level,
                    precision_config
                ),
            }

            dataset.append(question_entry)
            qid += 1

        return dataset

    @staticmethod
    def _build_question_text(
        params: dict,
        profile: str,
        precision_level: str,
        precision_config: dict
    ) -> str:
        """Build question text from hasegawa_mima_linear.md documentation

        Excludes Dummy Strategy section and developer-only content.
        """
        question_lines = [
            "# Hasegawa-Mima Linear Equation with RK4 and CG Solver",
            "",
            "## Introduction",
            "",
            "This simulation solves the linearized Hasegawa-Mima equation for drift wave dynamics in magnetized plasmas, using 4th-order Runge-Kutta time integration with a Conjugate Gradient solver for the Helmholtz equation:",
            "",
            "**Wall Time Constraint**: To prevent runaway simulations, a configurable wall time limit (default: 120 seconds) is enforced. Simulations that exceed this limit are terminated early and flagged as incomplete via the function call.",
            "",
            "**Governing equation:**",
            r"$$\frac{\partial q}{\partial t} + v_* \frac{\partial \phi}{\partial y} = 0$$",
            "",
            "Where:",
            r"$$q = \nabla^2 \phi - \phi$$",
            "",
            "**Physical variables:**",
            "",
            "- $\\phi$ = electrostatic potential",
            "- $q$ = generalized vorticity",
            "- $v_*$ = diamagnetic drift velocity",
            "- Domain is periodic in both x and y directions",
            "",
            "### Time Integration",
            "",
            "4th-order Runge-Kutta (RK4) method for temporal discretization:",
            "",
            r"$$q^{n+1} = q^n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$",
            "",
            "where $k_i$ are the RK4 stage evaluations of the RHS.",
            "",
            "**Simulation Duration:**",
            "",
            "- Recording interval: `record_dt = 1000.0` time units",
            "- Number of frames: `end_frame = 10`",
            r"- Total simulation time: $T = \text{record\_dt} \times \text{end\_frame} = 10{,}000$ time units",
            r"- Number of time steps: $N_{\text{steps}} = T / \Delta t$",
            "",
            "### Spatial Discretization",
            "",
            "Finite differences with periodic boundary conditions on a uniform 2D grid:",
            "",
            r"- Grid spacing: $\Delta x = \Delta y = L / N$",
            r"- 2D Laplacian operator: $\nabla^2 \phi$ discretized using 5-point stencil",
            r"- Derivative operator: $\partial \phi / \partial y$ using central differences",
            "",
            "### Helmholtz Solver",
            "",
            r"At each RK4 stage, solve the Helmholtz equation for $\phi$ given $q$:",
            "",
            r"$$(\nabla^2 - I)\phi = q$$",
            "",
            "Solved using sparse Conjugate Gradient (CG) method with:",
            "",
            "- **cg_atol**: Absolute tolerance for convergence",
            "- **cg_maxiter**: Maximum CG iterations (fixed to be: 1000)",
            "",
            "### Analytical Solution",
            "",
            "For validation, an exact spectral solution via 2D FFT is available:",
            "",
            r"$$\phi(t) = \mathcal{F}^{-1}\left[\hat{\phi}_0 \exp\left(i \frac{v_* k_y t}{1 + k^2}\right)\right]$$",
            "",
            r"where $\mathcal{F}$ denotes Fourier transform and $k^2 = k_x^2 + k_y^2$.",
            "",
            "## Test Cases",
            "",
            "The case key in the config file sets different initial conditions:",
            "",
        ]

        # Add profile-specific test case description
        # Note: params has flat structure, no envs_params
        case = params["case"]
        L = float(params["L"])
        v_star = float(params["v_star"])
        Dx = float(params["Dx"])

        if case == "monopole" or profile == "p1":
            question_lines.extend([
                "### Current Configuration: p1 - Monopole",
                "",
                "**Gaussian monopole centered in domain:**",
                "",
                r"$$\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2 + (y-L/2)^2}{2D_x^2}\right)$$",
                "",
                f"- **Domain size**: L = {L:.2f} (2π × 10)",
                f"- **Diamagnetic drift velocity**: v* = {v_star}",
                f"- **Initial condition spatial scale**: Dx = {Dx}",
            ])
        elif case == "dipole" or profile == "p2":
            question_lines.extend([
                "### Current Configuration: p2 - Dipole",
                "",
                "**Gaussian dipole (odd in x):**",
                "",
                r"$$\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2 + (y-L/2)^2}{2D_x^2}\right) \cdot \frac{x-L/2}{D_x}$$",
                "",
                f"- **Domain size**: L = {L:.2f} (2π × 10)",
                f"- **Diamagnetic drift velocity**: v* = {v_star}",
                f"- **Initial condition spatial scale**: Dx = {Dx}",
            ])
        elif case == "sin_x_gauss_y" or profile == "p3":
            question_lines.extend([
                "### Current Configuration: p3 - sin_x_gauss_y",
                "",
                "**Sinusoidal in x, Gaussian in y:**",
                "",
                r"$$\phi_0 = 0.1 \sin(0.2x) \exp\left(-\frac{(y-L/2)^2}{2D_x^2}\right)$$",
                "",
                f"- **Domain size**: L = {L:.2f} (2π × 10)",
                f"- **Diamagnetic drift velocity**: v* = {v_star}",
                f"- **Initial condition spatial scale**: Dx = {Dx}",
            ])
        elif case == "gauss_x_sin_y" or profile == "p4":
            question_lines.extend([
                "### Current Configuration: p4 - gauss_x_sin_y",
                "",
                "**Gaussian in x, sinusoidal in y:**",
                "",
                r"$$\phi_0 = 0.1 \exp\left(-\frac{(x-L/2)^2}{2D_x^2}\right) \sin(0.2y)$$",
                "",
                f"- **Domain size**: L = {L:.2f} (2π × 10)",
                f"- **Diamagnetic drift velocity**: v* = {v_star}",
                f"- **Initial condition spatial scale**: Dx = {Dx}",
            ])
        else:
            # Fallback for unknown case
            question_lines.extend([

                f"### Current Configuration: {profile} - {case}",
                "",
                "- Custom simulation configuration",
                f"- **Domain size**: L = {L}",
                f"- **Diamagnetic drift velocity**: v* = {v_star}",
            ])

        question_lines.extend([
            "",
            "The simulated results are considered correct if the L2 RMSE meets the precision-dependent tolerance compared to the analytical solution.",
            "",
            "### Convergence Method",
            "",
            "Convergence is verified by comparing numerical solution against the analytical (spectral) solution:",
            "",
            "- Run numerical simulation with given parameters (N, dt, cg_atol)",
            "- Run analytical solution using 2D FFT (exact, serves as reference)",
            "- Calculate L2 RMSE between numerical and analytical solutions across all frames",
            "- Converged if RMSE < tolerance threshold",
            "",
            "### Cost Calculation",
            "",
            "Computational cost for the numerical method is estimated as:",
            "",
            r"$$\text{Cost} = N_{\text{CG}} \times N^2 + N_{\text{matvec}} \times N^2$$",
            "",
            "where:",
            r"- $N_{\text{CG}}$ = total CG iterations across all time steps",
            r"- $N_{\text{matvec}}$ = total sparse matrix-vector multiply operations",
            r"- Each CG iteration and matvec operation costs roughly $O(N^2)$",
            "",
            "## Parameter Tuning Tasks",
            "",
            "### Tasks",
            "",
            "1. **N Spatial Grid Number**",
            "   - N is the grid resolution: $\\Delta x = \\Delta y = L / N$, where $L$ is domain size",
            "",
            "2. **dt Time Step Size**",
            "   - dt is the time step for RK4 integration",
            "",
            "3. **cg_atol**",
            "   - Absolute residual threshold for Conjugate Gradient solver convergence",
            "",
            f"**Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Tolerance RMSE**: ≤ {precision_config['tolerance_rmse']}",
        ])

        return "\n".join(question_lines)

    def _save_dataset(
        self,
        dataset: List[Dict],
        task: str,
        precision_level: str,
        zero_shot: bool
    ) -> None:
        """Save dataset to the new path format: data/hasegawa_mima_linear/{task}/{precision_level}/"""
        base_dir = Path("data") / "hasegawa_mima_linear" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename

        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all Hasegawa-Mima linear questions for all tasks and precision levels"""
    print("HASEGAWA-MIMA LINEAR QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/hasegawa_mima_linear/{{task}}/{{precision_level}}/")
    print(f"Tasks (all support iterative + zero-shot): N, dt, cg_atol")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types:")
    print(f"  - iterative_questions.json (for all tasks)")
    print(f"  - zero_shot_questions.json (for all tasks)")

    gen = HasegawaMimaLinearQuestionGenerator()
    gen.generate_all_questions()

    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()
