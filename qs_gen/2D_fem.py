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
FEM_2D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "fem2d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "fem2d"
    / "successful"
    / "tasks.json"
)

# Profile-specific precision levels for FEM2D
# Structure: PRECISION_LEVELS[profile][level_name]
PRECISION_LEVELS = {
    "p1": {
        "high": {
            "energy_tolerance": 0.005,
            "var_threshold": 0.015,
            "description": "High precision energy convergence for cantilever beam"
        },
        "medium": {
            "energy_tolerance": 0.010,
            "var_threshold": 0.030,
            "description": "Medium precision energy convergence for cantilever beam"
        },
        "low": {
            "energy_tolerance": 0.020,
            "var_threshold": 0.060,
            "description": "Low precision energy convergence for cantilever beam"
        },
    },
    "p2": {
        "high": {
            "energy_tolerance": 0.005,
            "var_threshold": 0.030,
            "description": "High precision energy convergence for vibration bar"
        },
        "medium": {
            "energy_tolerance": 0.010,
            "var_threshold": 0.060,
            "description": "Medium precision energy convergence for vibration bar"
        },
        "low": {
            "energy_tolerance": 0.020,
            "var_threshold": 0.120,
            "description": "Low precision energy convergence for vibration bar"
        },
    },
    "p3": {
        "high": {
            "energy_tolerance": 0.010,
            "var_threshold": 0.040,
            "description": "High precision energy convergence for twisting column"
        },
        "medium": {
            "energy_tolerance": 0.020,
            "var_threshold": 0.080,
            "description": "Medium precision energy convergence for twisting column"
        },
        "low": {
            "energy_tolerance": 0.040,
            "var_threshold": 0.160,
            "description": "Low precision energy convergence for twisting column"
        },
    },
    "p4": {
        "high": {
            "energy_tolerance": 0.004,
            "var_threshold": 0.012,
            "description": "High precision energy convergence for gentle vibration bar"
        },
        "medium": {
            "energy_tolerance": 0.008,
            "var_threshold": 0.024,
            "description": "Medium precision energy convergence for gentle vibration bar"
        },
        "low": {
            "energy_tolerance": 0.016,
            "var_threshold": 0.048,
            "description": "Low precision energy convergence for gentle vibration bar"
        },
    },
    "p5": {
        "high": {
            "energy_tolerance": 0.015,
            "var_threshold": 0.060,
            "description": "High precision energy convergence for strong twisting column"
        },
        "medium": {
            "energy_tolerance": 0.030,
            "var_threshold": 0.120,
            "description": "Medium precision energy convergence for strong twisting column"
        },
        "low": {
            "energy_tolerance": 0.060,
            "var_threshold": 0.240,
            "description": "Low precision energy convergence for strong twisting column"
        },
    },
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class TwoDFEMQuestionGenerator:
    """Generate FEM 2D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()

    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _infer_precision_level(self, profile: str, precision_config: Dict[str, float]) -> str:
        """Infer precision level from profile and precision_config values

        Args:
            profile: Profile name (p1-p5)
            precision_config: Dict with energy_tolerance and var_threshold

        Returns:
            Precision level name (high/medium/low)

        Raises:
            KeyError: If profile is not found
            ValueError: If precision_config doesn't match any known level for the profile
        """
        # Fail fast: check if profile exists
        if profile not in PRECISION_LEVELS:
            raise KeyError(
                f"Unknown profile '{profile}'. "
                f"Available profiles: {list(PRECISION_LEVELS.keys())}"
            )

        # Extract values (fail fast if keys are missing)
        try:
            energy_tol = precision_config["energy_tolerance"]
            var_thresh = precision_config["var_threshold"]
        except KeyError as e:
            raise KeyError(
                f"Missing required key in precision_config: {e}. "
                f"Required keys: 'energy_tolerance', 'var_threshold'"
            )

        # Match against known precision levels for this profile
        for level_name, level_config in PRECISION_LEVELS[profile].items():
            if (level_config["energy_tolerance"] == energy_tol and
                level_config["var_threshold"] == var_thresh):
                return level_name

        # Fail fast: if exact match not found, raise error for clarity
        raise ValueError(
            f"Unknown precision level for profile {profile}: "
            f"energy_tolerance={energy_tol}, var_threshold={var_thresh}. "
            f"Available levels: {PRECISION_LEVELS[profile]}"
        )

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["dx", "cfl"]
        # Both tasks support both iterative and zero-shot modes

        total_files = 0
        total_questions = 0

        print("=" * 80)
        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)

            for precision_level in ["high", "medium", "low"]:
                print(f"  {precision_level.upper()} precision:")

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

        # Filter tasks by target parameter and inferred precision level
        matching_tasks = []
        for t in self.tasks_data["tasks"]:
            if t["target_parameter"] == task:
                profile = t["profile"]
                # Infer precision level from precision_config
                inferred_level = self._infer_precision_level(profile, t["precision_config"])
                if inferred_level == precision_level:
                    matching_tasks.append(t)

        qid = 1
        for task_data in matching_tasks:
            profile = task_data["profile"]

            # Get precision config for this profile and level
            precision_config = PRECISION_LEVELS[profile][precision_level]

            # Load profile configuration
            profile_path = FEM_2D_CFG_DIR / f"{profile}.yaml"
            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            # Extract results
            results = task_data["results"]
            cost_history = results["cost_history"]
            param_history = results["parameter_history"]
            is_converged = results["converged"]

            # Calculate dummy cost according to requirements
            # For zero-shot: use the second-to-last cost
            # For iterative: sum all costs except the last one
            if zero_shot and len(cost_history) > 1:
                dummy_cost = cost_history[-2]
            elif len(cost_history) > 1:
                dummy_cost = sum(cost_history[:-1])
            else:
                dummy_cost = cost_history[0]

            # Find the best_params from parameter_history based on optimal_parameter_value
            optimal_value = results["optimal_parameter_value"]
            best_params = None

            # Search through parameter_history to find the matching parameter combination
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

            # Filter best_params to only include the 2 tunable parameters
            filtered_best_params = {
                key: best_params[key] for key in ["dx", "cfl"]
                if key in best_params
            }

            # Filter param_history to only include the 2 tunable parameters
            filtered_param_history = []
            for param_set in param_history:
                filtered_set = {
                    key: param_set[key] for key in ["dx", "cfl"]
                    if key in param_set
                }
                filtered_param_history.append(filtered_set)

            # Extract case from params for storage
            # Fail fast if case is missing
            try:
                case = params["case"]
            except KeyError:
                raise KeyError(
                    f"Missing required 'case' field in {profile}.yaml. "
                    f"Please ensure all profile configs have a 'case' field."
                )

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
        """Build question text from fem2d.md documentation

        Excludes Dummy Strategy section and developer-only content.
        """
        question_lines = [
            "# 2D Finite Element Method (FEM) with Implicit Newton Solver",
            "",
            "## Introduction",
            "",
            "This simulation solves solid mechanics problems using the **implicit** Finite Element Method (FEM) with a Newton solver and line search for nonlinear elasticity. The FEM discretizes the continuous solid mechanics problem into a finite-dimensional system of equations solved at each time step.",
            "",
            "**Governing Equations:**",
            "",
            "The FEM solves the momentum conservation equation:",
            "",
            r"$$\rho \frac{\partial^2 \mathbf{u}}{\partial t^2} = \nabla \cdot \boldsymbol{\sigma} + \rho \mathbf{g}$$",
            "",
            "Where:",
            "",
            "- $\\mathbf{u}$ = displacement field",
            "- $\\rho$ = density",
            "- $\\boldsymbol{\\sigma}$ = stress tensor (Cauchy stress)",
            "- $\\mathbf{g}$ = gravitational acceleration",
            "",
            "**Constitutive Relations:**",
            "",
            "The stress tensor is computed using **Corotational Elasticity**:",
            "",
            r"$$\mathbf{P} = 2\mu (\mathbf{F} - \mathbf{R}) + \lambda (J - 1) \mathbf{F}^{-T}$$",
            "",
            "Where:",
            "",
            "- $\\mathbf{P}$ = first Piola-Kirchhoff stress tensor",
            "- $\\mathbf{F}$ = deformation gradient",
            "- $\\mathbf{R}$ = rotation matrix from polar decomposition ($\\mathbf{F} = \\mathbf{R}\\mathbf{S}$)",
            "- $\\mu = \\frac{E}{2(1+\\nu)}$ = shear modulus",
            "- $\\lambda = \\frac{E\\nu}{(1+\\nu)(1-2\\nu)}$ = Lame's first parameter",
            "- $J = \\det(\\mathbf{F})$ = volume ratio",
            "- $E$ = Young's modulus",
            "- $\\nu$ = Poisson's ratio",
            "",
            "### Spatial Discretization",
            "",
            "The FEM uses a triangle mesh to discretize the domain:",
            "",
            "1. **Element Formulation**: Each triangle element maps from reference configuration to current configuration using shape functions",
            "2. **Energy Functional**: Total potential energy is assembled from elastic strain energy and external work",
            "3. **Force Computation**: Element forces are computed from stress tensor via virtual work principle",
            "4. **Stiffness Matrix**: Element stiffness matrices are assembled into global system",
            "",
            "### Temporal Discretization",
            "",
            "The time integration uses an **implicit backward Euler scheme** with Newton-Raphson solver:",
            "",
            r"$$\mathbf{x}^{n+1} = \mathbf{x}^n + \Delta t \mathbf{v}^n + \Delta t^2 \mathbf{M}^{-1} \mathbf{f}(\mathbf{x}^{n+1})$$",
            "",
            "Where:",
            "",
            "- $\\mathbf{x}^{n+1}$ = position at next time step (unknown)",
            "- $\\mathbf{v}^n$ = velocity at current time step",
            "- $\\mathbf{M}$ = mass matrix",
            "- $\\mathbf{f}$ = internal + external forces",
            r"- **Newton Convergence Criterion**: $\frac{|\Delta \mathbf{x}|}{\Delta t} < \text{newton\_v\_res\_tol}$, where $\Delta \mathbf{x}$ is the position correction from Newton iteration",
            "",
            "The solver uses line search to ensure descent direction and improve convergence robustness.",
            "",
            "## Test Cases",
            "",
        ]

        # Add profile-specific test case description
        case = params["case"]
        envs_params = params["envs_params"]

        if case == "cantilever_beam" or profile == "p1":
            Lx = float(envs_params["Lx"])
            Ly = float(envs_params["Ly"])
            E = float(params["E"])
            nu = float(params["nu"])
            rho = float(params["density"])
            gravity = float(envs_params["gravity"])
            record_dt = float(params["record_dt"])
            end_frame = int(params["end_frame"])
            end_time = record_dt * end_frame

            question_lines.extend([
                "### Current Configuration: p1 - Cantilever Beam Simulation",
                "",
                f"- **Domain**: {Lx} × {Ly} m",
                f"- **Material**: E = {E:.1e} Pa, ν = {nu}, ρ = {rho} kg/m³",
                f"- **Gravity**: {gravity} m/s²",
                "- **Initial conditions**: Beam fixed at left edge (x < small tolerance), initially at rest (v=0)",
                f"- **End time**: {end_time} s",
                "- **Tests**: Cantilever beam bending under gravity with large deformation",
            ])
        elif case == "vibration_bar" and (profile == "p2" or profile == "p4"):
            Lx = float(envs_params["Lx"])
            Ly = float(envs_params["Ly"])
            x_start = float(envs_params["x_start"])
            E = float(params["E"])
            nu = float(params["nu"])
            rho = float(params["density"])
            gravity = float(envs_params["gravity"])
            initial_velocity_amplitude = float(envs_params["initial_velocity_amplitude"])
            record_dt = float(params["record_dt"])
            end_frame = int(params["end_frame"])
            end_time = record_dt * end_frame

            if profile == "p2":
                profile_desc = "p2 - Vibration Bar Simulation"
            else:  # p4
                profile_desc = "p4 - Gentle Vibration Bar Simulation"

            question_lines.extend([
                f"### Current Configuration: {profile_desc}",
                "",
                f"- **Domain**: {Lx} × {Ly} m (effective material region starts at x={x_start} m)",
                f"- **Material**: E = {E} Pa, ν = {nu}, ρ = {rho} kg/m³",
                f"- **Gravity**: {gravity} (no gravity - energy conserving)",
                f"- **Initial conditions**: Sinusoidal velocity field $v_x = {initial_velocity_amplitude} \\sin(0.5\\pi x/L_x)$ where $L_x$ is effective length, left edge fixed (x < x_start + small tolerance)",
                f"- **End time**: {end_time} s",
                "- **Tests**: 1D elastic wave propagation and compression dynamics",
            ])
        elif case == "twisting_column" and (profile == "p3" or profile == "p5"):
            Lx = float(envs_params["Lx"])
            Ly = float(envs_params["Ly"])
            E = float(params["E"])
            nu = float(params["nu"])
            rho = float(params["density"])
            gravity = float(envs_params["gravity"])
            initial_twist_amplitude = float(envs_params["initial_twist_amplitude"])
            record_dt = float(params["record_dt"])
            end_frame = int(params["end_frame"])
            end_time = record_dt * end_frame

            if profile == "p3":
                profile_desc = "p3 - Twisting Column Simulation"
            else:  # p5
                profile_desc = "p5 - Strong Twisting Column Simulation"

            question_lines.extend([
                f"### Current Configuration: {profile_desc}",
                "",
                f"- **Domain**: {Lx} × {Ly} m (tall column)",
                f"- **Material**: E = {E:.1e} Pa, ν = {nu}, ρ = {rho} kg/m³",
                f"- **Gravity**: {gravity} (no gravity - energy conserving)",
                f"- **Initial conditions**: Rotational velocity field around center, amplitude increases with height $y$, bottom edge fixed (y < small tolerance)",
                f"- **Initial velocity**: $v_x = -A \\cdot (y - y_c) \\cdot (y/L_y)$, $v_y = A \\cdot (x - x_c) \\cdot (y/L_y)$ where $A$ = {initial_twist_amplitude} m/s",
                f"- **End time**: {end_time} s",
                "- **Tests**: 2D rotational dynamics and energy conservation in twisting motion",
            ])
        else:
            # Fallback for unknown case
            question_lines.extend([
                f"### Current Configuration: {profile} - {case}",
                "",
                "- Custom simulation configuration",
            ])

        question_lines.extend([
            "",
            "## Convergence Metrics",
            "",
            "The simulated results are evaluated using two types of metrics:",
            "",
            "### Self-Checking Metrics (Individual Simulation Validation)",
            "",
            "1. **Energy Conservation**:",
            r"   - Total energy variation over time: $\text{var} = \frac{\sigma(E_{\text{tot}})}{\text{mean}(|E_{\text{tot}}|) + \epsilon} < \text{var\_threshold}$",
            "   - Where $E_{\\text{tot}} = E_{\\text{kinetic}} + E_{\\text{elastic\\_potential}}$",
            "   - Threshold depends on precision level",
            "",
            "2. **Positivity Preservation**: Kinetic and elastic potential energies must be ≥ 0",
            "",
            "### Comparison Metrics (Between Adjacent Parameter Sets)",
            "",
            "When comparing two simulations with different parameter values:",
            "",
            "1. **Energy L2 Relative Difference**:",
            "   - For each energy type (kin, pot): $\\text{diff}_i = \\frac{||E_1^i - E_2^i||_2}{||E_1^i||_2 + ||E_2^i||_2 + \\epsilon}$",
            "   - Average: $\\text{avg\\_diff} = \\text{mean}(\\text{diff}_{\\text{kin}}, \\text{diff}_{\\text{pot}})$",
            "",
            "2. **Convergence Criterion**:",
            r"   - $\text{converged} = (\text{avg\_diff} < \text{energy\_tolerance}) \land \text{energy\_conserved}_1 \land \text{energy\_conserved}_2 \land \text{positivity}_1 \land \text{positivity}_2$",
            "",
            "## Parameter Tuning Tasks",
            "",
            "### Tasks",
            "",
            "1. **dx Grid Resolution Search**",
            "   - dx controls the mesh resolution: element size in the x-direction",
            "",
            "2. **cfl Search**",
            "   - cfl controls the time step size for temporal discretization",
            "",
            f"**Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Energy Tolerance**: ≤ {precision_config['energy_tolerance']}",
            f"- **Variance Threshold**: ≤ {precision_config['var_threshold']}",
        ])

        return "\n".join(question_lines)

    def _save_dataset(
        self,
        dataset: List[Dict],
        task: str,
        precision_level: str,
        zero_shot: bool
    ) -> None:
        """Save dataset to the path format: data/fem_2d/{task}/{precision_level}/"""
        base_dir = Path("data") / "fem_2d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename

        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all FEM 2D questions for all tasks and precision levels"""
    print("2D FEM (FINITE ELEMENT METHOD) QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/fem_2d/{{task}}/{{precision_level}}/")
    print(f"Tasks (all support iterative + zero-shot): dx, cfl")
    print(f"Precision levels: high, medium, low (profile-specific thresholds)")
    print(f"File types:")
    print(f"  - iterative_questions.json (for all tasks)")
    print(f"  - zero_shot_questions.json (for all tasks)")

    gen = TwoDFEMQuestionGenerator()
    gen.generate_all_questions()

    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()
