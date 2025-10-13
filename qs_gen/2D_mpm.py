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
MPM_2D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "unstruct_mpm"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "unstruct_mpm"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "high": {
        "energy_tolerance": 0.003,
        "var_threshold": 0.015,
        "description": "High precision energy convergence - very challenging"
    },
    "medium": {
        "energy_tolerance": 0.01,
        "var_threshold": 0.02,
        "description": "Medium precision energy convergence - challenging"
    },
    "low": {
        "energy_tolerance": 0.03,
        "var_threshold": 0.05,
        "description": "Low precision energy convergence - achievable"
    },
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class TwoDMPMQuestionGenerator:
    """Generate MPM 2D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()

    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _infer_precision_level(self, precision_config: Dict[str, float]) -> str:
        """Infer precision level from precision_config values

        Since tasks.json doesn't have precision_level field, we need to infer it
        from the energy_tolerance and var_threshold values.
        """
        energy_tol = precision_config.get("energy_tolerance")
        var_thresh = precision_config.get("var_threshold")

        # Match against known precision levels
        for level_name, level_config in PRECISION_LEVELS.items():
            if (level_config["energy_tolerance"] == energy_tol and
                level_config["var_threshold"] == var_thresh):
                return level_name

        # Fallback: if exact match not found, raise error for clarity
        raise ValueError(
            f"Unknown precision level: energy_tolerance={energy_tol}, "
            f"var_threshold={var_thresh}"
        )

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["nx", "n_part", "cfl"]
        # All three tasks support both iterative and zero-shot modes

        total_files = 0
        total_questions = 0

        print("=" * 80)
        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)

            for precision_level in PRECISION_LEVELS.keys():
                precision_config = PRECISION_LEVELS[precision_level]
                energy_tol = precision_config["energy_tolerance"]
                var_thresh = precision_config["var_threshold"]

                print(f"  {precision_level.upper()} precision:")
                print(f"    Energy tolerance: {energy_tol}, Var threshold: {var_thresh}")

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
                # Infer precision level from precision_config
                inferred_level = self._infer_precision_level(t["precision_config"])
                if inferred_level == precision_level:
                    matching_tasks.append(t)

        qid = 1
        for task_data in matching_tasks:
            profile = task_data["profile"]

            # Load profile configuration
            profile_path = MPM_2D_CFG_DIR / f"{profile}.yaml"
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

            # Filter best_params to only include the 3 tunable parameters
            filtered_best_params = {
                key: best_params[key] for key in ["nx", "n_part", "cfl"]
                if key in best_params
            }

            # Filter param_history to only include the 3 tunable parameters
            filtered_param_history = []
            for param_set in param_history:
                filtered_set = {
                    key: param_set[key] for key in ["nx", "n_part", "cfl"]
                    if key in param_set
                }
                filtered_param_history.append(filtered_set)

            # Extract case from params for storage
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
        """Build question text from unstruct_mpm.md documentation

        Excludes Dummy Strategy section and developer-only content.
        """
        question_lines = [
            "# Unstructured Material Point Method (MPM) with Taichi-based Particle Simulation",
            "",
            "## Introduction",
            "",
            "This simulation solves solid mechanics problems using the **explicit** Material Point Method (MPM) on unstructured mesh. The MPM is a hybrid Eulerian-Lagrangian method that combines the advantages of both approaches for simulating large deformation problems in solid mechanics.",
            "",
            "**Governing Equations:**",
            "",
            "The MPM solves the momentum conservation equation:",
            "",
            r"$$\frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla \mathbf{v} = \frac{1}{\rho} \nabla \cdot \boldsymbol{\sigma} + \mathbf{g}$$",
            "",
            "Where:",
            "",
            "- $\\mathbf{v}$ = velocity field",
            "- $\\rho$ = density",
            "- $\\boldsymbol{\\sigma}$ = stress tensor",
            "- $\\mathbf{g}$ = gravitational acceleration",
            "",
            "**Constitutive Relations:**",
            "",
            "The stress tensor is computed using **Corotational Elasticity**:",
            "",
            r"$$\boldsymbol{\sigma} = 2\mu (\mathbf{F} - \mathbf{R}) \mathbf{F}^T + \lambda J(J-1)\mathbf{I}$$",
            "",
            "Where:",
            "",
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
            "The unstructured MPM uses a background triangle mesh for solving the momentum equation and Lagrangian particles for tracking material properties:",
            "",
            "1. **Particle-to-Grid Transfer**: Material properties are transferred from particles to mesh vertices",
            "2. **Grid Update**: Momentum equation is solved on the mesh",
            "3. **Grid-to-Particle Transfer**: Updated velocities are transferred back to particles",
            "4. **Particle Update**: Particles are advected and their properties updated",
            "",
            "### Temporal Discretization",
            "",
            "The time integration uses an explicit scheme with CFL condition for stability:",
            "",
            r"$$\Delta t = \frac{\text{CFL}}{v_{\text{max,init}} / \Delta x}$$",
            "",
            "Where:",
            "",
            "- $v_{\\text{max,init}}$ = initial maximum velocity (upper bound, as the system has no new input forces)",
            "- $\\Delta x$ = characteristic grid spacing",
            "- **Note**: this solver uses **explicit** material point methods and uses the initial maximum velocity to calcualted the dt, which is later as fixed for stability.",
            "",
            "## Test Cases",
            "",
        ]

        # Add profile-specific test case description
        case = params["case"]
        envs_params = params["envs_params"]

        if case == "cantilever" or profile == "p1":
            Lx = float(envs_params["Lx"])
            Ly = float(envs_params["Ly"])
            E = float(envs_params["E"])
            nu = float(envs_params["nu"])
            rho = float(envs_params["p_rho"])
            gravity = float(envs_params["gravity"])
            end_time = float(envs_params["end_time"])

            question_lines.extend([
                "### Current Configuration: p1 - Cantilever Beam Simulation",
                "",
                f"- **Domain**: {Lx} × {Ly} m",
                f"- **Material**: E = {E:.1e} Pa, ν = {nu}, ρ = {rho} kg/m³",
                f"- **Gravity**: {gravity} m/s²",
                "- **Initial conditions**: Beam fixed at left edge (x=0 to 1 m), positioned at y=5 to 7 m, initially at rest (v=0)",
                "- **Initial max velocity**: 0.5 m/s (used for CFL calculation)",
                f"- **End time**: {end_time} s",
                "- **Tests**: Beam bending and large deformation under gravity",
            ])
        elif case == "vibration_bar" or profile == "p2":
            Lx = float(envs_params["Lx"])
            Ly = float(envs_params["Ly"])
            Lxe = float(envs_params["Lxe"])
            Lye = float(envs_params["Lye"])
            E = float(envs_params["E"])
            nu = float(envs_params["nu"])
            rho = float(envs_params["p_rho"])
            gravity = float(envs_params["gravity"])
            end_time = float(envs_params["end_time"])

            question_lines.extend([
                "### Current Configuration: p2 - Vibration Bar Simulation",
                "",
                f"- **Domain**: {Lx} × {Ly} m (effective material region: {Lxe} × {Lye} m starting at x=5 m)",
                f"- **Material**: E = {E} Pa, ν = {nu}, ρ = {rho} kg/m³",
                f"- **Gravity**: {gravity} (no gravity)",
                "- **Initial conditions**: Sinusoidal velocity field $v_x = 0.75 \\sin(0.5\\pi x/L_x)$ where $L_x$ is effective length",
                "- **Initial max velocity**: 1.0 m/s (used for CFL calculation)",
                f"- **End time**: {end_time} s",
                "- **Tests**: Elastic wave propagation and vibration modes",
            ])
        elif case == "disk_collision" or profile == "p3":
            Lx = float(envs_params["Lx"])
            Ly = float(envs_params["Ly"])
            R = float(envs_params["R"])
            E = float(envs_params["E"])
            nu = float(envs_params["nu"])
            rho = float(envs_params["p_rho"])
            gravity = float(envs_params["gravity"])
            end_time = float(envs_params["end_time"])

            question_lines.extend([
                "### Current Configuration: p3 - Disk Collision Simulation",
                "",
                f"- **Domain**: {Lx} × {Ly} m",
                f"- **Material**: E = {E} Pa, ν = {nu}, ρ = {rho} kg/m³",
                f"- **Gravity**: {gravity} (no gravity)",
                f"- **Initial conditions**: Two disks (radius R={R} m) moving toward each other with velocities (0.1, 0.1) and (-0.1, -0.1) m/s",
                "- **Initial max velocity**: 0.025 m/s (used for CFL calculation, accounting for collision dynamics)",
                f"- **End time**: {end_time} s",
                "- **Tests**: Impact dynamics and contact resolution",
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
            "   - Total energy variation over time using hybrid approach:",
            "     - When mean energy is small (< 1e-10): $\\text{var} = \\sigma(E_{\\text{tot}}) < \\text{var\\_threshold}$ (absolute threshold)",
            "     - When mean energy is significant: $\\text{var} = \\frac{\\sigma(E_{\\text{tot}})}{\\text{mean}(|E_{\\text{tot}}|)} < \\text{var\\_threshold}$ (relative threshold)",
            "   - Where $E_{\\text{tot}} = E_{\\text{kinetic}} + E_{\\text{potential}} + E_{\\text{gravitational}}$",
            "   - Threshold depends on precision level (high: 0.015, medium: 0.02, low: 0.05)",
            "   - Special handling for **disk_collision** case: only checks first 50% of time horizon as in this case the disks will collide each other causing diffusion.",
            "",
            "2. **Positivity Preservation**: kinetic and elastic potential energies must be ≥ 0",
            "",
            "3. **Wall Time Limit**: Simulation must complete within 600 seconds (10 minutes).",
            "   - This is checked internally by the solver: if the simulation exceeds the wall time limit during execution, it is immediately marked as **not converged** and stopped.",
            "",
            "### Comparison Metrics (Between Adjacent Parameter Sets)",
            "",
            "When comparing two simulations with different parameter values:",
            "",
            "1. **Energy L2 Relative Difference**:",
            "   - For each energy type (pot, kin, gra): $\\text{diff}_i = \\frac{||E_1^i - E_2^i||_2}{||E_1^i||_2 + ||E_2^i||_2 + \\epsilon}$",
            "   - Average: $\\text{avg\\_diff} = \\text{mean}(\\text{diff}_{\\text{pot}}, \\text{diff}_{\\text{kin}}, \\text{diff}_{\\text{gra}})$",
            "   - Note: Total energy (tot) is excluded from comparison to focus on individual energy components",
            "",
            "2. **Convergence Criterion**:",
            "   - $\\text{converged} = (\\text{avg\\_diff} < \\text{energy\\_tolerance}) \\land \\text{energy\\_conserved}_1 \\land \\text{energy\\_conserved}_2$",
            "   - Where energy_tolerance depends on precision level",
            "",
            "## Parameter Tuning Tasks",
            "",
            "### Tasks",
            "",
            "1. **nx Grid Resolution Search**",
            "   - nx controls the background grid resolution: $\\Delta x = L / nx$ where $L$ is domain length",
            "",
            "2. **n_part Particle Density Search**",
            "   - n_part controls the number of particles per grid cell",
            "",
            "3. **CFL Stability Search**",
            "   - CFL number controls time step size for temporal stability",
            "",
            "### Cost Calculation",
            "",
            "The computational cost is calculated as:",
            "",
            r"$$\text{Total Cost} = \sum_{\text{iteraton}}n_{\text{particles}} + \sum_{\text{each particle}} n_{\text{neighbor communications}}$$",
            "",
            "Where:",
            "",
            "- $n_{\\text{particles}}$ = total number of particles in the simulation",
            "- $n_{\\text{neighbor communications}}$ = number of particle-particle interactions for each particle within the support radius",
            "",
            "This cost metric captures:",
            "",
            "1. **Particle Density Cost**: Scales with total number of particles",
            "2. **Neighbor Communication Cost**: Accounts for particle-particle interactions within support radius (controlled by `radii` parameter, fixed at 1.0)",
            "3. **Spatial Complexity**: Reflects both discretization density (via `nx` and `n_part`) and interaction complexity (via neighbor search on unstructured mesh)",
            "",
            f"**Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Energy Tolerance**: ≤ {precision_config['energy_tolerance']}",
            f"- **Variance Threshold**: ≤ {precision_config['var_threshold']}",
            f"- **Description**: {precision_config['description']}"
        ])

        return "\n".join(question_lines)

    def _save_dataset(
        self,
        dataset: List[Dict],
        task: str,
        precision_level: str,
        zero_shot: bool
    ) -> None:
        """Save dataset to the new path format: data/mpm_2d/{task}/{precision_level}/"""
        base_dir = Path("data") / "mpm_2d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename

        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all MPM 2D questions for all tasks and precision levels"""
    print("2D MPM (MATERIAL POINT METHOD) QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/mpm_2d/{{task}}/{{precision_level}}/")
    print(f"Tasks (all support iterative + zero-shot): nx, n_part, cfl")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types:")
    print(f"  - iterative_questions.json (for all tasks)")
    print(f"  - zero_shot_questions.json (for all tasks)")

    gen = TwoDMPMQuestionGenerator()
    gen.generate_all_questions()

    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()
