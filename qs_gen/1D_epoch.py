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
EPOCH_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "epoch"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "epoch"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "low": {"tolerance_rmse": 0.36},
    "medium": {"tolerance_rmse": 0.33},
    "high": {"tolerance_rmse": 0.30},
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class OneDEpochQuestionGenerator:
    """Generate Epoch-1D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()

    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["dt_multiplier", "nx", "npart", "field_order", "particle_order"]
        total_files = 0
        total_questions = 0

        print("=" * 80)
        for task in tasks:
            print(f"\n📋 TASK: {task.upper()}")
            print("-" * 50)

            for precision_level in PRECISION_LEVELS.keys():
                tolerance = PRECISION_LEVELS[precision_level]["tolerance_rmse"]
                print(f"  🎯 {precision_level.upper()} precision (RMSE ≤ {tolerance}):")

                # Generate iterative questions
                dataset_iter = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=False)
                self._save_dataset(dataset_iter, task=task, precision_level=precision_level, zero_shot=False)

                # Generate zero-shot questions
                dataset_zero = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=True)
                self._save_dataset(dataset_zero, task=task, precision_level=precision_level, zero_shot=True)

                total_files += 2
                total_questions += len(dataset_iter) + len(dataset_zero)
                print(f"     ✓ Iterative: {len(dataset_iter):2d} questions")
                print(f"     ✓ Zero-shot: {len(dataset_zero):2d} questions")

        print("\n" + "=" * 80)
        print(f"🎉 SUMMARY: Generated {total_questions} questions across {total_files} files")
        print("=" * 80)

    def generate_question_dataset(self, task: str, precision_level: str, zero_shot: bool) -> List[Dict]:
        """Generate question dataset for specific task and precision level"""
        dataset: List[Dict] = []
        tolerance_rmse = PRECISION_LEVELS[precision_level]["tolerance_rmse"]

        # Handle mapping from correct naming to data source naming
        data_task_name = "dt_multipler" if task == "dt_multiplier" else task

        # Filter tasks by target parameter and precision level
        matching_tasks = [
            t for t in self.tasks_data["tasks"]
            if (t["target_parameter"] == data_task_name and
                t["precision_config"]["tolerance_rmse"] == tolerance_rmse)
        ]

        qid = 1
        for task_data in matching_tasks:
            profile = task_data["profile"]

            # Load profile configuration
            profile_path = EPOCH_CFG_DIR / f"{profile}.yaml"
            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            # Extract results
            results = task_data["results"]
            cost_history = results["cost_history"]
            param_history = results["parameter_history"]
            is_converged = results["converged"]

            # Calculate dummy cost according to requirements
            dummy_cost = cost_history[-2] if zero_shot else sum(cost_history[:-1])

            # Find the best_params from parameter_history based on optimal_parameter_value
            optimal_value = results["optimal_parameter_value"]
            best_params = None

            # Handle different optimal_parameter_value formats:
            # - dt_multiplier: list format [param_value, resolution]
            # - nx, npart, field_order, particle_order: single numeric value
            if task == "dt_multiplier":
                # For dt_multiplier, optimal_value is [param_value, resolution]
                target_param_value = optimal_value[0] if isinstance(optimal_value, list) else optimal_value
            else:
                # For other parameters, optimal_value is a single number
                target_param_value = optimal_value

            # Search through parameter_history to find the matching parameter combination
            for param_set in param_history:
                if isinstance(param_set, list):
                    # For tasks like dt_multiplier/field_order/particle_order, param_history is 2D: [[{params}, {params}], ...]
                    # We want the last element (highest resolution) of each iteration
                    param_dict = param_set[-1]
                else:
                    # For tasks like nx/npart, param_history is 1D: [{params}, {params}, ...]
                    param_dict = param_set

                # Map parameter names for epoch solver
                param_name_mapping = {
                    "dt_multiplier": "dt_mult",
                    "nx": "nx",
                    "npart": "npart",
                    "field_order": "field_Order",
                    "particle_order": "particle_order"
                }

                mapped_param_name = param_name_mapping.get(task, task)
                if param_dict.get(mapped_param_name) == target_param_value:
                    best_params = param_dict.copy()
                    break

            # If not found, this is an error in the data
            if best_params is None:
                raise ValueError(f"Could not find optimal parameter {task}={target_param_value} in parameter_history")

            # Build question entry
            question_entry = {
                "QID": qid,
                "profile": profile,
                "zero_shot": zero_shot,
                "target_parameter": task,
                "precision_level": precision_level,
                "tolerance_rmse": tolerance_rmse,
                "is_converged": is_converged,
                "dummy_cost": dummy_cost,
                "cost_history": cost_history,
                "best_params": best_params,
                "param_history": param_history,
                "question": self._build_question_text(params, precision_level, tolerance_rmse),
            }

            dataset.append(question_entry)
            qid += 1

        return dataset

    @staticmethod
    def _build_question_text(params: dict, precision_level: str, tolerance_rmse: float) -> str:
        """Build question text for Epoch PIC simulation"""
        question_lines = [
            "Problem: EPOCH 1D Simulation with Ionisation",
            "",
            "This simulation utilizes a Particle in Cell (PIC) code called EPOCH to solve the 1D ionisation of a Carbon target irradiated with an ultra-intense laser pulse.",
            "",
            "Initially the domain is loaded with pseudo-particles (that represent a number of real particles). The number of particles that are initially loaded into each cell is determined by the parameter npart.",
            "",
            "The general procedure that EPOCH and PIC methods follows:",
            "",
            "1. Based on the particles locations and velocities (allowing for the determination of rho and j, where the current is calculated via drho/dt + div(j) = 0 for charge conservation) interpolate onto the discrete grid determined by the parameter nx (which determines the number of grid cells). How the contributions of each particles are loaded onto the grid are based on the parameter particle_order (changing the shape of how many neighboring cells' properties are influenced by a particle in a specific cell due to the use of pseudo-particles).",
            "",
            "2. Based on the interpolated rho and j, Maxwell's Equations can be solved to find the electromagnetic fields on the grid.",
            "",
            "Maxwell's Equations:",
            r"$\nabla \cdot \vec{E}=\frac{\rho}{\epsilon_0}$",
            "",
            r"$\nabla \cdot \vec{B}=0$",
            "",
            r"$\nabla \times \vec{E}=-\frac{\partial \vec{B}}{\partial t}$",
            "",
            r"$\nabla \times \vec{B}=\mu_0(\vec{j}+\epsilon_0\frac{\partial \vec{E}}{\partial t})$",
            "",
            "Solving these equations are done with a finite difference method controlled by the parameter field_order (determining how many points to use for each grid cell's field calculation), along with a timestep determined by the parameter dt_multiplier.",
            "",
            "3. With the values of E(x,t) and B(x,t), the forces on the particles can be determined during the particle push as follows.",
            "",
            "Particle Equations of Motion:",
            r"$\frac{d\vec{p}}{dt}=q(\vec{E} +\vec{v} \times \vec{B} )$",
            "",
            r"$\frac{d\vec{x}}{dt}=\vec{v}=\frac{\vec{p}}{\gamma m}$",
            "",
            r"$\gamma=\sqrt{1+(\frac{|\vec{p}|}{mc})^2}$",
            "",
            "Solving these equations are done with a finite difference method in time, where the finite time step is determined by the parameter dt_multiplier (along with the grid size).",
            "",
            "In this case, field ionisation is used to ionise the carbon target, where the carbon is ionised from the effects of the laser. When the simulation detects the necessary energy conditions for a pseudoparticle of Carbon to be ionised, it changes the charge of that Carbon particle as necessary and generates a new pseudoparticle for the ejected electron.",
            "",
            "Physical Parameters:",
            f"- Domain length (L): {params['L']} μm",
            f"- Carbon target thickness (L_target): {params['L_target']} μm",
            f"- Normalized laser amplitude (a0): {params['a0']}",
            f"- Laser wavelength (laser_lambda): {params['laser_lambda']} μm",
            f"- Laser duration (laser_time): {params['laser_time']} fs",
            f"- Target density (n_target): {params['n_target']} n_crit",
            "",
            # "where:",
            # r"- The normalized laser amplitude $a_0=|e|E_0/m_ec\omega_0$ (|e| is the elementary charge of an electron, E_0 is the amplitude of the laser, m_e is the mass of the electron, c is the speed of light in a vacuum and ω_0=2πc/λ_0 is the laser's frequency with wavelength λ_0)",
            # r"- The critical plasma density $n_{cr}=\omega_0^2 m_e \epsilon_0 / |e|^2$",
            "",
            "Tunable Parameters:",
            "- **nx**: Number of grid points used in the 1D simulation, where nx is an integer. Defines the number of grid cells for spatial discretization.",
            "",
            "- **dt_multiplier**: EPOCH uses a CFL-based time step calculation.",
            "",
            "- **npart**: In the \"species\" block (listed as npart_per_cell), which defines the number of pseudoparticles to use in each cell. For this simulation, the only initial species is unionized carbon.",
            "",
            "- **field_order**: In the control block, which defines the finite difference method used to solve Maxwell's Equations during each step.",
            "",
            "- **particle_order**: Defines the particle weighting for the macro-particles.",
            "",
            "Convergence Check:",
            "- Errors between the simulation based on your solution and the simulation based on the self-refined solution are computed to assess convergence.",
            "- Convergence is confirmed if the following validation criteria are satisfied.",
            "",
            "Validation Criteria:",
            f"- **Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Required RMSE Tolerance**: ≤ {tolerance_rmse}",
            "- Relative RMSE must meet this tolerance compared to self-refined solution",
            "- Physical consistency: charge conservation and electromagnetic field evolution within acceptable bounds",
            "- Numerical stability: proper ionization dynamics and particle behavior"
        ]
        return "\n".join(question_lines)

    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        """Save dataset to the new path format: data/epoch_1d/{task}/{precision_level}/"""
        base_dir = Path("data") / "epoch_1d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename

        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all Epoch questions for all tasks and precision levels"""
    print("🚀 EPOCH PIC QUESTION GENERATOR")
    print("=" * 80)
    print(f"📂 Output directory: data/epoch_1d/{{task}}/{{precision_level}}/")
    print(f"📋 Tasks: dt_multiplier, nx, npart, field_order, particle_order")
    print(f"🎯 Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"📄 File types: iterative_questions.json, zero_shot_questions.json")

    gen = OneDEpochQuestionGenerator()
    gen.generate_all_questions()

    print(f"\n✅ All questions generated successfully!")


if __name__ == "__main__":
    main()