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
HEAT1D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "heat_1d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "heat_1d"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "low": {"tolerance_rmse": 0.01, "description": "Relaxed convergence criteria"},
    "medium": {"tolerance_rmse": 0.001, "description": "Moderate convergence criteria"},
    "high": {"tolerance_rmse": 0.0001, "description": "Most stringent convergence criteria"},
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class OneDHeatTransferQuestionGenerator:
    """Generate Heat-1D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()
        
    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["cfl", "n_space"]
        total_files = 0
        total_questions = 0
        
        print("=" * 80)
        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)
            
            for precision_level in PRECISION_LEVELS.keys():
                tolerance = PRECISION_LEVELS[precision_level]["tolerance_rmse"]
                description = PRECISION_LEVELS[precision_level]["description"]
                print(f"  {precision_level.upper()} precision (RMSE <= {tolerance}):")
                
                # Generate iterative questions
                dataset_iter = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=False)
                self._save_dataset(dataset_iter, task=task, precision_level=precision_level, zero_shot=False)
                
                # Generate zero-shot questions  
                dataset_zero = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=True)
                self._save_dataset(dataset_zero, task=task, precision_level=precision_level, zero_shot=True)
                
                total_files += 2
                total_questions += len(dataset_iter) + len(dataset_zero)
                print(f"      Iterative: {len(dataset_iter):2d} questions")
                print(f"      Zero-shot: {len(dataset_zero):2d} questions")
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: Generated {total_questions} questions across {total_files} files")
        print("=" * 80)

    def generate_question_dataset(self, task: str, precision_level: str, zero_shot: bool) -> List[Dict]:
        """Generate question dataset for specific task and precision level"""
        dataset: List[Dict] = []
        tolerance_rmse = PRECISION_LEVELS[precision_level]["tolerance_rmse"]
        
        # Filter tasks by target parameter and precision level
        matching_tasks = [
            t for t in self.tasks_data["tasks"] 
            if (t["target_parameter"] == task and 
                t["precision_config"]["tolerance_rmse"] == tolerance_rmse)
        ]
        
        qid = 1
        for task_data in matching_tasks:
            profile = task_data["profile"]
            
            # Load profile configuration
            profile_path = HEAT1D_CFG_DIR / f"{profile}.yaml"
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
            
            # Search through parameter_history to find the matching parameter combination
            for param_set in param_history:
                if param_set.get(task) == optimal_value:
                    best_params = param_set.copy()
                    break
            
            # If not found, this is an error in the data
            if best_params is None:
                raise ValueError(f"Could not find optimal parameter {task}={optimal_value} in parameter_history")
            
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
        question_lines = [
            "Problem: 1D Heat Conduction with Explicit Finite Difference Method",
            "",
            "This simulation solves the 1D heat conduction equation with mixed boundary conditions using an explicit finite difference scheme:",
            "",
            "Heat Equation:",
            r"$\frac{\partial T}{\partial t} = \alpha \frac{\partial^2 T}{\partial x^2}$",
            "",
            "Where:",
            "- $T$ = temperature",
            r"- $\alpha = \frac{k}{\rho c_p}$ = thermal diffusivity",
            "- $k$ = thermal conductivity",
            "- $\\rho$ = density",
            "- $c_p$ = specific heat capacity",
            "",
            "Boundary Conditions:",
            r"- Left boundary (x=0): Convection to ambient temperature: $\frac{k}{\Delta x}(T_1 - T_0) = h(T_0 - T_{\infty})$",
            r"- Right boundary (x=L): Adiabatic: $\frac{\partial T}{\partial x} = 0$",
            "",
            "Numerical Discretization:",
            "The spatial discretization uses explicit finite differences:",
            "",
            r"$T_i^{n+1} = T_i^n + \frac{\alpha \Delta t}{(\Delta x)^2} (T_{i-1}^n - 2T_i^n + T_{i+1}^n)$",
            "",
            "The time step is constrained by the CFL condition for diffusion:",
            r"$\Delta t = \frac{\text{CFL} \cdot (\Delta x)^2}{2\alpha}$",
            "",
            "Boundary Treatment:",
            r"- **Left boundary (convection)**: $T_0 = \frac{\frac{\Delta x}{k} T_1 + h T_{\infty}}{\frac{\Delta x}{k} + h}$",
            "- **Right boundary (adiabatic)**: $T_{N} = T_{N-1}$",
            "",
            "Parameter Information:",
            "- cfl: Courant-Friedrichs-Lewy number for temporal stability in diffusion problems",
            "- n_space: Number of spatial grid points, determines spatial resolution: $\\Delta x = L / n\\_space$",
            "",
            "Physical Parameters:",
            f"- Wall thickness: {params['L']} m",
            f"- Thermal conductivity: {params['k']} W/m-K",
            f"- Convection coefficient: {params['h']} W/m²-K",
            f"- Density: {params['rho']} kg/m³",
            f"- Specific heat capacity: {params['cp']} J/kg-K",
            f"- Ambient temperature: {params['T_inf']} K",
            f"- Initial temperature: {params['T_init']} K",
            "",
            "Convergence Check:",
            "- Errors between the simulation based on your solution and the simulation based on the self-refined solution are computed to assess convergence.",
            "- Convergence is confirmed if the following validation criteria are satisfied.",
            "",
            "Validation Criteria:",
            f"- **Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Required RMSE Tolerance**: ≤ {tolerance_rmse}",
            "- Relative RMSE must meet this tolerance compared to self-refined solution",
            "- Temperature positivity: temperature values must remain physically reasonable",
            "- Heat flux consistency: boundary heat fluxes should satisfy conservation principles"
        ]
        return "\n".join(question_lines)
    
    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        """Save dataset to the new path format: data/heat_1d/{task}/{precision_level}/"""
        base_dir = Path("data") / "heat_1d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename
        
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all Heat 1D questions for all tasks and precision levels"""
    print("HEAT 1D QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/heat_1d/{{task}}/{{precision_level}}/")
    print(f"Tasks: cfl, n_space")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types: iterative_questions.json, zero_shot_questions.json")
    
    gen = OneDHeatTransferQuestionGenerator()
    gen.generate_all_questions()
    
    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()