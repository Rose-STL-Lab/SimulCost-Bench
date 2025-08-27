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
HEAT2D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "heat_steady_2d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "heat_steady_2d"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "low": {"tolerance_rmse": 0.005, "description": "Relaxed convergence criteria"},
    "medium": {"tolerance_rmse": 0.0005, "description": "Moderate convergence criteria"},
    "high": {"tolerance_rmse": 0.0003, "description": "Most stringent convergence criteria"},
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class TwoDHeatTransferQuestionGenerator:
    """Generate Heat Steady 2D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()
        
    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["dx", "relax", "t_init", "error_threshold"]
        # Define which tasks support iterative mode based on search_type in yaml config
        iterative_tasks = ["dx", "error_threshold"]  # "iterative+0-shot"
        
        total_files = 0
        total_questions = 0
        
        print("=" * 80)
        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)
            
            for precision_level in PRECISION_LEVELS.keys():
                tolerance = PRECISION_LEVELS[precision_level]["tolerance_rmse"]
                print(f"  {precision_level.upper()} precision (RMSE <= {tolerance}):")
                
                # Generate iterative questions only for tasks that support it
                if task in iterative_tasks:
                    dataset_iter = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=False)
                    self._save_dataset(dataset_iter, task=task, precision_level=precision_level, zero_shot=False)
                    total_files += 1
                    total_questions += len(dataset_iter)
                    print(f"      Iterative: {len(dataset_iter):2d} questions")
                
                # Generate zero-shot questions for all tasks
                dataset_zero = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=True)
                self._save_dataset(dataset_zero, task=task, precision_level=precision_level, zero_shot=True)
                
                total_files += 1
                total_questions += len(dataset_zero)
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
            profile_path = HEAT2D_CFG_DIR / f"{profile}.yaml"
            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)
            
            # Extract results
            results = task_data["results"]
            cost_history = results["cost_history"]
            param_history = results["parameter_history"]
            is_converged = results["converged"]
            
            # Calculate dummy cost according to requirements
            if zero_shot:
                # For zero-shot, use the last cost if only one element, otherwise second to last
                dummy_cost = cost_history[-1] if len(cost_history) == 1 else cost_history[-2]
            else:
                # For iterative, sum all costs except the last one
                dummy_cost = sum(cost_history[:-1]) if len(cost_history) > 1 else 0
            
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
            "Problem: 2D Steady-State Heat Conduction with Jacobi Iteration and SOR",
            "",
            "This simulation solves 2D steady-state heat transfer problems using the Jacobi iteration method with Successive Over-Relaxation (SOR). The solver handles rectangular domains with fixed boundary conditions using an iterative approach.",
            "",
            "Governing Equation:",
            r"$\nabla^2 T = 0$",
            "",
            "Discretized Form (5-point stencil):",
            r"$T_{i,j} = \frac{1}{4}(T_{i-1,j} + T_{i+1,j} + T_{i,j-1} + T_{i,j+1})$",
            "",
            "SOR Update Formula:",
            r"$T_{i,j}^{new} = \omega \cdot T_{i,j}^{Jacobi} + (1-\omega) \cdot T_{i,j}^{old}$",
            "",
            "where $\\omega$ is the relaxation parameter.",
            "",
            "Numerical Method:",
            "The solution uses point-wise Jacobi iteration with SOR acceleration:",
            "",
            "1. **Jacobi Update**: Calculate new temperature based on neighboring values",
            "2. **SOR Relaxation**: Blend new and old values using relaxation parameter $\\omega$",
            "3. **Convergence Check**: Monitor RMSE between successive iterations",
            "4. **Boundary Enforcement**: Maintain fixed boundary conditions at each iteration",
            "",
            "Corner temperatures are set as the average of adjacent boundary values.",
            "",
            "Parameter Information:",
            "- **dx**: Grid spacing determines spatial resolution: $\\Delta x = L_x / n_x$, $\\Delta y = L_y / n_y$",
            "- **relax**: SOR relaxation parameter $\\omega$ affecting convergence speed and stability",
            "- **error_threshold**: Convergence criterion - iteration stops when RMSE between steps drops below threshold",
            "- **t_init**: Initial temperature field value for starting the iteration",
            "",
            "Physical Parameters:",
            f"- Domain width: $L_x = {params.get('Lx', 'Lx')}$ m",
            f"- Domain height: $L_y = {params.get('Ly', 'Ly')}$ m",
            f"- Top boundary temperature: {params.get('T_top', 'T_top')} K",
            f"- Bottom boundary temperature: {params.get('T_bottom', 'T_bottom')} K",
            f"- Left boundary temperature: {params.get('T_left', 'T_left')} K",
            f"- Right boundary temperature: {params.get('T_right', 'T_right')} K",
            "",
            "Grid Information:",
            "- Number of grid points: $n_x = L_x / dx + 1$, $n_y = L_y / dx + 1$",
            "- Total interior points: $(n_x - 2) \\times (n_y - 2)$",
            "- Computational cost scales approximately as $O(n_x \\times n_y \\times \\text{iterations})$",
            "",
            "Convergence Check:",
            "- Errors between the simulation based on your solution and the simulation based on the self-refined solution are computed to assess convergence.",
            "- Convergence is confirmed if the following validation criteria are satisfied.",
            "",
            "Validation Criteria:",
            f"- **Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Required RMSE Tolerance**: ≤ {tolerance_rmse}",
            "- Relative RMSE must meet this tolerance compared to self-refined solution",
            "- Temperature validity: All values finite and within boundary range",
            "- Gradient reasonableness: Temperature gradients remain physically reasonable"
        ]
        return "\n".join(question_lines)
    
    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        """Save dataset to the new path format: data/heat_2d/{task}/{precision_level}/"""
        base_dir = Path("data") / "heat_2d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename
        
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all Heat Steady 2D questions for all tasks and precision levels"""
    print("HEAT STEADY 2D QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/heat_2d/{{task}}/{{precision_level}}/")
    print(f"Tasks: dx, relax, t_init, error_threshold")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types: iterative_questions.json, zero_shot_questions.json")
    
    gen = TwoDHeatTransferQuestionGenerator()
    gen.generate_all_questions()
    
    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()