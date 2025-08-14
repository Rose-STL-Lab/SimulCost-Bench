import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from qs_gen.utils import *

# -----------------------------------------------------------------------------#
#  Global settings                                                             #
# -----------------------------------------------------------------------------#
EULER1D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "euler_1d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "euler_1d"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "low": {"tolerance_rmse": 0.08},
    "medium": {"tolerance_rmse": 0.02},
    "high": {"tolerance_rmse": 0.01},
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class OneDEulerQuestionGenerator:
    """Generate Euler-1D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()
        
    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["cfl", "beta", "k", "n_space"]
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
            profile_path = EULER1D_CFG_DIR / f"{profile}.yaml"
            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)
            case_name = params["case"]
            
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
                if isinstance(param_set, list):
                    # For tasks like beta/k, param_history is 2D: [[{params}, {params}], ...]
                    # We want the last element (highest resolution) of each iteration
                    param_dict = param_set[-1]
                else:
                    # For tasks like cfl/n_space, param_history is 1D: [{params}, {params}, ...]
                    param_dict = param_set
                    
                if param_dict.get(task) == optimal_value:
                    best_params = param_dict.copy()
                    break
            
            # If not found, this is an error in the data
            if best_params is None:
                raise ValueError(f"Could not find optimal parameter {task}={optimal_value} in parameter_history")
            
            # Build question entry
            question_entry = {
                "QID": qid,
                "profile": profile,
                "case": case_name,
                "zero_shot": zero_shot,
                "target_parameter": task,
                "precision_level": precision_level,
                "tolerance_rmse": tolerance_rmse,
                "is_converged": is_converged,
                "dummy_cost": dummy_cost,
                "cost_history": cost_history,
                "best_params": best_params,
                "param_history": param_history,
                "question": self._build_question_text(params),
            }
            
            dataset.append(question_entry)
            qid += 1
        
        return dataset

    @staticmethod
    def _build_question_text(params: dict) -> str:
        question_lines = [
            "Problem: Euler 1D Equations with 2nd Order MUSCL-Roe Method",
            "",
            "This simulation solves the 1D Euler equations for compressible inviscid flow, using a 2nd order MUSCL scheme with Roe flux and generalized superbee limiter:",
            "",
            "Conservative form:",
            r"$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = 0$",
            "",
            "Where the conservative variables and flux are:",
            r"$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho E \end{pmatrix}, \quad \mathbf{F} = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ u(\rho E + p) \end{pmatrix}$",
            "",
            "Primitive variables:",
            "- $\rho$ = density",
            "- $u$ = velocity",
            "- $p$ = pressure",
            "- $E$ = specific total energy",
            "",
            "Equation of state:",
            r"$p = (\gamma - 1) \rho \left(E - \frac{u^2}{2}\right)$",
            "",
            "where $\gamma$ is the ratio of specific heats.",
            "",
            "Spatial Discretization:",
            "The spatial discretization uses MUSCL reconstruction with blending parameter $k$:",
            "",
            r"$\mathbf{U}^L_{j+\frac{1}{2}} = \mathbf{U}_j + \frac{1+k}{4} \psi(r_{j}) (\mathbf{U}_{j+1} - \mathbf{U}_{j})$",
            "",
            r"$\mathbf{U}^R_{j+\frac{1}{2}} = \mathbf{U}_{j+1} - \frac{1+k}{4} \psi(r_{j+1}) (\mathbf{U}_{j+2} - \mathbf{U}_{j+1})$",
            "",
            "where $k$ is a blending coefficient between central ($k=1$) and upwind ($k=-1$) scheme, and $\psi(r)$ is the slope limiter function.",
            "",
            "Slope Limiting:",
            "The slope limiter uses a generalized superbee limiter:",
            "",
            r"$\psi(r) = \max\left[0, \max\left[\min(\beta r, 1), \min(r, \beta)\right]\right]$",
            "",
            "where $\beta$ is the limiter parameter controlling dissipation.",
            "",
            "The slope ratio $r$ at interface $j$ is defined as:",
            "",
            r"$r_{j} = \frac{\mathbf{U}_{j+1} - \mathbf{U}_{j}}{\mathbf{U}_{j+2} - \mathbf{U}_{j+1}}$",
            "",
            "This ratio indicates the local non-smoothness, which will be the input into the slope limiter to achieve the TVD condition.",
            "",
            "Flux Computation:",
            "The interface flux is computed using the Roe approximate Riemann solver:",
            "",
            r"$\mathbf{F}_{j+\frac{1}{2}} = \frac{1}{2}\left[\mathbf{F}(\mathbf{U}^L) + \mathbf{F}(\mathbf{U}^R)\right] - \frac{1}{2}|\mathbf{A}|(\mathbf{U}^R - \mathbf{U}^L)$",
            "",
            "where $|\mathbf{A}|$ is the Roe matrix with Roe-averaged quantities.",
            "",
            "Initial condition cases:",
            "- sod: Left: $\rho=1.0, u=0.0, p=1.0$; Right: $\rho=0.125, u=0.0, p=0.1$",
            "- lax: Left: $\rho=0.445, u=0.6977, p=3.528$; Right: $\rho=0.5, u=0.0, p=0.571$",
            "- mach_3: Left: $\rho=3.857, u=0.92, p=10.333$; Right: $\rho=1.0, u=3.55, p=1.0$",
            "",
            "Parameter Information:",
            "- cfl: Courant-Friedrichs-Lewy number, $CFL = \\frac{(|u| + c) \\Delta t}{\\Delta x}$ where $c = \\sqrt{\\gamma p/\\rho}$ is the speed of sound",
            "- beta: Limiter parameter for generalized superbee",
            "- k: Blending parameter between central and upwind fluxes",
            "- n_space: Number of grid cells for spatial discretization, determines spatial resolution: $\\Delta x = L / n\\_space$",
            "",
            "Physical Parameters:",
            f"- Domain length: {params['L']}",
            f"- Gamma (ratio of specific heats): {params['gamma']}",
            f"- Case: {params['case']}",
            "",
            "Convergence Check:",
            "- Errors between the simulation based on your solution and the simulation based on the self-refined solution are computed to assess convergence.",
            "- Convergence is confirmed if the following validation criteria are satisfied.",
            "",
            "Validation Criteria:",
            "- Relative RMSE meets the precision-dependent tolerance (low: 0.01, medium: 0.005, high: 0.0025) compared to reference solution",
            "- Positivity preservation: pressure and density must remain positive at all times",
            "- Shock speed consistency: pressure gradients should not exceed physical bounds"
        ]
        return "\n".join(question_lines)
    
    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        """Save dataset to the new path format: data/euler_1d/{task}/{precision_level}/"""
        base_dir = Path("data") / "euler_1d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename
        
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all Euler 1D questions for all tasks and precision levels"""
    print("🚀 EULER 1D QUESTION GENERATOR")
    print("=" * 80)
    print(f"📂 Output directory: data/euler_1d/{{task}}/{{precision_level}}/")
    print(f"📋 Tasks: cfl, beta, k, n_space")
    print(f"🎯 Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"📄 File types: iterative_questions.json, zero_shot_questions.json")
    
    gen = OneDEulerQuestionGenerator()
    gen.generate_all_questions()
    
    print(f"\n✅ All questions generated successfully!")


if __name__ == "__main__":
    main()
