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
BURGERS1D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "burgers_1d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "burgers_1d"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "low": {"tolerance_rmse": 0.08},
    "medium": {"tolerance_rmse": 0.04},
    "high": {"tolerance_rmse": 0.01},
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class OneDBurgersQuestionGenerator:
    """Generate Burgers-1D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()
        
    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["cfl", "k", "beta", "n_space"]
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
            profile_path = BURGERS1D_CFG_DIR / f"{profile}.yaml"
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
                    # For composite tasks like k/beta, param_history is 2D: [[{params}, {params}], ...]
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
                "question": self._build_question_text(params, precision_level, tolerance_rmse),
            }
            
            dataset.append(question_entry)
            qid += 1
        
        return dataset

    @staticmethod
    def _build_question_text(params: dict, precision_level: str, tolerance_rmse: float) -> str:
        question_lines = [
            "Problem: Burgers 1D Equation with 2nd Order Roe Method",
            "",
            "This simulation solves the 1D inviscid Burgers equation, which serves as a simplified model for compressible gas dynamics, using a 2nd order Roe method with generalized superbee limiter:",
            "",
            "Conservation form:",
            r"$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0$",
            "",
            "where $u$ is the conserved variable representing velocity in the Burgers equation context.",
            "",
            "Flux function:",
            r"$f(u) = \frac{u^2}{2}$",
            "",
            "Spatial Discretization:",
            "The spatial discretization uses MUSCL reconstruction with blending parameter $k$:",
            "",
            r"$u^L_{j+\frac{1}{2}} = u_j + \frac{1-k}{4} \delta^+ u_{j-\frac{1}{2}} + \frac{1+k}{4} \delta^- u_{j+\frac{1}{2}}$",
            "",
            r"$u^R_{j+\frac{1}{2}} = u_{j+1} - \frac{1+k}{4} \delta^+ u_{j+\frac{1}{2}} - \frac{1-k}{4} \delta^- u_{j+\frac{3}{2}}$",
            "",
            "where $k$ is a blending coefficient between central ($k=1$) and upwind ($k=-1$) scheme.",
            "",
            "Slope Limiting:",
            "The slope limiter uses a generalized superbee limiter:",
            "",
            r"$\psi(r) = \max\left[0, \max\left[\min(\beta r, 1), \min(r, \beta)\right]\right]$",
            "",
            "where $\beta$ is the limiter parameter controlling dissipation ($\beta \geq 1$).",
            "",
            "The slope ratio $r$ at interface $j$ is defined as:",
            "",
            r"$r_{j} = \frac{u_{j} - u_{j-1}}{u_{j+1} - u_{j}}$",
            "",
            "This ratio indicates the local non-smoothness, which will be the input into the slope limiter to achieve the TVD condition.",
            "",
            "Flux Computation:",
            "The interface flux is computed using the Roe approximate Riemann solver:",
            "",
            r"$F_{j+\frac{1}{2}} = \frac{1}{2}[f(u^L) + f(u^R)] - \frac{1}{2}|a|(u^R - u^L)$",
            "",
            r"where $a = \frac{1}{2}(u^L + u^R)$ is the Roe-averaged wave speed.",
            "",
            "Test Cases:",
            "",
            "The case key in the config file sets different initial conditions:",
            "",
            "1. **sin** - Sinusoidal wave: $u(x,0) = \\sin(2\\pi x/L) + 0.5$",
            "2. **rarefaction** - Rarefaction wave: $u(x,0) = -0.1$ for $x < L/2$, $u(x,0) = 0.5$ for $x \\geq L/2$",
            "3. **sod** - Modified Sod shock tube: $u(x,0) = 1.0$ for $x < L/2$, $u(x,0) = 0.1$ for $x \\geq L/2$",
            "4. **double_shock** - Two interacting shocks: $u(x,0) = 1.0$ for $x < L/3$, $u(x,0) = 0.5$ for $L/3 \\leq x < 2L/3$, $u(x,0) = 0.1$ for $x \\geq 2L/3$",
            "5. **blast** - Interacting blast waves: $u(x,0) = \\exp\\left(-\\frac{(x-L/4)^2}{2\\sigma^2}\\right) + 0.8\\cdot\\exp\\left(-\\frac{(x-3L/4)^2}{2\\sigma^2}\\right)$, where $\\sigma = L/20$",
            "",
            "Parameter Information:",
            "- cfl: Courant-Friedrichs-Lewy number, $CFL = u_{max} \\cdot \\frac{\\Delta t}{\\Delta x}$ where $u_{max}$ is the maximum wave speed",
            "- beta: Parameter for generalized superbee limiter strength ($\\beta \\geq 1$)",
            "- k: Blending parameter between central and upwind fluxes",
            "- n_space: Number of grid cells for spatial discretization, determines spatial resolution: $\\Delta x = L / n\\_space$",
            "",
            "Physical Parameters:",
            f"- Domain length: {params['L']}",
            f"- Case: {params['case']}",
            "",
            "Convergence Check:",
            "- Errors between the simulation based on your solution and the simulation based on the self-refined solution are computed to assess convergence.",
            "- Convergence is confirmed if the following validation criteria are satisfied:",
            "",
            "Validation Criteria:",
            f"- **Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Required RMSE Tolerance**: ≤ {tolerance_rmse}",
            "- Relative RMSE must meet this tolerance compared to reference solution",
            "- **Mass conservation**: the total integral remains constant over time",
            "- **Energy non-increasing**: the total energy $\\int u^2 dx$ should not increase",
            "- **Total Variation (TV) non-increasing**: enforces entropy stability",
            "- **Maximum principle satisfaction**: solution bounded by initial condition extrema"
        ]
        return "\n".join(question_lines)
    
    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        """Save dataset to the new path format: data/burgers_1d/{task}/{precision_level}/"""
        base_dir = Path("data") / "burgers_1d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename
        
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all Burgers 1D questions for all tasks and precision levels"""
    print("🚀 BURGERS 1D QUESTION GENERATOR")
    print("=" * 80)
    print(f"📂 Output directory: data/burgers_1d/{{task}}/{{precision_level}}/")
    print(f"📋 Tasks: cfl, k, beta, n_space")
    print(f"🎯 Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"📄 File types: iterative_questions.json, zero_shot_questions.json")
    
    gen = OneDBurgersQuestionGenerator()
    gen.generate_all_questions()
    
    print(f"\n✅ All questions generated successfully!")


if __name__ == "__main__":
    main()