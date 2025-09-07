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
NS_TRANSIENT_2D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "ns_transient_2d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "ns_transient_2d"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "high": {
        "norm_rmse_tolerance": 0.15,
        "description": "Very stringent convergence criteria - very challenging"
    },
    "medium": {
        "norm_rmse_tolerance": 0.3,
        "description": "Stringent convergence criteria - challenging"
    },
    "low": {
        "norm_rmse_tolerance": 0.6,
        "description": "Moderate convergence criteria - achievable but not trivial"
    },
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class TwoDTransientNavierStokesQuestionGenerator:
    """Generate NS transient 2D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()
        
    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["resolution", "cfl"]
        # Only resolution and cfl support iterative mode
        # relaxation_factor and residual_threshold only support zero-shot
        iterative_supported_tasks = ["resolution", "cfl"]
        zero_shot_only_tasks = ["relaxation_factor", "residual_threshold"]
        
        all_tasks = tasks + zero_shot_only_tasks
        
        total_files = 0
        total_questions = 0
        
        print("=" * 80)
        for task in all_tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)
            
            for precision_level in PRECISION_LEVELS.keys():
                precision_config = PRECISION_LEVELS[precision_level]
                norm_rmse_tol = precision_config["norm_rmse_tolerance"]
                
                print(f"  {precision_level.upper()} precision:")
                print(f"    Norm RMSE tolerance: {norm_rmse_tol}")
                
                # Generate iterative questions only for resolution and cfl
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
            profile = task_data["profile"]
            
            # Load profile configuration
            profile_path = NS_TRANSIENT_2D_CFG_DIR / f"{profile}.yaml"
            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)
            
            # Extract results
            results = task_data["results"]
            cost_history = results["cost_history"]
            param_history = results["parameter_history"]
            is_converged = results["converged"]
            
            # Calculate dummy cost according to requirements
            dummy_cost = cost_history[-2] if zero_shot and len(cost_history) > 1 else sum(cost_history[:-1]) if len(cost_history) > 1 else cost_history[0]
            
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
            
            # Filter best_params to only include the 4 tunable parameters
            filtered_best_params = {
                key: best_params[key] for key in ["resolution", "cfl", "relaxation_factor", "residual_threshold"]
                if key in best_params
            }
            
            # Filter param_history to only include the 4 tunable parameters
            filtered_param_history = []
            for param_set in param_history:
                filtered_set = {
                    key: param_set[key] for key in ["resolution", "cfl", "relaxation_factor", "residual_threshold"]
                    if key in param_set
                }
                filtered_param_history.append(filtered_set)
            
            # Build question entry
            question_entry = {
                "QID": qid,
                "profile": profile,
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
        question_lines = [
            "# Navier-Stokes Transient 2D Equations with Taichi-based Fluid Simulation",
            "",
            "## Introduction",
            "",
            "This simulation solves the 2D transient incompressible Navier-Stokes equations using a Taichi-based fluid simulation framework with configurable boundary conditions and numerical schemes:",
            "",
            "**Continuity equation:**",
            r"$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$",
            "",
            "**Momentum equations:**",
            r"$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \frac{1}{Re} \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$",
            "",
            r"$$\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \frac{1}{Re} \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$",
            "",
            "Where:",
            "- $u, v$ = velocity components in x, y directions",
            "- $p$ = pressure",
            "- $Re$ = Reynolds number",
            "- $t$ = time",
            "",
            "### Numerical Method",
            "",
            "The simulation uses:",
            "1. **Taichi framework**: High-performance GPU/CPU computation with automatic differentiation",
            "2. **Staggered grid**: Pressure at cell centers, velocities at cell faces",
            "3. **CIP (Constrained Interpolation Profile) scheme**: High-order advection scheme for stability",
            "4. **Pressure correction**: SIMPLE-like algorithm with configurable relaxation factor",
            "5. **Time stepping**: CFL-controlled time step for temporal stability",
            "6. **Vorticity confinement**: Optional artificial viscosity for numerical stability",
            "",
            "### Domain Configuration",
            "",
            "- **Aspect Ratio**: Fixed at 0.5 (y/x), meaning the domain is twice as wide as it is tall",
            "- **Domain Resolution**: x_resolution = 2 × resolution, y_resolution = resolution",
            "- **CFL Calculation**: $\\Delta t = \\text{CFL} \\times \\Delta x$ where $\\Delta x = 1/\\text{resolution}$",
            "- **Maximum Wall Time**: 1200 seconds (20 minutes) per simulation",
            "",
            "## Test Cases",
            "",
            "### Current Configuration",
            "",
        ]
        
        # Add boundary condition specific description based on boundary_condition parameter
        bc = params.get('boundary_condition', 1)
        reynolds = params.get('reynolds_num', 1000.0)
        
        if bc == 1:
            question_lines.extend([
                "**Simple Circular Obstacle**",
                "- Single circular obstacle in center of channel",
                "- Uniform inlet velocity, pressure outlet",
                "- Clean flow separation and wake formation"
            ])
        elif bc == 2:
            question_lines.extend([
                "**Multiple Obstacles with Steps**",
                "- Complex maze-like geometry with multiple rectangular obstacles",
                "- Stepped flow path with alternating obstacle placement",
                "- Tests flow through complex geometric constraints"
            ])
        elif bc == 3:
            question_lines.extend([
                "**Random Circular Obstacles**",
                "- 100 randomly placed circular obstacles (seed=123 for reproducibility)",
                "- Dense obstacle field testing flow through irregular patterns",
                "- Tests robustness to geometric complexity"
            ])
        elif bc == 4:
            question_lines.extend([
                "**Dual Inlet/Outlet Configuration**",
                "- Two separate inlet streams (top and bottom)",
                "- Single central outlet",
                "- Tests flow mixing and interaction between streams"
            ])
        elif bc == 5:
            question_lines.extend([
                "**Complex Obstacle Array**",
                "- Dense array of rectangular obstacles in systematic pattern",
                "- Multiple flow paths with varying widths",
                "- Tests flow through highly constrained geometries"
            ])
        elif bc == 6:
            question_lines.extend([
                "**Dragon-Shaped Obstacle**",
                "- Complex artistic obstacle loaded from PNG image file",
                "- Irregular, organic shape testing flow around complex boundaries",
                "- Tests numerical robustness with highly irregular geometry"
            ])
        else:
            question_lines.extend([
                f"**Custom Boundary Condition {bc}**",
                "- Complex geometry with various obstacles"
            ])
        
        question_lines.extend([
            "",
            "### Physical Parameters",
            f"- **Boundary Condition**: {bc}",
            f"- **Reynolds Number**: {reynolds}",
            # f"- **Advection Scheme**: {params.get('advection_scheme', 'cip')}",
            # f"- **Vorticity Confinement**: {params.get('vorticity_confinement', 0.0)}",
            # f"- **Total Runtime**: {params.get('total_runtime', 1.0)}",
            "",
            "## Parameter Tuning Tasks",
            "",
            "### Tasks",
            "",
            "1. **Resolution Convergence Search**",
            "   - **resolution**: Grid resolution determining spatial discretization quality",
            "",
            "2. **CFL Optimization**",
            "   - **cfl**: Courant-Friedrichs-Lewy number controlling time step stability",
            "",
            "3. **Relaxation Factor Optimization (0-shot)**",
            "   - **relaxation_factor**: Pressure correction relaxation factor controlling convergence rate",
            "",
            "4. **Residual Threshold Optimization (0-shot)**",
            "   - **residual_threshold**: Pressure solver convergence threshold",
            "",
            "### Convergence Criteria",
            "The solution is considered converged when:",
            "**Normalized velocity RMSE**: $\\text{RMSE}(\\|\\vec{v}\\|) < \\text{norm\\_rmse\\_tolerance}$",
            "",
            f"**Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Normalized RMSE Tolerance**: ≤ {precision_config['norm_rmse_tolerance']}",
            f"- **Description**: {precision_config['description']}"
        ])
        return "\n".join(question_lines)
    
    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        """Save dataset to the new path format: data/ns_transient_2d/{task}/{precision_level}/"""
        base_dir = Path("data") / "ns_transient_2d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename
        
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all NS transient 2D questions for all tasks and precision levels"""
    print("2D TRANSIENT NAVIER-STOKES QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/ns_transient_2d/{{task}}/{{precision_level}}/")
    print(f"Iterative tasks: resolution, cfl")
    print(f"Zero-shot only tasks: relaxation_factor, residual_threshold")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types:")
    print(f"  - iterative_questions.json (for resolution and cfl)")
    print(f"  - zero_shot_questions.json (for all tasks)")
    
    gen = TwoDTransientNavierStokesQuestionGenerator()
    gen.generate_all_questions()
    
    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()