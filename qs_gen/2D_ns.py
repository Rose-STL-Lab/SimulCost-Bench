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
NS2D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "ns_channel_2d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "ns_channel_2d"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "low": {
        "mass_tolerance": 1.0e-04,
        "u_rmse_tolerance": 0.11,
        "v_rmse_tolerance": 0.05,
        "p_rmse_tolerance": 0.4,
        "description": "Moderate convergence criteria - achievable but not trivial"
    },
    "medium": {
        "mass_tolerance": 1.0e-06,
        "u_rmse_tolerance": 0.02,
        "v_rmse_tolerance": 0.005,
        "p_rmse_tolerance": 0.20,
        "description": "Stringent convergence criteria - challenging"
    },
    "high": {
        "mass_tolerance": 1.0e-08,
        "u_rmse_tolerance": 0.008,
        "v_rmse_tolerance": 0.001,
        "p_rmse_tolerance": 0.10,
        "description": "Extremely stringent convergence criteria - very challenging"
    },
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class TwoDNavierStokesQuestionGenerator:
    """Generate NS-2D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()
        
    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", 
                "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"]
        # Tasks that support both iterative and zero-shot
        iterative_supported_tasks = ["mesh_x", "mesh_y"]
        
        total_files = 0
        total_questions = 0
        
        print("=" * 80)
        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)
            
            for precision_level in PRECISION_LEVELS.keys():
                precision_config = PRECISION_LEVELS[precision_level]
                mass_tol = precision_config["mass_tolerance"]
                u_rmse_tol = precision_config["u_rmse_tolerance"]
                v_rmse_tol = precision_config["v_rmse_tolerance"]
                p_rmse_tol = precision_config["p_rmse_tolerance"]
                
                print(f"  {precision_level.upper()} precision:")
                print(f"    Mass tolerance: {mass_tol}")
                print(f"    U RMSE tolerance: {u_rmse_tol}")
                print(f"    V RMSE tolerance: {v_rmse_tol}")
                print(f"    P RMSE tolerance: {p_rmse_tol}")
                
                # Generate iterative questions only for mesh_x and mesh_y
                if task in iterative_supported_tasks:
                    dataset_iter = self.generate_question_dataset(task=task, precision_level=precision_level, zero_shot=False)
                    self._save_dataset(dataset_iter, task=task, precision_level=precision_level, zero_shot=False)
                    total_files += 1
                    total_questions += len(dataset_iter)
                    print(f"      Iterative: {len(dataset_iter):2d} questions")
                else:
                    print(f"      Iterative: Not supported for {task}")
                
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
            profile_path = NS2D_CFG_DIR / f"{profile}.yaml"
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
                "precision_config": precision_config,
                "is_converged": is_converged,
                "dummy_cost": dummy_cost,
                "cost_history": cost_history,
                "best_params": best_params,
                "param_history": param_history,
                "aspect_ratio": task_data.get("non_target_parameters", {}).get("aspect_ratio"),
                "question": self._build_question_text(params, precision_level, precision_config),
            }
            
            dataset.append(question_entry)
            qid += 1
        
        return dataset

    @staticmethod
    def _build_question_text(params: dict, precision_level: str, precision_config: dict) -> str:
        question_lines = [
            "Problem: 2D Navier-Stokes Channel Flow with SIMPLE Algorithm",
            "",
            "This simulation solves the 2D steady incompressible Navier-Stokes equations using the SIMPLE (Semi-Implicit Method for Pressure Linked Equations) algorithm on a staggered finite volume grid:",
            "",
            "**Continuity equation:**",
            r"$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$",
            "",
            "**Momentum equations:**",
            r"$$\rho \left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) = -\frac{\partial p}{\partial x} + \mu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$",
            "",
            r"$$\rho \left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) = -\frac{\partial p}{\partial y} + \mu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$",
            "",
            "Where:",
            "- $u, v$ = velocity components in x, y directions",
            "- $p$ = pressure",
            "- $\\rho$ = density (constant for incompressible flow)",
            "- $\\mu$ = dynamic viscosity",
            "",
            "### Numerical Method",
            "",
            "The SIMPLE algorithm uses:",
            "1. **Staggered grid**: Pressure at cell centers, velocities at cell faces",
            "2. **Under-relaxation**: Factors $\\omega_u$, $\\omega_v$, $\\omega_p$ for stability",
            "3. **Iterative convergence**: Based on mass conservation and velocity/pressure residuals",
            "4. **Geometric flexibility**: Supports various channel geometries with obstacles",
            "",
            "## Test Cases",
            "",
            f"**{params['boundary_condition']}**:",
        ]
        
        # Add boundary condition specific description
        if params['boundary_condition'] == "channel_flow":
            question_lines.extend([
                "- Geometry: Standard rectangular channel",
                "- Boundary: Uniform inlet velocity, no-slip walls, pressure outlet",
                "- Reynolds: Low to moderate"
            ])
        elif params['boundary_condition'] == "back_stair_flow":
            question_lines.extend([
                "- Geometry: Channel with backward-facing step",
                "- Boundary: Uniform inlet velocity, no-slip walls, pressure outlet", 
                "- Reynolds: Low to moderate"
            ])
        elif params['boundary_condition'] == "expansion_channel":
            question_lines.extend([
                "- Geometry: Channel with gradual expansion",
                "- Boundary: Uniform inlet velocity, no-slip walls, pressure outlet",
                "- Reynolds: Low to moderate"
            ])
        elif params['boundary_condition'] == "cube_driven_flow":
            question_lines.extend([
                "- Geometry: Channel with cubic obstacle",
                "- Boundary: Uniform inlet velocity, no-slip walls, pressure outlet",
                "- Reynolds: Low to moderate"
            ])
        
        question_lines.extend([
            "",
            "## Parameter Tuning Tasks",
            "",
            "### Tasks",
            "",
            "1. **Mesh Resolution Convergence Search**",
            "   - **mesh_x**: Number of grid cells in x-direction, determines spatial resolution along channel length",
            "   - **mesh_y**: Number of grid cells in y-direction, determines spatial resolution across channel width",
            "",
            "2. **Under-Relaxation Factor Optimization**",
            "   - **omega_u**: Under-relaxation factor for u-velocity",
            "   - **omega_v**: Under-relaxation factor for v-velocity",
            "   - **omega_p**: Under-relaxation factor for pressure",
            "",
            "3. **Convergence Threshold Optimization**",
            "   - **diff_u_threshold**: Convergence threshold for u-velocity iterations",
            "   - **diff_v_threshold**: Convergence threshold for v-velocity iterations",
            "   - **res_iter_v_threshold**: Residual threshold for inner v-velocity iterations",
            "",
            "Physical Parameters:",
            f"- Channel length: {params['length']} m",
            f"- Channel breadth: {params['breadth']} m",
            f"- Dynamic viscosity: {params['mu']} Pa⋅s",
            f"- Fluid density: {params['rho']} kg/m³",
            f"- Wall height: {params['other_params']['wall_height']}",
            f"- Wall width: {params['other_params']['wall_width']}",
            f"- Wall start height: {params['other_params']['wall_start_height']}",
            f"- Wall start width: {params['other_params']['wall_start_width']}",
            "",
            "",
            "### Convergence Criteria",
            "The simulated results are considered correct if the relative RMSE meets the precision-dependent tolerance compared to reference solution, and the solution satisfies the convergence criteria:",
            "",
            "1. **Mass conservation**: $\\left|\\sum \\text{mass flux}\\right| < \\text{mass\\_tolerance}$",
            "2. **Velocity convergence**: $\\text{RMSE}(u) < \\text{u\\_rmse\\_tolerance}$, $\\text{RMSE}(v) < \\text{v\\_rmse\\_tolerance}$",
            "3. **Pressure convergence**: $\\text{RMSE}(p) < \\text{p\\_rmse\\_tolerance}$",
            "4. **Mass conservation**: Global mass balance must be satisfied",
            "5. **Physical realizability**: Solution must satisfy no-slip boundary conditions and conservation principles",
            "",
            f"**Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Mass Conservation Tolerance**: ≤ {precision_config['mass_tolerance']:.0e}",
            f"- **U-Velocity RMSE Tolerance**: ≤ {precision_config['u_rmse_tolerance']}",
            f"- **V-Velocity RMSE Tolerance**: ≤ {precision_config['v_rmse_tolerance']}",
            f"- **Pressure RMSE Tolerance**: ≤ {precision_config['p_rmse_tolerance']}"
        ])
        return "\n".join(question_lines)
    
    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        """Save dataset to the new path format: data/ns_channel_2d/{task}/{precision_level}/"""
        base_dir = Path("data") / "ns_channel_2d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)
        
        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename
        
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all NS 2D questions for all tasks and precision levels"""
    print("2D NAVIER-STOKES QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/ns_channel_2d/{{task}}/{{precision_level}}/")
    print(f"Tasks: mesh_x, mesh_y, omega_u, omega_v, omega_p, diff_u_threshold, diff_v_threshold, res_iter_v_threshold")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types:")
    print(f"  - iterative_questions.json (only for mesh_x and mesh_y)")
    print(f"  - zero_shot_questions.json (for all tasks)")
    
    gen = TwoDNavierStokesQuestionGenerator()
    gen.generate_all_questions()
    
    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()