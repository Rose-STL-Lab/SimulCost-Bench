import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import Dict, Any
from qs_gen.utils import *
import time
import argparse
from costsci_tools.gen_cfgs.ns_channel_2d import create_ns_channel_profiles
from costsci_tools.dummy_sols.ns_channel_2d import (
    grid_search_mesh_x, grid_search_mesh_y, grid_search_omega_u, grid_search_omega_v, 
    grid_search_omega_p, grid_search_diff_u_threshold, grid_search_diff_v_threshold,
    grid_search_res_iter_v_threshold, rule_based_search, PARAM_RULES
)
import yaml

NS2D_CFG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "costsci_tools", "run_configs", "ns_channel_2d"
)

class twoD_NSQuestionGenerator():
    """Generate training dataset for 2D Navier-Stokes channel flow problems"""
    def __init__(self, search_mode="rule"):
        np.random.seed(42)
        self.search_mode = search_mode

    def generate_question_dataset(self, num_samples: int, task: str, zero_shot: bool, max_iter: int = 10) -> List[Dict]:
        dataset: List[Dict] = []

        mass_tolerance = 1e-4
        u_rmse_tolerance = 3e-2
        v_rmse_tolerance = 3e-2
        p_rmse_tolerance = 3e-2

        default = {
            "mesh_x": 100,
            "mesh_y": 20,
            "omega_u": 0.5,
            "omega_v": 0.5,
            "omega_p": 0.5,
            "diff_u_threshold": 1e-4,
            "diff_v_threshold": 1e-4,
            "res_iter_v_threshold": 1e-4,
        }

        for idx in range(1, num_samples + 1):
            profile_name = f"p{idx}"
            profile_path = os.path.join(NS2D_CFG_DIR, f"{profile_name}.yaml")

            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            # Get boundary condition for global variable
            boundary_condition = params["boundary_condition"]
            
            # Set up global boundary_condition variable for dummy functions
            import costsci_tools.dummy_sols.ns_channel_2d as dummy_mod
            dummy_mod.boundary_condition = boundary_condition

            # Choose search method based on search_mode
            if self.search_mode == "rule":
                fixed_params = {
                    'mesh_x': default["mesh_x"],
                    'mesh_y': default["mesh_y"],
                    'omega_u': default["omega_u"],
                    'omega_v': default["omega_v"],
                    'omega_p': default["omega_p"],
                    'diff_u_threshold': default["diff_u_threshold"],
                    'diff_v_threshold': default["diff_v_threshold"],
                    'res_iter_v_threshold': default["res_iter_v_threshold"],
                }
                initial_value = default[task.lower()]
                is_converged, best_param, cost_history, param_history = rule_based_search(
                    profile_name, task.lower(), initial_value, max_iter,
                    params["length"], params["breadth"], mass_tolerance,
                    u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance,
                    **fixed_params
                )
                param_tag = f"best_{task.lower()}"
            # Task-specific parameter search for grid mode
            elif task.lower() == "mesh_x":
                mesh_x_values = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
                is_converged, best_param, cost_history, param_history = grid_search_mesh_x(
                    profile=profile_name,
                    mesh_x_values=mesh_x_values,
                    mesh_y=default["mesh_y"],
                    omega_u=default["omega_u"],
                    omega_v=default["omega_v"],
                    omega_p=default["omega_p"],
                    diff_u_threshold=default["diff_u_threshold"],
                    diff_v_threshold=default["diff_v_threshold"],
                    res_iter_v_threshold=default["res_iter_v_threshold"],
                    length=params["length"],
                    breadth=params["breadth"],
                    mass_tolerance=mass_tolerance,
                    u_rmse_tolerance=u_rmse_tolerance,
                    v_rmse_tolerance=v_rmse_tolerance,
                    p_rmse_tolerance=p_rmse_tolerance
                )
                param_tag = "best_mesh_x"
            elif task.lower() == "mesh_y":
                mesh_y_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                is_converged, best_param, cost_history, param_history = grid_search_mesh_y(
                    profile=profile_name,
                    mesh_x=default["mesh_x"],
                    mesh_y_values=mesh_y_values,
                    omega_u=default["omega_u"],
                    omega_v=default["omega_v"],
                    omega_p=default["omega_p"],
                    diff_u_threshold=default["diff_u_threshold"],
                    diff_v_threshold=default["diff_v_threshold"],
                    res_iter_v_threshold=default["res_iter_v_threshold"],
                    length=params["length"],
                    breadth=params["breadth"],
                    mass_tolerance=mass_tolerance,
                    u_rmse_tolerance=u_rmse_tolerance,
                    v_rmse_tolerance=v_rmse_tolerance,
                    p_rmse_tolerance=p_rmse_tolerance
                )
                param_tag = "best_mesh_y"
            elif task.lower() == "omega_u":
                omega_u_values = [0.1 * i for i in range(1, 11)]
                is_converged, best_param, cost_history, param_history = grid_search_omega_u(
                    profile=profile_name,
                    mesh_x=default["mesh_x"],
                    mesh_y=default["mesh_y"],
                    omega_u_values=omega_u_values,
                    omega_v=default["omega_v"],
                    omega_p=default["omega_p"],
                    diff_u_threshold=default["diff_u_threshold"],
                    diff_v_threshold=default["diff_v_threshold"],
                    res_iter_v_threshold=default["res_iter_v_threshold"],
                    length=params["length"],
                    breadth=params["breadth"],
                    mass_tolerance=mass_tolerance,
                    u_rmse_tolerance=u_rmse_tolerance,
                    v_rmse_tolerance=v_rmse_tolerance,
                    p_rmse_tolerance=p_rmse_tolerance
                )
                param_tag = "best_omega_u"
            elif task.lower() == "omega_v":
                omega_v_values = [0.1 * i for i in range(1, 11)]
                is_converged, best_param, cost_history, param_history = grid_search_omega_v(
                    profile=profile_name,
                    mesh_x=default["mesh_x"],
                    mesh_y=default["mesh_y"],
                    omega_u=default["omega_u"],
                    omega_v_values=omega_v_values,
                    omega_p=default["omega_p"],
                    diff_u_threshold=default["diff_u_threshold"],
                    diff_v_threshold=default["diff_v_threshold"],
                    res_iter_v_threshold=default["res_iter_v_threshold"],
                    length=params["length"],
                    breadth=params["breadth"],
                    mass_tolerance=mass_tolerance,
                    u_rmse_tolerance=u_rmse_tolerance,
                    v_rmse_tolerance=v_rmse_tolerance,
                    p_rmse_tolerance=p_rmse_tolerance
                )
                param_tag = "best_omega_v"
            elif task.lower() == "omega_p":
                omega_p_values = [0.1 * i for i in range(1, 11)]
                is_converged, best_param, cost_history, param_history = grid_search_omega_p(
                    profile=profile_name,
                    mesh_x=default["mesh_x"],
                    mesh_y=default["mesh_y"],
                    omega_u=default["omega_u"],
                    omega_v=default["omega_v"],
                    omega_p_values=omega_p_values,
                    diff_u_threshold=default["diff_u_threshold"],
                    diff_v_threshold=default["diff_v_threshold"],
                    res_iter_v_threshold=default["res_iter_v_threshold"],
                    length=params["length"],
                    breadth=params["breadth"],
                    mass_tolerance=mass_tolerance,
                    u_rmse_tolerance=u_rmse_tolerance,
                    v_rmse_tolerance=v_rmse_tolerance,
                    p_rmse_tolerance=p_rmse_tolerance
                )
                param_tag = "best_omega_p"
            elif task.lower() == "diff_u_threshold":
                diff_u_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                is_converged, best_param, cost_history, param_history = grid_search_diff_u_threshold(
                    profile=profile_name,
                    mesh_x=default["mesh_x"],
                    mesh_y=default["mesh_y"],
                    omega_u=default["omega_u"],
                    omega_v=default["omega_v"],
                    omega_p=default["omega_p"],
                    diff_u_values=diff_u_values,
                    diff_v_threshold=default["diff_v_threshold"],
                    res_iter_v_threshold=default["res_iter_v_threshold"],
                    length=params["length"],
                    breadth=params["breadth"],
                    mass_tolerance=mass_tolerance,
                    u_rmse_tolerance=u_rmse_tolerance,
                    v_rmse_tolerance=v_rmse_tolerance,
                    p_rmse_tolerance=p_rmse_tolerance
                )
                param_tag = "best_diff_u_threshold"
            elif task.lower() == "diff_v_threshold":
                diff_v_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                is_converged, best_param, cost_history, param_history = grid_search_diff_v_threshold(
                    profile=profile_name,
                    mesh_x=default["mesh_x"],
                    mesh_y=default["mesh_y"],
                    omega_u=default["omega_u"],
                    omega_v=default["omega_v"],
                    omega_p=default["omega_p"],
                    diff_u_threshold=default["diff_u_threshold"],
                    diff_v_values=diff_v_values,
                    res_iter_v_threshold=default["res_iter_v_threshold"],
                    length=params["length"],
                    breadth=params["breadth"],
                    mass_tolerance=mass_tolerance,
                    u_rmse_tolerance=u_rmse_tolerance,
                    v_rmse_tolerance=v_rmse_tolerance,
                    p_rmse_tolerance=p_rmse_tolerance
                )
                param_tag = "best_diff_v_threshold"
            elif task.lower() == "res_iter_v_threshold":
                res_iter_v_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                is_converged, best_param, cost_history, param_history = grid_search_res_iter_v_threshold(
                    profile=profile_name,
                    mesh_x=default["mesh_x"],
                    mesh_y=default["mesh_y"],
                    omega_u=default["omega_u"],
                    omega_v=default["omega_v"],
                    omega_p=default["omega_p"],
                    diff_u_threshold=default["diff_u_threshold"],
                    diff_v_threshold=default["diff_v_threshold"],
                    res_iter_v_values=res_iter_v_values,
                    length=params["length"],
                    breadth=params["breadth"],
                    mass_tolerance=mass_tolerance,
                    u_rmse_tolerance=u_rmse_tolerance,
                    v_rmse_tolerance=v_rmse_tolerance,
                    p_rmse_tolerance=p_rmse_tolerance
                )
                param_tag = "best_res_iter_v_threshold"
            else:
                raise ValueError(f"Unknown task: {task.lower()}")

            if best_param is not None:
                print(f"\nRecommended {param_tag}: {best_param}, the total cost is {cost_history}")
            else:
                print(f"\nNo convergent {param_tag} found within the given iterations, the total cost is {sum(cost_history[:-1])}")

            dummy_cost = cost_history[-2] if zero_shot else sum(cost_history[:-1])

            question = self._build_question_text(dummy_cost, params, task, boundary_condition)
            dataset.append(
                {
                    "QID": idx,
                    "profile": profile_name,
                    "boundary_condition": boundary_condition,
                    "zero_shot": zero_shot,
                    "is_converged": is_converged,
                    "dummy_cost": dummy_cost,
                    "cost_history": cost_history,
                    param_tag: best_param,
                    "param_history": param_history,
                    "question": question,
                }
            )

        return dataset

    def _build_question_text(self, cost: int, params: Dict[str, Any], task: str, boundary_condition: str) -> str:
        question_lines = [
            "Problem: Steady 2D Navier–Stokes — Channel Flow (SIMPLE, Staggered FVM)",
            "",
            "This simulation solves the steady, incompressible 2D Navier–Stokes equations for channel flow using the SIMPLE algorithm on a staggered finite-volume grid:",
            "",
            "Momentum (component-wise):",
            "ρ(u·∇)u = -∇p + μ∇²u",
            "",
            "Continuity:",
            "∇·u = 0",
            "",
            "Discretization & Coupling:",
            "- Grid: staggered. Pressure p is stored at cell centers; u and v at faces.",
            "- Viscous terms use second-order central differences; convection uses upwind/central per solver internals.",
            "- Pressure–velocity coupling: SIMPLE with under-relaxation on u, v, and p.",
            "- The pressure correction step enforces mass conservation.",
            "",
            "Convergence Criteria:",
            "Outer loop stops when:",
            "- Mass residual is below mass_tolerance",
            "- Velocity/pressure RMSE drop below their tolerances",
            "",
            f"Boundary Conditions & Profile: {boundary_condition}",
        ]
        
        if boundary_condition == "channel_flow":
            question_lines.extend([
                "- No-slip walls (u=v=0)",
                "- Prescribed inlet profile (often parabolic or uniform)",
                "- Fixed/zero-gradient outlet pressure",
            ])
        elif boundary_condition == "back_stair_flow":
            question_lines.extend([
                "- Backward-facing step geometry",
                "- No-slip walls with step discontinuity",
                "- Inlet velocity profile and outlet pressure condition",
            ])
        elif boundary_condition == "expansion_channel":
            question_lines.extend([
                "- Channel expansion geometry",
                "- No-slip walls with geometric expansion",
                "- Inlet velocity profile and outlet pressure condition",
            ])
        elif boundary_condition == "cube_driven_flow":
            question_lines.extend([
                "- Lid-driven cavity flow in cubic domain",
                "- Moving top wall with specified velocity",
                "- No-slip conditions on remaining walls",
            ])
        else:
            question_lines.extend([
                "- Custom geometry boundary conditions",
                "- No-slip walls where applicable",
                "- Specified inlet/outlet conditions",
            ])
        
        question_lines.extend([
            "",
            "Tunable Parameters:",
            "- mesh_x, mesh_y: Number of cells in x and y directions",
            "- omega_u, omega_v: Under-relaxation factors for u and v velocity components", 
            "- omega_p: Under-relaxation factor for pressure correction",
            "- diff_u_threshold, diff_v_threshold: Per-iteration update thresholds for velocity components",
            "- res_iter_v_threshold: Schedule for pressure/velocity residual gate",
            "",
            "Physical Parameters:",
            f"- Domain length: {params['length']}",
            f"- Domain breadth: {params['breadth']}",
            f"- Density (ρ): {params['rho']}",
            f"- Dynamic viscosity (μ): {params['mu']}",
            "",
            "Global Stopping & Iteration Budget:",
            "- mass_tolerance: Mass residual target (default 1e-4)",
            "- u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance: Field RMSE targets (default 3e-2)",
            "",
            "Validation Criteria:",
            "- Mass conservation: Global mass residual ≤ mass_tolerance",
            "- Field stability: RMSE between successive outer iterations for u, v, p falls below tolerances",
            "- Cost tracking: Iteration count and/or wall time recorded",
            "- Convergence flags: Mass and field stability criteria must both be satisfied",
        ])
        
        return "\n".join(question_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("-t", "--task", type=str, 
                        choices=["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p",
                                "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"], 
                        default="mesh_x", help="Task of problem to solve")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode")
    parser.add_argument("-s", "--search_mode", type=str, 
                        choices=["grid", "rule"], default="rule",
                        help="Choose search mode: grid (fixed list) or rule (rule-based)")
    parser.add_argument("-m", "--max_iter", type=int, default=10,
                        help="Maximum iterations for rule-based search")

    args = parser.parse_args()
    num_samples = args.num_samples
    task = args.task
    zero_shot = args.zero_shot
    search_mode = args.search_mode
    max_iter = args.max_iter
    flag = "zero_shot" if zero_shot else "iterative"

    dataset_dir = f"data/2D_ns/{task}"
    question_file = f"{dataset_dir}/{flag}_{search_mode}_question.json"
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"[INFO] Generating {num_samples} samples for task '{task}'")
    print(f"[INFO] Zero-shot mode: {'ON' if zero_shot else 'OFF'}")
    print(f"[INFO] Search mode: {search_mode}")
    if search_mode == "rule":
        print(f"[INFO] Max iterations: {max_iter}")
    print(f"[INFO] Output path: {question_file}")

    base_profile_path = "costsci_tools/run_configs/ns_channel_2d/p1.yaml"
    if not os.path.isfile(base_profile_path):
        raise FileNotFoundError(f"Base profile path not found: {base_profile_path}")

    # Generate additional profiles if needed (beyond p1)
    # Create num_samples profiles for each of the 4 boundary conditions
    if num_samples > 0:
        random_profiles = create_ns_channel_profiles(
            num_profiles_per_bc=num_samples,
            base_profile_path=base_profile_path,
            solver_name="ns_channel_2d"
        )
        
        # Update num_samples to reflect total profiles (p1 + 4*num_samples)
        total_samples = 1 + 4 * num_samples
    else:
        # Only use p1 if num_samples is 0
        total_samples = 1

    generator = twoD_NSQuestionGenerator(search_mode=search_mode)
    dataset = generator.generate_question_dataset(
        num_samples=total_samples,
        task=task,
        zero_shot=zero_shot,
        max_iter=max_iter
    )
    save_dataset(dataset, question_file)