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
    grid_search_res_iter_v_threshold
)
import yaml

NS2D_CFG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "costsci_tools", "run_configs", "ns_channel_2d"
)

class twoD_NSQuestionGenerator():
    """Generate training dataset for 2D Navier-Stokes channel flow problems"""
    def __init__(self):
        np.random.seed(42)

    def generate_question_dataset(self, num_samples: int, task: str, zero_shot: bool) -> List[Dict]:
        dataset: List[Dict] = []

        mass_tolerance = 1e-8
        u_rmse_tolerance = 1e-3
        v_rmse_tolerance = 1e-3
        p_rmse_tolerance = 1e-3

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

            # Task-specific parameter search
            if task.lower() == "mesh_x":
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
            "Problem: 2D Steady-State Navier-Stokes Equations with SIMPLE Algorithm",
            "",
            "This simulation solves the incompressible, steady-state Navier-Stokes equations in 2D using Finite Volume Method (FVM) on a staggered grid with pressure-velocity coupling via the SIMPLE algorithm.",
            "",
            "Governing Equations:",
            "- Momentum (u and v components):",
            r"  $\rho (u \cdot \nabla u) = -\nabla p + \mu \nabla^2 u$",
            r"  $\rho (v \cdot \nabla v) = -\nabla p + \mu \nabla^2 v$",
            "- Continuity (mass conservation):",
            r"  $\nabla \cdot u = 0$",
            "",
            "where:",
            "- $u, v$ = velocity components in x and y directions",
            "- $p$ = pressure",
            "- $\rho$ = density",
            "- $\mu$ = dynamic viscosity",
            "",
            "Numerical Method:",
            "The SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm is used for pressure-velocity coupling:",
            "1. Solve momentum equations with guessed pressure field",
            "2. Solve pressure correction equation from continuity",
            "3. Correct velocities and pressure using relaxation factors",
            "4. Iterate until convergence",
            "",
            "Spatial Discretization:",
            "- Finite Volume Method (FVM) on staggered grid",
            "- Velocity components stored at cell faces",
            "- Pressure stored at cell centers",
            "- Central differencing for diffusion terms",
            "- Upwind differencing for convection terms",
            "",
            f"Boundary Conditions: {boundary_condition}",
        ]
        
        if boundary_condition == "channel_flow":
            question_lines.extend([
                "- Inlet: Parabolic velocity profile (u = 1 at center, u = 0 at walls)",
                "- Outlet: Fixed pressure (p = 0)",
                "- Walls: No-slip condition (u = v = 0)",
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
            "- mesh_x, mesh_y: Grid resolution in x and y directions",
            "- omega_u, omega_v: Relaxation factors for u and v velocity correction",
            "- omega_p: Relaxation factor for pressure correction",
            "- diff_u_threshold: Threshold for u-velocity update residual",
            "- diff_v_threshold: Threshold for v-velocity update residual", 
            "- res_iter_v_threshold: Residual norm threshold for pressure correction",
            "",
            "Physical Parameters:",
            f"- Domain length: {params['length']}",
            f"- Domain breadth: {params['breadth']}",
            f"- Density (ρ): {params['rho']}",
            f"- Dynamic viscosity (μ): {params['mu']}",
            "",
            "Convergence Criteria:",
            "The simulation is considered converged when all of the following criteria are satisfied:",
            "- Mass conservation: |∇·u| < 1e-8 (mass residual tolerance)",
            "- Velocity field stability: RMSE changes < 1e-3 for u, v, and p fields",
            "- Steady-state achievement: residuals decrease monotonically",
            "- Physical validity: velocities remain bounded and pressure field is smooth",
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

    args = parser.parse_args()
    num_samples = args.num_samples
    task = args.task
    zero_shot = args.zero_shot
    flag = "zero_shot" if zero_shot else "iterative"

    dataset_dir = f"data/2D_ns/{task}"
    question_file = f"{dataset_dir}/{flag}_question.json"
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"[INFO] Generating {num_samples} samples for task '{task}'")
    print(f"[INFO] Zero-shot mode: {'ON' if zero_shot else 'OFF'}")
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

    generator = twoD_NSQuestionGenerator()
    dataset = generator.generate_question_dataset(
        num_samples=total_samples,
        task=task,
        zero_shot=zero_shot
    )
    save_dataset(dataset, question_file)