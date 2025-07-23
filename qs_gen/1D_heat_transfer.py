import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import Dict, Any
# from qs_gen.base import QuestionGenerator
from qs_gen.utils import *
import time
# from api_call.heat_transfer.wall_heat_transfer_solver import dummy_strategy
import argparse
from costsci_tools.gen_cfgs.heat_1d import create_heat1d_profiles
from costsci_tools.dummy_sols.heat_1d import find_convergent_cfl, find_convergent_n_space
import yaml

HEAT1D_CFG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "costsci_tools", "run_configs", "heat_1d"
)

class oneD_HeatTransferQuestionGenerator():
    """Generate training dataset for heat transfer problems"""
    def __init__(self):
        np.random.seed(42)

    def generate_question_dataset(self, num_samples: int, task: str, zero_shot: bool) -> List[Dict]:
        dataset: List[Dict] = []

        tolerance = 1e-4
        max_iter  = 20

        for idx in range(1, num_samples + 1):
            profile_name = f"p{idx}"
            profile_path = os.path.join(HEAT1D_CFG_DIR, f"{profile_name}.yaml")

            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            if task.lower() == "cfl":
                is_converged, best_param, cost_history, param_history = find_convergent_cfl(
                    profile=profile_name,
                    initial_cfl=1.0,
                    initial_n_space=100,
                    tolerance=tolerance,
                    max_iter=max_iter,
                )
                param_tag = "best_CFL"
            elif task.lower() == "n_space":
                is_converged, best_param, cost_history, param_history = find_convergent_n_space(
                    profile=profile_name,
                    initial_n_space=10,
                    cfl=1.0,
                    tolerance=tolerance,
                    max_iter=max_iter,
                )
                param_tag = "best_n_space"

            if best_param is not None:
                print(f"\nRecommended {param_tag}: {best_param}, the total cost is {cost_history}")
            else:
                print(f"\nNo convergent {param_tag} found within the given iterations, the total cost is {sum(cost_history[:-1])}")

            dummy_cost = cost_history[-2] if zero_shot else sum(cost_history[:-1])

            question = self.generate_question(dummy_cost, params, task)
            dataset.append(
                {
                    "QID":    idx,
                    "profile": profile_name,
                    "zero_shot": zero_shot,
                    "is_converged": is_converged,
                    "dummy_cost":  dummy_cost,
                    "cost_history": cost_history,
                    param_tag: best_param,
                    "param_history": param_history,
                    "question":  question,
                }
            )

        return dataset

    def generate_question(self, cost: int, params: Dict[str, Any], task: str) -> str:
        error_type = "Courant-Friedrichs-Lewy" if task == "cfl" else "n_space"
        question_lines = [
            "Problem: 1D transient heat conduction in a wall with:",
            f"- Wall thickness: {params['L']:.6f} m",
            f"- Left boundary: Convection (h = {params['h']:.3f} W/m^2-K, "
            f"T_inf = {params['T_inf']:.2f} C)",
            f"- Right boundary: Insulated (zero heat flux)",
            f"- Initial temperature: {params['T_init']:.2f} C (uniform)",
            f"- Thermal conductivity: {params['k']:.4f} W/m-K",
            f"- Specific heat: {params['cp']:.1f} J/kg-K",
            f"- Density: {params['rho']:.1f} kg/m^3",
            f"- Recording interval: {params['record_dt']:.4f} s",
            "",
            "Convergence criteria:",
            f"- {error_type} L2 error: the T gradient at the left tip between the adjacent parameters",
            "The criteria must be satisfied for convergence.",
            "- L2 ≤ 1e-4",
        ]

        return "\n".join(question_lines)

    # def generate_cost_sequence(self):
    # # Get unique process ID
    #     process_id = os.getpid()
    #     # Create unique subdirectory
    #     cache_dir = f"tool_result/process_{process_id}"
    #     os.makedirs(cache_dir, exist_ok=True)
        
    #     # Generate unique filename with timestamp
    #     timestamp = int(time.time() * 1000)
    #     params = self.generate_random_parameters()
    #     cost, sequence = dummy_strategy(
    #         **params,
    #         cache_file=f"{cache_dir}/qid_{timestamp}.h5",
    #         config_file=f"{cache_dir}/qid_{timestamp}.json",
    #         convergence_threshold=1e-4
    #     )
    #     return cost, sequence, params


if __name__ == "__main__":
    # Create generator and generate dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("-t", "--task", type=str, choices=["cfl", "n_space"], default="n_space",
                        help="Task of problem to solve")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                    help="Enable zero-shot mode")
    
    args = parser.parse_args()
    num_samples = args.num_samples
    task = args.task
    zero_shot = args.zero_shot
    flag = "zero_shot" if zero_shot else "iterative"

    dataset_dir = f"data/1D_heat_transfer/{task}"

    question_file = f"{dataset_dir}/{flag}_question.json"
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"[INFO] Generating {num_samples} samples for task '{task}'")
    print(f"[INFO] Zero-shot mode: {'ON' if zero_shot else 'OFF'}")
    print(f"[INFO] Output path: {question_file}")
    
    base_profile_path = "costsci_tools/run_configs/heat_1d/p1.yaml"
    if not os.path.isfile(base_profile_path):
        raise FileNotFoundError(f"Base profile path not found: {base_profile_path}")

    generator = oneD_HeatTransferQuestionGenerator()
    random_profiles = create_heat1d_profiles(num_profiles=num_samples-1, base_profile_path=base_profile_path, solver_name="heat_1d")
    dataset = generator.generate_question_dataset(num_samples=num_samples, task=task, zero_shot=zero_shot)
    save_dataset(dataset, question_file)  
