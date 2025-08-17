import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import Dict, Any
from qs_gen.utils import *
import time
import argparse
from costsci_tools.gen_cfgs.heat_steady_2d import create_heat1d_profiles
from costsci_tools.dummy_sols.heat_steady_2d import find_optimal_dx, grid_search_relax, grid_search_T_init, find_optimal_error_threshold
import yaml

HEAT2D_CFG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "costsci_tools", "run_configs", "heat_steady_2d"
)

class twoD_HeatTransferQuestionGenerator():
    """Generate training dataset for heat transfer problems"""
    def __init__(self):
        np.random.seed(42)

    def generate_question_dataset(self, num_samples: int, task: str, zero_shot: bool) -> List[Dict]:
        dataset: List[Dict] = []

        tolerance = 1e-3
        max_iter  = 10

        for idx in range(1, num_samples + 1):
            profile_name = f"p{idx}"
            profile_path = os.path.join(HEAT2D_CFG_DIR, f"{profile_name}.yaml")

            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            if task.lower() == "dx":
                is_converged, best_param, cost_history, param_history = find_optimal_dx(
                    profile=profile_name,
                    initial_dx=0.1,
                    relax=1.0,
                    error_threshold=1e-7,
                    T_init=0.25,
                    tolerance=tolerance,
                    max_iter=max_iter,
                )
                param_tag = "best_dx"
            elif task.lower() == "relax":
                # relax_values = [1.1]
                relax_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

                # cost_history is a dictionary here
                is_converged, best_param, cost_history, param_history = grid_search_relax(
                    profile=profile_name,
                    dx=0.005,
                    relax_values=relax_values,
                    error_threshold=1e-7,
                    T_init=0.25,
                    max_iter=max_iter,
                )
                param_tag = "optimal_relaxation_factor"
            elif task.lower() == "t_init":
                T_init_values = [
                    0.05,
                    0.1,
                    0.15,
                    0.2,
                    0.25,
                    0.3,
                    0.35,
                    0.4,
                    0.45,
                    0.5,
                    0.55,
                    0.6,
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                ]

                # cost_history is a dictionary here
                is_converged, best_param, cost_history, param_history = grid_search_T_init(
                    profile=profile_name,
                    dx=0.005,
                    relax=1.0,
                    error_threshold=1e-7,
                    T_init_values=T_init_values,
                    max_iter=max_iter,
                )
                param_tag = "optimal_initial_temperature"
            elif task.lower() == "error_threshold":
                is_converged, best_param, cost_history, param_history = find_optimal_error_threshold(
                    profile=profile_name,
                    dx=0.005,
                    relax=1.0,
                    initial_error=1e-5,
                    T_init=0.25,
                    tolerance=tolerance,
                    max_iter=max_iter,
                )
                param_tag = "best_error_threshold"

            if best_param is not None:
                print(f"\nRecommended {param_tag}: {best_param}, the total cost is {cost_history}")
            else:
                print(f"\nNo convergent {param_tag} found within the given iterations, the total cost is {sum(cost_history[:-1])}")

            if task.lower() == "relax" or task.lower() == "t_init":
                dummy_cost = cost_history[best_param]
            else:
                dummy_cost = cost_history[-2] if zero_shot else sum(cost_history[:-1])

            question = self.generate_question(dummy_cost, params, task)
            dataset.append(
                {
                    "QID":    idx,
                    "profile": profile_name,
                    "zero_shot": zero_shot,
                    "is_converged": is_converged,
                    "dummy_cost": dummy_cost,
                    "cost_history": cost_history,
                    param_tag: best_param,
                    "param_history": param_history,
                    "question":  question,
                }
            )

        return dataset

    def generate_question(self, cost: int, params: Dict[str, Any], task: str) -> str:
        question_lines = [
            "Problem: Steady State Heat Transfer in 2D",
            "",
            "Physical Parameters:",
            f"- T_top: {params['T_top']}",
            f"- T_bottom: {params['T_bottom']}",
            f"- T_left: {params['T_left']}",
            f"- T_right: {params['T_right']}",
            "",
            "Convergence criteria:",
            "- L2 ≤ 1e-3",
        ]

        return "\n".join(question_lines)

if __name__ == "__main__":
    # Create generator and generate dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("-t", "--task", type=str, choices=["dx", "relax", "t_init", "error_threshold"],
                    default="dx", help="Task of problem to solve")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode")

    args = parser.parse_args()
    num_samples = args.num_samples
    task = args.task

    # Force zero_shot to True for specific tasks
    zero_shot = args.zero_shot or task in ["relax", "t_init"]
    flag = "zero_shot" if zero_shot else "iterative"

    dataset_dir = f"data/2D_heat_transfer/{task}"
    question_file = f"{dataset_dir}/{flag}_question.json"
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"[INFO] Generating {num_samples} samples for task '{task}'")
    print(f"[INFO] Zero-shot mode: {'ON' if zero_shot else 'OFF'}")
    print(f"[INFO] Output path: {question_file}")

    base_profile_path = "costsci_tools/run_configs/heat_steady_2d/p1.yaml"
    if not os.path.isfile(base_profile_path):
        raise FileNotFoundError(f"Base profile path not found: {base_profile_path}")

    generator = twoD_HeatTransferQuestionGenerator()
    random_profiles = create_heat1d_profiles(
        num_profiles=num_samples - 1,
        base_profile_path=base_profile_path,
        solver_name="heat_steady_2d"
    )
    dataset = generator.generate_question_dataset(
        num_samples=num_samples,
        task=task,
        zero_shot=zero_shot
    )
    save_dataset(dataset, question_file)
