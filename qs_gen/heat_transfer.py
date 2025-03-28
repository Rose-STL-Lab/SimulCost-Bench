import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import Dict, Any
from qs_gen.base import QuestionGenerator
from qs_gen.utils import *
import time
from api_call.heat_transfer.wall_heat_transfer_solver import dummy_strategy
import argparse


class HeatTransferQuestionGenerator(QuestionGenerator):
    """Generate training dataset for heat transfer problems"""
    def __init__(self, num_workers: int=10):
        super().__init__(num_workers)
        np.random.seed(int(time.time()))

    def generate_random_parameters(self) -> Dict[str, Any]:
        """Generate physically reasonable random parameters"""
        # Randomly select a material
        log_h_min = np.log10(0.1)
        log_h_max = np.log10(100)
        log_h = np.random.uniform(log_h_min, log_h_max)
        h = round(10**log_h, 2)  # Round to 1 decimal place
        L = round(np.random.uniform(0.1, 0.2), 3)
        k = round(np.random.uniform(0.5, 1), 2)
        rho = round(np.random.uniform(1000, 2000))
        cp = round(np.random.uniform(800, 1000))
        T_inf = round(np.random.uniform(4, 20))
        T_init = round(np.random.uniform(21, 30))
        t_final = round(np.random.uniform(900, 3600))

        return {
            "L": L,
            "k": k,
            "h": h,
            "rho": rho,
            "cp": cp,
            "T_inf": T_inf,
            "T_init": T_init,
            "t_final": t_final,
        }

    def generate_question(self, budget: int, params: Dict[str, Any]) -> str:
        """Generate the question text"""
        question = (
            f"Problem: 1D transient heat conduction in a wall with:\n"
            f"- Wall thickness: {params['L']:f} m\n"
            f"- Left boundary: Convection (h={params['h']:f} W/m\u00b2\u00b7K, T_\u221E={params['T_inf']:f}C)\n"
            f"- Right boundary: Insulated (zero heat flux)\n"
            f"- Initial temperatue: {params['T_init']:f}C uniformly\n"
            f"- Thermal conductivity: {params['k']:f} W/m\u00b7K\n"
            f"- Specific heat: {params['cp']:f} J/kg\u00b7K\n"
            f"- Density: {params['rho']:f} kg/m\u00b3\n"
            f"- End time: {params['t_final']:f} s\n"
        )
        convergence_criteria = (
            f"Convergence criteria:\n"
            f"1. Spatial: L2 error < 1e-4\u00b0C\n"
            f"2. Temporal: L2 error < 1e-4\u00b0C\n(where L2 error would be average squared difference across all grid points at each time step)\n"
            f"(Note: Both spatial and temporal L2 errors must be less than 1e-4\u00b0C to be considered converged)\n"
        )
        cost_budget = (
            f"Cost budget: {budget}\n"
        )
        question = question + convergence_criteria + cost_budget
        return question

    def generate_cost_sequence(self):
    # Get unique process ID
        process_id = os.getpid()
        # Create unique subdirectory
        cache_dir = f"tool_result/process_{process_id}"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = int(time.time() * 1000)
        params = self.generate_random_parameters()
        cost, sequence = dummy_strategy(
            **params,
            cache_file=f"{cache_dir}/qid_{timestamp}.h5",
            config_file=f"{cache_dir}/qid_{timestamp}.json",
            convergence_threshold=1e-4
        )
        return cost, sequence, params


if __name__ == "__main__":
    # Create generator and generate dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=5,
                        help="Number of samples to generate")
    args = parser.parse_args()
    dataset_dir = "data/heat_transfer"
    question_file = f"{dataset_dir}/question.json"
    os.makedirs(dataset_dir, exist_ok=True)
    generator = HeatTransferQuestionGenerator(num_workers=10)
    num_samples = args.num_samples
    
    dataset = generator.generate_quesion_dataset(num_samples=num_samples)
    save_dataset(dataset, question_file)  



