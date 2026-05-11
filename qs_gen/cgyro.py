import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from qs_gen.utils import *


CGYRO_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "cgyro"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "cgyro"
    / "successful"
    / "tasks.json"
)

# CGYRO precision levels are keyed on comparison_tolerance (eigenvalue L∞ bound).
# These match costsci_tools/docs/cgyro.md: low=1e-3, medium=1e-4, high=1e-5.
PRECISION_LEVELS = {
    "low": {"tolerance_rmse": 1e-3},
    "medium": {"tolerance_rmse": 1e-4},
    "high": {"tolerance_rmse": 1e-5},
}

TASKS = ["n_radial", "n_theta", "n_xi", "n_energy", "freq_tol", "delta_t"]


class CgyroQuestionGenerator:
    """Generate CGYRO training-set questions from pre-computed tasks.json."""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()

    def _load_tasks_data(self) -> Dict:
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def generate_all_questions(self) -> None:
        total_files = 0
        total_questions = 0

        print("=" * 80)
        for task in TASKS:
            print(f"\n📋 TASK: {task.upper()}")
            print("-" * 50)

            for precision_level in PRECISION_LEVELS.keys():
                tolerance = PRECISION_LEVELS[precision_level]["tolerance_rmse"]
                print(f"  🎯 {precision_level.upper()} precision (Δeig ≤ {tolerance:.0e}):")

                dataset_iter = self.generate_question_dataset(task, precision_level, zero_shot=False)
                self._save_dataset(dataset_iter, task, precision_level, zero_shot=False)

                dataset_zero = self.generate_question_dataset(task, precision_level, zero_shot=True)
                self._save_dataset(dataset_zero, task, precision_level, zero_shot=True)

                total_files += 2
                total_questions += len(dataset_iter) + len(dataset_zero)
                print(f"     ✓ Iterative: {len(dataset_iter):2d} questions")
                print(f"     ✓ Zero-shot: {len(dataset_zero):2d} questions")

        print("\n" + "=" * 80)
        print(f"🎉 SUMMARY: Generated {total_questions} questions across {total_files} files")
        print("=" * 80)

    def generate_question_dataset(self, task: str, precision_level: str, zero_shot: bool) -> List[Dict]:
        dataset: List[Dict] = []
        tolerance = PRECISION_LEVELS[precision_level]["tolerance_rmse"]

        matching_tasks = [
            t for t in self.tasks_data["tasks"]
            if (t["target_parameter"] == task and
                t["precision_config"].get("comparison_tolerance") == tolerance)
        ]

        qid = 1
        for task_data in matching_tasks:
            profile = task_data["profile"]

            profile_path = CGYRO_CFG_DIR / f"{profile}.yaml"
            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            results = task_data["results"]
            cost_history = results["cost_history"]
            param_history = results["parameter_history"]
            is_converged = results["converged"]

            # 0-shot reference cost = cost of the penultimate (finest-but-one)
            # run; iterative = sum of all runs except the final verification.
            dummy_cost = cost_history[-2] if (zero_shot and len(cost_history) >= 2) else sum(cost_history[:-1]) if len(cost_history) > 1 else cost_history[0]

            optimal_value = results["optimal_parameter_value"]
            best_params = None
            for param_set in param_history:
                if isinstance(param_set, list):
                    param_dict = param_set[-1]
                else:
                    param_dict = param_set
                if param_dict.get(task) == optimal_value:
                    best_params = dict(param_dict)
                    break

            if best_params is None:
                raise ValueError(
                    f"Could not find optimal {task}={optimal_value} in parameter_history for {profile}"
                )

            dataset.append({
                "QID": qid,
                "profile": profile,
                "zero_shot": zero_shot,
                "target_parameter": task,
                "precision_level": precision_level,
                "tolerance_rmse": tolerance,
                "is_converged": is_converged,
                "dummy_cost": dummy_cost,
                "cost_history": cost_history,
                "best_params": best_params,
                "param_history": param_history,
                "question": self._build_question_text(profile, params, precision_level, tolerance),
            })
            qid += 1

        return dataset

    @staticmethod
    def _build_question_text(profile: str, params: dict, precision_level: str, tolerance: float) -> str:
        """Build question text for a CGYRO linear simulation."""
        rmin = params.get("RMIN", params.get("rmin"))
        ky = params.get("KY", params.get("ky"))
        question_lines = [
            "Problem: CGYRO Linear Gyrokinetic Simulation",
            "",
            "This simulation uses the local-spectral gyrokinetic code CGYRO to solve the nonlinear gyrokinetic equations governing turbulent transport in magnetized fusion plasmas. We treat only linear simulations (no nonlinear mode coupling), resolving the most unstable linear eigenmode of a DIII-D-like tokamak discharge.",
            "",
            "General procedure:",
            "",
            "1. The Fokker-Planck–Maxwell system is reduced via the gyrokinetic approximation (strong magnetization, scale separation). Fast gyromotion is averaged out, yielding a 5D phase-space system for the gyrocenter distribution function.",
            "",
            "2. Perpendicular directions are represented in Fourier space; the field-aligned coordinate is on a field-aligned grid; velocity space is on a 2D grid in pitch angle and particle energy. This converts the integro-differential system into a coupled algebraic system per species and Fourier mode.",
            "",
            "3. Time integration is operator-split: collisionless terms explicit, collisional terms implicit. At each step, moments of the distribution function solve the Maxwell field equations (quasineutrality, Ampère).",
            "",
            "4. Linear simulations are run until the most-unstable eigenvalue ω + iγ stabilises. The real part ω is the mode frequency; its sign identifies ion-mode vs electron-mode propagation. The imaginary part γ is the growth rate.",
            "",
            f"Plasma Configuration ({profile}):",
            f"- Scaled minor radius rmin = r/a: {rmin}",
            f"- Normalized poloidal wavenumber ky: {ky}",
            "",
            "Tunable Parameters:",
            "- **n_radial**: Number of radial wavenumber points (radial Fourier harmonics retained). Integer.",
            "- **n_theta**: Number of poloidal grid points. Integer.",
            "- **n_xi**: Number of Legendre pseudospectral mesh points in pitch angle. Integer.",
            "- **n_energy**: Number of generalized-Laguerre pseudospectral mesh points in energy. Integer.",
            "- **freq_tol**: Eigenvalue convergence tolerance for the linear solve. Positive real.",
            "- **delta_t**: Initial simulation timestep (adaptively refined to meet the error tolerance). Positive real.",
            "",
            "Convergence Check:",
            "- The simulation is considered internally converged when eigenvalues meet CGYRO's freq_tol threshold.",
            "- Cross-resolution convergence is assessed by comparing the final eigenvalue of the current run against a self-refined run (doubled resolution for integer parameters; halved tolerance or timestep for freq_tol/delta_t).",
            "- Convergence is confirmed if max(|Δω|, |Δγ|) ≤ tolerance.",
            "",
            "Validation Criteria:",
            f"- **Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Required Eigenvalue L∞ Tolerance**: ≤ {tolerance:.0e}",
            "- The real frequency ω and growth rate γ must both agree within the tolerance across refinement levels.",
        ]
        return "\n".join(question_lines)

    def _save_dataset(self, dataset: List[Dict], task: str, precision_level: str, zero_shot: bool) -> None:
        base_dir = Path("data") / "cgyro" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)


def main() -> None:
    print("🚀 CGYRO QUESTION GENERATOR")
    print("=" * 80)
    print(f"📂 Output directory: data/cgyro/{{task}}/{{precision_level}}/")
    print(f"📋 Tasks: {', '.join(TASKS)}")
    print(f"🎯 Precision levels: {list(PRECISION_LEVELS.keys())}")

    CgyroQuestionGenerator().generate_all_questions()
    print(f"\n✅ All questions generated successfully!")


if __name__ == "__main__":
    main()
