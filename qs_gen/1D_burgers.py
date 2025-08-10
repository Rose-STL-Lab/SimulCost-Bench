"""
Generate Burgers-1D training questions and save them in the required
folder / file layout.

Directory layout created (example):

data/
└── burgers_1d/
    ├── cfl/
    │   ├── sin/
    │   │   ├── iterative_questions.json
    │   │   └── zero_shot_questions.json
    │   ├── rarefaction/ …
    │   └── (5 cases total)
    ├── k/
    │   ├── sin/k_questions.json
    │   └── …
    └── w/
        ├── sin/w_questions.json
        └── …
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
import argparse

from qs_gen.utils import *
from costsci_tools.dummy_sols.burgers_1d import (
    find_convergent_cfl,
    find_optimal_k,
    find_optimal_w,
)

# -----------------------------------------------------------------------------#
#  Global settings & combinations                                               #
# -----------------------------------------------------------------------------#
BURGERS1D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "burgers_1d"
)

# Fixed (w, k) combinations. Total: 45 + 15 + 15 = 75 questions
COMBOS = {
    "cfl": [  # 9 × 5 profile = 45
        {"w": 1.0, "k": 1},
        {"w": 1.0, "k": 0},
        {"w": 1.0, "k": -1},
        {"w": 1.5, "k": 1},
        {"w": 1.5, "k": 0},
        {"w": 1.5, "k": -1},
        {"w": 2.0, "k": 1},
        {"w": 2.0, "k": 0},
        {"w": 2.0, "k": -1},
    ],
    "k": [  # 3 × 5 profile = 15
        {"w": 1.0},
        {"w": 1.5},
        {"w": 2.0},
    ],
    "w": [  # 3 × 5 profile = 15
        {"k": -1},
        {"k": 0},
        {"k": 1},
    ],
}


# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class OneDBurgersQuestionGenerator:
    """Generate Burgers-1D training-set questions"""

    def __init__(self) -> None:
        np.random.seed(42)

    # --------------------------------------------------------------------- #
    def generate_question_dataset(self, task: str, zero_shot: bool) -> List[Dict]:
        """Return a list of question dicts with QID reset per *case*"""
        dataset: List[Dict] = []

        tol_inf = 5e-2
        tol_l2 = 5e-3

        # --- five profiles: p1 … p5 (each profile == one physical case) --- #
        for profile_idx in range(1, 6):
            profile_name = f"p{profile_idx}"
            profile_path = BURGERS1D_CFG_DIR / f"{profile_name}.yaml"

            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            case_name = params["case"]
            qid_case = 1                     # ★ reset QID for this case ★

            # ---- iterate over (w,k) / (w) / (k) combos ------------------- #
            for combo in COMBOS[task]:
                if task == "cfl":
                    k_val, w_val = combo["k"], combo["w"]
                    (
                        is_converged,
                        best_param,
                        cost_history,
                        param_history,
                    ) = find_convergent_cfl(
                        profile=profile_name,
                        cfl=1.0,
                        k=k_val,
                        w=w_val,
                        tolerance_infity=tol_inf,
                        tolerance_2=tol_l2,
                    )
                    param_tag = "best_cfl"

                elif task == "k":
                    w_val = combo["w"]
                    (
                        is_converged,
                        best_param,
                        cost_history,
                        param_history,
                    ) = find_optimal_k(
                        profile=profile_name,
                        w=w_val,
                        tolerance_infity=tol_inf,
                        tolerance_2=tol_l2,
                    )
                    param_tag = "best_k"

                elif task == "w":
                    k_val = combo["k"]
                    (
                        is_converged,
                        best_param,
                        cost_history,
                        param_history,
                    ) = find_optimal_w(
                        profile=profile_name,
                        k=k_val,
                        tolerance_infity=tol_inf,
                        tolerance_2=tol_l2,
                    )
                    param_tag = "best_w"
                else:
                    raise ValueError(f"Unknown task: {task}")

                dummy_cost = (
                    cost_history[-2] if zero_shot else sum(cost_history[:-1])
                )

                if task == "cfl":
                    dataset.append(
                        {
                            "QID": qid_case,           # ★ per-case local ID ★
                            "profile": profile_name,
                            "case": case_name,
                            "zero_shot": zero_shot,
                            "is_converged": is_converged,
                            "dummy_cost": dummy_cost,
                            "cost_history": cost_history,
                            param_tag: best_param,
                            "param_history": param_history,
                            "question": self._build_question_text(
                                dummy_cost, params, task
                            ),
                        }
                    )
                elif task == "k":
                    dataset.append(
                        {
                            "QID": qid_case,
                            "profile": profile_name,
                            "case": case_name,
                            "zero_shot": zero_shot,
                            "optimal_is_converged": is_converged,
                            "dummy_cost": dummy_cost,
                            "optimal_cost_history": cost_history,
                            "best_k": best_param[0],
                            "best_cfl": best_param[1],
                            "param_history": param_history,
                            "question": self._build_question_text(
                                dummy_cost, params, task
                            ),
                        }
                    )
                elif task == "w":
                    dataset.append(
                        {
                            "QID": qid_case,
                            "profile": profile_name,
                            "case": case_name,
                            "zero_shot": zero_shot,
                            "optimal_is_converged": is_converged,
                            "dummy_cost": dummy_cost,
                            "optimal_cost_history": cost_history,
                            "best_w": best_param[0],
                            "best_cfl": best_param[1],
                            "param_history": param_history,
                            "question": self._build_question_text(
                                dummy_cost, params, task
                            ),
                        }
                    )
                else:
                    raise ValueError(f"Unknown task: {task}")

                qid_case += 1                     # ★ increment local ID ★

        return dataset

    @staticmethod
    def _build_question_text(cost: float, params: dict, task: str) -> str:
        question_lines = [
            "Problem: Burgers 1D Equation with 2nd Order Roe Method",
            "",
            "This simulation solves the 1D inviscid Burgers equation, a simplified model for compressible gas dynamics, using a 2nd order Roe method with minmod limiter:",
            "",
            "∂u/∂t + u ∂u/∂x = 0",
            "",
            "Spatial discretization:",
            "u_{j+1/2}^l = u_{j+1} - κ/4 * δ⁺u_{j-1/2} + (1 + κ)/4 * δ⁻u_{j+1/2}",
            "u_{j+1/2}^r = u_j - (1 + κ)/4 * δ⁺u_{j+1/2} + (1 - κ)/4 * δ⁻u_{j+3/2}",
            "",
            "κ is the blending coefficient: κ=1 for central, κ=-1 for upwind scheme.",
            "",
            "Slope limiter:",
            "δ⁻u_{j+1/2} = minmod(ω * (u_j - u_{j-1}) / Δx, (u_{j+1} - u_j) / Δx)",
            "δ⁺u_{j+1/2} = minmod((u_{j+1} - u_j) / Δx, ω * (u_{j+2} - u_{j+1}) / Δx)",
            "ω is a generalization parameter controlling farther side slope influence.",
            "",
            "Initial condition cases:",
            "- sin: u(x,0) = sin(2πx / L) + 0.5",
            "- rarefaction: u(x,0) = -0.1 if x < L/2, 0.5 otherwise",
            "- sod: u(x,0) = 1.0 if x < L/2, 0.1 otherwise",
            "- double_shock: u(x,0) = 1.0 (x < L/3), 0.5 (L/3 ≤ x < 2L/3), 0.1 (x ≥ 2L/3)",
            "- blast: u(x,0) = exp(-(x-L/4)^2/(2σ^2)) + 0.8*exp(-(x-3L/4)^2/(2σ^2)), with σ=L/20",
            "",
            "Physical Parameters:",
            f"- Domain length: {params['L']}",
            f"- Case: {params['case']}",
            "",
            "Convergence Check:",
            "- The current CFL value will be halved (veri_cfl = cfl / 2), and the simulation is rerun with veri_cfl.",
            "- Errors between the two simulations are computed to assess convergence.",
            "- Convergence is confirmed if the following validation criteria are satisfied.",
            "",
            "Validation Criteria:",
            "- L2 norm ≤ 5e-3 and L∞ norm ≤ 5e-2",
            "- Mass conservation: total integral constant over time",
            "- Energy non-increasing: ∫u² dx must not increase",
            "- Total Variation (TV) non-increasing: ∑|u_{i+1} - u_i| decreases",
            "- Maximum principle: values remain within initial min/max"
        ]
        return "\n".join(question_lines)
    
# -----------------------------------------------------------------------------#
#  Saving utilities                                                            #
# -----------------------------------------------------------------------------#
def _group_by_case(records: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for rec in records:
        grouped.setdefault(rec["case"], []).append(rec)
    return grouped

def save_dataset_by_case(
    records: List[Dict], task: str, zero_shot: bool
) -> None:
    """Write JSON files into data/burgers_1d/<task>/<case>/ ..."""
    base_dir = Path("data") / "burgers_1d" / task
    case_groups = _group_by_case(records)

    for case_name, recs in case_groups.items():
        out_dir = base_dir / case_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if task in ["cfl", "k", "w"]:
            fname = (
                "zero_shot_questions.json"
                if zero_shot
                else "iterative_questions.json"
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        out_path = out_dir / fname
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(recs, fp, ensure_ascii=False, indent=2)

        print(
            f"[{datetime.now():%H:%M:%S}] Saved {len(recs):2d} "
            f"→ {out_path.as_posix()}"
        )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["cfl", "k", "w"],
        default="cfl",
        help="Target parameter to optimise",
    )
    parser.add_argument(
        "-z",
        "--zero_shot",
        action="store_true",
        help="Generate zero-shot versions "
        "(iterative if not set, only meaningful for task=cfl)",
    )
    args = parser.parse_args()

    task = args.task.lower()
    zero_shot = args.zero_shot

    print(f"[INFO] Task: {task.upper()}   Zero-shot: {zero_shot}")

    gen = OneDBurgersQuestionGenerator()
    dataset = gen.generate_question_dataset(task=task, zero_shot=zero_shot)
    save_dataset_by_case(dataset, task=task, zero_shot=zero_shot)


if __name__ == "__main__":
    main()
