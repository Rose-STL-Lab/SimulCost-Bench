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
from costsci_tools.dummy_sols.euler_1d import (
    find_convergent_cfl,
    find_optimal_beta,
    find_optimal_k,
)

# -----------------------------------------------------------------------------#
#  Global settings & combinations                                               #
# -----------------------------------------------------------------------------#
EULER1D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "euler_1d"
)

# 固定的 (beta, k) 组合
COMBOS = {
    "cfl": [  # 3 × 3 profile = 9
        {"beta": 0.5, "k": -1.0},
        {"beta": 0.5, "k": 0.0},
        {"beta": 0.5, "k": 1.0},
        {"beta": 1.0, "k": -1.0},
        {"beta": 1.0, "k": 0.0},
        {"beta": 1.0, "k": 1.0},
        {"beta": 2.0, "k": -1.0},
        {"beta": 2.0, "k": 0.0},
        {"beta": 2.0, "k": 1.0},
    ],
    "beta": [  # 3 × 3 profile = 9
        {"k": -1.0},
        {"k": 0.0},
        {"k": 1.0},
    ],
    "k": [  # 3 × 3 profile = 9
        {"beta": 0.5},
        {"beta": 1.0},
        {"beta": 2.0},
    ],
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class OneDEulerQuestionGenerator:
    """Generate Euler-1D training-set questions"""

    def __init__(self) -> None:
        np.random.seed(42)

    # --------------------------------------------------------------------- #
    def generate_question_dataset(self, task: str, zero_shot: bool) -> List[Dict]:
        """Return a list of question dicts with QID reset per *case*"""
        dataset: List[Dict] = []

        tol_linf = 0.2
        tol_rmse = 0.02

        # --- euler_1d profile (currently only one case: sod) --- #
        profile_name = "p1"
        profile_path = EULER1D_CFG_DIR / f"{profile_name}.yaml"

        with open(profile_path, "r", encoding="utf-8") as fp:
            params: Dict[str, Any] = yaml.safe_load(fp)

        case_name = params["case"]
        qid_case = 1  # ★ reset QID for this case ★

        # ---- iterate over (beta, k) / (beta) / (k) combos ------------------- #
        for combo in COMBOS[task]:
            if task == "cfl":
                beta_val, k_val = combo["beta"], combo["k"]
                (
                    is_converged,
                    best_param,
                    cost_history,
                    param_history,
                ) = find_convergent_cfl(
                    profile=profile_name,
                    cfl=1.0,
                    beta=beta_val,
                    k=k_val,
                    tolerance_linf=tol_linf,
                    tolerance_rmse=tol_rmse,
                )
                param_tag = "best_cfl"

            elif task == "beta":
                k_val = combo["k"]
                (
                    is_converged,
                    best_param,
                    cost_history,
                    param_history,
                ) = find_optimal_beta(
                    profile=profile_name,
                    k=k_val,
                    tolerance_linf=tol_linf,
                    tolerance_rmse=tol_rmse,
                )
                param_tag = "best_beta"

            elif task == "k":
                beta_val = combo["beta"]
                (
                    is_converged,
                    best_param,
                    cost_history,
                    param_history,
                ) = find_optimal_k(
                    profile=profile_name,
                    beta=beta_val,
                    tolerance_linf=tol_linf,
                    tolerance_rmse=tol_rmse,
                )
                param_tag = "best_k"
            else:
                raise ValueError(f"Unknown task: {task}")

            dummy_cost = (
                cost_history[-2] if zero_shot else sum(cost_history[:-1])
            )

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

            qid_case += 1  # ★ increment local ID ★

        return dataset

    @staticmethod
    def _build_question_text(cost: float, params: dict, task: str) -> str:
        question_lines = [
            "Problem: Euler 1D Equations with 2nd Order MUSCL-Roe Method",
            "",
            "This simulation solves the 1D Euler equations for compressible inviscid flow, using a 2nd order MUSCL scheme with Roe flux and generalized superbee limiter.",
            "",
            "Conservative form:",
            r"$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = 0$",
            "where the conservative variables and flux are:",
            r"$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho E \end{pmatrix}, \quad \mathbf{F} = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ u(\rho E + p) \end{pmatrix}$",
            "",
            "Primitive variables:",
            "- $\rho$ = density",
            "- $u$ = velocity",
            "- $p$ = pressure",
            "- $E$ = specific total energy",
            "",
            "Equation of state:",
            r"$p = (\gamma - 1) \rho \left(E - \frac{u^2}{2}\right)$",
            "where $\gamma$ is the ratio of specific heats.",
            "",
            "Spatial Discretization:",
            "The spatial discretization uses MUSCL reconstruction with blending parameter $k$:",
            r"$\mathbf{U}^L_{j+\frac{1}{2}} = \mathbf{U}_j + \frac{1+k}{4} \psi(r_{j}) (\mathbf{U}_{j+1} - \mathbf{U}_{j})$",
            r"$\mathbf{U}^R_{j+\frac{1}{2}} = \mathbf{U}_{j+1} - \frac{1+k}{4} \psi(r_{j+1}) (\mathbf{U}_{j+2} - \mathbf{U}_{j+1})$",
            "where $k$ is a blending coefficient between central ($k=1$) and upwind ($k=-1$) scheme, and $\psi(r)$ is the slope limiter function.",
            "",
            "Slope Limiting:",
            "The slope limiter uses a generalized superbee limiter:",
            r"$\psi(r) = \max\left[0, \max\left[\min(\beta r, 1), \min(r, \beta)\right]\right]$",
            "where $\beta$ is the limiter parameter controlling dissipation.",
            "The slope ratio $r$ at interface $j$ is defined as:",
            r"$r_{j} = \frac{\mathbf{U}_{j+1} - \mathbf{U}_{j}}{\mathbf{U}_{j+2} - \mathbf{U}_{j+1}}$",
            "This ratio indicates the local non-smoothness, which will be the input into the slope limiter to achieve the TVD condition.",
            "",
            "Flux Computation:",
            "The interface flux is computed using the Roe approximate Riemann solver:",
            r"$\mathbf{F}_{j+\frac{1}{2}} = \frac{1}{2}\left[\mathbf{F}(\mathbf{U}^L) + \mathbf{F}(\mathbf{U}^R)\right] - \frac{1}{2}|\mathbf{A}|(\mathbf{U}^R - \mathbf{U}^L)$",
            "where $|\mathbf{A}|$ is the Roe matrix with Roe-averaged quantities.",
            "",
            "Test Case:",
            "Initial condition case (Sod Shock Tube):",
            "- Left: $\rho=1.0, u=0.0, p=1.0$",
            "- Right: $\rho=0.125, u=0.0, p=0.1$",
            "",
            "Physical Parameters:",
            f"- Gamma: {params['gamma']}",
            f"- Domain length: {params['L']}",
            f"- Case: {params['case']}",
            "",
            "Convergence Check:",
            "- The current CFL value will be halved (veri_cfl = cfl / 2), and the simulation is rerun with veri_cfl.",
            "- Errors between the two simulations are computed to assess convergence.",
            "- Convergence is confirmed if the following validation criteria are satisfied.",
            "",
            "Validation Criteria:",
            "- RMSE ≤ 5e-3 and L∞ norm ≤ 5e-2",
            "- Mass conservation: total integral of density remains constant over time",
            "- Energy conservation: the total energy ∫ρE dx should remain constant",
            "- Positivity preservation: pressure and density must remain positive at all times",
            "- Shock speed consistency: pressure gradients should not exceed physical bounds"
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
    """Write JSON files into data/euler_1d/<task>/<case>/ ..."""
    base_dir = Path("data") / "euler_1d" / task
    case_groups = _group_by_case(records)

    for case_name, recs in case_groups.items():
        out_dir = base_dir / case_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if task in ["cfl", "beta", "k"]:
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
        choices=["cfl", "beta", "k"],
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

    gen = OneDEulerQuestionGenerator()
    dataset = gen.generate_question_dataset(task=task, zero_shot=zero_shot)
    save_dataset_by_case(dataset, task=task, zero_shot=zero_shot)


if __name__ == "__main__":
    main()
