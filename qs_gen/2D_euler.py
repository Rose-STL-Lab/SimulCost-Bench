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
EULER_2D_CFG_DIR = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "run_configs"
    / "euler_2d"
)

TASKS_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "costsci_tools"
    / "dataset"
    / "euler_2d"
    / "successful"
    / "tasks.json"
)

PRECISION_LEVELS = {
    "high": {
        "tolerance_rmse": 0.008,
        "description": "High precision convergence - very challenging"
    },
    "medium": {
        "tolerance_rmse": 0.04,
        "description": "Medium precision convergence - challenging"
    },
    "low": {
        "tolerance_rmse": 0.08,
        "description": "Low precision convergence - achievable"
    },
}

TESTCASE_NAMES = {
    0: "central_explosion",
    1: "stair_flow",
    2: "sod_tube",
    3: "lax_tube",
    4: "mach_3",
    5: "strong_tube",
    6: "high_mach",
    7: "interact_blast",
    8: "rarefaction"
}

# -----------------------------------------------------------------------------#
#  Question generator                                                           #
# -----------------------------------------------------------------------------#
class TwoDEulerQuestionGenerator:
    """Generate Euler 2D training-set questions from pre-computed results"""

    def __init__(self) -> None:
        np.random.seed(42)
        self.tasks_data = self._load_tasks_data()

    def _load_tasks_data(self) -> Dict:
        """Load pre-computed tasks results with fail-fast validation"""
        with open(TASKS_JSON_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        # Validate metadata
        if "metadata" not in data:
            raise KeyError("Missing required field 'metadata' in tasks.json")
        if "tasks" not in data:
            raise KeyError("Missing required field 'tasks' in tasks.json")

        return data

    def _validate_task_entry(self, task_data: Dict) -> None:
        """Fail-fast validation of task entry fields"""
        required_fields = [
            "task_id", "profile", "precision_level", "tolerance_rmse",
            "target_parameter", "non_target_parameters", "is_converged",
            "optimal_parameters", "cost_history", "parameter_history"
        ]

        for field in required_fields:
            if field not in task_data:
                raise KeyError(
                    f"Missing required field '{field}' in task entry "
                    f"{task_data.get('task_id', 'unknown')}"
                )

        # Validate target_parameter
        valid_targets = ["n_grid_x", "cfl", "cg_tolerance"]
        target = task_data["target_parameter"]
        if target not in valid_targets:
            raise ValueError(
                f"Invalid target_parameter '{target}' in task {task_data['task_id']}. "
                f"Must be one of {valid_targets}"
            )

        # Validate precision_level
        precision_level = task_data["precision_level"]
        if precision_level not in PRECISION_LEVELS:
            raise ValueError(
                f"Invalid precision_level '{precision_level}' in task {task_data['task_id']}. "
                f"Must be one of {list(PRECISION_LEVELS.keys())}"
            )

        # Validate tolerance_rmse matches precision level
        expected_tolerance = PRECISION_LEVELS[precision_level]["tolerance_rmse"]
        actual_tolerance = task_data["tolerance_rmse"]
        if abs(actual_tolerance - expected_tolerance) > 1e-10:
            raise ValueError(
                f"Tolerance mismatch in task {task_data['task_id']}: "
                f"precision_level='{precision_level}' expects tolerance_rmse={expected_tolerance}, "
                f"but got {actual_tolerance}"
            )

        # Validate parameter_history is not empty
        if not task_data["parameter_history"]:
            raise ValueError(
                f"Empty parameter_history in task {task_data['task_id']}"
            )

    def generate_all_questions(self) -> None:
        """Generate questions for all tasks and precision levels"""
        tasks = ["n_grid_x", "cfl", "cg_tolerance"]
        # All three tasks support both iterative and zero-shot modes

        total_files = 0
        total_questions = 0

        print("=" * 80)
        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)

            for precision_level in PRECISION_LEVELS.keys():
                precision_config = PRECISION_LEVELS[precision_level]
                tolerance_rmse = precision_config["tolerance_rmse"]

                print(f"  {precision_level.upper()} precision:")
                print(f"    Tolerance RMSE: {tolerance_rmse}")

                # Generate iterative questions (all tasks support iterative)
                dataset_iter = self.generate_question_dataset(
                    task=task,
                    precision_level=precision_level,
                    zero_shot=False
                )
                if dataset_iter:  # Only save if there are questions
                    self._save_dataset(
                        dataset_iter,
                        task=task,
                        precision_level=precision_level,
                        zero_shot=False
                    )
                    total_files += 1
                    total_questions += len(dataset_iter)
                    print(f"      Iterative: {len(dataset_iter):2d} questions")
                else:
                    print(f"      Iterative: 0 questions (no data)")

                # Generate zero-shot questions (all tasks support zero-shot)
                dataset_zero = self.generate_question_dataset(
                    task=task,
                    precision_level=precision_level,
                    zero_shot=True
                )
                if dataset_zero:  # Only save if there are questions
                    self._save_dataset(
                        dataset_zero,
                        task=task,
                        precision_level=precision_level,
                        zero_shot=True
                    )
                    total_files += 1
                    total_questions += len(dataset_zero)
                    print(f"      Zero-shot: {len(dataset_zero):2d} questions")
                else:
                    print(f"      Zero-shot: 0 questions (no data)")

        print("\n" + "=" * 80)
        print(f"SUMMARY: Generated {total_questions} questions across {total_files} files")
        print("=" * 80)

    def generate_question_dataset(
        self,
        task: str,
        precision_level: str,
        zero_shot: bool
    ) -> List[Dict]:
        """Generate question dataset for specific task and precision level"""
        dataset: List[Dict] = []
        precision_config = PRECISION_LEVELS[precision_level]

        # Filter tasks by target parameter and precision level
        matching_tasks = []
        for t in self.tasks_data["tasks"]:
            # Validate task entry first (fail-fast)
            self._validate_task_entry(t)

            if (t["target_parameter"] == task and
                t["precision_level"] == precision_level):
                matching_tasks.append(t)

        qid = 1
        for task_data in matching_tasks:
            profile = task_data["profile"]

            # Load profile configuration
            profile_path = EULER_2D_CFG_DIR / f"{profile}.yaml"
            with open(profile_path, "r", encoding="utf-8") as fp:
                params: Dict[str, Any] = yaml.safe_load(fp)

            # Extract results
            cost_history = task_data["cost_history"]
            param_history = task_data["parameter_history"]
            is_converged = task_data["is_converged"]

            # Calculate dummy cost according to requirements
            # For zero-shot: use the second-to-last cost
            # For iterative: sum all costs except the last one
            if zero_shot and len(cost_history) > 1:
                dummy_cost = cost_history[-2]
            elif len(cost_history) > 1:
                dummy_cost = sum(cost_history[:-1])
            else:
                dummy_cost = cost_history[0]

            # Find the best_params from parameter_history based on optimal_parameters
            optimal_parameters = task_data["optimal_parameters"]
            best_params = None

            # Search through parameter_history to find the matching parameter combination
            for param_set in param_history:
                if param_set.get(task) == optimal_parameters[task]:
                    best_params = param_set.copy()
                    break

            # If not found, this is an error in the data
            if best_params is None:
                raise ValueError(
                    f"Could not find optimal parameter {task}={optimal_parameters[task]} "
                    f"in parameter_history for task {task_data['task_id']}"
                )

            # Filter best_params to only include the 3 tunable parameters
            filtered_best_params = {
                key: best_params[key] for key in ["n_grid_x", "cfl", "cg_tolerance"]
                if key in best_params
            }

            # Filter param_history to only include the 3 tunable parameters
            filtered_param_history = []
            for param_set in param_history:
                filtered_set = {
                    key: param_set[key] for key in ["n_grid_x", "cfl", "cg_tolerance"]
                    if key in param_set
                }
                filtered_param_history.append(filtered_set)

            # Extract testcase from params and convert to name
            testcase_num = params["testcase"]
            testcase = TESTCASE_NAMES[testcase_num]

            # Build question entry
            question_entry = {
                "QID": qid,
                "profile": profile,
                "case": testcase,
                "zero_shot": zero_shot,
                "target_parameter": task,
                "precision_level": precision_level,
                "precision_config": precision_config,
                "is_converged": is_converged,
                "dummy_cost": dummy_cost,
                "cost_history": cost_history,
                "best_params": filtered_best_params,
                "param_history": filtered_param_history,
                "question": self._build_question_text(
                    params,
                    profile,
                    precision_level,
                    precision_config
                ),
            }

            dataset.append(question_entry)
            qid += 1

        return dataset

    @staticmethod
    def _build_question_text(
        params: dict,
        profile: str,
        precision_level: str,
        precision_config: dict
    ) -> str:
        """Build question text from euler_2d.md documentation

        Excludes Dummy Strategy section and developer-only content.
        """
        question_lines = [
            "# Euler 2D Equations with Advection-Projection Method",
            "",
            "## Introduction",
            "",
            "This simulation solves the 2D compressible Euler equations for inviscid gas dynamics using an advection-projection method with high-order WENO reconstruction and TVD Runge-Kutta time integration:",
            "",
            "**Conservative form:**",
            r"$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = 0$$",
            "",
            "Where the conservative variables and flux are:",
            r"$$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho E \end{pmatrix}, \quad \mathbf{F} = \begin{pmatrix} \rho \mathbf{u} \\ \rho u \mathbf{u} + p\mathbf{e}_x \\ \rho v \mathbf{u} + p\mathbf{e}_y \\ \mathbf{u}(\rho E + p) \end{pmatrix}$$",
            "",
            "**Primitive variables:**",
            "",
            "- $\\rho$ = density",
            "- $u, v$ = velocity components in x and y directions",
            "- $p$ = pressure",
            "- $E$ = specific total energy",
            "",
            "**Equation of state (ideal gas):**",
            r"$$p = (\gamma - 1) \rho \left(E - \frac{u^2 + v^2}{2}\right)$$",
            "",
            "where $\\gamma = 1.4$ is the ratio of specific heats for air.",
            "",
            "### Numerical Method: Advection-Projection",
            "",
            "The solver uses a fractional-step method that splits the computation into two stages per time step:",
            "",
            "1. **Advection Step**: Update conservative variables via pure advection, as if there is no influence of the pressure",
            "   - High-order WENO reconstruction (3rd order) for spatial discretization",
            "   - Local Lax-Friedrichs Riemann solver for flux computation",
            "   - TVD Runge-Kutta (3rd order) for time integration",
            "",
            "2. **Projection Step**: Enforce divergence-free constraint on velocity field",
            "   - Solve pressure Poisson equation: $\\nabla^2 p = \\text{RHS}$, where RHS is derived to account for the influence of the internal pressure",
            "   - Use Conjugate Gradient (CG) iterative solver",
            "",
            "### Spatial Discretization",
            "",
            "The spatial discretization uses:",
            "",
            "- **Cartesian grid**: Uniform spacing $\\Delta x = \\Delta y = 1 / N_x$",
            "- **WENO reconstruction**: Weighted Essentially Non-Oscillatory scheme for high accuracy near discontinuities",
            "  - WENO3: 3rd order accuracy",
            "- **Riemann solver**: Local Lax-Friedrichs flux at cell interfaces",
            "- **Ghost layers**: 2-cell boundary layers for accurate boundary conditions",
            "",
            "### Temporal Discretization",
            "",
            "Time integration uses TVD Runge-Kutta schemes:",
            "",
            "- **TVDRK3**: 3rd order, 3 substeps per time step",
            "",
            "**Adaptive time stepping** with CFL condition:",
            r"$$\Delta t = \text{CFL} \cdot \frac{2 \Delta x}{|u|+\sqrt{u^2 + 4c^2}}$$",
            "",
            "where $c = \\sqrt{\\gamma p / \\rho}$ is the local speed of sound.",
            "",
            "Minimum timestep bounds (for extreme shocks):",
            "",
            "- Standard: $\\Delta t_{\\text{min}} = 10^{-10}$",
            "- Strong shocks (testcase 5): $\\Delta t_{\\text{min}} = 10^{-11}$",
            "- High Mach (testcase 6): $\\Delta t_{\\text{min}} = 10^{-8}$",
            "",
            "### Boundary Conditions",
            "",
            "The solver supports four boundary condition types:",
            "",
            "- **FREE**: Free boundaries (extrapolation from interior)",
            "- **BOUND**: Wall boundaries (no-slip/slip)",
            "- **INLET**: Prescribed inflow with fixed conservative variables",
            "- **GAS**: Interior gas cells (default)",
            "",
            "## Test Cases",
            "",
            "The solver provides 9 test cases spanning true 2D flows and pseudo-1D problems. Test cases 0-1 are genuine 2D problems, while cases 2-8 mimic 1D problems using thin 2D grids (aspect ratio = 1/Nx).",
            "",
        ]

        # Add profile-specific test case description based on testcase
        testcase = params["testcase"]

        if testcase == 0 or profile == "p1":
            question_lines.extend([
                "### True 2D Problems",
                "",
                "**Case 0 (p1): Central Explosion**",
                "",
                "- Circular high-pressure region in center of domain",
                "- Aspect ratio: 1.0 (square domain)",
                "- Domain: $[-0.5, 0.5] \\times [-0.5, 0.5]$",
                "- Initial conditions:",
                "  - Center (radius $r \\leq 0.04N_x$): $\\rho=1.0$, $p=2.5$, $u=v=0$",
                "  - Ambient: $\\rho=0.125$, $p=0.25$, $u=v=0$",
                "- Boundary: FREE on all sides",
                "- Duration: 10 frames × 0.075s = 0.75s",
                "- Tests: Radial symmetry, circular shock propagation",
            ])
        elif testcase == 1 or profile == "p2":
            question_lines.extend([
                "### True 2D Problems",
                "",
                "**Case 1 (p2): Stair Flow**",
                "",
                "- Supersonic flow over step geometry",
                "- Aspect ratio: 0.25 (wide domain, 4:1)",
                "- Domain: $[-0.5, 0.5] \\times [-0.125, 0.125]$",
                "- Initial conditions:",
                "  - Uniform supersonic flow: $\\rho=1.4$, $p=1.0$, $u=3.0$, $v=0$",
                "- Geometry: Step at $x \\geq 0.25$, $y < 0.25$ (blocked)",
                "- Boundary: INLET (left), BOUND (top/bottom/step)",
                "- Duration: 10 frames × 0.075s = 0.75s",
                "- Tests: Shock reflection, flow separation, oblique shocks",
            ])
        elif testcase == 2 or profile == "p3":
            question_lines.extend([
                "### Pseudo-1D Problems (Thin 2D Grids)",
                "",
                "These test cases use aspect ratio = 1/Nx to create thin grids with ny=2 interior cells, effectively mimicking 1D behavior while using the 2D solver. Top and bottom boundaries are walls (BOUND).",
                "",
                "**Case 2 (p3): Sod Shock Tube**",
                "",
                "- Classic Riemann problem",
                "- Left ($x < 0$): $\\rho=1.0$, $p=2.5$, $u=v=0$",
                "- Right ($x \\geq 0$): $\\rho=0.125$, $p=0.25$, $u=v=0$",
                "- Boundary: BOUND (top/bottom), FREE (left/right)",
                "- Duration: 10 frames × 0.02s = 0.2s",
            ])
        elif testcase == 3 or profile == "p4":
            question_lines.extend([
                "### Pseudo-1D Problems (Thin 2D Grids)",
                "",
                "These test cases use aspect ratio = 1/Nx to create thin grids with ny=2 interior cells, effectively mimicking 1D behavior while using the 2D solver. Top and bottom boundaries are walls (BOUND).",
                "",
                "**Case 3 (p4): Lax Shock Tube**",
                "",
                "- Similar to Sod with inlet BC",
                "- Left ($x < 0$): $\\rho=0.445$, $E=8.9284$, $\\rho u=0.31061$, $\\rho v=0$",
                "- Right ($x \\geq 0$): $\\rho=0.5$, $E=1.425$, $\\rho u=0$, $\\rho v=0$",
                "- Boundary: INLET (left), BOUND (top/bottom)",
                "- Duration: 10 frames × 0.02s = 0.2s",
            ])
        elif testcase == 4 or profile == "p5":
            question_lines.extend([
                "### Pseudo-1D Problems (Thin 2D Grids)",
                "",
                "These test cases use aspect ratio = 1/Nx to create thin grids with ny=2 interior cells, effectively mimicking 1D behavior while using the 2D solver. Top and bottom boundaries are walls (BOUND).",
                "",
                "**Case 4 (p5): Mach 3 Problem**",
                "",
                "- High Mach number shock",
                "- Left ($x < 0$): $\\rho=3.857$, $E=27.46478$, $\\rho u=3.54844$, $\\rho v=0$",
                "- Right ($x \\geq 0$): $\\rho=1.0$, $E=8.80125$, $\\rho u=3.55$, $\\rho v=0$",
                "- Boundary: INLET (left), FREE (right), BOUND (top/bottom)",
                "- Duration: 10 frames × 0.02s = 0.2s",
            ])
        elif testcase == 5 or profile == "p6":
            question_lines.extend([
                "### Pseudo-1D Problems (Thin 2D Grids)",
                "",
                "These test cases use aspect ratio = 1/Nx to create thin grids with ny=2 interior cells, effectively mimicking 1D behavior while using the 2D solver. Top and bottom boundaries are walls (BOUND).",
                "",
                "**Case 5 (p6): Strong Shock Tube**",
                "",
                "- Extreme pressure ratio ($2.5 \\times 10^{10}$)",
                "- Left ($x < 0$): $\\rho=1.0$, $E=2.5 \\times 10^{10}$, $\\rho u=0$, $\\rho v=0$",
                "- Right ($x \\geq 0$): $\\rho=0.125$, $E=0.25$, $\\rho u=0$, $\\rho v=0$",
                "- Boundary: BOUND (top/bottom), FREE (left/right)",
                "- Duration: 10 frames × 0.02s = 0.2s",
                "- Special: Minimum timestep $\\Delta t_{\\text{min}} = 10^{-11}$ for stability",
            ])
        elif testcase == 6 or profile == "p7":
            question_lines.extend([
                "### Pseudo-1D Problems (Thin 2D Grids)",
                "",
                "These test cases use aspect ratio = 1/Nx to create thin grids with ny=2 interior cells, effectively mimicking 1D behavior while using the 2D solver. Top and bottom boundaries are walls (BOUND).",
                "",
                "**Case 6 (p7): High Mach Problem**",
                "",
                "- Very high velocity shock (Mach $\\approx 20000$)",
                "- Left ($x < 0$): $\\rho=10.0$, $E=20001250$, $\\rho u=20000$, $\\rho v=0$",
                "- Right ($x \\geq 0$): $\\rho=20.0$, $E=1250$, $\\rho u=0$, $\\rho v=0$",
                "- Boundary: INLET (left), FREE (right), BOUND (top/bottom)",
                "- Duration: 10 frames × 0.02s = 0.2s",
                "- Special: Minimum timestep $\\Delta t_{\\text{min}} = 10^{-8}$ for stability",
            ])
        elif testcase == 7 or profile == "p8":
            question_lines.extend([
                "### Pseudo-1D Problems (Thin 2D Grids)",
                "",
                "These test cases use aspect ratio = 1/Nx to create thin grids with ny=2 interior cells, effectively mimicking 1D behavior while using the 2D solver. Top and bottom boundaries are walls (BOUND).",
                "",
                "**Case 7 (p8): Interacting Blast Shock**",
                "",
                "- Two blast waves colliding",
                "- Left 10% ($x < -0.4$): $\\rho=1.0$, $E=2500$, $\\rho u=0$, $\\rho v=0$",
                "- Middle 80%: $\\rho=1.0$, $E=0.025$, $\\rho u=0$, $\\rho v=0$",
                "- Right 10% ($x > 0.4$): $\\rho=1.0$, $E=250$, $\\rho u=0$, $\\rho v=0$",
                "- Boundary: BOUND on all sides (closed tube)",
                "- Duration: 10 frames × 0.02s = 0.2s",
            ])
        elif testcase == 8 or profile == "p9":
            question_lines.extend([
                "### Pseudo-1D Problems (Thin 2D Grids)",
                "",
                "These test cases use aspect ratio = 1/Nx to create thin grids with ny=2 interior cells, effectively mimicking 1D behavior while using the 2D solver. Top and bottom boundaries are walls (BOUND).",
                "",
                "**Case 8 (p9): Symmetric Rarefaction Waves**",
                "",
                "- Opposite velocity collision creating vacuum",
                "- Left ($x < 0$): $\\rho=1.0$, $E=3.0$, $\\rho u=-2.0$, $\\rho v=0$",
                "- Right ($x \\geq 0$): $\\rho=1.0$, $E=3.0$, $\\rho u=2.0$, $\\rho v=0$",
                "- Boundary: INLET (left/right), BOUND (top/bottom)",
                "- Duration: 10 frames × 0.015s = 0.15s",
            ])
        else:
            # Fallback for unknown testcase
            question_lines.extend([
                f"### Current Configuration: {profile} - Test Case {testcase}",
                "",
                "- Custom simulation configuration",
            ])

        question_lines.extend([
            "",
            "### Convergence Criteria",
            "",
            "The simulated results are considered correct if the Normalized RMSE meets the precision-dependent tolerance compared to a reference solution (typically adjacent finer grid or tighter CG tolerance):",
            "",
            r"$$\text{NRMSE} = \frac{\|\mathbf{f}_1 - \mathbf{f}_2\|_2}{\max(|\mathbf{f}_1|)}$$",
            "",
            "The final RMSE is the average of NRMSE over density and pressure fields:",
            r"$$\text{RMSE} = \frac{\text{NRMSE}(\rho) + \text{NRMSE}(p)}{2}$$",
            "",
            "When comparing different grid resolutions, the coarser solution is interpolated to the finer grid before computing NRMSE. Pressure is vertex-centered (at $i \\cdot \\Delta x$), while density and velocity are cell-centered (at $(i+0.5) \\cdot \\Delta x$).",
            "",
            "## Parameter Tuning Tasks",
            "",
            "### Tasks",
            "",
            "1. **n_grid_x Convergence Search**",
            "   - n_grid_x determines spatial resolution: $\\Delta x = \\Delta y = 1 / n\\_grid\\_x$",
            "",
            "2. **CFL Convergence Search**",
            "   - CFL (Courant-Friedrichs-Lewy) number controls timestep: $\\Delta t = \\text{CFL} \\cdot \\Delta x / (|u| + c)$",
            "",
            "3. **cg_tolerance Optimization**",
            "   - Convergence tolerance for CG solver in projection step",
            "",
            "### Cost Calculation",
            "",
            "The computational cost is tracked as:",
            "",
            r"$$\text{Total Cost} = N_{\text{cells}} \times (N_{\text{steps}} + N_{\text{CG\_iters}})$$",
            "",
            "Where:",
            "",
            "- $N_{\\text{cells}} = N_x \\times N_y$ = total number of grid cells",
            "- $N_{\\text{steps}}$ = total number of advection steps taken",
            "- $N_{\\text{CG\\_iters}}$ = total CG iterations across all projection steps",
            "",
            f"**Current Problem Precision Level**: {precision_level.upper()}",
            f"- **Tolerance RMSE**: ≤ {precision_config['tolerance_rmse']}",
        ])

        return "\n".join(question_lines)

    def _save_dataset(
        self,
        dataset: List[Dict],
        task: str,
        precision_level: str,
        zero_shot: bool
    ) -> None:
        """Save dataset to the new path format: data/euler_2d/{task}/{precision_level}/"""
        base_dir = Path("data") / "euler_2d" / task / precision_level
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = "zero_shot_questions.json" if zero_shot else "iterative_questions.json"
        out_path = base_dir / filename

        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(dataset, fp, ensure_ascii=False, indent=2)

def main() -> None:
    """Generate all Euler 2D questions for all tasks and precision levels"""
    print("2D EULER EQUATION QUESTION GENERATOR")
    print("=" * 80)
    print(f"Output directory: data/euler_2d/{{task}}/{{precision_level}}/")
    print(f"Tasks (all support iterative + zero-shot): n_grid_x, cfl, cg_tolerance")
    print(f"Precision levels: {list(PRECISION_LEVELS.keys())}")
    print(f"File types:")
    print(f"  - iterative_questions.json (for all tasks)")
    print(f"  - zero_shot_questions.json (for all tasks)")

    gen = TwoDEulerQuestionGenerator()
    gen.generate_all_questions()

    print("\nAll questions generated successfully!")


if __name__ == "__main__":
    main()
