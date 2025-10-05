import sys
import os
# Add project root to path (go up 3 levels: twoD_ns_transient_icl.py -> additional_exp -> dataset_gen -> SimulCost-Bench)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset_gen.base import DatasetGenerator
import json
from inference import save_result
import argparse
from utils.param_compatibility import fetch_param

zero_shot_HUMAN_CODE = r"""
def forward(self, data: dict):
    # Extract input data
    messages = data["messages"]
    qid = data["QID"]

    # Initialize experiment manager
    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=[
            "resolution",
            "cfl",
            "relaxation_factor",
            "residual_threshold"
        ],
        norm_rmse_tolerance=data.get("norm_rmse_tolerance")
    )

    # Set up experiment agent
    experiment_instruction = "Given the problem, you should use the tool call to run the experiment."
    experiment_agent = self.get_experiment_agent()

    # Zero-Shot
    tool_reason, tool_name, tool_args = experiment_agent.query(messages, experiment_instruction)
    messages.append({
        "role": "assistant",
        "content": json.dumps({
            "tool_reason": tool_reason,
            "tool_name": tool_name,
            "tool_args": tool_args
        })
    })

    # Execute tool and inject results
    tool_result, acc_cost = experiment_agent.execute_tool(tool_reason=tool_reason, tool_name=tool_name, tool_args=tool_args, tool_manager=experiment_manager, profile=data["profile"])
    messages.append({
        "role": "user",
        "content": json.dumps(tool_result)
    })

    # Collect true history from tool manager
    param_seq = experiment_manager.get_param_sequence()
    cost_seq = experiment_manager.get_cost_sequence()

    summary_data = {
        "QID": qid,
        "is_converged": tool_result.get("is_converged", False),
        "times": len(param_seq),
        "param_sequence": param_seq,
        "accumulated_cost": experiment_manager.accumulated_cost,
        "cost_sequence": cost_seq
    }

    tool_df = experiment_manager.get_tool_call_df()
    return summary_data, tool_df
"""

# Best parameters from data/ns_transient_2d directory for ICL examples (first question from each precision level)
BEST_PARAMS_DATA = {
    "cfl": {
        "low": {"resolution": 200, "cfl": 0.2, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 274080000},
        "medium": {"resolution": 200, "cfl": 0.2, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 274080000},
        "high": {"resolution": 200, "cfl": 0.2, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 274080000}
    },
    "resolution": {
        "low": {"resolution": 50, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 31575000},
        "medium": {"resolution": 50, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 31575000},
        "high": {"resolution": 200, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 1042640000}
    },
    "relaxation_factor": {
        "low": {"resolution": 200, "cfl": 0.05, "relaxation_factor": 0.9, "residual_threshold": 0.01, "simulation_cost": 688720000},
        "medium": {"resolution": 200, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 788960000},
        "high": {"resolution": 200, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 788960000}
    },
    "residual_threshold": {
        "low": {"resolution": 200, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.005, "simulation_cost": 1042640000},
        "medium": {"resolution": 200, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.005, "simulation_cost": 1042640000},
        "high": {"resolution": 200, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.005, "simulation_cost": 1042640000}
    }
}

def generate_icl_examples_for_precision(task: str, precision_level: str, no_cost: bool = False) -> list:
    """Generate ICL examples for a specific task and precision level (one converged, one non-converged)"""
    examples = []

    # Add one converged example for the specific precision level
    if precision_level in BEST_PARAMS_DATA[task]:
        params = BEST_PARAMS_DATA[task][precision_level].copy()
        if "simulation_cost" not in params:
            raise ValueError(f"Missing simulation_cost in BEST_PARAMS_DATA[{task}][{precision_level}]")
        simulation_cost = params.pop("simulation_cost")
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        if no_cost:
            example = (
                f"In a previous simulation ({precision_level} precision level) with parameters {param_str}, "
                f"the result was is_converged=true."
            )
        else:
            example = (
                f"In a previous simulation ({precision_level} precision level) with parameters {param_str}, "
                f"the result was is_converged=true, simulation_cost={simulation_cost}."
            )
        examples.append(example)

    # Add one non-converged example for the specific precision level
    non_converged_examples = {
        "cfl": {
            "low": {"resolution": 200, "cfl": 0.2, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 206960000},
            "medium": {"resolution": 400, "cfl": 0.2, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 2075520000},
            "high": {"resolution": 400, "cfl": 0.2, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 1970560000}
        },
        "resolution": {
            "low": {"resolution": 50, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 13970000},
            "medium": {"resolution": 50, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 13970000},
            "high": {"resolution": 100, "cfl": 0.05, "relaxation_factor": 1.3, "residual_threshold": 0.01, "simulation_cost": 155660000}
        },
        "relaxation_factor": {
            "low": {},
            "medium": {},
            "high": {}
        },
        "residual_threshold": {
            "low": {},
            "medium": {},
            "high": {}
        }
    }

    # Skip non-converged examples for relaxation_factor and residual_threshold tasks
    if task not in ["relaxation_factor", "residual_threshold"]:
        if task in non_converged_examples and precision_level in non_converged_examples[task]:
            params = non_converged_examples[task][precision_level].copy()
            if "simulation_cost" in params:
                simulation_cost = params.pop("simulation_cost")
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                if no_cost:
                    example = (
                        f"In a previous simulation ({precision_level} precision level) with parameters {param_str}, "
                        f"the result was is_converged=false."
                    )
                else:
                    example = (
                        f"In a previous simulation ({precision_level} precision level) with parameters {param_str}, "
                        f"the result was is_converged=false, simulation_cost={simulation_cost}."
                    )
                examples.append(example)

    return examples

def generate_icl_examples(task: str, precision_levels: list) -> list:
    """Generate ICL examples for backward compatibility - uses all precision levels"""
    examples = []
    for precision in precision_levels:
        examples.extend(generate_icl_examples_for_precision(task, precision))
    return examples

def generate_specific_icl_examples_with_header(task: str, precision_level: str, no_cost: bool = False) -> list:
    """Generate specific ICL examples with proper numbering and single disclaimer"""
    raw_examples = generate_icl_examples_for_precision(task, precision_level, no_cost)
    formatted_examples = []

    for i, example in enumerate(raw_examples, 1):
        formatted_examples.append(f"**Reference Example {i}:**\n{example}")

    # Add disclaimer once at the end
    disclaimer = (
        "\nNote that your problem may have different settings (case, precision requirements, etc.), "
        "so the same parameters could yield completely different results. "
        "This is only provided as a reference to understand the scale of values - do not treat it as ground truth.\n"
    )
    formatted_examples.append(disclaimer)

    return formatted_examples

def generate_uniform_icl_examples(task: str, no_cost: bool = False) -> list:
    """Generate uniform ICL examples (all 6 examples: 3 precision levels × 2 examples each) with proper numbering"""
    all_examples = []
    example_counter = 1

    for precision in ["low", "medium", "high"]:
        raw_examples = generate_icl_examples_for_precision(task, precision, no_cost)
        for example in raw_examples:
            all_examples.append(f"**Reference Example {example_counter}:**\n{example}")
            example_counter += 1

    # Add disclaimer once at the end
    disclaimer = (
        "\nNote that your problem may have different settings (case, precision requirements, etc.), "
        "so the same parameters could yield completely different results. "
        "This is only provided as a reference to understand the scale of values - do not treat it as ground truth.\n"
    )
    all_examples.append(disclaimer)

    return all_examples

# Legacy ICL examples - kept for backward compatibility but simplified
CFL_ICL_EXAMPLES = generate_icl_examples("cfl", ["low", "medium", "high"])

RESOLUTION_ICL_EXAMPLES = generate_icl_examples("resolution", ["low", "medium", "high"])

RELAXATION_FACTOR_ICL_EXAMPLES = generate_icl_examples("relaxation_factor", ["low", "medium", "high"])

RESIDUAL_THRESHOLD_ICL_EXAMPLES = generate_icl_examples("residual_threshold", ["low", "medium", "high"])

def get_icl_examples(task: str) -> list:
    """Get ICL examples for a specific task"""
    examples_map = {
        "cfl": CFL_ICL_EXAMPLES,
        "resolution": RESOLUTION_ICL_EXAMPLES,
        "relaxation_factor": RELAXATION_FACTOR_ICL_EXAMPLES,
        "residual_threshold": RESIDUAL_THRESHOLD_ICL_EXAMPLES
    }
    return examples_map.get(task, CFL_ICL_EXAMPLES)  # fallback to CFL examples

iterative_HUMAN_CODE = r"""
def forward(self, data: dict):
    messages = data['messages']
    qid      = data['QID']

    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=['resolution', 'cfl', 'relaxation_factor', 'residual_threshold'],
        norm_rmse_tolerance=data.get('norm_rmse_tolerance')
    )

    exp_instr = (
        'Given the problem, you should use the tool call to run the experiment. '
        'When you think the experiment can be stopped, set should_stop to true, '
        'otherwise set it to false.'
    )
    agent = LLMAgentBase(
        ['tool_reason', 'tool_name', 'tool_args', 'should_stop'],
        'Experiment Agent',
        self.logger
    )

    last_valid_tool_result = None  # Track the last successful tool result
    tool_result = None  # Initialize to avoid undefined variable

    for attempt in range(10):
        tool_reason, tool_name, tool_args, should_stop = agent.query(messages, exp_instr)
        messages.append({
            'role': 'assistant',
            'content': json.dumps({
                'tool_reason': tool_reason,
                'tool_name':  tool_name,
                'tool_args':  tool_args,
                'should_stop': should_stop
            })
        })

        if attempt == 0:
            self.logger.info('\n\n\n')
            self.logger.info('========== 🧐 The model begins to solve a new problem ==========')
        self.logger.info(f'========== The {attempt + 1} attempt of the model ==========')
        self.logger.info(f'should_stop: {should_stop}')

        stop_flag = bool(should_stop) if isinstance(should_stop, bool) else str(should_stop).lower() == 'true'
        if stop_flag:
            self.logger.info('========== 🎯 The model stops the experiment! ==========')
            break

        tool_result, acc_cost = agent.execute_tool(
            tool_reason=tool_reason,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_manager=experiment_manager,
            profile=data['profile']
        )

        # Track the last valid tool result (not containing error)
        if 'error' not in tool_result:
            last_valid_tool_result = tool_result

        messages.append({'role': 'user', 'content': json.dumps(tool_result)})

    param_seq  = experiment_manager.get_param_sequence()
    cost_seq   = experiment_manager.get_cost_sequence()

    # Use the last valid tool result if available, otherwise use a default empty result
    if last_valid_tool_result is not None:
        final_tool_result = last_valid_tool_result
    elif tool_result is not None:
        final_tool_result = tool_result
    else:
        # If no tool was ever executed (immediate should_stop=True), create a default result
        final_tool_result = {'is_converged': False, 'error': 'No tool execution - immediate stop'}

    summary = {
        'QID': qid,
        'is_converged': final_tool_result.get('is_converged', False),
        'times': len(param_seq),
        'param_sequence': param_seq,
        'accumulated_cost': experiment_manager.accumulated_cost,
        'cost_sequence': cost_seq,
    }
    tool_df = experiment_manager.get_tool_call_df()
    return summary, tool_df
"""

def list_cases(task_dir: str):
    """Return all cases in the task directory"""
    return [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]

def build_cfl_workflow(zero_shot: bool, resolution0: int, relaxation_factor0: float, residual_threshold0: float, precision_level: str = "low", use_uniform_icl: bool = False, no_cost: bool = False) -> str:
    """Build the workflow for CFL optimization"""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("cfl", no_cost)
    else:
        icl_examples = generate_specific_icl_examples_with_header("cfl", precision_level, no_cost)
    icl_text = "\n".join(icl_examples)

    header = (
        "CFL (Courant-Friedrichs-Lewy) number controls the time step stability: "
        "$\\Delta t = \\text{CFL} \\times \\Delta x$ where $\\Delta x = 1/\\text{resolution}$.\n"
        "You may **only** change `cfl`.\n"
        f"The values are: resolution={resolution0}, relaxation_factor={relaxation_factor0}, residual_threshold={residual_threshold0}. **You must not change them!**\n"
        "\n" + icl_text
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for cfl.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that ensures temporal stability while maintaining computational efficiency.\n"
            "Step 1: Make your best **one-shot** guess for cfl.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of cfl, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine cfl based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your solution."
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_resolution_workflow(zero_shot: bool, cfl0: float, relaxation_factor0: float, residual_threshold0: float, precision_level: str = "low", use_uniform_icl: bool = False, no_cost: bool = False) -> str:
    """Build the workflow for resolution parameter optimization"""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("resolution", no_cost)
    else:
        icl_examples = generate_specific_icl_examples_with_header("resolution", precision_level, no_cost)
    icl_text = "\n".join(icl_examples)

    header = (
        "resolution determines the spatial discretization quality: "
        "Domain Resolution: x_resolution = 2 × resolution, y_resolution = resolution.\n"
        "You may **only** change `resolution`.\n"
        f"The values are: cfl={cfl0}, relaxation_factor={relaxation_factor0}, residual_threshold={residual_threshold0}. **You must not change them!**\n"
        "\n" + icl_text
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for resolution.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that provides adequate spatial resolution while keeping computational cost reasonable.\n"
            "Step 1: Make your best **one-shot** guess for resolution.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of resolution, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine resolution based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your resolution."
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_relaxation_factor_workflow(zero_shot: bool, resolution0: int, cfl0: float, residual_threshold0: float, precision_level: str = "low", use_uniform_icl: bool = False, no_cost: bool = False) -> str:
    """relaxation_factor-task: 0-shot select relaxation_factor"""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("relaxation_factor", no_cost)
    else:
        icl_examples = generate_specific_icl_examples_with_header("relaxation_factor", precision_level, no_cost)
    icl_text = "\n".join(icl_examples)

    header = (
        "relaxation_factor controls the pressure correction relaxation factor controlling convergence rate.\n"
        "You may **only** change `relaxation_factor`.\n"
        f"The values are: resolution={resolution0}, cfl={cfl0}, residual_threshold={residual_threshold0}. **You must not change them!**\n"
        "\n" + icl_text
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick relaxation_factor.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1: Choose relaxation_factor based on domain knowledge (only one chance).\n"
            "Step 2: Call the convergence test function.\n"
            "Step 3: You may refine relaxation_factor at most 10 times.\n"
            "Step 4: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_residual_threshold_workflow(zero_shot: bool, resolution0: int, cfl0: float, relaxation_factor0: float, precision_level: str = "low", use_uniform_icl: bool = False, no_cost: bool = False) -> str:
    """residual_threshold-task: 0-shot select residual_threshold"""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("residual_threshold", no_cost)
    else:
        icl_examples = generate_specific_icl_examples_with_header("residual_threshold", precision_level, no_cost)
    icl_text = "\n".join(icl_examples)

    header = (
        "residual_threshold controls the pressure solver convergence threshold.\n"
        "You may **only** change `residual_threshold`.\n"
        f"The values are: resolution={resolution0}, cfl={cfl0}, relaxation_factor={relaxation_factor0}. **You must not change them!**\n"
        "\n" + icl_text
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick residual_threshold.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1: Choose residual_threshold based on domain knowledge (only one chance).\n"
            "Step 2: Call the convergence test function.\n"
            "Step 3: You may refine residual_threshold at most 10 times.\n"
            "Step 4: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

class twoD_NS_Transient_DatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameter for solving the 2D transient incompressible Navier–Stokes equations "
            "using a staggered grid with CIP scheme for advection and pressure correction method. This simulation models "
            "incompressible fluid flow with configurable boundary conditions. You should try to minimize the total cost "
            "incurred by function calls, but your primary goal is to successfully meet the convergence criteria. "
            "You should always use the tool call function to finish the problem."
        )

        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."

        system_prompt = zero_shot_system_prompt if zero_shot else iterative_system_prompt

        return system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", choices=["cfl", "resolution", "relaxation_factor", "residual_threshold"],
                        help="Task to solve (if not specified, generates all tasks)")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode (if not specified, generates both modes)")
    args = parser.parse_args()

    # If no specific task is provided, generate all tasks
    if args.task:
        tasks = [args.task]
    else:
        tasks = ["cfl", "resolution", "relaxation_factor", "residual_threshold"]

    # If no specific mode is provided, generate both modes
    if args.zero_shot:
        modes = [True]  # Only zero-shot
    else:
        modes = [False, True]  # Both iterative and zero-shot

    # Define three ICL dataset types
    icl_types = [
        {"name": "accuracy_focused", "use_uniform": False, "no_cost": False},
        {"name": "cost_excluded", "use_uniform": False, "no_cost": True},
        {"name": "full", "use_uniform": True, "no_cost": False}
    ]

    print("🚀 NAVIER-STOKES TRANSIENT 2D ICL DATASET GENERATOR")
    print("=" * 80)
    print(f"📋 Tasks: {tasks}")
    print(f"🎯 Modes: {'zero-shot only' if len(modes) == 1 else 'iterative + zero-shot'}")
    print(f"🧠 ICL Types: accuracy_focused, cost_excluded, full")
    print(f"📂 Output base directory: data/icl/ns_transient_2d/")
    print(f"📂 Source directory: data/ns_transient_2d/{{task}}/{{precision_level}}/")

    total_files = 0

    for icl_type in icl_types:
        icl_name = icl_type["name"]
        use_uniform_icl = icl_type["use_uniform"]
        no_cost = icl_type["no_cost"]

        print(f"\n{'='*80}")
        print(f"🔄 Generating ICL Type: {icl_name.upper()}")
        print(f"{'='*80}")

        for task in tasks:
            print(f"\n📋 TASK: {task.upper()}")
            print("-" * 50)

            task_dir = f"data/ns_transient_2d/{task}"

            # Get precision levels from the new structure
            precision_levels = []
            if os.path.exists(task_dir):
                precision_levels = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
            if not precision_levels:
                precision_levels = ["low", "medium", "high"]  # fallback

            generator = twoD_NS_Transient_DatasetGenerator(
                f"tool_documentation/ns_transient_2d/{task}.json"
            )

            for precision_level in precision_levels:
                print(f"  🎯 {precision_level.upper()} precision:")

                # Create output directory for this ICL type and precision level
                out_dir = f"data/icl/ns_transient_2d/{icl_name}/{precision_level}"
                os.makedirs(out_dir, exist_ok=True)

                for zflag in modes:
                    flag = "zero_shot" if zflag else "iterative"
                    question_file = f"{task_dir}/{precision_level}/{flag}_questions.json"

                    if not os.path.exists(question_file):
                        continue

                    with open(question_file, "r") as f:
                        questions = json.load(f)

                    dataset_entries = []

                    for idx, q in enumerate(questions):
                        if task == "cfl":
                            resolution0 = fetch_param(q["param_history"][0], "resolution")
                            relaxation_factor0 = fetch_param(q["param_history"][0], "relaxation_factor")
                            residual_threshold0 = fetch_param(q["param_history"][0], "residual_threshold")
                            wf = build_cfl_workflow(zflag, resolution0, relaxation_factor0, residual_threshold0, precision_level, use_uniform_icl, no_cost)
                        elif task == "resolution":
                            cfl0 = fetch_param(q["param_history"][0], "cfl")
                            relaxation_factor0 = fetch_param(q["param_history"][0], "relaxation_factor")
                            residual_threshold0 = fetch_param(q["param_history"][0], "residual_threshold")
                            wf = build_resolution_workflow(zflag, cfl0, relaxation_factor0, residual_threshold0, precision_level, use_uniform_icl, no_cost)
                        elif task == "relaxation_factor":
                            resolution0 = fetch_param(q["param_history"][0], "resolution")
                            cfl0 = fetch_param(q["param_history"][0], "cfl")
                            residual_threshold0 = fetch_param(q["param_history"][0], "residual_threshold")
                            wf = build_relaxation_factor_workflow(zflag, resolution0, cfl0, residual_threshold0, precision_level, use_uniform_icl, no_cost)
                        elif task == "residual_threshold":
                            resolution0 = fetch_param(q["param_history"][0], "resolution")
                            cfl0 = fetch_param(q["param_history"][0], "cfl")
                            relaxation_factor0 = fetch_param(q["param_history"][0], "relaxation_factor")
                            wf = build_residual_threshold_workflow(zflag, resolution0, cfl0, relaxation_factor0, precision_level, use_uniform_icl, no_cost)

                        # Create a modified question with norm_rmse_tolerance at top level for compatibility
                        q_modified = q.copy()
                        if "precision_config" in q and "norm_rmse_tolerance" in q["precision_config"]:
                            q_modified["norm_rmse_tolerance"] = q["precision_config"]["norm_rmse_tolerance"]

                        single_ds = generator.generate_dataset(wf, [q_modified], zflag)[0]

                        # Update dataset entry to only include required fields
                        # Extract norm_rmse_tolerance from precision_config if it exists
                        norm_rmse_tolerance = None
                        if "precision_config" in q and "norm_rmse_tolerance" in q["precision_config"]:
                            norm_rmse_tolerance = q["precision_config"]["norm_rmse_tolerance"]

                        filtered_ds = {
                            "QID": single_ds.get("QID", q.get("QID")),
                            "profile": q.get("profile"),
                            "case": q.get("case"),
                            "zero_shot": q.get("zero_shot"),
                            "target_parameter": q.get("target_parameter"),
                            "precision_level": q.get("precision_level"),
                            "norm_rmse_tolerance": norm_rmse_tolerance,
                            "messages": single_ds.get("messages")
                        }

                        dataset_entries.append(filtered_ds)

                        if idx == 0:
                            human_code = zero_shot_HUMAN_CODE if zflag else iterative_HUMAN_CODE
                            agent = {"workflow": wf, "code": human_code}
                            archive_file = f"{out_dir}/{task}_{flag}_agent.json"
                            with open(archive_file, "w") as f:
                                json.dump([agent], f, indent=4)

                    dataset_file = f"{out_dir}/{task}_{flag}_dataset.json"
                    save_result(dataset_entries, dataset_file)
                    print(f"     ✓ {flag.capitalize()}: {len(dataset_entries):2d} entries -> {dataset_file}")
                    total_files += 1

    print("\n" + "=" * 80)
    print(f"🎉 SUMMARY: Generated {total_files} dataset files")
    print("=" * 80)

if __name__ == "__main__":
    main()