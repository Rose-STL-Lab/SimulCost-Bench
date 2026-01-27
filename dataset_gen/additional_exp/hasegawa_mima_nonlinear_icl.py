import sys
import os
# Add project root to path (go up 3 levels: hasegawa_mima_nonlinear_icl.py -> additional_exp -> dataset_gen -> SimulCost-Bench)
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
            "N",
            "dt"
        ],
        tolerance_rmse=data.get("tolerance_rmse")
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

# Best parameters from data/hasegawa_mima_nonlinear directory for ICL examples (one per precision level)
# TODO: Fill in the actual values
BEST_PARAMS_DATA = {
    "N": {
        "low": {"N": 32, "dt": 2.5, "simulation_cost": 1474693120},
        "medium": {"N": 64, "dt": 2.5, "simulation_cost": 7078526976},
        "high": {"N": 128, "dt": 2.5, "simulation_cost": 33033125888}
    },
    "dt": {
        "low": {"N": 32, "dt": 40, "simulation_cost": 92293120},
        "medium": {"N": 64, "dt": 40, "simulation_cost": 443006976},
        "high": {"N": 128, "dt": 40, "simulation_cost": 2067365888}
    }
}

def generate_icl_examples_for_precision(task: str, precision_level: str, no_cost: bool = False) -> list:
    """Generate ICL examples for a specific task and precision level (one converged, one non-converged)"""
    examples = []

    # Add one converged example for the specific precision level
    if precision_level in BEST_PARAMS_DATA[task]:
        params = BEST_PARAMS_DATA[task][precision_level].copy()
        if params.get("simulation_cost") is None:
            # Skip if values not filled in
            return examples
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
    # TODO: Fill in the actual values
    non_converged_examples = {
        "N": {
            "low": {"N": 128, "dt": 40, "simulation_cost": 2067365888},
            "medium": {"N": 32, "dt": 2.5, "simulation_cost": 1474693120},
            "high": {"N": 64, "dt": 2.5, "simulation_cost": 7078526976}
        },
        "dt": {
            "low": {"N": 128, "dt": 0.1, "simulation_cost": 244044365824},
            "medium": {"N": 128, "dt": 0.2, "simulation_cost": 240188784640},
            "high": {"N": 128, "dt": 0.1, "simulation_cost": 245365571584}
        }
    }

    if task in non_converged_examples and precision_level in non_converged_examples[task]:
        params = non_converged_examples[task][precision_level].copy()
        if params.get("simulation_cost") is None:
            # Skip if values not filled in
            return examples
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
    if formatted_examples:
        disclaimer = (
            "\nNote that your problem may have different settings (case, precision requirements, etc.), "
            "so the same parameters could yield completely different results. "
            "This is only provided as a reference to understand the scale of values - do not treat it as ground truth.\n"
        )
        formatted_examples.append(disclaimer)

    return formatted_examples

def generate_uniform_icl_examples(task: str, no_cost: bool = False) -> list:
    """Generate uniform ICL examples (all 6 examples: 3 precision levels x 2 examples each) with proper numbering"""
    all_examples = []
    example_counter = 1

    for precision in ["low", "medium", "high"]:
        raw_examples = generate_icl_examples_for_precision(task, precision, no_cost)
        for example in raw_examples:
            all_examples.append(f"**Reference Example {example_counter}:**\n{example}")
            example_counter += 1

    # Add disclaimer once at the end
    if all_examples:
        disclaimer = (
            "\nNote that your problem may have different settings (case, precision requirements, etc.), "
            "so the same parameters could yield completely different results. "
            "This is only provided as a reference to understand the scale of values - do not treat it as ground truth.\n"
        )
        all_examples.append(disclaimer)

    return all_examples

# Legacy ICL examples - kept for backward compatibility but simplified
N_ICL_EXAMPLES = generate_icl_examples("N", ["low", "medium", "high"])
DT_ICL_EXAMPLES = generate_icl_examples("dt", ["low", "medium", "high"])

def get_icl_examples(task: str) -> list:
    """Get ICL examples for a specific task"""
    examples_map = {
        "N": N_ICL_EXAMPLES,
        "dt": DT_ICL_EXAMPLES
    }
    return examples_map.get(task, N_ICL_EXAMPLES)  # fallback to N examples

iterative_HUMAN_CODE = r"""
def forward(self, data: dict):
    messages = data['messages']
    qid      = data['QID']

    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=['N', 'dt'],
        tolerance_rmse=data.get('tolerance_rmse')
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
            self.logger.info('========== The model begins to solve a new problem ==========')
        self.logger.info(f'========== The {attempt + 1} attempt of the model ==========')
        self.logger.info(f'should_stop: {should_stop}')

        stop_flag = bool(should_stop) if isinstance(should_stop, bool) else str(should_stop).lower() == 'true'
        if stop_flag:
            self.logger.info('========== The model stops the experiment! ==========')
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

def build_N_workflow(zero_shot: bool, dt0: float, precision_level: str = "low", use_uniform_icl: bool = False, no_cost: bool = False) -> str:
    """Build the workflow for the N parameter optimization"""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("N", no_cost)
    else:
        icl_examples = generate_specific_icl_examples_with_header("N", precision_level, no_cost)
    icl_text = "\n".join(icl_examples)

    header = (
        "N (Grid resolution) determines the spatial discretization resolution: "
        "the total number of grid cells in each direction (N x N grid).\n"
        "You may **only** change `N`.\n"
        f"The fixed value is: dt={dt0}. **You must not change it!**\n"
    )
    if icl_text:
        header += "\n" + icl_text

    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for N.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that provides adequate spatial resolution while keeping computational cost reasonable.\n"
            "Step 1: Make your best **one-shot** guess for N.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of N, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine N based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your N.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_dt_workflow(zero_shot: bool, N0: int, precision_level: str = "low", use_uniform_icl: bool = False, no_cost: bool = False) -> str:
    """Build the workflow for dt parameter optimization"""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("dt", no_cost)
    else:
        icl_examples = generate_specific_icl_examples_with_header("dt", precision_level, no_cost)
    icl_text = "\n".join(icl_examples)

    header = (
        "dt (Time step size) determines the temporal discretization resolution "
        "for the RK4 (4th-order Runge-Kutta) time integration scheme.\n"
        "You may **only** change `dt`.\n"
        f"The fixed value is: N={N0}. **You must not change it!**\n"
    )
    if icl_text:
        header += "\n" + icl_text

    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for dt.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances temporal accuracy and computational efficiency.\n"
            "Step 1: Make your best **one-shot** guess for dt.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of dt, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine dt based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

class HasegawaMima_DatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameter, solving the Hasegawa-Mima nonlinear equation "
            "using a pseudo-spectral method with RK4 time integration on a uniform grid. "
            "This serves as a model for 2D plasma turbulence in magnetically confined fusion devices, "
            "simulating the evolution of electrostatic potential fluctuations due to drift-wave instabilities. "
            "You should try to minimize the total cost incurred by function calls, but your "
            "primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem."
        )

        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."

        system_prompt = zero_shot_system_prompt if zero_shot else iterative_system_prompt

        return system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", choices=["N", "dt"],
                        help="Task to solve (if not specified, generates all tasks)")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode (if not specified, generates both modes)")
    args = parser.parse_args()

    # If no specific task is provided, generate all tasks
    if args.task:
        tasks = [args.task]
    else:
        tasks = ["N", "dt"]

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

    print("HASEGAWA-MIMA NONLINEAR ICL DATASET GENERATOR")
    print("=" * 80)
    print(f"Tasks: {tasks}")
    print(f"Modes: {'zero-shot only' if len(modes) == 1 else 'iterative + zero-shot'}")
    print(f"ICL Types: accuracy_focused, cost_excluded, full")
    print(f"Output base directory: data/icl/hasegawa_mima_nonlinear/")
    print(f"Source directory: data/hasegawa_mima_nonlinear/{{task}}/{{precision_level}}/")

    total_files = 0

    for icl_type in icl_types:
        icl_name = icl_type["name"]
        use_uniform_icl = icl_type["use_uniform"]
        no_cost = icl_type["no_cost"]

        print(f"\n{'='*80}")
        print(f"Generating ICL Type: {icl_name.upper()}")
        print(f"{'='*80}")

        for task in tasks:
            print(f"\nTASK: {task.upper()}")
            print("-" * 50)

            task_dir = f"data/hasegawa_mima_nonlinear/{task}"

            # Get precision levels from the new structure
            precision_levels = []
            if os.path.exists(task_dir):
                precision_levels = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
            if not precision_levels:
                precision_levels = ["low", "medium", "high"]  # fallback

            generator = HasegawaMima_DatasetGenerator(
                f"tool_documentation/hasegawa_mima_nonlinear/{task}.json"
            )

            for precision_level in precision_levels:
                print(f"  {precision_level.upper()} precision:")

                # Create output directory for this ICL type and precision level
                out_dir = f"data/icl/hasegawa_mima_nonlinear/{icl_name}/{precision_level}"
                os.makedirs(out_dir, exist_ok=True)

                for zflag in modes:
                    flag = "zero_shot" if zflag else "iterative"
                    question_file = f"{task_dir}/{precision_level}/{flag}_questions.json"

                    if not os.path.exists(question_file):
                        print(f"     [!] Question file not found: {question_file}")
                        continue

                    with open(question_file, "r") as f:
                        questions = json.load(f)

                    dataset_entries = []

                    for idx, q in enumerate(questions):
                        # Get fixed parameters from the first parameter set
                        first_params = q["param_history"][0]

                        # Build workflow based on task type
                        if task == "N":
                            dt0 = fetch_param(first_params, "dt")
                            wf = build_N_workflow(zflag, dt0, precision_level, use_uniform_icl, no_cost)
                        elif task == "dt":
                            N0 = fetch_param(first_params, "N")
                            wf = build_dt_workflow(zflag, N0, precision_level, use_uniform_icl, no_cost)

                        single_ds = generator.generate_dataset(wf, [q], zflag)[0]

                        # Update dataset entry to only include required fields
                        precision_config = q.get("precision_config", {})
                        filtered_ds = {
                            "QID": single_ds.get("QID", q.get("QID")),
                            "profile": q.get("profile"),
                            "case": q.get("case"),
                            "zero_shot": q.get("zero_shot"),
                            "target_parameter": q.get("target_parameter"),
                            "precision_level": q.get("precision_level"),
                            "tolerance_rmse": precision_config.get("tolerance_rmse"),
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
                    print(f"     {flag.capitalize()}: {len(dataset_entries):2d} entries -> {dataset_file}")
                    total_files += 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: Generated {total_files} dataset files")
    print("=" * 80)

if __name__ == "__main__":
    main()
