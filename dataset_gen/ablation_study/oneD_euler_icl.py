import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            "cfl",
            "beta",
            "k",
            "n_space"
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

# Best parameters from data/euler_1d directory for ICL examples (one per precision level)
BEST_PARAMS_DATA = {
    "cfl": {
        "low": {"cfl": 1.0, "beta": 1.0, "k": -1.0, "n_space": 256, "simulation_cost": 30464},
        "medium": {"cfl": 0.5, "beta": 1.0, "k": -1.0, "n_space": 256, "simulation_cost": 58368},
        "high": {"cfl": 0.25, "beta": 1.0, "k": -1.0, "n_space": 256, "simulation_cost": 115968}
    },
    "beta": {
        "low": {"n_space": 256, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 115968},
        "medium": {"n_space": 1024, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 1846272},
        "high": {"n_space": 4096, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 29425664}
    },
    "k": {
        "low": {"n_space": 256, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 115968},
        "medium": {"n_space": 256, "cfl": 0.25, "beta": 1.0, "k": 0.4, "simulation_cost": 116736},
        "high": {"n_space": 1024, "cfl": 0.25, "beta": 1.0, "k": 0.4, "simulation_cost": 1849344}
    },
    "n_space": {
        "low": {"n_space": 256, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 115968},
        "medium": {"n_space": 1024, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 1846272},
        "high": {"n_space": 4096, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 29425664}
    }
}

def generate_icl_examples_for_precision(task: str, precision_level: str) -> list:
    """Generate ICL examples for a specific task and precision level (one converged, one non-converged)"""
    examples = []

    # Add one converged example for the specific precision level
    if precision_level in BEST_PARAMS_DATA[task]:
        params = BEST_PARAMS_DATA[task][precision_level].copy()
        if "simulation_cost" not in params:
            raise ValueError(f"Missing simulation_cost in BEST_PARAMS_DATA[{task}][{precision_level}]")
        simulation_cost = params.pop("simulation_cost")
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        example = (
            f"In a previous simulation ({precision_level} precision level) with parameters {param_str}, "
            f"the result was is_converged=true, simulation_cost={simulation_cost}."
        )
        examples.append(example)

    # Add one non-converged example for the specific precision level
    non_converged_examples = {
        "cfl": {
            "low": {"cfl": 0.8, "beta": 1.5, "k": 1.0, "n_space": 256, "simulation_cost": 44800},
            "medium": {"cfl": 0.6, "beta": 1.5, "k": 1.0, "n_space": 256, "simulation_cost": 52992},
            "high": {"cfl": 0.5, "beta": 1.5, "k": 1.0, "n_space": 256, "simulation_cost": 60928}
        },
        "beta": {
            "low": {"n_space": 25, "cfl": 0.25, "beta": 1.5, "k": 1.0, "simulation_cost": 1225},
            "medium": {"n_space": 100, "cfl": 0.25, "beta": 1.5, "k": -1.0, "simulation_cost": 17700},
            "high": {"n_space": 200, "cfl": 0.25, "beta": 1.5, "k": -1.0, "simulation_cost": 71000}
        },
        "k": {
            "low": {"n_space": 50, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 6000},
            "medium": {"n_space": 100, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 17700},
            "high": {"n_space": 200, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 71000}
        },
        "n_space": {
            "low": {"n_space": 10, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 260},
            "medium": {"n_space": 50, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 4600},
            "high": {"n_space": 100, "cfl": 0.25, "beta": 1.0, "k": -1.0, "simulation_cost": 17700}
        }
    }

    if task in non_converged_examples and precision_level in non_converged_examples[task]:
        params = non_converged_examples[task][precision_level].copy()
        if "simulation_cost" not in params:
            raise ValueError(f"Missing simulation_cost in non_converged_examples[{task}][{precision_level}]")
        simulation_cost = params.pop("simulation_cost")
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
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

def generate_specific_icl_examples_with_header(task: str, precision_level: str) -> list:
    """Generate specific ICL examples with proper numbering and single disclaimer"""
    raw_examples = generate_icl_examples_for_precision(task, precision_level)
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

def generate_uniform_icl_examples(task: str) -> list:
    """Generate uniform ICL examples (all 6 examples: 3 precision levels × 2 examples each) with proper numbering"""
    all_examples = []
    example_counter = 1

    for precision in ["low", "medium", "high"]:
        raw_examples = generate_icl_examples_for_precision(task, precision)
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

BETA_ICL_EXAMPLES = generate_icl_examples("beta", ["low", "medium", "high"])

K_ICL_EXAMPLES = generate_icl_examples("k", ["low", "medium", "high"])

N_SPACE_ICL_EXAMPLES = generate_icl_examples("n_space", ["low", "medium", "high"])

def get_icl_examples(task: str) -> list:
    """Get ICL examples for a specific task"""
    examples_map = {
        "cfl": CFL_ICL_EXAMPLES,
        "beta": BETA_ICL_EXAMPLES,
        "k": K_ICL_EXAMPLES,
        "n_space": N_SPACE_ICL_EXAMPLES
    }
    return examples_map.get(task, CFL_ICL_EXAMPLES)  # fallback to CFL examples

iterative_HUMAN_CODE = r"""
def forward(self, data: dict):
    messages = data['messages']
    qid      = data['QID']

    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=['cfl', 'beta', 'k', 'n_space'],
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

def build_cfl_workflow(zero_shot: bool, k0: float, beta0: float, n_space0: int, precision_level: str = "low", use_uniform_icl: bool = False) -> str:
    """Build the workflow for the first question based on the k0 and beta0"""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("cfl")
    else:
        icl_examples = generate_specific_icl_examples_with_header("cfl", precision_level)
    icl_text = "\n".join(icl_examples)

    header = (
        "CFL (Courant-Friedrichs-Lewy) number is defined as: "
        "$CFL = \\frac{(|u| + c) \\Delta t}{\\Delta x}$ where $c = \\sqrt{\\gamma \\frac{p}{\\rho}}$ is the speed of sound.\n"
        "You may **only** change `cfl`.\n"
        f"The value of k is **{k0}**, beta is **{beta0}**, n_space is **{n_space0}**. **You must not change them!**\n"
        "\n" + icl_text
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for cfl.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.\n"
            "Step 1: Make your best **one-shot** guess for cfl.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of cfl, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine cfl based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your resolution."
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_n_space_workflow(zero_shot: bool, k0: float, beta0: float, cfl0: float, precision_level: str = "low", use_uniform_icl: bool = False) -> str:
    """Build the workflow for n_space parameter optimization"""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("n_space")
    else:
        icl_examples = generate_specific_icl_examples_with_header("n_space", precision_level)
    icl_text = "\n".join(icl_examples)

    header = (
        "n_space (Number of grid cells) determines the spatial discretization resolution: "
        "$\\Delta x = L / n\\_space$ where L is the domain length.\n"
        "You may **only** change `n_space`.\n"
        f"The value of k is **{k0}**, beta is **{beta0}**, cfl is **{cfl0}**. **You must not change them!**\n"
        "\n" + icl_text
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for n_space.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that provides adequate spatial resolution while keeping computational cost reasonable.\n"
            "Step 1: Make your best **one-shot** guess for n_space.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of n_space, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine n_space based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your resolution."
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_k_workflow(zero_shot: bool, beta0: float, cfl0: float, precision_level: str = "low", use_uniform_icl: bool = False) -> str:
    """k-task: 0-shot select k, then only adjust n_space; beta is fixed."""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("k")
    else:
        icl_examples = generate_specific_icl_examples_with_header("k", precision_level)
    icl_text = "\n".join(icl_examples)

    header = (
        "This is a *composite* search task.\n"
        "You first pick **k** (blending parameter for MUSCL reconstruction) *once* and must never "
        "change it afterwards. The value of **k** determines the blend between central differencing (k=1) "
        "and upwind differencing (k=-1).\n"
        f"The parameter **beta is fixed at {beta0}**, cfl is **{cfl0}** and must not be changed.\n"
        "After choosing k, you may only modify `n_space`.\n"
        "\n" + icl_text
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick both *k* and *n_space*.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1  Choose k based on domain knowledge (only one chance).\n"
            "Step 2  Choose an initial coarse n_space.\n"
            "Step 3  Call the convergence test function, and refine n_space based on the feedback from the simulation.\n"
            "Step 4  You may refine n_space at most 10 times.\n"
            "Step 5  If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body


def build_beta_workflow(zero_shot: bool, k0: float, cfl0: float, precision_level: str = "low", use_uniform_icl: bool = False) -> str:
    """beta-task: 0-shot select beta, then only adjust n_space; k is fixed."""
    if use_uniform_icl:
        icl_examples = generate_uniform_icl_examples("beta")
    else:
        icl_examples = generate_specific_icl_examples_with_header("beta", precision_level)
    icl_text = "\n".join(icl_examples)

    header = (
        "This is a *composite* search task.\n"
        "You first pick **beta** (slope-limiter parameter for generalized superbee) *once* and must never "
        "change it afterwards.\n"
        f"The parameter **k is fixed at {k0}**, cfl is **{cfl0}** and must not be changed.\n"
        "After choosing beta, you may only modify `n_space`.\n"
        "\n" + icl_text
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick both *beta* and *n_space*.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1  Choose beta based on domain knowledge (only one chance).\n"
            "Step 2  Choose an initial n_space.\n"
            "Step 3  Call the convergence test function, and refine n_space based on the feedback from the simulation.\n"
            "Step 4  You may refine n_space at most 10 times.\n"
            "Step 5  If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

class oneD_Euler_DatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameter, solving the 1D Euler equations for compressible inviscid flow, "
            "using a 2nd order MUSCL scheme with Roe flux and generalized superbee limiter. This serves as a simplified model "
            "for compressible fluid dynamics. You should try to minimize the total cost incurred by function calls, but your "
            "primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem."
        )

        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."
        

        system_prompt = zero_shot_system_prompt if zero_shot else iterative_system_prompt

        return system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", choices=["cfl", "k", "beta", "n_space"],
                        help="Task to solve (if not specified, generates all tasks)")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode (if not specified, generates both modes)")
    icl_group = parser.add_mutually_exclusive_group()
    icl_group.add_argument("--specific", action="store_true",
                          help="Use precision-specific ICL examples (current behavior)")
    icl_group.add_argument("--uniform", action="store_true",
                          help="Use uniform ICL examples (all 6 examples: 3 precision levels × 2 examples each)")
    args = parser.parse_args()
    
    # If no specific task is provided, generate all tasks
    if args.task:
        tasks = [args.task]
    else:
        tasks = ["cfl", "n_space", "k", "beta"]
    
    # If no specific mode is provided, generate both modes
    if args.zero_shot:
        modes = [True]  # Only zero-shot
    else:
        modes = [False, True]  # Both iterative and zero-shot

    # Determine ICL mode - default to specific if neither is specified
    use_uniform_icl = args.uniform
    if not args.specific and not args.uniform:
        use_uniform_icl = False  # Default to specific mode

    print("🚀 EULER 1D ICL DATASET GENERATOR")
    print("=" * 80)
    print(f"📋 Tasks: {tasks}")
    print(f"🎯 Modes: {'zero-shot only' if len(modes) == 1 else 'iterative + zero-shot'}")
    print(f"🧠 ICL Mode: {'uniform (6 examples)' if use_uniform_icl else 'specific (2 examples per precision)'}")
    print(f"📂 Output directory: data/euler_1d_icl/human_write/{{precision_level}}/")
    print(f"📂 Source directory: data/euler_1d_icl/{{task}}/{{precision_level}}/")
    
    total_files = 0
    
    for task in tasks:
        print(f"\n📋 TASK: {task.upper()}")
        print("-" * 50)
        
        task_dir = f"data/euler_1d_icl/{task}"
        
        # Get precision levels from the new structure
        precision_levels = []
        if os.path.exists(task_dir):
            precision_levels = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        if not precision_levels:
            precision_levels = ["low", "medium", "high"]  # fallback

        generator = oneD_Euler_DatasetGenerator(
            f"tool_documentation/euler_1d/{task}.json"
        )

        for precision_level in precision_levels:
            print(f"  🎯 {precision_level.upper()} precision:")
            
            # Create output directory for this precision level
            out_dir = f"data/euler_1d_icl/human_write/{precision_level}"
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
                    if task == "cfl":
                        k0 = fetch_param(q["param_history"][0], "k")
                        beta0 = fetch_param(q["param_history"][0], "beta")
                        n_space0 = fetch_param(q["param_history"][0], "n_space")
                        wf = build_cfl_workflow(zflag, k0, beta0, n_space0, precision_level, use_uniform_icl)
                    elif task == "n_space":
                        k0 = fetch_param(q["param_history"][0], "k")
                        beta0 = fetch_param(q["param_history"][0], "beta")
                        cfl0 = fetch_param(q["param_history"][0], "cfl")
                        wf = build_n_space_workflow(zflag, k0, beta0, cfl0, precision_level, use_uniform_icl)
                    elif task == "k":
                        beta0 = fetch_param(q["param_history"][0][0], "beta")
                        cfl0 = fetch_param(q["param_history"][0][0], "cfl")
                        wf = build_k_workflow(zflag, beta0, cfl0, precision_level, use_uniform_icl)
                    elif task == "beta":
                        k0 = fetch_param(q["param_history"][0][0], "k")
                        cfl0 = fetch_param(q["param_history"][0][0], "cfl")
                        wf = build_beta_workflow(zflag, k0, cfl0, precision_level, use_uniform_icl)

                    single_ds = generator.generate_dataset(wf, [q], zflag)[0]
                    
                    # Update dataset entry to only include required fields
                    filtered_ds = {
                        "QID": single_ds.get("QID", q.get("QID")),
                        "profile": q.get("profile"),
                        "case": q.get("case"),
                        "zero_shot": q.get("zero_shot"),
                        "target_parameter": q.get("target_parameter"),
                        "precision_level": q.get("precision_level"),
                        "tolerance_rmse": q.get("tolerance_rmse"),
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
