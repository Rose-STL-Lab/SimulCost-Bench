import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.base import DatasetGenerator
import json
from inference import save_result
import argparse
from utils.param_compatibility import fetch_param

def build_n_grid_x_workflow(zero_shot: bool, cfl0: float, cg_tolerance0: float) -> str:
    """Build the workflow for the n_grid_x parameter optimization"""
    header = (
        "n_grid_x (Grid resolution in x-direction) determines the spatial discretization resolution: "
        "the number of grid cells in the x-direction. The cell size is dx = 1/n_grid_x.\n"
        "You may **only** change `n_grid_x`.\n"
        f"The fixed values are: cfl={cfl0}, cg_tolerance={cg_tolerance0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for n_grid_x.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that provides adequate spatial resolution while keeping computational cost reasonable.\n"
            "Step 1: Make your best **one-shot** guess for n_grid_x.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of n_grid_x, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine n_grid_x based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your n_grid_x.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_cfl_workflow(zero_shot: bool, n_grid_x0: int, cg_tolerance0: float) -> str:
    """Build the workflow for cfl parameter optimization"""
    header = (
        "cfl (CFL number - Courant-Friedrichs-Lewy condition) determines the time step stability constraint "
        "for explicit time integration: dt = cfl * dx / max_wave_speed.\n"
        "You may **only** change `cfl`.\n"
        f"The fixed values are: n_grid_x={n_grid_x0}, cg_tolerance={cg_tolerance0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for cfl.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances time step stability and computational efficiency.\n"
            "Step 1: Make your best **one-shot** guess for cfl.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of cfl, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine cfl based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_cg_tolerance_workflow(zero_shot: bool, n_grid_x0: int, cfl0: float) -> str:
    """Build the workflow for cg_tolerance parameter optimization"""
    header = (
        "cg_tolerance (Conjugate gradient solver tolerance) determines the convergence accuracy "
        "of the iterative CG solver used in the pressure projection step.\n"
        "You may **only** change `cg_tolerance`.\n"
        f"The fixed values are: n_grid_x={n_grid_x0}, cfl={cfl0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for cg_tolerance.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances solution accuracy and computational cost.\n"
            "Step 1: Make your best **one-shot** guess for cg_tolerance.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of cg_tolerance, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine cg_tolerance based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

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
            "n_grid_x",
            "cfl",
            "cg_tolerance"
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

iterative_HUMAN_CODE = r"""
def forward(self, data: dict):
    messages = data['messages']
    qid      = data['QID']

    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=[
            'n_grid_x',
            'cfl',
            'cg_tolerance'
        ],
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

class twoD_Euler_DatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameter, solving the 2D compressible Euler equations "
            "for inviscid gas dynamics using an advection-projection method with high-order WENO reconstruction "
            "and TVD Runge-Kutta time integration. "
            "This serves as a model for 2D compressible fluid flow problems including shock waves and rarefaction waves. "
            "You should try to minimize the total cost incurred by function calls, but your "
            "primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem."
        )

        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."

        system_prompt = zero_shot_system_prompt if zero_shot else iterative_system_prompt

        return system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", choices=["n_grid_x", "cfl", "cg_tolerance"],
                        help="Task to solve (if not specified, generates all tasks)")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode (if not specified, generates both modes)")
    args = parser.parse_args()

    # If no specific task is provided, generate all tasks
    if args.task:
        tasks = [args.task]
    else:
        tasks = ["n_grid_x", "cfl", "cg_tolerance"]

    # If no specific mode is provided, generate both modes
    if args.zero_shot:
        modes = [True]  # Only zero-shot
    else:
        modes = [False, True]  # Both iterative and zero-shot

    print("🚀 EULER 2D DATASET GENERATOR")
    print("=" * 80)
    print(f"📋 Tasks: {tasks}")
    print(f"🎯 Modes: {'zero-shot only' if len(modes) == 1 else 'iterative + zero-shot'}")

    total_files = 0

    for task in tasks:
        print(f"\n📋 TASK: {task.upper()}")
        print("-" * 50)

        task_dir = f"data/euler_2d/{task}"

        # Get precision levels from the new structure
        precision_levels = []
        if os.path.exists(task_dir):
            precision_levels = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        if not precision_levels:
            precision_levels = ["low", "medium", "high"]  # fallback

        generator = twoD_Euler_DatasetGenerator(
            f"tool_documentation/euler_2d/{task}.json"
        )

        for precision_level in precision_levels:
            print(f"  🎯 {precision_level.upper()} precision:")

            # Create output directory for this precision level
            out_dir = f"data/euler_2d/human_write/{precision_level}"
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
                    if task == "n_grid_x":
                        cfl0 = fetch_param(first_params, "cfl")
                        cg_tolerance0 = fetch_param(first_params, "cg_tolerance")
                        wf = build_n_grid_x_workflow(zflag, cfl0, cg_tolerance0)
                    elif task == "cfl":
                        n_grid_x0 = fetch_param(first_params, "n_grid_x")
                        cg_tolerance0 = fetch_param(first_params, "cg_tolerance")
                        wf = build_cfl_workflow(zflag, n_grid_x0, cg_tolerance0)
                    elif task == "cg_tolerance":
                        n_grid_x0 = fetch_param(first_params, "n_grid_x")
                        cfl0 = fetch_param(first_params, "cfl")
                        wf = build_cg_tolerance_workflow(zflag, n_grid_x0, cfl0)

                    single_ds = generator.generate_dataset(wf, [q], zflag)[0]

                    # Update dataset entry to only include required fields
                    precision_config = q.get("precision_config", {})
                    filtered_ds = {
                        "QID": single_ds.get("QID", q.get("QID")),
                        "profile": q.get("profile"),
                        "case": q.get("case"),  # Include case field for euler_2d
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
                print(f"     ✓ {flag.capitalize()}: {len(dataset_entries):2d} entries -> {dataset_file}")
                total_files += 1

    print("\n" + "=" * 80)
    print(f"🎉 SUMMARY: Generated {total_files} dataset files")
    print("=" * 80)

if __name__ == "__main__":
    main()
