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
            "nx",
            "dt_multiplier",
            "npart",
            "field_order",
            "particle_order"
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
        focused_parameters=['nx', 'dt_multiplier', 'npart', 'field_order', 'particle_order'],
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

def build_nx_workflow(zero_shot: bool, dt_multiplier0: float, npart0: int, field_order0: int, particle_order0: int) -> str:
    """Build the workflow for nx parameter optimization"""
    header = (
        "nx (Number of grid cells) determines the spatial discretization resolution for the 1D EPOCH PIC simulation: "
        "$\\Delta x = L / nx$ where L is the domain length.\n"
        "You may **only** change `nx`.\n"
        f"The value of dt_multiplier is **{dt_multiplier0}**, npart is **{npart0}**, field_order is **{field_order0}**, particle_order is **{particle_order0}**. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for nx.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that provides adequate spatial resolution while keeping computational cost reasonable.\n"
            "Step 1: Make your best **one-shot** guess for nx.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of nx, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine nx based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your resolution."
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_npart_workflow(zero_shot: bool, nx0: int, dt_multiplier0: float, field_order0: int, particle_order0: int) -> str:
    """Build the workflow for npart parameter optimization"""
    header = (
        "npart (Number of particles per cell) determines the particle density for the 1D EPOCH PIC simulation. "
        "Higher npart values provide better statistical accuracy but increase computational cost.\n"
        "You may **only** change `npart`.\n"
        f"The value of nx is **{nx0}**, dt_multiplier is **{dt_multiplier0}**, field_order is **{field_order0}**, particle_order is **{particle_order0}**. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for npart.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that provides adequate particle statistics while keeping computational cost reasonable.\n"
            "Step 1: Make your best **one-shot** guess for npart.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of npart, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine npart based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your resolution."
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_dt_multiplier_workflow(zero_shot: bool, npart0: int, field_order0: int, particle_order0: int) -> str:
    """dt_multiplier-task: 0-shot select dt_multiplier, then only adjust nx; other params are fixed."""
    header = (
        "This is a *composite* search task.\n"
        "You first pick **dt_multiplier** (time step multiplier) *once* and must never "
        "change it afterwards. The dt_multiplier determines the time step size as a fraction of the CFL limit.\n"
        f"The parameters **npart is fixed at {npart0}**, field_order is **{field_order0}**, particle_order is **{particle_order0}** and must not be changed.\n"
        "After choosing dt_multiplier, you may only modify `nx`.\n"
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick both *dt_multiplier* and *nx*.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1  Choose dt_multiplier based on domain knowledge (only one chance).\n"
            "Step 2  Choose an initial coarse nx.\n"
            "Step 3  Call the convergence test function, and refine nx based on the feedback from the simulation.\n"
            "Step 4  You may refine nx at most 10 times.\n"
            "Step 5  If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_field_order_workflow(zero_shot: bool, dt_multiplier0: float, npart0: int, particle_order0: int) -> str:
    """field_order-task: 0-shot select field_order, then only adjust nx; other params are fixed."""
    header = (
        "This is a *composite* search task.\n"
        "You first pick **field_order** (field interpolation order) *once* and must never "
        "change it afterwards. The field_order determines the polynomial order for field interpolation.\n"
        f"The parameters **dt_multiplier is fixed at {dt_multiplier0}**, npart is **{npart0}**, particle_order is **{particle_order0}** and must not be changed.\n"
        "After choosing field_order, you may only modify `nx`.\n"
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick both *field_order* and *nx*.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1  Choose field_order based on domain knowledge (only one chance).\n"
            "Step 2  Choose an initial coarse nx.\n"
            "Step 3  Call the convergence test function, and refine nx based on the feedback from the simulation.\n"
            "Step 4  You may refine nx at most 10 times.\n"
            "Step 5  If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_particle_order_workflow(zero_shot: bool, dt_multiplier0: float, npart0: int, field_order0: int) -> str:
    """particle_order-task: 0-shot select particle_order, then only adjust nx; other params are fixed."""
    header = (
        "This is a *composite* search task.\n"
        "You first pick **particle_order** (particle shape function order) *once* and must never "
        "change it afterwards. The particle_order determines the polynomial order for particle shape functions.\n"
        f"The parameters **dt_multiplier is fixed at {dt_multiplier0}**, npart is **{npart0}**, field_order is **{field_order0}** and must not be changed.\n"
        "After choosing particle_order, you may only modify `nx`.\n"
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick both *particle_order* and *nx*.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1  Choose particle_order based on domain knowledge (only one chance).\n"
            "Step 2  Choose an initial coarse nx.\n"
            "Step 3  Call the convergence test function, and refine nx based on the feedback from the simulation.\n"
            "Step 4  You may refine nx at most 10 times.\n"
            "Step 5  If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

class oneD_Epoch_DatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameter for 1D EPOCH particle-in-cell (PIC) simulations, "
            "which model plasma physics phenomena by tracking charged particles and solving electromagnetic fields. "
            "This serves as a fundamental tool for plasma physics research and laser-plasma interaction studies. "
            "You should try to minimize the total cost incurred by function calls, but your "
            "primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem."
        )

        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."

        system_prompt = zero_shot_system_prompt if zero_shot else iterative_system_prompt
        return system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", choices=["nx", "npart", "dt_multiplier", "field_order", "particle_order"],
                        help="Task to solve (if not specified, generates all tasks)")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode (if not specified, generates both modes)")
    args = parser.parse_args()

    # If no specific task is provided, generate all tasks
    if args.task:
        tasks = [args.task]
    else:
        tasks = ["nx", "npart", "dt_multiplier", "field_order", "particle_order"]

    # If no specific mode is provided, generate both modes
    if args.zero_shot:
        modes = [True]  # Only zero-shot
    else:
        modes = [False, True]  # Both iterative and zero-shot

    print("🚀 EPOCH 1D DATASET GENERATOR")
    print("=" * 80)
    print(f"📋 Tasks: {tasks}")
    print(f"🎯 Modes: {'zero-shot only' if len(modes) == 1 else 'iterative + zero-shot'}")

    total_files = 0

    for task in tasks:
        print(f"\n📋 TASK: {task.upper()}")
        print("-" * 50)

        # Handle the inconsistent naming (dt_multipler vs dt_multiplier)
        task_dir_name = "dt_multipler" if task == "dt_multiplier" else task
        task_dir = f"data/epoch_1d/{task_dir_name}"

        # Get precision levels from the new structure
        precision_levels = []
        if os.path.exists(task_dir):
            precision_levels = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        if not precision_levels:
            precision_levels = ["low", "medium", "high"]  # fallback

        generator = oneD_Epoch_DatasetGenerator(
            f"tool_documentation/epoch_1d/{task}.json"
        )

        for precision_level in precision_levels:
            print(f"  🎯 {precision_level.upper()} precision:")

            # Create output directory for this precision level
            out_dir = f"data/epoch_1d/human_write/{precision_level}"
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
                    if task == "nx":
                        dt_multiplier0 = fetch_param(q["param_history"][0], "dt_mult")
                        npart0 = fetch_param(q["param_history"][0], "npart")
                        field_order0 = fetch_param(q["param_history"][0], "field_Order")
                        particle_order0 = fetch_param(q["param_history"][0], "particle_order")
                        wf = build_nx_workflow(zflag, dt_multiplier0, npart0, field_order0, particle_order0)
                    elif task == "npart":
                        nx0 = fetch_param(q["param_history"][0], "nx")
                        dt_multiplier0 = fetch_param(q["param_history"][0], "dt_mult")
                        field_order0 = fetch_param(q["param_history"][0], "field_Order")
                        particle_order0 = fetch_param(q["param_history"][0], "particle_order")
                        wf = build_npart_workflow(zflag, nx0, dt_multiplier0, field_order0, particle_order0)
                    elif task == "dt_multiplier":
                        npart0 = fetch_param(q["param_history"][0][0], "npart")
                        field_order0 = fetch_param(q["param_history"][0][0], "field_Order")
                        particle_order0 = fetch_param(q["param_history"][0][0], "particle_order")
                        wf = build_dt_multiplier_workflow(zflag, npart0, field_order0, particle_order0)
                    elif task == "field_order":
                        dt_multiplier0 = fetch_param(q["param_history"][0][0], "dt_mult")
                        npart0 = fetch_param(q["param_history"][0][0], "npart")
                        particle_order0 = fetch_param(q["param_history"][0][0], "particle_order")
                        wf = build_field_order_workflow(zflag, dt_multiplier0, npart0, particle_order0)
                    elif task == "particle_order":
                        dt_multiplier0 = fetch_param(q["param_history"][0][0], "dt_mult")
                        npart0 = fetch_param(q["param_history"][0][0], "npart")
                        field_order0 = fetch_param(q["param_history"][0][0], "field_Order")
                        wf = build_particle_order_workflow(zflag, dt_multiplier0, npart0, field_order0)

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