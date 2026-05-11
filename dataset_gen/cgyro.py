import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from dataset_gen.base import DatasetGenerator
from inference import save_result
from utils.param_compatibility import fetch_param


TASKS = ["n_radial", "n_theta", "n_xi", "n_energy", "freq_tol", "delta_t"]


zero_shot_HUMAN_CODE = r"""
def forward(self, data: dict):
    messages = data["messages"]
    qid = data["QID"]

    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=[
            "n_radial",
            "n_theta",
            "n_xi",
            "n_energy",
            "freq_tol",
            "delta_t",
        ],
        tolerance_rmse=data.get("tolerance_rmse"),
    )

    experiment_instruction = "Given the problem, you should use the tool call to run the experiment."
    experiment_agent = self.get_experiment_agent()

    tool_reason, tool_name, tool_args = experiment_agent.query(messages, experiment_instruction)
    messages.append({
        "role": "assistant",
        "content": json.dumps({
            "tool_reason": tool_reason,
            "tool_name": tool_name,
            "tool_args": tool_args,
        }),
    })

    tool_result, acc_cost = experiment_agent.execute_tool(
        tool_reason=tool_reason, tool_name=tool_name, tool_args=tool_args,
        tool_manager=experiment_manager, profile=data["profile"],
    )
    messages.append({"role": "user", "content": json.dumps(tool_result)})

    param_seq = experiment_manager.get_param_sequence()
    cost_seq = experiment_manager.get_cost_sequence()

    summary_data = {
        "QID": qid,
        "is_converged": tool_result.get("is_converged", False),
        "times": len(param_seq),
        "param_sequence": param_seq,
        "accumulated_cost": experiment_manager.accumulated_cost,
        "cost_sequence": cost_seq,
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
        focused_parameters=['n_radial', 'n_theta', 'n_xi', 'n_energy', 'freq_tol', 'delta_t'],
        tolerance_rmse=data.get('tolerance_rmse'),
    )

    exp_instr = (
        'Given the problem, you should use the tool call to run the experiment. '
        'When you think the experiment can be stopped, set should_stop to true, '
        'otherwise set it to false.'
    )
    agent = LLMAgentBase(
        ['tool_reason', 'tool_name', 'tool_args', 'should_stop'],
        'Experiment Agent',
        self.logger,
    )

    last_valid_tool_result = None
    tool_result = None

    for attempt in range(10):
        tool_reason, tool_name, tool_args, should_stop = agent.query(messages, exp_instr)
        messages.append({
            'role': 'assistant',
            'content': json.dumps({
                'tool_reason': tool_reason,
                'tool_name':  tool_name,
                'tool_args':  tool_args,
                'should_stop': should_stop,
            }),
        })

        stop_flag = bool(should_stop) if isinstance(should_stop, bool) else str(should_stop).lower() == 'true'
        if stop_flag:
            break

        tool_result, acc_cost = agent.execute_tool(
            tool_reason=tool_reason, tool_name=tool_name, tool_args=tool_args,
            tool_manager=experiment_manager, profile=data['profile'],
        )
        if 'error' not in tool_result:
            last_valid_tool_result = tool_result
        messages.append({'role': 'user', 'content': json.dumps(tool_result)})

    param_seq = experiment_manager.get_param_sequence()
    cost_seq  = experiment_manager.get_cost_sequence()

    if last_valid_tool_result is not None:
        final_tool_result = last_valid_tool_result
    elif tool_result is not None:
        final_tool_result = tool_result
    else:
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


def _fixed_block(exclude: str, fixed: dict) -> str:
    """Format the non-target parameter block for the workflow header."""
    kept = {k: v for k, v in fixed.items() if k != exclude}
    parts = [f"{k} is **{v}**" for k, v in kept.items()]
    return ", ".join(parts)


def build_single_param_workflow(task: str, zero_shot: bool, fixed: dict) -> str:
    header = (
        f"The target parameter is **{task}**. "
        f"You may **only** change `{task}`.\n"
        f"The other parameters are fixed: {_fixed_block(task, fixed)}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            f"You have only one opportunity to choose an optimal value for {task}.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to pick a value that yields converged CGYRO eigenvalues at minimal cost.\n"
            f"Step 1: Make your best **one-shot** guess for {task}.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            f"Step 1: Estimate an initial fairly coarse choice of {task}, as you will gradually refine and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            f"Step 3: Refine {task} based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your resolution.\n"
            "Step 5: If you think the experiment can be stopped, respond with the final response format and make no further function calls. If you reach the 10th refinement, perform a convergence check immediately after that refinement and then respond with the final response format regardless of outcome."
        )
    return header + body


class CgyroDatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameter for linear CGYRO gyrokinetic simulations, "
            "which model turbulent transport in magnetized fusion plasmas. "
            "You should try to minimize the total cost incurred by function calls, but your "
            "primary goal is to successfully meet the convergence criteria. "
            "You should always use the tool call function to finish the problem."
        )
        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."
        return zero_shot_system_prompt if zero_shot else iterative_system_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", choices=TASKS, help="Task to generate (default: all)")
    parser.add_argument("-z", "--zero_shot", action="store_true", help="Zero-shot only (default: both)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    modes = [True] if args.zero_shot else [False, True]

    print("🚀 CGYRO DATASET GENERATOR")
    print("=" * 80)
    print(f"📋 Tasks: {tasks}")
    print(f"🎯 Modes: {'zero-shot only' if len(modes) == 1 else 'iterative + zero-shot'}")

    total_files = 0
    for task in tasks:
        print(f"\n📋 TASK: {task.upper()}")
        print("-" * 50)

        task_dir = f"data/cgyro/{task}"
        precision_levels = []
        if os.path.exists(task_dir):
            precision_levels = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        if not precision_levels:
            precision_levels = ["low", "medium", "high"]

        generator = CgyroDatasetGenerator(f"tool_documentation/cgyro/{task}.json")

        for precision_level in precision_levels:
            print(f"  🎯 {precision_level.upper()} precision:")
            out_dir = f"data/cgyro/human_write/{precision_level}"
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
                    first_params = q["param_history"][0]
                    if isinstance(first_params, list):
                        first_params = first_params[0]
                    fixed = {
                        "n_radial": fetch_param(first_params, "n_radial"),
                        "n_theta":  fetch_param(first_params, "n_theta"),
                        "n_xi":     fetch_param(first_params, "n_xi"),
                        "n_energy": fetch_param(first_params, "n_energy"),
                        "freq_tol": fetch_param(first_params, "freq_tol"),
                        "delta_t":  fetch_param(first_params, "delta_t"),
                    }
                    wf = build_single_param_workflow(task, zflag, fixed)

                    single_ds = generator.generate_dataset(wf, [q], zflag)[0]
                    filtered_ds = {
                        "QID": single_ds.get("QID", q.get("QID")),
                        "profile": q.get("profile"),
                        "case": q.get("case"),
                        "zero_shot": q.get("zero_shot"),
                        "target_parameter": q.get("target_parameter"),
                        "precision_level": q.get("precision_level"),
                        "tolerance_rmse": q.get("tolerance_rmse"),
                        "messages": single_ds.get("messages"),
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
