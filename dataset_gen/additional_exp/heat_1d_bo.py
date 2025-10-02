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
            "n_space"
        ],
        tolerance_rmse=data.get("tolerance_rmse")
    )

    # Set up experiment agent with heat_1d_bo dataset name for special handling
    experiment_instruction = "Given the problem, you should use the tool call to run the experiment."
    experiment_agent = self.get_experiment_agent(dataset_name="heat_1d_bo")

    # Zero-Shot with profile passed directly
    tool_reason, tool_name, tool_args = experiment_agent.query(messages, experiment_instruction, data["profile"])
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
            "cfl",
            "n_space"
        ],
        tolerance_rmse=data.get("tolerance_rmse")
    )

    # Set up experiment agent with heat_1d_bo dataset name for special handling
    agent = self.get_experiment_agent(["tool_reason", "tool_name", "tool_args", "should_stop"], dataset_name="heat_1d_bo")

    exp_instr = "Given the current problem and history, decide on the next experiment to run or if you should stop."

    last_valid_tool_result = None  # Track the last successful tool result
    tool_result = None  # Initialize to avoid undefined variable

    for attempt in range(10):
        tool_reason, tool_name, tool_args, should_stop = agent.query(messages, exp_instr, data['profile'])
        messages.append({
            'role': 'assistant',
            'content': json.dumps({
                'tool_reason': tool_reason,
                'tool_name': tool_name,
                'tool_args': tool_args,
                'should_stop': should_stop
            })
        })

        if should_stop:
            break

        try:
            tool_result, cost = agent.execute_tool(
                tool_reason=tool_reason,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_manager=experiment_manager,
                profile=data['profile']
            )

            if tool_result.get("is_converged", False):
                last_valid_tool_result = tool_result
                messages.append({
                    'role': 'user',
                    'content': json.dumps(tool_result)
                })
                break
            else:
                messages.append({
                    'role': 'user',
                    'content': json.dumps(tool_result)
                })
        except Exception as e:
            messages.append({
                'role': 'user',
                'content': json.dumps({"error": f"Tool execution failed: {str(e)}"})
            })

    # Use the last valid result if available, otherwise the latest result
    final_result = last_valid_tool_result if last_valid_tool_result else tool_result

    param_seq = experiment_manager.get_param_sequence()
    cost_seq = experiment_manager.get_cost_sequence()

    summary_data = {
        "QID": qid,
        "is_converged": final_result.get("is_converged", False) if final_result else False,
        "times": len(param_seq),
        "param_sequence": param_seq,
        "accumulated_cost": experiment_manager.accumulated_cost,
        "cost_sequence": cost_seq,
        "profile": data['profile']
    }

    tool_df = experiment_manager.get_tool_call_df()
    return summary_data, tool_df
"""

class heat1D_BO_DatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        # Load the template from the doc file
        with open(self.doc_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

        base_template = doc_data.get('instruction_template', '')

        if zero_shot:
            mode_instruction = (
                "You have only one opportunity to choose optimal values.\n"
                "No trial-and-error or iterative optimization is permitted.\n"
            )
        else:
            mode_instruction = (
                "You may iteratively refine your parameters based on feedback.\n"
                "You have at most 10 total opportunities to refine your solution.\n"
            )

        return f"{base_template}\n\n{mode_instruction}"

def list_cases(task_dir: str):
    """Return all cases in the task directory"""
    return [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]

def build_cfl_workflow(zero_shot: bool, n_space0: int) -> str:
    """Build the workflow for the CFL parameter optimization"""
    header = (
        "CFL (Courant-Friedrichs-Lewy) number is defined for diffusion problems as: "
        "$CFL = \\frac{\\alpha \\Delta t}{(\\Delta x)^2}$ where $\\alpha$ is the thermal diffusivity.\n"
        "You may **only** change `cfl`.\n"
        f"The value of n_space is **{n_space0}**. **You must not change it!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for cfl.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that ensures temporal stability while keeping computational cost reasonable.\n"
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
        )
    return header + body

def build_n_space_workflow(zero_shot: bool, cfl0: float) -> str:
    """Build the workflow for the n_space parameter optimization"""
    header = (
        "n_space is the number of spatial grid points for discretizing the domain.\n"
        "You may **only** change `n_space`.\n"
        f"The value of cfl is **{cfl0}**. **You must not change it!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for n_space.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that ensures spatial accuracy while keeping computational cost reasonable.\n"
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
        )
    return header + body

def main():
    """Main function that generates the complete heat_1d_bo dataset"""
    tasks = ["cfl", "n_space"]
    modes = [True, False]  # [zero_shot, iterative]

    print("🚀 HEAT TRANSFER 1D BO DATASET GENERATOR")
    print("=" * 80)
    print(f"📋 Tasks: {tasks}")
    print(f"🎯 Modes: {'zero-shot only' if len(modes) == 1 else 'iterative + zero-shot'}")

    total_files = 0

    for task in tasks:
        print(f"\n📋 TASK: {task.upper()}")
        print("-" * 50)

        # Use heat_1d data directory since heat_1d_bo shares the same data
        task_dir = f"data/heat_1d/{task}"

        # Get precision levels from the new structure
        precision_levels = []
        if os.path.exists(task_dir):
            precision_levels = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        if not precision_levels:
            precision_levels = ["low", "medium", "high"]  # fallback

        generator = heat1D_BO_DatasetGenerator(
            f"tool_documentation/oneD_heat_transfer/{task}.json"
        )

        for precision_level in precision_levels:
            print(f"  🎯 {precision_level.upper()} precision:")

            # Create output directory for this precision level under heat_1d_bo
            out_dir = f"data/heat_1d_bo/human_write/{precision_level}"
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
                        n_space0 = fetch_param(q["param_history"][0], "n_space")
                        wf = build_cfl_workflow(zflag, n_space0)
                    elif task == "n_space":
                        cfl0 = fetch_param(q["param_history"][0], "cfl")
                        wf = build_n_space_workflow(zflag, cfl0)

                    single_ds = generator.generate_dataset(wf, [q], zflag)[0]

                    # Update dataset entry to only include required fields
                    filtered_ds = {
                        "QID": single_ds.get("QID", q.get("QID")),
                        "profile": q.get("profile"),
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
    print(f"🎉 SUMMARY: Generated {total_files} dataset files for heat_1d_bo")
    print("=" * 80)

if __name__ == "__main__":
    main()