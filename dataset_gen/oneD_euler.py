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
            "current_cfl",
            "beta",
            "k"
        ]
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
        focused_parameters=['current_cfl', 'beta', 'k']
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

def build_cfl_workflow(zero_shot: bool, k0: float, beta0: float) -> str:
    """Build the workflow for the first question based on the k0 and beta0"""
    header = (
        "CFL (Courant-Friedrichs-Lewy) number is defined as: "
        "$CFL = \\frac{(|u| + c) \\Delta t}{\\Delta x}$ where $c = \\sqrt{\\gamma \\frac{p}{\\rho}}$ is the speed of sound.\n"
        "You may **only** change `cfl`.\n"
        f"The value of k is **{k0}**, beta is **{beta0}**. **You must not change them!**\n"
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

def build_k_workflow(zero_shot: bool, beta0: float) -> str:
    """k-task: 0-shot select k, then only adjust cfl; beta is fixed."""
    header = (
        "This is a *composite* search task.\n"
        "You first pick **k** (blending parameter for MUSCL reconstruction) *once* and must never "
        "change it afterwards. The value of **k** determines the blend between central differencing (k=1) "
        "and upwind differencing (k=-1).\n"
        f"The parameter **beta is fixed at {beta0}** and must not be changed.\n"
        "After choosing k, you may only modify `cfl`.\n"
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick both *k* and *cfl*.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1  Choose k based on domain knowledge (only one chance).\n"
            "Step 2  Choose an initial coarse cfl.\n"
            "Step 3  Call the convergence test function, and refine cfl based on the feedback from the simulation.\n"
            "Step 4  You may refine cfl at most 10 times.\n"
            "Step 5  If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body


def build_beta_workflow(zero_shot: bool, k0: float) -> str:
    """beta-task: 0-shot select beta, then only adjust cfl; k is fixed."""
    header = (
        "This is a *composite* search task.\n"
        "You first pick **beta** (slope-limiter parameter for generalized superbee) *once* and must never "
        "change it afterwards.\n"
        f"The parameter **k is fixed at {k0}** and must not be changed.\n"
        "After choosing beta, you may only modify `cfl`.\n"
    )
    if zero_shot:
        body = (
            "You have **one shot** to pick both *beta* and *cfl*.\n"
            "Then call the Convergence Test Function and report the result."
        )
    else:
        body = (
            "Step 1  Choose beta based on domain knowledge (only one chance).\n"
            "Step 2  Choose an initial cfl.\n"
            "Step 3  Call the convergence test function, and refine cfl based on the feedback from the simulation.\n"
            "Step 4  You may refine cfl at most 10 times.\n"
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
    parser.add_argument("-t", "--task", choices=["cfl", "k", "beta"],
                        default="cfl", help="Task to solve")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode")
    args = parser.parse_args()
    task  = args.task
    zflag = args.zero_shot
    flag  = "zero_shot" if zflag else "iterative"

    task_dir = f"data/euler_1d/{task}"
    cases = list_cases(task_dir)
    if not cases:
        cases = [""]

    out_dir = "data/euler_1d/human_write"
    os.makedirs(out_dir, exist_ok=True)

    generator = oneD_Euler_DatasetGenerator(
        f"tool_documentation/euler_1d/{task}.json"
    )

    for case in cases:
        case_prefix   = f"{case}/" if case else ""
        question_file = f"{task_dir}/{case_prefix}{flag}_questions.json"

        with open(question_file, "r") as f:
            questions = json.load(f)

        dataset_entries = []

        for idx, q in enumerate(questions):
            if task == "cfl":
                k0 = fetch_param(q["param_history"][0], "k")
                beta0 = fetch_param(q["param_history"][0], "beta")
                wf = build_cfl_workflow(zflag, k0, beta0)
            elif task == "k":
                beta0 = fetch_param(q["param_history"][0][0], "beta")
                wf = build_k_workflow(zflag, beta0)
            elif task == "beta":
                k0 = fetch_param(q["param_history"][0][0], "k")
                wf = build_beta_workflow(zflag, k0)

            single_ds = generator.generate_dataset(wf, [q], zflag)[0]
            single_ds["profile"] = q.get("profile")
            single_ds["zero_shot"] = q.get("zero_shot")
            single_ds["case"] = q.get("case")

            dataset_entries.append(single_ds)

            if idx == 0:
                human_code = zero_shot_HUMAN_CODE if zflag else iterative_HUMAN_CODE
                agent = {"workflow": wf, "code": human_code}
                archive_file = f"{out_dir}/{task}_{case or 'default'}_{flag}_agent.json"
                with open(archive_file, "w") as f:
                    json.dump([agent], f, indent=4)

        dataset_file = f"{out_dir}/{task}_{case or 'default'}_{flag}_dataset.json"
        save_result(dataset_entries, dataset_file)
        print(f"[✓] {case or 'default'} done -> {dataset_file}")

if __name__ == "__main__":
    main()
