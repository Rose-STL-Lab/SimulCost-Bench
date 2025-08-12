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
    messages = data["messages"]
    qid = data["QID"]

    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=[
            "mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p",
            "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"
        ]
    )

    experiment_instruction = "Given the problem, you should use the tool call to run the experiment."
    experiment_agent = self.get_experiment_agent()
    
    tool_reason, tool_name, tool_args = experiment_agent.query(messages, experiment_instruction)
    messages.append({
        "role": "assistant",
        "content": json.dumps({
            "tool_reason": tool_reason,
            "tool_name": tool_name,
            "tool_args": tool_args
        })
    })

    tool_result, acc_cost = experiment_agent.execute_tool(
        tool_reason=tool_reason, tool_name=tool_name, tool_args=tool_args, 
        tool_manager=experiment_manager, profile=data["profile"])
    messages.append({
        "role": "user",
        "content": json.dumps(tool_result)
    })

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
    qid = data['QID']

    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=[
            'mesh_x', 'mesh_y', 'omega_u', 'omega_v', 'omega_p',
            'diff_u_threshold', 'diff_v_threshold', 'res_iter_v_threshold'
        ]
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

    last_valid_tool_result = None
    tool_result = None
    
    for attempt in range(10):
        tool_reason, tool_name, tool_args, should_stop = agent.query(messages, exp_instr)
        messages.append({
            'role': 'assistant',
            'content': json.dumps({
                'tool_reason': tool_reason,
                'tool_name': tool_name,
                'tool_args': tool_args,
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
        
        if 'error' not in tool_result:
            last_valid_tool_result = tool_result
            
        messages.append({'role': 'user', 'content': json.dumps(tool_result)})

    param_seq = experiment_manager.get_param_sequence()
    cost_seq = experiment_manager.get_cost_sequence()
    
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

def list_cases(task_dir: str):
    """Return all cases in the task directory"""
    return [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]

def build_workflow(task: str, zero_shot: bool, defaults: dict) -> str:
    """Build workflow based on task type"""
    task_descriptions = {
        "mesh_x": "mesh resolution in x-direction (mesh_x)",
        "mesh_y": "mesh resolution in y-direction (mesh_y)", 
        "omega_u": "under-relaxation factor for u-velocity (omega_u)",
        "omega_v": "under-relaxation factor for v-velocity (omega_v)",
        "omega_p": "under-relaxation factor for pressure (omega_p)",
        "diff_u_threshold": "convergence threshold for u-velocity (diff_u_threshold)",
        "diff_v_threshold": "convergence threshold for v-velocity (diff_v_threshold)",
        "res_iter_v_threshold": "residual threshold for pressure/velocity coupling (res_iter_v_threshold)"
    }
    
    fixed_params = {k: v for k, v in defaults.items() if k != task}
    fixed_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
    
    header = (
        "You are solving 2D Navier-Stokes channel flow using SIMPLE algorithm on a staggered finite-volume grid.\n"
        f"Your task is to optimize the {task_descriptions[task]} to achieve convergence with minimal computational cost.\n"
        f"Fixed parameters: {fixed_str}\n"
        f"You may **only** change `{task}`.\n"
    )
    
    if zero_shot:
        body = (
            f"You have only one opportunity to choose an optimal value for {task}.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a parameter value that achieves convergence while minimizing computational cost.\n"
            f"Step 1: Make your best **one-shot** guess for {task}.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            f"Step 1: Start with an initial value for {task} as you will gradually refine it.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            f"Step 3: Refine {task} based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, respond with the final format and make no further function calls. "
            "If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; "
            "then, regardless of convergence status, respond with the final format and make no further function calls."
        )
    return header + body

class twoD_NS_DatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template"""
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameters for solving the 2D Navier-Stokes equations for incompressible channel flow, "
            "using the SIMPLE algorithm on a staggered finite-volume grid. This serves as a fundamental model "
            "for computational fluid dynamics. You should try to minimize the total cost incurred by function calls, but your "
            "primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem."
        )

        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."
        return zero_shot_system_prompt if zero_shot else iterative_system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", 
                        choices=["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", 
                                "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
                        default="mesh_x", help="Task to solve")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode")
    args = parser.parse_args()
    task = args.task
    zflag = args.zero_shot
    flag = "zero_shot" if zflag else "iterative"

    task_dir = f"data/2D_ns/{task}"
    cases = list_cases(task_dir)
    if not cases:
        cases = [""]

    out_dir = "data/2D_ns/human_write"
    os.makedirs(out_dir, exist_ok=True)

    generator = twoD_NS_DatasetGenerator(
        f"tool_documentation/ns_channel_2d/{task}.json"
    )

    for case in cases:
        case_prefix = f"{case}/" if case else ""
        question_file = f"{task_dir}/{case_prefix}{flag}_question.json"

        with open(question_file, "r") as f:
            questions = json.load(f)

        dataset_entries = []

        for idx, q in enumerate(questions):
            defaults = {
                "mesh_x": fetch_param(q["param_history"][0], "mesh_x", 100),
                "mesh_y": fetch_param(q["param_history"][0], "mesh_y", 20),
                "omega_u": fetch_param(q["param_history"][0], "omega_u", 0.5),
                "omega_v": fetch_param(q["param_history"][0], "omega_v", 0.5),
                "omega_p": fetch_param(q["param_history"][0], "omega_p", 0.5),
                "diff_u_threshold": fetch_param(q["param_history"][0], "diff_u_threshold", 0.0001),
                "diff_v_threshold": fetch_param(q["param_history"][0], "diff_v_threshold", 0.0001),
                "res_iter_v_threshold": fetch_param(q["param_history"][0], "res_iter_v_threshold", 0.0001)
            }

            wf = build_workflow(task, zflag, defaults)

            single_ds = generator.generate_dataset(wf, [q], zflag)[0]
            single_ds["profile"] = q.get("profile")
            single_ds["zero_shot"] = q.get("zero_shot")
            single_ds["case"] = q.get("case")
            single_ds["boundary_condition"] = q.get("boundary_condition")

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