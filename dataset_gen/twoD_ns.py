import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.base import DatasetGenerator
import json
from inference import save_result
import argparse
from utils.param_compatibility import fetch_param

def build_mesh_x_workflow(zero_shot: bool, mesh_y0: int, omega_u0: float, omega_v0: float, omega_p0: float, 
                          diff_u_threshold0: float, diff_v_threshold0: float, res_iter_v_threshold0: str, aspect_ratio: float) -> str:
    """Build the workflow for the mesh_x parameter optimization"""
    header = (
        "mesh_x (Grid resolution in x-direction) determines the spatial discretization resolution in the x-direction: "
        "the number of grid cells along the channel length.\n"
        f"**CRITICAL CONSTRAINT**: You must maintain the aspect ratio mesh_y/mesh_x = {aspect_ratio}. "
        f"This means if you choose mesh_x = X, then mesh_y must be {aspect_ratio} * X = {int(aspect_ratio * 100)}% of X.\n"
        "You may **only** change `mesh_x`.\n"
        f"The fixed values are: mesh_y (calculated from aspect ratio), omega_u={omega_u0}, omega_v={omega_v0}, "
        f"omega_p={omega_p0}, diff_u_threshold={diff_u_threshold0}, diff_v_threshold={diff_v_threshold0}, "
        f"res_iter_v_threshold={res_iter_v_threshold0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for mesh_x.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that provides adequate spatial resolution while keeping computational cost reasonable.\n"
            "Step 1: Make your best **one-shot** guess for mesh_x.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of mesh_x, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine mesh_x based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your mesh resolution.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_mesh_y_workflow(zero_shot: bool, mesh_x0: int, omega_u0: float, omega_v0: float, omega_p0: float, 
                          diff_u_threshold0: float, diff_v_threshold0: float, res_iter_v_threshold0: str, aspect_ratio: float) -> str:
    """Build the workflow for mesh_y parameter optimization"""
    header = (
        "mesh_y (Grid resolution in y-direction) determines the spatial discretization resolution in the y-direction: "
        "the number of grid cells across the channel width.\n"
        f"**CRITICAL CONSTRAINT**: You must maintain the aspect ratio mesh_y/mesh_x = {aspect_ratio}. "
        f"This means if you choose mesh_y = Y, then mesh_x must be Y / {aspect_ratio} = {int(100/aspect_ratio)}% of Y.\n"
        "You may **only** change `mesh_y`.\n"
        f"The fixed values are: mesh_x (calculated from aspect ratio), omega_u={omega_u0}, omega_v={omega_v0}, "
        f"omega_p={omega_p0}, diff_u_threshold={diff_u_threshold0}, diff_v_threshold={diff_v_threshold0}, "
        f"res_iter_v_threshold={res_iter_v_threshold0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for mesh_y.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that provides adequate spatial resolution while keeping computational cost reasonable.\n"
            "Step 1: Make your best **one-shot** guess for mesh_y.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of mesh_y, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine mesh_y based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your mesh resolution.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_omega_u_workflow(zero_shot: bool, mesh_x0: int, mesh_y0: int, omega_v0: float, omega_p0: float,
                          diff_u_threshold0: float, diff_v_threshold0: float, res_iter_v_threshold0: str) -> str:
    """Build the workflow for omega_u parameter optimization"""
    header = (
        "omega_u (Under-relaxation factor for u-velocity) determines the convergence speed and stability "
        "of the u-velocity component in the SIMPLE algorithm.\n"
        "You may **only** change `omega_u`.\n"
        f"The fixed values are: mesh_x={mesh_x0}, mesh_y={mesh_y0}, omega_v={omega_v0}, omega_p={omega_p0}, "
        f"diff_u_threshold={diff_u_threshold0}, diff_v_threshold={diff_v_threshold0}, "
        f"res_iter_v_threshold={res_iter_v_threshold0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for omega_u.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances convergence speed and stability.\n"
            "Step 1: Make your best **one-shot** guess for omega_u.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of omega_u, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine omega_u based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_omega_v_workflow(zero_shot: bool, mesh_x0: int, mesh_y0: int, omega_u0: float, omega_p0: float,
                          diff_u_threshold0: float, diff_v_threshold0: float, res_iter_v_threshold0: str) -> str:
    """Build the workflow for omega_v parameter optimization"""
    header = (
        "omega_v (Under-relaxation factor for v-velocity) determines the convergence speed and stability "
        "of the v-velocity component in the SIMPLE algorithm.\n"
        "You may **only** change `omega_v`.\n"
        f"The fixed values are: mesh_x={mesh_x0}, mesh_y={mesh_y0}, omega_u={omega_u0}, omega_p={omega_p0}, "
        f"diff_u_threshold={diff_u_threshold0}, diff_v_threshold={diff_v_threshold0}, "
        f"res_iter_v_threshold={res_iter_v_threshold0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for omega_v.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances convergence speed and stability.\n"
            "Step 1: Make your best **one-shot** guess for omega_v.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of omega_v, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine omega_v based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_omega_p_workflow(zero_shot: bool, mesh_x0: int, mesh_y0: int, omega_u0: float, omega_v0: float,
                          diff_u_threshold0: float, diff_v_threshold0: float, res_iter_v_threshold0: str) -> str:
    """Build the workflow for omega_p parameter optimization"""
    header = (
        "omega_p (Under-relaxation factor for pressure) determines the convergence speed and stability "
        "of the pressure correction in the SIMPLE algorithm.\n"
        "You may **only** change `omega_p`.\n"
        f"The fixed values are: mesh_x={mesh_x0}, mesh_y={mesh_y0}, omega_u={omega_u0}, omega_v={omega_v0}, "
        f"diff_u_threshold={diff_u_threshold0}, diff_v_threshold={diff_v_threshold0}, "
        f"res_iter_v_threshold={res_iter_v_threshold0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for omega_p.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances convergence speed and stability.\n"
            "Step 1: Make your best **one-shot** guess for omega_p.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of omega_p, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine omega_p based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_diff_u_threshold_workflow(zero_shot: bool, mesh_x0: int, mesh_y0: int, omega_u0: float, omega_v0: float, omega_p0: float,
                                   diff_v_threshold0: float, res_iter_v_threshold0: str) -> str:
    """Build the workflow for diff_u_threshold parameter optimization"""
    header = (
        "diff_u_threshold (Convergence threshold for u-velocity iterations) determines when the u-velocity "
        "iterative solver stops: the iteration stops when the difference between successive iterations "
        "drops below this threshold.\n"
        "You may **only** change `diff_u_threshold`.\n"
        f"The fixed values are: mesh_x={mesh_x0}, mesh_y={mesh_y0}, omega_u={omega_u0}, omega_v={omega_v0}, "
        f"omega_p={omega_p0}, diff_v_threshold={diff_v_threshold0}, "
        f"res_iter_v_threshold={res_iter_v_threshold0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for diff_u_threshold.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances solution accuracy and computational cost.\n"
            "Step 1: Make your best **one-shot** guess for diff_u_threshold.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of diff_u_threshold, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine diff_u_threshold based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_diff_v_threshold_workflow(zero_shot: bool, mesh_x0: int, mesh_y0: int, omega_u0: float, omega_v0: float, omega_p0: float,
                                   diff_u_threshold0: float, res_iter_v_threshold0: str) -> str:
    """Build the workflow for diff_v_threshold parameter optimization"""
    header = (
        "diff_v_threshold (Convergence threshold for v-velocity iterations) determines when the v-velocity "
        "iterative solver stops: the iteration stops when the difference between successive iterations "
        "drops below this threshold.\n"
        "You may **only** change `diff_v_threshold`.\n"
        f"The fixed values are: mesh_x={mesh_x0}, mesh_y={mesh_y0}, omega_u={omega_u0}, omega_v={omega_v0}, "
        f"omega_p={omega_p0}, diff_u_threshold={diff_u_threshold0}, "
        f"res_iter_v_threshold={res_iter_v_threshold0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for diff_v_threshold.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances solution accuracy and computational cost.\n"
            "Step 1: Make your best **one-shot** guess for diff_v_threshold.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of diff_v_threshold, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine diff_v_threshold based on the feedback from the simulation.\n"
            "Step 4: You have at most 10 total opportunities to refine your parameter.\n"
            "Step 5: If you think the experiment can be stopped, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_res_iter_v_threshold_workflow(zero_shot: bool, mesh_x0: int, mesh_y0: int, omega_u0: float, omega_v0: float, omega_p0: float,
                                       diff_u_threshold0: float, diff_v_threshold0: float) -> str:
    """Build the workflow for res_iter_v_threshold parameter optimization"""
    header = (
        "res_iter_v_threshold (Residual threshold for inner v-velocity iterations) determines the stopping "
        "criterion for inner v-velocity iterations in the SIMPLE algorithm.\n"
        "You may **only** change `res_iter_v_threshold`.\n"
        f"The fixed values are: mesh_x={mesh_x0}, mesh_y={mesh_y0}, omega_u={omega_u0}, omega_v={omega_v0}, "
        f"omega_p={omega_p0}, diff_u_threshold={diff_u_threshold0}, "
        f"diff_v_threshold={diff_v_threshold0}. **You must not change them!**\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose an optimal value for res_iter_v_threshold.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that balances solution accuracy and computational cost.\n"
            "Step 1: Make your best **one-shot** guess for res_iter_v_threshold.\n"
            "Step 2: Call the Convergence Test Function and check if converged.\n"
            "Step 3: Output final answer with no further tool calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial choice of res_iter_v_threshold, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: Refine res_iter_v_threshold based on the feedback from the simulation.\n"
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
            "mesh_x",
            "mesh_y",
            "omega_u",
            "omega_v", 
            "omega_p",
            "diff_u_threshold",
            "diff_v_threshold",
            "res_iter_v_threshold"
        ],
        mass_tolerance=data.get("mass_tolerance"),
        u_rmse_tolerance=data.get("u_rmse_tolerance"),
        v_rmse_tolerance=data.get("v_rmse_tolerance"),
        p_rmse_tolerance=data.get("p_rmse_tolerance")
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
            'mesh_x',
            'mesh_y', 
            'omega_u',
            'omega_v',
            'omega_p',
            'diff_u_threshold',
            'diff_v_threshold',
            'res_iter_v_threshold'
        ],
        mass_tolerance=data.get('mass_tolerance'),
        u_rmse_tolerance=data.get('u_rmse_tolerance'),
        v_rmse_tolerance=data.get('v_rmse_tolerance'),
        p_rmse_tolerance=data.get('p_rmse_tolerance')
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

class twoD_NavierStokes_DatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameter, solving the 2D steady incompressible Navier-Stokes equations using "
            "the SIMPLE (Semi-Implicit Method for Pressure Linked Equations) algorithm on a staggered finite volume grid. "
            "This serves as a model for 2D fluid flow problems with complex channel geometries. "
            "You should try to minimize the total cost incurred by function calls, but your "
            "primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem."
        )

        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."
        
        system_prompt = zero_shot_system_prompt if zero_shot else iterative_system_prompt

        return system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", choices=["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", 
                                                "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
                        help="Task to solve (if not specified, generates all tasks)")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Enable zero-shot mode (if not specified, generates both modes)")
    args = parser.parse_args()
    
    # If no specific task is provided, generate all tasks
    if args.task:
        tasks = [args.task]
    else:
        tasks = ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", 
                "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"]
    
    # If no specific mode is provided, generate both modes
    if args.zero_shot:
        modes = [True]  # Only zero-shot
    else:
        modes = [False, True]  # Both iterative and zero-shot

    print("🚀 NAVIER-STOKES 2D DATASET GENERATOR")
    print("=" * 80)
    print(f"📋 Tasks: {tasks}")
    print(f"🎯 Modes: {'zero-shot only' if len(modes) == 1 else 'iterative + zero-shot'}")
    
    total_files = 0
    
    for task in tasks:
        print(f"\n📋 TASK: {task.upper()}")
        print("-" * 50)
        
        task_dir = f"data/ns_channel_2d/{task}"
        
        # Get precision levels from the new structure
        precision_levels = []
        if os.path.exists(task_dir):
            precision_levels = [d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))]
        if not precision_levels:
            precision_levels = ["low", "medium", "high"]  # fallback

        generator = twoD_NavierStokes_DatasetGenerator(
            f"tool_documentation/ns_channel_2d/{task}.json"
        )

        for precision_level in precision_levels:
            print(f"  🎯 {precision_level.upper()} precision:")
            
            # Create output directory for this precision level
            out_dir = f"data/ns_channel_2d/human_write/{precision_level}"
            os.makedirs(out_dir, exist_ok=True)
            
            for zflag in modes:
                flag = "zero_shot" if zflag else "iterative"
                
                # Skip iterative mode for tasks other than mesh_x and mesh_y (they only have zero-shot)
                if not zflag and task not in ["mesh_x", "mesh_y"]:
                    continue
                    
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
                    aspect_ratio = q.get("aspect_ratio")
                    
                    # Build workflow based on task type
                    if task == "mesh_x":
                        mesh_y0 = fetch_param(first_params, "mesh_y")
                        omega_u0 = fetch_param(first_params, "omega_u")
                        omega_v0 = fetch_param(first_params, "omega_v")
                        omega_p0 = fetch_param(first_params, "omega_p")
                        diff_u_threshold0 = fetch_param(first_params, "diff_u_threshold")
                        diff_v_threshold0 = fetch_param(first_params, "diff_v_threshold")
                        res_iter_v_threshold0 = fetch_param(first_params, "res_iter_v_threshold")
                        wf = build_mesh_x_workflow(zflag, mesh_y0, omega_u0, omega_v0, omega_p0, 
                                                  diff_u_threshold0, diff_v_threshold0, res_iter_v_threshold0, aspect_ratio)
                    elif task == "mesh_y":
                        mesh_x0 = fetch_param(first_params, "mesh_x")
                        omega_u0 = fetch_param(first_params, "omega_u")
                        omega_v0 = fetch_param(first_params, "omega_v")
                        omega_p0 = fetch_param(first_params, "omega_p")
                        diff_u_threshold0 = fetch_param(first_params, "diff_u_threshold")
                        diff_v_threshold0 = fetch_param(first_params, "diff_v_threshold")
                        res_iter_v_threshold0 = fetch_param(first_params, "res_iter_v_threshold")
                        wf = build_mesh_y_workflow(zflag, mesh_x0, omega_u0, omega_v0, omega_p0, 
                                                  diff_u_threshold0, diff_v_threshold0, res_iter_v_threshold0, aspect_ratio)
                    elif task == "omega_u":
                        mesh_x0 = fetch_param(first_params, "mesh_x")
                        mesh_y0 = fetch_param(first_params, "mesh_y")
                        omega_v0 = fetch_param(first_params, "omega_v")
                        omega_p0 = fetch_param(first_params, "omega_p")
                        diff_u_threshold0 = fetch_param(first_params, "diff_u_threshold")
                        diff_v_threshold0 = fetch_param(first_params, "diff_v_threshold")
                        res_iter_v_threshold0 = fetch_param(first_params, "res_iter_v_threshold")
                        wf = build_omega_u_workflow(zflag, mesh_x0, mesh_y0, omega_v0, omega_p0, 
                                                   diff_u_threshold0, diff_v_threshold0, res_iter_v_threshold0)
                    elif task == "omega_v":
                        mesh_x0 = fetch_param(first_params, "mesh_x")
                        mesh_y0 = fetch_param(first_params, "mesh_y")
                        omega_u0 = fetch_param(first_params, "omega_u")
                        omega_p0 = fetch_param(first_params, "omega_p")
                        diff_u_threshold0 = fetch_param(first_params, "diff_u_threshold")
                        diff_v_threshold0 = fetch_param(first_params, "diff_v_threshold")
                        res_iter_v_threshold0 = fetch_param(first_params, "res_iter_v_threshold")
                        wf = build_omega_v_workflow(zflag, mesh_x0, mesh_y0, omega_u0, omega_p0, 
                                                   diff_u_threshold0, diff_v_threshold0, res_iter_v_threshold0)
                    elif task == "omega_p":
                        mesh_x0 = fetch_param(first_params, "mesh_x")
                        mesh_y0 = fetch_param(first_params, "mesh_y")
                        omega_u0 = fetch_param(first_params, "omega_u")
                        omega_v0 = fetch_param(first_params, "omega_v")
                        diff_u_threshold0 = fetch_param(first_params, "diff_u_threshold")
                        diff_v_threshold0 = fetch_param(first_params, "diff_v_threshold")
                        res_iter_v_threshold0 = fetch_param(first_params, "res_iter_v_threshold")
                        wf = build_omega_p_workflow(zflag, mesh_x0, mesh_y0, omega_u0, omega_v0, 
                                                   diff_u_threshold0, diff_v_threshold0, res_iter_v_threshold0)
                    elif task == "diff_u_threshold":
                        mesh_x0 = fetch_param(first_params, "mesh_x")
                        mesh_y0 = fetch_param(first_params, "mesh_y")
                        omega_u0 = fetch_param(first_params, "omega_u")
                        omega_v0 = fetch_param(first_params, "omega_v")
                        omega_p0 = fetch_param(first_params, "omega_p")
                        diff_v_threshold0 = fetch_param(first_params, "diff_v_threshold")
                        res_iter_v_threshold0 = fetch_param(first_params, "res_iter_v_threshold")
                        wf = build_diff_u_threshold_workflow(zflag, mesh_x0, mesh_y0, omega_u0, omega_v0, omega_p0, 
                                                            diff_v_threshold0, res_iter_v_threshold0)
                    elif task == "diff_v_threshold":
                        mesh_x0 = fetch_param(first_params, "mesh_x")
                        mesh_y0 = fetch_param(first_params, "mesh_y")
                        omega_u0 = fetch_param(first_params, "omega_u")
                        omega_v0 = fetch_param(first_params, "omega_v")
                        omega_p0 = fetch_param(first_params, "omega_p")
                        diff_u_threshold0 = fetch_param(first_params, "diff_u_threshold")
                        res_iter_v_threshold0 = fetch_param(first_params, "res_iter_v_threshold")
                        wf = build_diff_v_threshold_workflow(zflag, mesh_x0, mesh_y0, omega_u0, omega_v0, omega_p0, 
                                                            diff_u_threshold0, res_iter_v_threshold0)
                    elif task == "res_iter_v_threshold":
                        mesh_x0 = fetch_param(first_params, "mesh_x")
                        mesh_y0 = fetch_param(first_params, "mesh_y")
                        omega_u0 = fetch_param(first_params, "omega_u")
                        omega_v0 = fetch_param(first_params, "omega_v")
                        omega_p0 = fetch_param(first_params, "omega_p")
                        diff_u_threshold0 = fetch_param(first_params, "diff_u_threshold")
                        diff_v_threshold0 = fetch_param(first_params, "diff_v_threshold")
                        wf = build_res_iter_v_threshold_workflow(zflag, mesh_x0, mesh_y0, omega_u0, omega_v0, omega_p0, 
                                                                diff_u_threshold0, diff_v_threshold0)

                    single_ds = generator.generate_dataset(wf, [q], zflag)[0]
                    
                    # Update dataset entry to only include required fields
                    precision_config = q.get("precision_config", {})
                    filtered_ds = {
                        "QID": single_ds.get("QID", q.get("QID")),
                        "profile": q.get("profile"),
                        "zero_shot": q.get("zero_shot"),
                        "target_parameter": q.get("target_parameter"),
                        "precision_level": q.get("precision_level"),
                        "aspect_ratio": q.get("aspect_ratio"),
                        "mass_tolerance": precision_config.get("mass_tolerance"),
                        "u_rmse_tolerance": precision_config.get("u_rmse_tolerance"),
                        "v_rmse_tolerance": precision_config.get("v_rmse_tolerance"),
                        "p_rmse_tolerance": precision_config.get("p_rmse_tolerance"),
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