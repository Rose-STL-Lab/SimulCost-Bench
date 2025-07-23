import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.base import DatasetGenerator
import json
from inference import save_result
import argparse

def build_n_space_workflow(zero_shot: bool) -> str:
    """Build the workflow for n_space task"""
    header = (
        "You need to choose a reasonable value for n_space the number of spatial segments to solve a given PDE problem.\n"
        "The value of cfl is 1.0; You don't need to change it.\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose a reasonable value for n_space.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.\n"
            "Step 1: You must make your best one-shot guess based solely on your domain knowledge.\n"
            "Step 2: Call the Convergence Test Function; check if the solution has converged.\n"
            "Step 3: Respond using the final response format and make no further function calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of n_space (the number of segments in length), as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: If not converged, refine n_space based on the trajectory of previous errors, and the distance to the convergence threshold.\n"
            "Step 4: You have at most 10 total opportunities to refine your resolution. **After every single refinement**, you must call the Convergence Test Function to check if the solution has converged.\n"
            "Step 5: If you think the experiment can be stopped before the 10th refinement, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

def build_cfl_workflow(zero_shot: bool) -> str:
    """Build the workflow for cfl task"""
    header = (
        "You need to choose a reasonable value for cfl, the Courant-Friedrichs-Lewy condition which establishes a relationship between temporal and spatial discretization, to solve a given PDE problem.\n"
        "The value of n_space is 100; You don't need to change it.\n"
    )
    if zero_shot:
        body = (
            "You have only one opportunity to choose a reasonable value for cfl.\n"
            "No trial-and-error or iterative optimization is permitted.\n"
            "Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.\n"
            "Step 1: You must make your best one-shot guess based solely on your domain knowledge.\n"
            "Step 2: Call the Convergence Test Function; check if the solution has converged.\n"
            "Step 3: Respond using the final response format and make no further function calls."
        )
    else:
        body = (
            "Step 1: Estimate an initial fairly coarse choice of cfl, as you will gradually refine the solution and check convergence.\n"
            "Step 2: Call the Convergence Test Function; check if converged.\n"
            "Step 3: If not converged, refine cfl based on the trajectory of previous errors, and the distance to the convergence threshold.\n"
            "Step 4: You have at most 10 total opportunities to refine your resolution. **After every single refinement**, you must call the Convergence Test Function to check if the solution has converged.\n"
            "Step 5: If you think the experiment can be stopped before the 10th refinement, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."
        )
    return header + body

zero_shot_HUMAN_CODE = r"""
def forward(self, data: dict):
    # Extract input data
    messages = data['messages']
    qid      = data['QID']

    # Initialize experiment manager
    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=['cfl', 'n_space']
    )

    # Set up experiment agent
    experiment_instruction = (
        'Given the problem, you should use the tool call to run the experiment.'
    )
    experiment_agent = self.get_experiment_agent()
    
    # Zero-Shot
    tool_reason, tool_name, tool_args = experiment_agent.query(
        messages, experiment_instruction
    )
    messages.append({
        'role': 'assistant',
        'content': json.dumps({
            'tool_reason': tool_reason,
            'tool_name':  tool_name,
            'tool_args':  tool_args
        })
    })

    # Execute tool and inject results
    tool_result, acc_cost = experiment_agent.execute_tool(
        tool_reason, tool_name, tool_args, experiment_manager, qid
    )
    messages.append({'role': 'user', 'content': json.dumps(tool_result)})

    # Collect true history from tool manager
    param_seq = experiment_manager.get_param_sequence()
    cost_seq  = experiment_manager.get_cost_sequence()

    summary_data = {
        'QID': qid,
        'is_converged': tool_result.get('is_converged', False),
        'times': len(param_seq),
        'param_sequence': param_seq,
        'accumulated_cost': experiment_manager.accumulated_cost,
        'cost_sequence': cost_seq,
    }

    tool_df = experiment_manager.get_tool_call_df()
    return summary_data, tool_df
"""

iterative_HUMAN_CODE = r"""
def forward(self, data: dict):
    # Extract input data
    messages = data['messages']
    qid      = data['QID']

    # Initialize experiment manager and state
    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=['cfl', 'n_space']
    )

    # Set up experiment agent
    experiment_instruction = (
        'Given the problem, you should use the tool call to run the experiment. '
        'When you think the experiment can be stopped, set should_stop to true, '
        'otherwise set it to false.'
    )
    experiment_agent = self.get_experiment_agent(['tool_reason', 'tool_name', 'tool_args', 'should_stop'])
    
    # Main interaction loop
    for attempt in range(10):
        # Query agent for next action and inject query
        tool_reason, tool_name, tool_args, should_stop = experiment_agent.query(
            messages, experiment_instruction
        )
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
        
        # Execute tool and inject results from tool
        tool_result, acc_cost = experiment_agent.execute_tool(
            tool_reason, tool_name, tool_args, experiment_manager, qid
        )
        messages.append({'role': 'user', 'content': json.dumps(tool_result)})
    
    param_seq = experiment_manager.get_param_sequence()
    cost_seq  = experiment_manager.get_cost_sequence()
    summary_data = {
        'QID': qid,
        'is_converged': tool_result['is_converged'],
        'times': len(param_seq),
        'param_sequence': param_seq,
        'accumulated_cost': experiment_manager.accumulated_cost,
        'cost_sequence': cost_seq,
    }

    tool_df = experiment_manager.get_tool_call_df()
    return summary_data, tool_df
"""

class oneD_HeatTransferDatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        zero_shot_system_prompt = (
            "Your task is to find the coarsest grid resolution that achieves convergence in a 1D heat transfer simulation, subject to a specified cost budget. You must minimize the total cost incurred by function calls, but your primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem.\n"
        )

        iterative_system_prompt = (
            "Your task is to find the coarsest grid resolution that achieves convergence in a 1D heat transfer simulation, subject to a specified cost budget. You must minimize the total cost incurred by function calls, but your primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem. And the maximum number of your function calls is 10.\n"
        )

        system_prompt = zero_shot_system_prompt if zero_shot else iterative_system_prompt

        return system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, default="n_space",
                        help="Task of problem to solve")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                    help="Enable zero-shot mode")
    args = parser.parse_args()
    task = args.task
    zero_shot = args.zero_shot
    flag = "zero_shot" if zero_shot else "iterative"
    
    # Generate the human written workflow
    question_file = f"data/1D_heat_transfer/{task}/{flag}_question.json"

    os.makedirs("data/1D_heat_transfer/human_write", exist_ok=True)
    dataset_file = f"data/1D_heat_transfer/human_write/{task}_{flag}_dataset.json"
    archive_file = f"data/1D_heat_transfer/human_write/{task}_{flag}_agent.json"

    if task == "n_space":
        workflow = build_n_space_workflow(zero_shot)
    else:
        workflow = build_cfl_workflow(zero_shot)

    human_code = zero_shot_HUMAN_CODE if zero_shot else iterative_HUMAN_CODE

    agent = {
        "workflow": workflow,
        "code": human_code
    }
    with open(archive_file, 'w') as f:
        json.dump([agent], f, indent=4)
    with open(question_file, "r") as f:
        questions = json.load(f)
    # Generate the dataset
    generator = oneD_HeatTransferDatasetGenerator(f"tool_documentation/oneD_heat_transfer/{task}.json")
    dataset = generator.generate_dataset(workflow, questions, zero_shot)
    save_result(dataset, dataset_file)

if __name__ == "__main__":
    main()
    