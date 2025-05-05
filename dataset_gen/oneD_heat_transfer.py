import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.base import DatasetGenerator
import json
from inference import save_result
import argparse

n_space_zero_shot_HUMAN_WORKFLOW = """
You have only one opportunity to choose a reasonable value for n_space the number of spatial segments to solve a given PDE problem.
No trial-and-error or iterative optimization is permitted.
Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.
Please strike a balance between being too conservative and too aggressive:
- If n_space is too small, the process may fail to converge.
- If it's too large, the cost may increase dramatically.
The value of cfl is 1.0; You don't need to change it.
Step 1: You must make your best one-shot guess based solely on your domain knowledge.
Step 2: Call the PDE function; store the problem configuration in the config file.
Step 3: Call the Convergence Test Function; check if the solution has converged.
Step 4: Respond using the final response format and make no further PDE calls.
"""

n_space_iterative_HUMAN_WORKFLOW = """
Choose a reasonable value for n_space the number of spatial segments to solve a given PDE problem.
The value of cfl is 1.0; You don't need to change it.
Step 1: Estimate an initial fairly coarse choice of n_space (the number of segments in length), as you will gradually refine the solution and check convergence.
Step 2: Call the PDE function; store the question config in the config file.
Step 3: Call the Convergence Test Function; check if converged.
Step 4: If not converged, refine n_space based on the trajectory of previous errors, and the distance to the convergence threshold. You should always refine resolution by multiplying, and the multiplier depends on your previous analysis.
Step 5: You have at most 10 total opportunities to refine your resolution. **After every single refinement**, you must call the Convergence Test Function to check if the solution has converged.
Step 6: If the solution converges on or before the 10th refinement, you must respond with the final response format and make no further PDE calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further PDE calls."""

cfl_zero_shot_HUMAN_WORKFLOW = """
You have only one opportunity to choose a reasonable value for cfl, the number of Courant-Friedrichs-Lewy condition which establishes a relationship between temporal and spatial discretization, to solve a given PDE problem.
No trial-and-error or iterative optimization is permitted.
Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.
Please strike a balance between being too conservative and too aggressive:
- If cfl is too large, the process may fail to converge.
- If it's too small, the cost may increase dramatically.
The value of n_space is 100; You don't need to change it.
Step 1: You must make your best one-shot guess based solely on your domain knowledge.
Step 2: Call the PDE function; store the problem configuration in the config file.
Step 3: Call the Convergence Test Function; check if the solution has converged.
Step 4: Respond using the final response format and make no further PDE calls.
"""

cfl_iterative_HUMAN_WORKFLOW = """
Choose a reasonable value for cfl, the number of Courant-Friedrichs-Lewy condition which establishes a relationship between temporal and spatial discretization, to solve a given PDE problem.
The value of n_space is 100; You don't need to change it.
Step 1: Estimate an initial fairly coarse choice of cfl, as you will gradually refine the solution and check convergence.
Step 2: Call the PDE function; store the question config in the config file.
Step 3: Call the Convergence Test Function; check if converged.
Step 4: If not converged, refine cfl based on the trajectory of previous errors, and the distance to the convergence threshold. You should always refine resolution by division, and the multiplier depends on your previous analysis.
Step 5: You have at most 10 total opportunities to refine your resolution. **After every single refinement**, you must call the Convergence Test Function to check if the solution has converged.
Step 6: If the solution converges on or before the 10th refinement, you must respond with the final response format and make no further PDE calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further PDE calls."""

zero_shot_HUMAN_CODE = """def forward(self, data: dict):
    # Extract input data
    messages = data["messages"]
    qid = data["QID"]
    dummy_cost = data["dummy_cost"]

    # Initialize experiment manager and state
    experiment_manager = ToolCallManager(self.logger, qid, dummy_cost=dummy_cost)

    # Set up experiment agent
    experiment_instruction = "Given the problem, you should use the tool call to run the experiment."
    experiment_agent = LLMAgentBase(["tool_reason", "tool_name", "tool_args"], "Experiment Agent")
    
    # Zero-Shot
    tool_reason, tool_name, tool_args = experiment_agent.query(messages, experiment_instruction)
    messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args})})
    # Execute tool and inject results from tool
    tool_result, acc_cost = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager, qid)
    messages.append({"role": "user", "content": json.dumps(tool_result)})
    
    # Set up experiment summary agent
    summary_instruction = "Given the process of the experiment, you should use the tool call to summarize the experiment "
    tool_reason, tool_name, tool_args = experiment_agent.query(messages, summary_instruction)
    messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args})})
    tool_result, _ = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager, qid)
    tool_df = experiment_manager.get_tool_call_df()
    # Return final result (summary)
    return tool_result, tool_df
"""

iterative_HUMAN_CODE = """def forward(self, data: dict):
    # Extract input data
    messages = data["messages"]
    qid = data["QID"]
    dummy_cost = data["dummy_cost"]

    # Initialize experiment manager and state
    experiment_manager = ToolCallManager(self.logger, qid, dummy_cost=dummy_cost)

    # Set up experiment agent
    experiment_instruction = "Given the problem, you should use the tool call to run the experiment."
    experiment_agent = LLMAgentBase(["tool_reason", "tool_name", "tool_args"], "Experiment Agent")
    
    # Main interaction loop
    for _ in range(10):
        # Query agent for next action and inject query
        tool_reason, tool_name, tool_args = experiment_agent.query(messages, experiment_instruction)
        messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args})})
        # Execute tool and inject results from tool
        tool_result, acc_cost = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager, qid)
        messages.append({"role": "user", "content": json.dumps(tool_result)})
        # Continue conversation if not in summary phase
        if tool_result.get("is_converged"):
            break
    
    # Set up experiment summary agent
    summary_instruction = "Given the process of the experiment, you should use the tool call to summarize the experiment "
    tool_reason, tool_name, tool_args = experiment_agent.query(messages, summary_instruction)
    messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args})})
    tool_result, _ = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager, qid)
    tool_df = experiment_manager.get_tool_call_df()
    # Return final result (summary)
    return tool_result, tool_df
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
        workflow = n_space_zero_shot_HUMAN_WORKFLOW if zero_shot else n_space_iterative_HUMAN_WORKFLOW
        human_code = zero_shot_HUMAN_CODE if zero_shot else iterative_HUMAN_CODE
    else:
        workflow = cfl_zero_shot_HUMAN_WORKFLOW if zero_shot else cfl_iterative_HUMAN_WORKFLOW
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
    generator = oneD_HeatTransferDatasetGenerator(f"tool_documentation/oneD_heat_transfer_{task}.json")
    dataset = generator.generate_dataset(workflow, questions, zero_shot)
    save_result(dataset, dataset_file)

if __name__ == "__main__":
    main()
    