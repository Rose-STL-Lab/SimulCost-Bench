import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.base import DatasetGenerator
import json
from inference import save_result

HUMAN_WORKFLOW = """
Step 1: Estimate an initial fairly coarse choice of n_space (the number of segments in length) and n_time (the number of segments in time), as you will gradually refine the solution and check convergence.
Step 2: Call the PDE function → store the question config in the config file.
Step 3: Call the Convergence Test Function → check if converged.
Step 4: If not converged, refine n_space and/or n_time based on the trajectory of previous errors, and the distance to the convergence threshold. You should always refine resolution by multiplying, and the multiplier depends on your previous analysis.
Step 5: You have at most 10 total opportunities to refine your resolution. **After every single refinement**, you must call the Convergence Test Function to check if the solution has converged.
Step 6: If the solution converges on or before the 10th refinement, you must respond with the final response format and make no further PDE calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further PDE calls."""

HUMAN_CODE = """def forward(self, data: dict):
    # Extract input data
    messages = data["messages"]
    qid = data["QID"]
    budget = data["budget"]

    # Initialize experiment manager and state
    experiment_manager = ToolCallManager(self.logger, qid, budget=budget)

    # Set up experiment agent
    experiment_instruction = "Given the problem, you should use the tool call to run the experiment."
    experiment_agent = LLMAgentBase(["tool_reason", "tool_name", "tool_args"], "Experiment Agent")
    
    # Main interaction loop
    while True:
        # Query agent for next action and inject query
        tool_reason, tool_name, tool_args = experiment_agent.query(messages, experiment_instruction)
        messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args})})
        # Execute tool and inject results from tool
        tool_result, acc_cost = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager)
        messages.append({"role": "user", "content": json.dumps(tool_result)})
        # Continue conversation if not in summary phase
        if acc_cost >= budget or tool_result.get("is_experiment_ended", True):
            break
    
    # Set up experiment summary agent
    summary_instruction = "Given the process of the experiment, you should use the tool call to summarize the experiment "
    tool_reason, tool_name, tool_args = experiment_agent.query(messages, summary_instruction)
    messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args})})
    tool_result, _ = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager)
    tool_df = experiment_manager.get_tool_call_df()
    # Return final result (summary)
    return tool_result, tool_df
"""

class HeatTransferDatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        system_prompt = (
            "Your task is to find the coarsest grid resolution that achieves convergence in a 1D heat transfer simulation, subject to a specified cost budget. You must minimize the total cost incurred by function calls, but your primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem. And the maximum number of your function calls is 15.\n"
        )
        return system_prompt

def main():
    # Generate the human written workflow
    question_file = f"data/heat_transfer/question.json"
    os.makedirs("data/heat_transfer/human_write", exist_ok=True)
    dataset_file = f"data/heat_transfer/human_write/dataset.json"
    archive_file = f"data/heat_transfer/human_write/agent.json"
    agent = {
        "workflow": HUMAN_WORKFLOW,
        "code": HUMAN_CODE
    }
    with open(archive_file, 'w') as f:
        json.dump([agent], f, indent=4)
    with open(question_file, "r") as f:
        questions = json.load(f)
    # Generate the dataset
    generator = HeatTransferDatasetGenerator("tool_documentation/heat_transfer.json")
    dataset = generator.generate_dataset(HUMAN_WORKFLOW, questions)
    save_result(dataset, dataset_file)

if __name__ == "__main__":
    main()
    