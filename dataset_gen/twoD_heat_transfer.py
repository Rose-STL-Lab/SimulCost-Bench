import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_gen.base import DatasetGenerator
import json
from inference import save_result
import argparse

dx_zero_shot_HUMAN_WORKFLOW = """
The grid resolution determines the spatial discretization accuracy. 
A finer grid (smaller dx) provides more accurate solutions but increases computational cost.
The convergence metric is the temperature distribution at the middle (vertical) line.
You have only one opportunity to choose an optimal value for dx, to find an optimal grid resolution.
No trial-and-error or iterative optimization is permitted.
Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.
Please strike a balance between being too conservative and too aggressive:
- If dx is too large, the process may fail to converge.
- If it's too small, the cost may increase dramatically.
The value of relax is 1.0, T_init is 0.25, error_threshold is 1e-7. You must not change them!
THE ONLY CHANGABLE PARAMETER IS dx!
Step 1: You must make your best one-shot guess based solely on your domain knowledge.
Step 2: Call the Convergence Test Function; check if the solution has converged.
Step 3: Respond using the final response format and make no further function calls.
"""

dx_iterative_HUMAN_WORKFLOW = """
The grid resolution determines the spatial discretization accuracy. 
A finer grid (smaller dx) provides more accurate solutions but increases computational cost.
The convergence metric is the temperature distribution at the middle (vertical) line.
Choose a reasonable value for dx, to find an optimal grid resolution.
The value of relax is 1.0, T_init is 0.25, error_threshold is 1e-7. You must not change them!
THE ONLY CHANGABLE PARAMETER IS dx!
Step 1: Estimate an initial fairly coarse choice of dx, as you will gradually refine the solution and check convergence.
Step 2: Call the Convergence Test Function; check if converged.
Step 3: If not converged, refine dx based on the trajectory of previous errors, and the distance to the convergence threshold.
Step 4: You have at most 10 total opportunities to refine your resolution. **After every single refinement**, you must call the Convergence Test Function to check if the solution has converged.
Step 5: If you think the experiment can be stopped before the 10th refinement, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."""

error_threshold_zero_shot_HUMAN_WORKFLOW = """
The error threshold determines when to stop the Jacobi iteration process. 
The convergence metric is the temperature distribution at the middle (vertical) line.
You have only one opportunity to choose an optimal value for the error_threshold.
No trial-and-error or iterative optimization is permitted.
Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.
Please strike a balance between being too conservative and too aggressive:
- If the error_threshold is too large, the process may fail to converge.
- If it's too small, the cost may increase dramatically.
The value of dx is 0.005, relax is 1.0, T_init is 0.25. You must not change them!
THE ONLY CHANGABLE PARAMETER IS error_threshold!
Step 1: You must make your best one-shot guess based solely on your domain knowledge.
Step 2: Call the Convergence Test Function; check if the solution has converged.
Step 3: Respond using the final response format and make no further function calls.
"""

error_threshold_iterative_HUMAN_WORKFLOW = """
The error threshold determines when to stop the Jacobi iteration process. 
The convergence metric is the temperature distribution at the middle (vertical) line.
Choose a reasonable value for error_threshold which is likely to converge, while also keeping the cost from becoming too high.
The value of dx is 0.005, relax is 1.0, T_init is 0.25. You must not change them!
THE ONLY CHANGABLE PARAMETER IS error_threshold!
Step 1: Estimate an initial fairly coarse choice of error_threshold, as you will gradually refine the solution and check convergence.
Step 2: Call the Convergence Test Function; check if converged.
Step 3: If not converged, refine error_threshold based on the trajectory of previous errors, and the distance to the convergence threshold.
Step 4: You have at most 10 total opportunities to refine your resolution. **After every single refinement**, you must call the Convergence Test Function to check if the solution has converged.
Step 5: If you think the experiment can be stopped before the 10th refinement, you must respond with the final response format and make no further function calls. If you reach the 10th refinement, you **must** still perform a convergence check immediately after that refinement; then, regardless of whether it is converged or not, respond with the final response format and make no further function calls."""

relax_zero_shot_HUMAN_WORKFLOW = """
The relaxation factor affects convergence speed of the SOR method. Optimal values typically lie between 0.05 and 1.95.
Note for relax ratio (SOR), a bad choice may lead to NAN/INFITY solution or unable to converge after max number of iterations.
You have only one opportunity to choose an optimal value for relax.
No trial-and-error or iterative optimization is permitted.
Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.
Please strike a balance between being too conservative and too aggressive:
- If relax is too large, the process may fail to converge.
- If it's too small, the cost may increase dramatically.
The value of dx is 0.005, T_init is 0.25, error_threshold is 1e-7. You must not change them!
THE ONLY CHANGABLE PARAMETER IS relax!
Step 1: You must make your best one-shot guess based solely on your domain knowledge.
Step 2: Call the Convergence Test Function; check if the solution has converged.
Step 3: Respond using the final response format and make no further function calls.
"""

t_init_zero_shot_HUMAN_WORKFLOW = """
The initial temperature field can affect convergence speed.
Note for t_init, a bad choice may lead to NAN/INFITY solution or unable to converge after max number of iterations.
You have only one opportunity to choose an optimal value for t_init.
No trial-and-error or iterative optimization is permitted.
Your goal is to select a value that is likely to converge, while also keeping the cost from becoming too high.
The value of dx is 0.005, relax is 1.0, error_threshold is 1e-7. You must not change them!
THE ONLY CHANGABLE PARAMETER IS t_init!
Step 1: You must make your best one-shot guess based solely on your domain knowledge.
Step 2: Call the Convergence Test Function; check if the solution has converged.
Step 3: Respond using the final response format and make no further function calls.
"""

zero_shot_HUMAN_CODE = """def forward(self, data: dict):
    # Extract input data
    messages = data["messages"]
    qid = data["QID"]

    # Initialize experiment manager
    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=[
            "current_dx",
            "current_relax",
            "current_t_init",
            "current_error_threshold"
        ]
    )

    # Set up experiment agent
    experiment_instruction = "Given the problem, you should use the tool call to run the experiment."
    experiment_agent = LLMAgentBase(["tool_reason", "tool_name", "tool_args"], "Experiment Agent")
    
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
    tool_result, acc_cost = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager, qid)
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

iterative_HUMAN_CODE = """def forward(self, data: dict):
    # Extract input data
    messages = data["messages"]
    qid = data["QID"]

    # Initialize experiment manager and state
    experiment_manager = ToolCallManager(
        self.logger,
        qid,
        focused_parameters=[
            "current_dx",
            "current_relax",
            "current_t_init",
            "current_error_threshold"
        ]
    )

    # Set up experiment agent
    experiment_instruction = "Given the problem, you should use the tool call to run the experiment. When you think the experiment can be stopped, set should_stop to true, otherwise set it to false."
    experiment_agent = LLMAgentBase(["tool_reason", "tool_name", "tool_args", "should_stop"], "Experiment Agent")
    
    # Main interaction loop
    for attempt in range(10):
        # Query agent for next action and inject query
        tool_reason, tool_name, tool_args, should_stop = experiment_agent.query(messages, experiment_instruction)
        messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args, "should_stop": should_stop})})
        
        if attempt == 0: self.logger.info(f"========== 🧐 The model begins to solve a new problem ==========")
        self.logger.info(f"========== The {attempt + 1} attempt of the model ==========")
        self.logger.info(f"should_stop: {should_stop}")
        if isinstance(should_stop, bool):
            stop_flag = should_stop
        else:
            stop_flag = str(should_stop).lower() == "true"

        if stop_flag:
            self.logger.info("========== 🎯 The model stops the experiment! ==========")
            break
        
        # Execute tool and inject results from tool
        tool_result, acc_cost = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager, qid)
        messages.append({"role": "user", "content": json.dumps(tool_result)})
    
    param_seq = experiment_manager.get_param_sequence()
    cost_seq = experiment_manager.get_cost_sequence()
    summary_data = {
        "QID": qid,
        "is_converged": tool_result["is_converged"],
        "times": len(param_seq),
        "param_sequence": param_seq,
        "accumulated_cost": experiment_manager.accumulated_cost,
        "cost_sequence": cost_seq
    }

    tool_df = experiment_manager.get_tool_call_df()
    return summary_data, tool_df
"""

class twoD_HeatTransferDatasetGenerator(DatasetGenerator):
    def __init__(self, doc_file: str):
        super().__init__(doc_file)

    def get_instruction_template(self, zero_shot) -> str:
        """Get the standardized instruction template using loaded JSON file"""
        zero_shot_system_prompt = (
            "Your task is to find the optimal parameter, solving 2D steady-state heat transfer problems using the Jacobi iteration method with Successive Over-Relaxation (SOR). The simulation models a square plate with fixed boundary conditions: the top boundary is held at temperature 1.0, while all other boundaries are held at temperature 0.0. You should try to minimize the total cost incurred by function calls, but your primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem."
        )

        iterative_system_prompt = zero_shot_system_prompt + "\nAnd the maximum number of your function calls is 10."
        

        system_prompt = zero_shot_system_prompt if zero_shot else iterative_system_prompt

        return system_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, choices=["dx", "relax", "t_init", "error_threshold"],
                    default="dx", help="Task of problem to solve")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                    help="Enable zero-shot mode")
    args = parser.parse_args()
    task = args.task

    # Force zero_shot to True for specific tasks
    zero_shot = args.zero_shot or task in ["relax", "t_init"]
    flag = "zero_shot" if zero_shot else "iterative"
    
    # Generate the human written workflow
    question_file = f"data/2D_heat_transfer/{task}/{flag}_question.json"

    os.makedirs("data/2D_heat_transfer/human_write", exist_ok=True)
    dataset_file = f"data/2D_heat_transfer/human_write/{task}_{flag}_dataset.json"
    archive_file = f"data/2D_heat_transfer/human_write/{task}_{flag}_agent.json"

    if task == "dx":
        workflow =  dx_zero_shot_HUMAN_WORKFLOW if zero_shot else dx_iterative_HUMAN_WORKFLOW
    elif task == "relax":
        workflow = relax_zero_shot_HUMAN_WORKFLOW
    elif task == "t_init":
        workflow = t_init_zero_shot_HUMAN_WORKFLOW
    elif task == "error_threshold":
        workflow = error_threshold_zero_shot_HUMAN_WORKFLOW if zero_shot else error_threshold_iterative_HUMAN_WORKFLOW

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
    generator = twoD_HeatTransferDatasetGenerator(f"tool_documentation/twoD_heat_transfer/{task}.json")
    dataset = generator.generate_dataset(workflow, questions, zero_shot)
    save_result(dataset, dataset_file)

if __name__ == "__main__":
    main()
    