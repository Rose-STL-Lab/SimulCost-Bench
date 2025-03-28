import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from inference import find_json
from typing import Dict, List

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Workflow**\ndescribe the workflow step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, data: dict) -> Dict[str, Any]:
    # Your code here
    return answer,
    "workflow": "Workflow of your proposed agent"
"""
}

Tool = {
    "thought": "A tool is a function that can be called to perform a specific task. It can be used to perform complex computations or access external resources.",
    "name": "Tool Agent",
    "code": """def forward(self, data: dict):
    messages = data["messages"]
    qid = data["QID"]

    tool_manager = ToolCallManager(self.logger, qid)

    # Instruction for tool call and execution
    tool_instruction = "Given the problem, you should use the tool call to run the experiment"
    tool_agent = LLMAgentBase(["tool_reason", "tool_name", "tool_args"], "Tool Agent")
    tool_reason, tool_name, tool_args = tool_agent.query(messages, tool_instruction)
    messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args})})
    tool_result, _ = tool_agent.execute_tool(tool_reason, tool_name, tool_args, tool_manager)
    tool_df = tool_manager.get_tool_call_df()

    return tool_result, tool_df
    """,
    "workflow": """Step 1: Initialize Components
    - Create ToolCallManager with QID and logger
    - Load message history from input data
    Step 2: Query Tool Agent and Update Context
    - Request LLM proposal (tool_reason, tool_name, tool_args)
    - Append JSON tool proposal to message history
    Step 3: Execute & Monitor
    - Run tool through ToolCallManager
    Step 4: Deliver Result
    - Return formatted tool result and tool call dataframe
    """
}

Experiment = {
    "thought": "Using iteraively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution. And you should return a summary of the experiemnt when it is finished",
    "name": "Experiment Agent",
    "code": """def forward(self, data: dict):
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
        if tool_result["is_experiment_ended"] or acc_cost >= budget:
            break
    
    # Set up experiment summary agent
    summary_instruction = "Given the process of the experiment, you should use the tool call to summarize the experiment "
    tool_reason, tool_name, tool_args = experiment_agent.query(messages, summary_instruction)
    messages.append({"role": "assistant", "content": json.dumps({"tool_reason": tool_reason, "tool_name": tool_name, "tool_args": tool_args})})
    tool_result, _ = experiment_agent.execute_tool(tool_reason, tool_name, tool_args, experiment_manager)
    tool_df = experiment_manager.get_tool_call_df()
    # Return final result (summary)
    return tool_result, tool_df
    """,
    "workflow": """Step 1: Initialize Components
    - Create ToolCallManager with QID and logger
    - Load message history from input data
    Step 2: Configure Iterative Agent
    - Set dynamic experiment instruction for LLM
    Step 3: Query Action Cycle
    - Request LLM proposal (tool_reason, tool_name, tool_args)
    - Append JSON tool proposal to message history
    Step 4: Execute & Process Feedback
    - Run tool through ToolCallManager
    - Inject results back as user response
    Step 5: Evaluate Continuation
    - Check is_experiment_ended flag in results
    - If continuing -> Repeat Steps 3-4
    - If finished -> Initiate Summary
    Step 6: Deliver Result
    - Return formatted tool result and tool call dataframe
    """
}
# Configure LLM with temperature=0 for deterministic output


# Create structured prompt template
base = """You are a computational physics expert. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent preforimnig well on solving the described PDE convergence problem. Your agent should be able to generate a code and a workflow that is optimized for the given problem.

Problem Description:
Your task is to find the coarsest grid resolution that achieves convergence in a 1D heat transfer simulation, subject to a specified cost budget. You must minimize the total cost incurred by function calls, but your primary goal is to successfully meet the convergence criteria. You should always use the tool call function to finish the problem.

Available functions: 
{"type": "function", "function": {"name": "run_1d_heat_transfer_PDE_exp", "description": "Conduct a 1D heat transfer PDE simulation and evaluate its spatial/temporal convergence. This experiment will conduct both PDE solving and convergence testing in one function. The big O complexity is O(n_space * n_time), and the actual cost is 5 * n_space * n_time. Results will be stored in an HDF5 file, the experiment configuration in a JSON file, and a convergence report will provide the spatial and temporal convergence information. It will return {\"spatial_l2_error\": float, \"temporal_l2_error\": float, \"is_experiment_ended\": bool, \"accumulated_cost\": int}.", "parameters": {"type": "object", "properties": {"L": {"type": "float", "description": "Wall thickness in meters"}, "k": {"type": "float", "description": "Thermal conductivity in W/(m\u00b7K)"}, "h": {"type": "float", "description": "Heat transfer coefficient in W/(m\u00b2\u00b7K)"}, "rho": {"type": "float", "description": "Density of wall material in kg/m\u00b3"}, "cp": {"type": "float", "description": "Specific heat capacity in J/(kg\u00b7K)"}, "T_inf": {"type": "float", "description": "Ambient temperature in Celsius"}, "T_init": {"type": "float", "description": "Initial temperature in Celsius"}, "t_final": {"type": "float", "description": "Final time for simulation in seconds"}, "n_space": {"type": "integer", "description": "Number of spatial segments"}, "n_time": {"type": "integer", "description": "Number of time segments"}, "cache_file": {"type": "string", "description": "HDF5 file path (.h5) to store PDE results (e.g. tool_result/qid_{qid}.h5)"}, "config_file": {"type": "string", "description": "JSON file path (.json) to store experiment config (e.g. tool_result/qid_{qid}.json)"}, "convergence_threshold": {"type": "float", "description": "Threshold to decide if spatial/temporal convergence is satisfied"}}, "required": ["L", "k", "h", "rho", "cp", "T_inf", "T_init", "t_final", "n_space", "n_time", "cache_file", "config_file", "convergence_threshold"]}}}
{"type": "function", "function": {"name": "get_heat_transfer_exp_summary", "description": "Once converged or if the budget is exceeded, get the summary of 1D heat transfer PDE simulation. It will return {\"converged\": bool, \"times\": int, \"sequence\": [{\"n_space\": int, \"n_time\": int}, ...], \"accumulated_cost\": int}", "parameters": {"type": "object", "properties": {"converged": {"type": "boolean", "description": "Whether the simulation has converged (not dependent on budget)"}, "out_of_budget": {"type": "boolean", "description": "Whether the simulation has run out of budget"}, "times": {"type": "integer", "description": "Number of times the PDE experiment function has been called"}, "sequence": {"type": "array", "description": "Sequence of n_space and n_time you tryed (should also include the out of budget ones if any)", "items": {"type": "object", "properties": {"n_space": {"type": "integer", "description": "Number of spatial segments"}, "n_time": {"type": "integer", "description": "Number of time segments"}}, "required": ["n_space", "n_time"]}}}, "required": ["converged", "out_of_budget", "times", "sequence"]}}} 
QID: 0 
Problem: 1D transient heat conduction in a wall with:
- Wall thickness: 0.190000 m
- Left boundary: Convection (h=8.950000 W/m²·K, T_∞=17.000000C)
- Right boundary: Insulated (zero heat flux)
- Initial temperatue: 23.000000C uniformly
- Thermal conductivity: 0.930000 W/m·K
- Specific heat: 917.000000 J/kg·K
- Density: 1979.000000 kg/m³
- End time: 2455.000000 s
Convergence criteria:
1. Spatial: L2 error < 1e-4°C
2. Temporal: L2 error < 1e-4°C
(where L2 error would be average squared difference across all grid points at each time step)
(Note: Both spatial and temporal L2 errors must be less than 1e-4°C to be considered converged)
Cost budget: 160778240

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
import json
import pandas as pd
from typing import List, Dict, Any
import logging

FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the JSON format that contains the following keys: {str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"

def find_json(response: str) -> Dict[str, Any]:
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        response = response[start_idx:end_idx]
        return json.loads(response)
    except Exception as e:
        error_msg = f"Error in find_json: {str(e)}"
        return {"error": error_msg}

class LLMAgentBase():
    def __init__(self, output_field: List[str], agent_name: str):
        self.agent_name = agent_name
        self.output_field = output_field
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            seed=42,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm.bind(response_format={"type": "json_object"})
        
    def generate_output_instruction(self, instruction):
        request_keys = ",".join(self.output_field)
        return f"{instruction}\n{FORMAT_INST(request_keys)}"

    def query(self, messages: list[dict], instruction: str) -> list:
        messages[-1]["content"] += "\n" + self.generate_output_instruction(instruction)
        json_response = self.llm.invoke(messages).content.strip()
        json_dict = find_json(json_response)

        output_infos = []
        for value in json_dict.values():
            output_infos.append(value)

        return output_infos

    def execute_tool(self, tool_reason: str, tool_name: str, tool_args: Dict[str, Any], tool_manager: ToolCallManager) -> dict:
        tool_result, acc_cost = tool_manager.execute_tool_call(tool_reason, tool_name, tool_args)
        
        return tool_result, acc_cost
    
class ToolCallManager:
    def __init__(self, logger: logging.Logger, qid: int, focused_parameters: List[str] = None, budget: int = 100):
        self.logger = logger
        self.tool_call_df = pd.DataFrame()
        # Record only the focused parameters, and the other parameters will be ignored
        self.focused_parameters = focused_parameters
        self.qid = qid
        self.accumulated_cost = 0
        self.budget = budget

    def execute_tool_call(self, tool_reason: str, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        \"""
        Execute a tool call from the model's output.
        \"""
        try:
            self.logger.info(f"Received tool call: {json.dumps({'tool_reason': tool_reason, 'tool_name': tool_name, 'tool_args': tool_args})}")
            func = globals()[tool_name]
            result = func(tool_args, self.accumulated_cost)
            self.accumulated_cost = result['accumulated_cost']
            self.logger.info(f"Tool call result: {json.dumps(result)}, Accumulated cost: {self.accumulated_cost}")
            if not tool_name.endswith('summary'):
                self._record_tool_call(tool_name, tool_args, tool_reason, result)

            return result, self.accumulated_cost

        except Exception as e:
            error_msg = f"Error executing tool call (QID={self.qid}): {str(e)}"
            self.logger.error(error_msg)
            # Return a dict with an 'error' key, so the caller can detect failure
            return {"error": error_msg}, True

    def _record_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_reason: str, result: Dict[str, Any]) -> None:
        \"""
        Record tool call details in the DataFrame.
        \"""
        try:
            focus_args = {}
            if self.focused_parameters is None:
                focus_args = tool_args
            else:
                for param in self.focused_parameters:
                    if param in tool_args:
                        focus_args[param] = tool_args[param]

            row_data = {
                "QID": self.qid,
                "tool_name": tool_name,
                "tool_args": str(focus_args),
                "tool_reason": tool_reason,
            }
            for key, value in result.items():
                row_data[f"{key}"] = str(value)

            new_row = pd.DataFrame([row_data])    
            self.tool_call_df = pd.concat([self.tool_call_df, new_row], ignore_index=True)

        except Exception as e:
            self.logger.error(f"Error recording tool call: {str(e)}")

    def get_tool_call_df(self) -> pd.DataFrame:
        \"""
        Return the tool call DataFrame.
        \"""
        if self.tool_call_df.empty:
            return pd.DataFrame(columns=['qid', 'step', 'tool_name', 'parameters'])
        return self.tool_call_df

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    def forward(self, data: dict) -> Dict[str, Any]:
        \"""
        Placeholder method for processing task information.
        
        Args:
        - data (dict): A dictionary containing task information, include "messages", "budget", "QID"
        
        Returns:
        - Answer (dict): Your FINAL Answer. Return either a namedtuple Info or a string of answers.
        \"""
        pass
```

# Discovered architecture archive
Here is the archive of the discovered architectures:
[ARCHIVE]

Your GOAL is to generate a new agent that improve the existing agent to MAXIMIZE the success rate and MINIMIZE the cost (prevent out of budget).

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture. 
The third key ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.
The last key ("workflow") should contain a natural language description of your agent's forward function logic. Ensure the workflow description explicitly mirrors the code's execution sequence and control flow.
Workflow Formatting Rules:
1. Number steps sequentially as "Step X: [Title]"
2. Use hyphen bullets (-) for main actions
3. Use plus bullets (+) for conditional substeps
4. Format formulas as italicized natural language
5. Separate steps with one empty line
6. Prohibit markdown symbols
7. Use imperative verbs to start bullets

Here is an example of the output format for the next agent architecture:

[EXAMPLE]
You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. 
"""

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""
def get_init_archive():
    return [Tool, Experiment]

def get_init_prompt(current_archive):
    archive_str = ""
    for idx, archive in enumerate(current_archive):
        archive_str = archive_str + f"Workflow {idx}: {archive}\n" 
    prompt = base.replace("[ARCHIVE]", archive_str)

    return system_prompt, prompt

def query(messages: List[Dict], provider: str = "gemini", model_name: str = "gemini-1.5-pro"):
    if provider == "openai":
        llm = ChatOpenAI(
            model_name=model_name,
            seed=42,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    response = llm.invoke(messages).content.strip()
    
    return find_json(response)
