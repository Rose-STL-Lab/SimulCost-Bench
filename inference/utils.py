import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import json
# from api_call import *
from tool_call import *
from typing import List, Dict, Any, Tuple
import pandas as pd

import numpy as np
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def setup_logging(filename: str = None) -> logging.Logger:
    """Setup logging configuration with filtered handlers"""
    # Clear previous log file's contents
    if filename:
        open(filename, 'w').close()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(message)s')

    # Create filters
    class ConsoleFilter(logging.Filter):
        def filter(self, record):
            return not getattr(record, 'no_console', False)

    class FileFilter(logging.Filter):
        def filter(self, record):            
            return not getattr(record, 'no_file', False)

    # Console handler with filter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ConsoleFilter())
    logger.addHandler(console_handler)

    # File handler with filter
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(FileFilter())
        logger.addHandler(file_handler)

    return logger

# save dataset in json format
def save_result(dataset: List[Dict], filename: str):
    dir = filename[:filename.rindex('/')]
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(filename, "w") as f:
        json.dump(dataset, f, indent=4, cls=NumpyEncoder)

def find_json(response: str) -> Dict[str, Any]:
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        response = response[start_idx:end_idx]
        return json.loads(response)
    except Exception as e:
        error_msg = f"Error in find_json: {str(e)}"
        return {"error": error_msg}

class ToolCallManager:
    def __init__(self, logger: logging.Logger, qid: int, focused_parameters: List[str] = None, dummy_cost: int = 100):
        self.logger = logger
        self.tool_call_df = pd.DataFrame()
        # Record only the focused parameters, and the other parameters will be ignored
        self.focused_parameters = focused_parameters
        self.qid = qid
        self.accumulated_cost = 0
        self.dummy_cost = dummy_cost

    def execute_tool_call(self, tool_reason: str, tool_name: str, tool_args: Dict[str, Any], qid: int) -> Tuple[Dict[str, Any], int]:
        """Execute a tool call from the model's output."""
        try:
            self.logger.info(
               "Received tool call: " +
                json.dumps(
                    {"tool_reason": tool_reason,
                     "tool_name": tool_name,
                     "tool_args": tool_args},
                    cls=NumpyEncoder
                )
            )

            func = globals()[tool_name]
            profile = f"p{qid}"
            # print("tool_args: ", tool_args)
            # print("accumulated_cost: ", self.accumulated_cost)
            # print("profile: ", profile)

            if tool_name == "get_heat_transfer_exp_summary":
                result = func(
                    **tool_args,
                    qid=qid,
                    accumulated_cost=self.accumulated_cost
                )
            elif tool_name in ["check_converge_cfl", "check_converge_n_space"]:
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    current_n_space=tool_args["n_space"],
                    current_cfl=tool_args["cfl"],
                    tolerance=1e-4
                )
            elif tool_name == "get_twoD_heat_transfer_exp_summary":
                result = func(
                    **tool_args,
                    qid=qid,
                    accumulated_cost=self.accumulated_cost
                )
            elif tool_name in ["check_converge_dx", "check_converge_relax", "check_converge_t_init", "check_converge_error_threshold"]:
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    current_dx=tool_args["current_dx"],
                    current_relax=tool_args["current_relax"],
                    current_t_init=tool_args["current_t_init"],
                    current_error_threshold=tool_args["current_error_threshold"],
                    tolerance=1e-3
                )

            self.accumulated_cost = result['accumulated_cost']
            self.logger.info(f"Tool call result: {json.dumps(result, cls=NumpyEncoder)}")
            if not tool_name.endswith('summary'):
                self._record_tool_call(tool_name, tool_args, tool_reason, result)

            return result, self.accumulated_cost

        except Exception as e:
            error_msg = f"Error executing tool call (QID={self.qid}): {str(e)}"
            self.logger.error(error_msg)
            # Return a dict with an 'error' key, so the caller can detect failure
            return {"error": error_msg}, self.accumulated_cost

    def _record_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_reason: str, result: Dict[str, Any]) -> None:
        """Record tool call details in the DataFrame."""
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
        """Return the tool call DataFrame."""
        if self.tool_call_df.empty:
            return pd.DataFrame(columns=['qid', 'step', 'tool_name', 'parameters'])
        return self.tool_call_df

def add_example_data(dataset: List[Dict], example_data: List[Dict]):
    example = example_data[1]["messages"]
    for data in dataset:
        data["messages"][0]["content"] += "\n" + example

    return dataset