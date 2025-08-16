import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import json
from tool_call import *
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from logging import LoggerAdapter
from utils.param_compatibility import fetch_param
import pdb

TOOL_NAME_KEYS = {
    "heat_1d_check_converge_cfl": ["n_space", "cfl"],
    "heat_1d_check_converge_n_space": ["n_space", "cfl"],
    "heat_2d_check_converge_dx": ["current_dx", "current_relax", "current_t_init", "current_error_threshold"],
    "heat_2d_check_converge_relax": ["current_dx", "current_relax", "current_t_init", "current_error_threshold"],
    "heat_2d_check_converge_t_init": ["current_dx", "current_relax", "current_t_init", "current_error_threshold"],
    "heat_2d_check_converge_error_threshold": ["current_dx", "current_relax", "current_t_init", "current_error_threshold"],
    "burgers_1d_solve": ["current_cfl", "k", "w"],
    "euler_1d_check_converge_cfl": ["cfl", "beta", "k", "n_space"],
    "euler_1d_check_converge_beta": ["cfl", "beta", "k", "n_space"],
    "euler_1d_check_converge_k": ["cfl", "beta", "k", "n_space"],
    "euler_1d_check_converge_n_space": ["cfl", "beta", "k", "n_space"],
    "ns_2d_check_converge_mesh_x": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
    "ns_2d_check_converge_mesh_y": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
    "ns_2d_check_converge_omega_u": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
    "ns_2d_check_converge_omega_v": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
    "ns_2d_check_converge_omega_p": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
    "ns_2d_check_converge_diff_u_threshold": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
    "ns_2d_check_converge_diff_v_threshold": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
    "ns_2d_check_converge_res_iter_v_threshold": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"],
    "ns_2d_check_converge_parameter": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"]
}

def extract_parameters_regex(response_text: str, required_keys: List[str]) -> Dict[str, Any]:
    """
    Extract parameters using rule-based regex methods from response text.
    
    Args:
        response_text: The raw response text from the model
        required_keys: List of parameter keys to extract
    
    Returns:
        Dictionary with extracted parameters, or empty dict if extraction fails
    """
    import re
    
    params = {}
    
    # Try to extract parameters from the response text
    for key in required_keys:
        # Look for the key in various formats within the response
        key_patterns = [
            # Match "key": "value" or "key": value 
            rf'["\']?{key}["\']?\s*:\s*["\']?([0-9.]+)["\']?',
            # Match key: "value" or key: value
            rf'{key}\s*:\s*["\']?([0-9.]+)["\']?',
            # Match key = value
            rf'{key}\s*=\s*["\']?([0-9.]+)["\']?',
            # More flexible: key followed by any non-letter chars then number
            rf'{key}[^a-zA-Z]*?([0-9.]+)'
        ]
        
        for key_pattern in key_patterns:
            key_matches = re.findall(key_pattern, response_text, re.IGNORECASE)
            if key_matches:
                try:
                    # Use the first match found
                    value = key_matches[0]
                    if key == "n_space":
                        params[key] = int(float(value))
                    else:
                        params[key] = float(value)
                    break
                except ValueError:
                    continue
    
    return params


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def setup_logging(filename: str = None, resume: bool = False) -> logging.Logger:
    """Setup logging configuration with filtered handlers"""
    # Clear previous log file's contents only if not resuming
    if filename and not resume:
        open(filename, 'w').close()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(qid_prefix)s%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    class QIDFilter(logging.Filter):
        """Automatically add qid_prefix format for LogRecord without qid field"""
        def filter(self, record):
            if not hasattr(record, 'qid') or record.qid == '-':
                record.qid_prefix = ''
            else:
                record.qid_prefix = f'QID={record.qid} - '
            return True

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
    console_handler.addFilter(QIDFilter())
    logger.addHandler(console_handler)

    # File handler with filter
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(FileFilter())
        file_handler.addFilter(QIDFilter())
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

def find_json_robust(response: str):
    import json
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start_idx:end_idx]
        obj = json.loads(json_str)
        return obj

    except Exception as e:
        # Fallback: try the original find_json function
        try:
            return find_json(response)
        except Exception as e2:
            error_msg = f"Error in find_json_robust: {str(e)}; Fallback error: {str(e2)}"
            return {"error": error_msg}
            

class ToolCallManager:
    def __init__(self, base_logger: logging.Logger, qid: int,
                 focused_parameters: List[str] = None, tolerance_rmse: float = None):
        self.logger = LoggerAdapter(base_logger, {'qid': qid})
        self.tool_call_df = pd.DataFrame()
        # Record only the focused parameters, and the other parameters will be ignored
        self.focused_parameters = focused_parameters
        self.qid = qid
        self.param_sequence = []
        self.cost_sequence  = [] 
        self.accumulated_cost = 0
        self.tolerance_rmse = tolerance_rmse

    def execute_tool_call(self, tool_reason: str, tool_name: str, tool_args: Dict[str, Any], profile: int) -> Tuple[Dict[str, Any], int]:
        """Execute a tool call from the model's output."""
        try:
            self.logger.info(
               "🛠 Received tool call: " +
                json.dumps(
                    {"tool_reason": tool_reason,
                     "tool_name": tool_name,
                     "tool_args": tool_args},
                    cls=NumpyEncoder,
                    indent=2
                )
            )

            if tool_args and not isinstance(tool_args, dict):
                tool_args = extract_parameters_regex(tool_args, TOOL_NAME_KEYS[tool_name])

            # Check for validation issues and provide detailed error messages
            validation_errors = []
            tool_reason_missing = False
            
            # Check tool_reason but don't add to validation_errors (allow missing)
            if not tool_reason or (isinstance(tool_reason, str) and tool_reason.strip() == ""):
                tool_reason_missing = True
                self.logger.warning("tool_reason is empty or missing, but proceeding with simulation as tool_name and tool_args are available")
                
            # Check critical fields that must be present
            if not tool_name or not tool_name.strip():
                validation_errors.append("tool_name is empty or missing")
            
            if not tool_args or not isinstance(tool_args, dict):
                validation_errors.append("tool_args is empty, missing, or not a dictionary")
            
            # Only raise error if critical fields are missing
            if validation_errors:
                error_msg = f"Tool call validation failed: {'; '.join(validation_errors)}. This likely indicates the model's response was missing required fields during JSON parsing/validation."
                raise ValueError(error_msg)
            
            # Provide a default value for tool_reason if missing to ensure clean logging
            if tool_reason_missing:
                tool_reason = "[tool_reason was missing from model response, proceeding with simulation]"

            func = globals()[tool_name]
            if "heat_2d" in tool_name:
                profile = f"p{profile}"
            else:
                profile = f"{profile}"

            if tool_name in ["heat_1d_check_converge_cfl", "heat_1d_check_converge_n_space"]:
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    n_space=fetch_param(tool_args, "n_space", "current_n_space"),
                    cfl=fetch_param(tool_args, "cfl", "current_cfl"),
                    tolerance=1e-4
                )
            elif tool_name in ["heat_2d_check_converge_dx", "heat_2d_check_converge_relax", "heat_2d_check_converge_t_init", "heat_2d_check_converge_error_threshold"]:
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    current_dx=fetch_param(tool_args, "current_dx", "dx"),
                    current_relax=fetch_param(tool_args, "current_relax", "relax"),
                    current_t_init=fetch_param(tool_args, "current_t_init", "t_init", "T_init"),
                    current_error_threshold=fetch_param(tool_args, "current_error_threshold", "error_threshold"),
                    tolerance=1e-3
                )
            elif tool_name in ["burgers_1d_solve"]:
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    current_cfl=fetch_param(tool_args, "current_cfl", "cfl"),
                    k=fetch_param(tool_args, "k"),
                    w=fetch_param(tool_args, "w"),
                    linf_tolerance=5e-2,
                    rmse_tolerance=5e-3,
                )
            elif tool_name in ["euler_1d_check_converge_cfl", "euler_1d_check_converge_beta", "euler_1d_check_converge_k", "euler_1d_check_converge_n_space"]:
                # Use tolerance_rmse from dataset - required field
                if self.tolerance_rmse is None:
                    raise ValueError(f"tolerance_rmse is required for euler_1d tools but was not provided in dataset (QID={self.qid})")
                tolerance = self.tolerance_rmse
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    cfl=fetch_param(tool_args, "cfl"),
                    beta=fetch_param(tool_args, "beta"),
                    k=fetch_param(tool_args, "k"),
                    n_space=fetch_param(tool_args, "n_space"),
                    rmse_tolerance=tolerance
                )
            elif tool_name in [
                "ns_2d_check_converge_mesh_x", "ns_2d_check_converge_mesh_y", "ns_2d_check_converge_omega_u", 
                "ns_2d_check_converge_omega_v", "ns_2d_check_converge_omega_p", "ns_2d_check_converge_diff_u_threshold", 
                "ns_2d_check_converge_diff_v_threshold", "ns_2d_check_converge_res_iter_v_threshold", "ns_2d_check_converge_parameter"
            ]:
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    mesh_x=fetch_param(tool_args, "mesh_x"),
                    mesh_y=fetch_param(tool_args, "mesh_y"),
                    omega_u=fetch_param(tool_args, "omega_u"),
                    omega_v=fetch_param(tool_args, "omega_v"),
                    omega_p=fetch_param(tool_args, "omega_p"),
                    diff_u_threshold=fetch_param(tool_args, "diff_u_threshold"),
                    diff_v_threshold=fetch_param(tool_args, "diff_v_threshold"),
                    res_iter_v_threshold=fetch_param(tool_args, "res_iter_v_threshold"),
                    mass_tolerance=1e-4,
                    u_rmse_tolerance=3e-2,
                    v_rmse_tolerance=3e-2,
                    p_rmse_tolerance=3e-2,
                )

            prev_cost = self.accumulated_cost
            self.accumulated_cost = result['accumulated_cost']

            step_cost = self.accumulated_cost - prev_cost
            self.cost_sequence.append(step_cost)

            self.logger.info(f"✅ Tool call result: {json.dumps(result, cls=NumpyEncoder, indent=2)}")

            # if not tool_name.endswith('summary'):
            self._record_tool_call(tool_name, tool_args, tool_reason, result)

            return result, self.accumulated_cost

        except Exception as e:
            error_msg = f"Error executing tool call (QID={self.qid}): {str(e)}"
            self.logger.error(error_msg)
            # For failed tool calls, still record an empty attempt to avoid param_sequence being empty
            # This ensures param_sequence is never completely empty, preventing IndexError in evaluation
            if not self.param_sequence:  # Only add if param_sequence is completely empty
                self.param_sequence.append({})
                self.cost_sequence.append(0)
            # Return a dict with all required fields to prevent KeyError in downstream processing
            return {
                "error": error_msg,
                "is_converged": False,
                "accumulated_cost": self.accumulated_cost
            }, self.accumulated_cost

    def _record_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_reason: str, result: Dict[str, Any]) -> None:
        """Record tool call details in the DataFrame."""
        try:
            focus_args = {}
            focus_args = tool_args

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

            self.param_sequence.append(focus_args)

        except Exception as e:
            self.logger.error(f"Error recording tool call: {str(e)}")
    
    def get_tool_call_df(self) -> pd.DataFrame:
        """Return the tool call DataFrame."""
        if self.tool_call_df.empty:
            return pd.DataFrame(columns=['qid', 'step', 'tool_name', 'parameters'])
        return self.tool_call_df
    
    def get_param_sequence(self) -> List[Dict[str, Any]]:
        return self.param_sequence
    
    def get_cost_sequence(self) -> List[int]:
        return self.cost_sequence

def add_example_data(dataset: List[Dict], example_data: List[Dict]):
    example = example_data[1]["messages"]
    for data in dataset:
        data["messages"][0]["content"] += "\n" + example

    return dataset