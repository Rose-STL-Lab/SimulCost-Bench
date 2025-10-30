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
    "heat_2d_check_converge_dx": ["dx", "relax", "t_init", "error_threshold"],
    "heat_2d_check_converge_relax": ["dx", "relax", "t_init", "error_threshold"],
    "heat_2d_check_converge_t_init": ["dx", "relax", "t_init", "error_threshold"],
    "heat_2d_check_converge_error_threshold": ["dx", "relax", "t_init", "error_threshold"],
    "burgers_1d_check_converge_cfl": ["cfl", "beta", "k", "n_space"],
    "burgers_1d_check_converge_beta": ["cfl", "beta", "k", "n_space"],
    "burgers_1d_check_converge_k": ["cfl", "beta", "k", "n_space"],
    "burgers_1d_check_converge_n_space": ["cfl", "beta", "k", "n_space"],
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
    "ns_transient_2d_check_converge_resolution": ["resolution", "cfl", "relaxation_factor", "residual_threshold"],
    "ns_transient_2d_check_converge_cfl": ["resolution", "cfl", "relaxation_factor", "residual_threshold"],
    "ns_transient_2d_check_converge_relaxation_factor": ["resolution", "cfl", "relaxation_factor", "residual_threshold"],
    "ns_transient_2d_check_converge_residual_threshold": ["resolution", "cfl", "relaxation_factor", "residual_threshold"],
    "epoch_1d_check_converge_dt_multiplier": ["nx", "dt_multiplier", "npart", "field_order", "particle_order"],
    "epoch_1d_check_converge_nx": ["nx", "dt_multiplier", "npart", "field_order", "particle_order"],
    "epoch_1d_check_converge_npart": ["nx", "dt_multiplier", "npart", "field_order", "particle_order"],
    "epoch_1d_check_converge_field_order": ["nx", "dt_multiplier", "npart", "field_order", "particle_order"],
    "epoch_1d_check_converge_particle_order": ["nx", "dt_multiplier", "npart", "field_order", "particle_order"],
    "mpm_2d_check_converge_nx": ["nx", "npart", "cfl"],
    "mpm_2d_check_converge_npart": ["nx", "npart", "cfl"],
    "mpm_2d_check_converge_cfl": ["nx", "npart", "cfl"],
    "diff_react_1d_check_converge_cfl": ["n_space", "cfl", "tol"],
    "diff_react_1d_check_converge_n_space": ["n_space", "cfl", "tol"],
    "diff_react_1d_check_converge_tol": ["n_space", "cfl", "tol"],
    "euler_2d_check_converge_cfl": ["n_grid_x", "cfl", "cg_tolerance"],
    "euler_2d_check_converge_n_grid_x": ["n_grid_x", "cfl", "cg_tolerance"],
    "euler_2d_check_converge_cg_tolerance": ["n_grid_x", "cfl", "cg_tolerance"]
}



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


def find_json_robust(response: str):
    """
    Robust JSON extraction with multiple fallback strategies.
    Prioritizes extracting the first complete JSON object.
    """
    import json
    
    # Input validation
    if not response or not isinstance(response, str):
        return {"error": "Invalid input: response is empty or not a string"}
    
    # Strategy 1: Extract first complete JSON object with proper brace matching
    try:
        brace_count = 0
        start_idx = -1
        
        for i, char in enumerate(response):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    try:
                        json_str = response[start_idx:i+1]
                        obj = json.loads(json_str)
                        # Validate that it contains expected fields
                        if isinstance(obj, dict) and any(key in obj for key in ['tool_name', 'tool_args', 'tool_reason']):
                            return obj
                    except json.JSONDecodeError:
                        # This JSON object is malformed, continue searching
                        pass
                    # Reset for next potential JSON object
                    start_idx = -1
    except Exception:
        pass
    
    # Strategy 2: Simple boundary matching (first { to last })
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            obj = json.loads(json_str)
            return obj
    except Exception:
        pass
    
    # Strategy 3: Try to extract JSON from each line separately
    try:
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and any(key in obj for key in ['tool_name', 'tool_args', 'tool_reason']):
                        return obj
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    # Final fallback: return error with detailed message
    error_msg = f"All JSON extraction strategies failed. Response preview: {response[:200]}..."
    return {"error": error_msg}
            

class ToolCallManager:
    def __init__(self, base_logger: logging.Logger, qid: int,
                 focused_parameters: List[str] = None, tolerance_rmse: float = None,
                 mass_tolerance: float = None, u_rmse_tolerance: float = None,
                 v_rmse_tolerance: float = None, p_rmse_tolerance: float = None,
                 norm_rmse_tolerance: float = None, energy_tolerance: float = None,
                 var_threshold: float = None):
        self.logger = LoggerAdapter(base_logger, {'qid': qid})
        self.tool_call_df = pd.DataFrame()
        # Record only the focused parameters, and the other parameters will be ignored
        self.focused_parameters = focused_parameters
        self.qid = qid
        self.param_sequence = []
        self.cost_sequence  = []
        self.accumulated_cost = 0
        self.tolerance_rmse = tolerance_rmse
        # NS_2D specific tolerances
        self.mass_tolerance = mass_tolerance
        self.u_rmse_tolerance = u_rmse_tolerance
        self.v_rmse_tolerance = v_rmse_tolerance
        self.p_rmse_tolerance = p_rmse_tolerance
        # NS_Transient_2D specific tolerance
        self.norm_rmse_tolerance = norm_rmse_tolerance
        # MPM_2D specific tolerances
        self.energy_tolerance = energy_tolerance
        self.var_threshold = var_threshold

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

            if tool_name in ["heat_1d_check_converge_cfl", "heat_1d_check_converge_n_space"]:
                # Use tolerance_rmse from dataset - required field
                if self.tolerance_rmse is None:
                    raise ValueError(f"tolerance_rmse is required for heat_1d tools but was not provided in dataset (QID={self.qid})")
                tolerance = self.tolerance_rmse
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    n_space=fetch_param(tool_args, "n_space"),
                    cfl=fetch_param(tool_args, "cfl"),
                    tolerance=tolerance
                )
            elif tool_name in ["heat_2d_check_converge_dx", "heat_2d_check_converge_relax", "heat_2d_check_converge_t_init", "heat_2d_check_converge_error_threshold"]:
                # Use tolerance_rmse from dataset - required field
                if self.tolerance_rmse is None:
                    raise ValueError(f"tolerance_rmse is required for heat_2d tools but was not provided in dataset (QID={self.qid})")
                tolerance = self.tolerance_rmse
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    dx=fetch_param(tool_args, "dx"),
                    relax=fetch_param(tool_args, "relax"),
                    t_init=fetch_param(tool_args, "t_init", "T_init"),
                    error_threshold=fetch_param(tool_args, "error_threshold"),
                    tolerance=tolerance
                )
            elif tool_name in ["burgers_1d_check_converge_cfl", "burgers_1d_check_converge_beta", "burgers_1d_check_converge_k", "burgers_1d_check_converge_n_space"]:
                # Use tolerance_rmse from dataset - required field
                if self.tolerance_rmse is None:
                    raise ValueError(f"tolerance_rmse is required for burgers_1d tools but was not provided in dataset (QID={self.qid})")
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
                "ns_2d_check_converge_diff_v_threshold", "ns_2d_check_converge_res_iter_v_threshold"
            ]:
                # Use ns_2d specific tolerance values from dataset - required fields
                if None in [self.mass_tolerance, self.u_rmse_tolerance, self.v_rmse_tolerance, self.p_rmse_tolerance]:
                    raise ValueError(f"All tolerance values (mass_tolerance, u_rmse_tolerance, v_rmse_tolerance, p_rmse_tolerance) are required for ns_2d tools but some were not provided in dataset (QID={self.qid})")
                
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
                    mass_tolerance=self.mass_tolerance,
                    u_rmse_tolerance=self.u_rmse_tolerance,
                    v_rmse_tolerance=self.v_rmse_tolerance,
                    p_rmse_tolerance=self.p_rmse_tolerance,
                )
            elif tool_name in [
                "ns_transient_2d_check_converge_resolution", "ns_transient_2d_check_converge_cfl", 
                "ns_transient_2d_check_converge_relaxation_factor", "ns_transient_2d_check_converge_residual_threshold"
            ]:
                # Use norm_rmse_tolerance from dataset - required field
                if self.norm_rmse_tolerance is None:
                    raise ValueError(f"norm_rmse_tolerance is required for ns_transient_2d tools but was not provided in dataset (QID={self.qid})")
                
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    resolution=fetch_param(tool_args, "resolution"),
                    cfl=fetch_param(tool_args, "cfl"),
                    relaxation_factor=fetch_param(tool_args, "relaxation_factor"),
                    residual_threshold=fetch_param(tool_args, "residual_threshold"),
                    norm_rmse_tolerance=self.norm_rmse_tolerance,
                )
            elif tool_name in [
                "epoch_1d_check_converge_dt_multiplier", "epoch_1d_check_converge_nx",
                "epoch_1d_check_converge_npart", "epoch_1d_check_converge_field_order",
                "epoch_1d_check_converge_particle_order"
            ]:
                # Use tolerance_rmse from dataset - required field
                if self.tolerance_rmse is None:
                    raise ValueError(f"tolerance_rmse is required for epoch_1d tools but was not provided in dataset (QID={self.qid})")
                tolerance = self.tolerance_rmse
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    nx=fetch_param(tool_args, "nx"),
                    dt_multiplier=fetch_param(tool_args, "dt_multiplier"),
                    npart=fetch_param(tool_args, "npart"),
                    field_order=fetch_param(tool_args, "field_order"),
                    particle_order=fetch_param(tool_args, "particle_order"),
                    tolerance=tolerance
                )
            elif tool_name in [
                "mpm_2d_check_converge_nx", "mpm_2d_check_converge_npart", "mpm_2d_check_converge_cfl"
            ]:
                # Use energy_tolerance and var_threshold from dataset - required fields
                if None in [self.energy_tolerance, self.var_threshold]:
                    raise ValueError(f"Both energy_tolerance and var_threshold are required for mpm_2d tools but some were not provided in dataset (QID={self.qid})")

                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    nx=fetch_param(tool_args, "nx"),
                    npart=fetch_param(tool_args, "npart", "n_part"),
                    cfl=fetch_param(tool_args, "cfl"),
                    energy_tolerance=self.energy_tolerance,
                    var_threshold=self.var_threshold,
                )
            elif tool_name in [
                "diff_react_1d_check_converge_cfl", "diff_react_1d_check_converge_n_space",
                "diff_react_1d_check_converge_tol"
            ]:
                # diff_react_1d uses task-specific tolerance from tolerance_rmse dict
                if self.tolerance_rmse is None:
                    raise ValueError(f"tolerance_rmse is required for diff_react_1d tools but was not provided in dataset (QID={self.qid})")

                # Extract task name from tool name (e.g., "diff_react_1d_check_converge_cfl" -> "cfl")
                task = tool_name.replace("diff_react_1d_check_converge_", "")

                # tolerance_rmse is a dict like {"n_space": 0.05, "cfl": 0.05, "tol": 1e-05}
                if not isinstance(self.tolerance_rmse, dict):
                    raise ValueError(f"tolerance_rmse for diff_react_1d must be a dict, got {type(self.tolerance_rmse)} (QID={self.qid})")

                if task not in self.tolerance_rmse:
                    raise ValueError(f"Task '{task}' not found in tolerance_rmse dict: {self.tolerance_rmse} (QID={self.qid})")

                rmse_tolerance = self.tolerance_rmse[task]

                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    n_space=fetch_param(tool_args, "n_space"),
                    cfl=fetch_param(tool_args, "cfl"),
                    tol=fetch_param(tool_args, "tol"),
                    rmse_tolerance=rmse_tolerance
                )
            elif tool_name in [
                "euler_2d_check_converge_cfl", "euler_2d_check_converge_n_grid_x",
                "euler_2d_check_converge_cg_tolerance"
            ]:
                # Use tolerance_rmse from dataset - required field
                if self.tolerance_rmse is None:
                    raise ValueError(f"tolerance_rmse is required for euler_2d tools but was not provided in dataset (QID={self.qid})")
                tolerance = self.tolerance_rmse
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    n_grid_x=fetch_param(tool_args, "n_grid_x"),
                    cfl=fetch_param(tool_args, "cfl"),
                    cg_tolerance=fetch_param(tool_args, "cg_tolerance"),
                    rmse_tolerance=tolerance
                )
            else:
                # Critical else branch to handle unrecognized tool names
                raise ValueError(f"Unrecognized tool name: '{tool_name}'. This tool is not supported by the current implementation (QID={self.qid})")

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