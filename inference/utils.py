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
from logging import LoggerAdapter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# class SeparatorFilter(logging.Filter):
#     """
#     每当 QID 从上一条日志变成新的值时，
#     在消息前插入空行或分隔线，便于视觉分组。
#     """
#     _last_qid = None

#     def filter(self, record: logging.LogRecord) -> bool:
#         current_qid = getattr(record, 'qid', '-')
#         if current_qid != self._last_qid:
#             self._last_qid = current_qid
#             # 下面两行二选一：空行 or 分隔线
#             # record.msg = '\n' + record.msg
#             record.msg = '\n' + '=' * 100 + '\n' + record.msg
#         return True

def setup_logging(filename: str = None) -> logging.Logger:
    """Setup logging configuration with filtered handlers"""
    # Clear previous log file's contents
    if filename:
        open(filename, 'w').close()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] QID=%(qid)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    class QIDFilter(logging.Filter):
        """给没有 qid 字段的 LogRecord 自动补 '-'，保证 formatter 不报错"""
        def filter(self, record):
            if not hasattr(record, 'qid'):
                record.qid = '-'
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
    # console_handler.addFilter(SeparatorFilter()) 
    logger.addHandler(console_handler)

    # File handler with filter
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(FileFilter())
        file_handler.addFilter(QIDFilter())
        # file_handler.addFilter(SeparatorFilter()) 
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
    def __init__(self, base_logger: logging.Logger, qid: int,
                 focused_parameters: List[str] = None):
        self.logger = LoggerAdapter(base_logger, {'qid': qid})
        self.tool_call_df = pd.DataFrame()
        # Record only the focused parameters, and the other parameters will be ignored
        self.focused_parameters = focused_parameters
        self.qid = qid
        self.param_sequence = []
        self.cost_sequence  = [] 
        self.accumulated_cost = 0

    def execute_tool_call(self, tool_reason: str, tool_name: str, tool_args: Dict[str, Any], qid: int) -> Tuple[Dict[str, Any], int]:
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

            func = globals()[tool_name]
            profile = f"p{qid}"

            if tool_name in ["check_converge_cfl", "check_converge_n_space"]:
                result = func(
                    accumulated_cost=self.accumulated_cost,
                    profile=profile,
                    current_n_space=tool_args["n_space"],
                    current_cfl=tool_args["cfl"],
                    tolerance=1e-4
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
            # Return a dict with an 'error' key, so the caller can detect failure
            return {"error": error_msg}, self.accumulated_cost

    def _record_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_reason: str, result: Dict[str, Any]) -> None:
        """Record tool call details in the DataFrame."""
        try:
            focus_args = {}
            # if self.focused_parameters is None:
            focus_args = tool_args
            # else:
            #     for param in self.focused_parameters:
            #         if param in tool_args:
            #             focus_args[param] = tool_args[param]

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

            # if not tool_name.endswith("summary"):           # 过滤掉总结类工具
            self.param_sequence.append(focus_args)      # focus_args 已只保留关心的字段

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