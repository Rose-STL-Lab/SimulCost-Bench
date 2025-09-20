from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen3:
    def __init__(self, model_path: str):
        """
        Initialize your custom model.
        
        Args:
            model_path (str): Path to your model files/weights
        """
        self.model_path = model_path
        
        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def invoke(self, messages: list[dict]) -> str:
        """
        Perform inference on the given messages.
        
        Args:
            messages (list[dict]): List of message dictionaries
                Each message should have the format:
                {"role": "system|user|assistant", "content": "message content"}
        
        Returns:
            str: JSON-formatted string containing the model's response
        """
        # print("================================================")
        # print("Messages:", messages)
        # print("================================================")
        # Prepare the model input        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        
        # Parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print("thinking content:", thinking_content)
        # print("Qwen3 Response:", response)
        
        return response
    
# _PROBLEM_TO_TOOL_NAME = {
#     "1D_heat_transfer":{
#         "cfl":"heat_1d_check_converge_cfl",
#         "n_space":"heat_1d_check_converge_n_space"
#     },
#     "2D_heat_transfer":{
        
#     },
#     "1D_burgers":{
        
#     },
#     "euler_1d":{
#         "cfl":"euler_1d_check_converge_cfl",
#         "beta":"euler_1d_check_converge_beta",
#         "k":"euler_1d_check_converge_k",
#         "n_space":"euler_1d_check_converge_n_space"
#     },
#     "ns_channel_2d":{
#         "mesh_x":"ns_2d_check_converge_mesh_x"
#     },
#     "ns_transient_2d":{
#         "resolution":"ns_transient_2d_check_converge_resolution"
#     }
# }

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import copy
# from methods import *

# class BOAgent:
#     def __init__(self, model_path='/home/leo/workspace/SimulCost-Bench/custom_model/bo_configs/config.yaml'):
#         '''
#         Bayesian Optimization base class.
        
#         Here model_path is a yaml config recording BO parameters.
#         '''
#         from omegaconf import OmegaConf
        
#         cfg = OmegaConf.load(model_path)
#         self.cfg = cfg
        
#         # Import BO class from methods
#         from methods import BO
#         self.bo = BO(model_path)
        
#     def invoke(self, messages):
#         """Main entry point for Bayesian Optimization agent inference.
        
#         Args:
#             messages (list): List of message dictionaries
            
#         Returns:
#             str: JSON-formatted response with optimization results
#         """
#         # Extract problem parameters from messages
#         problem, task, profile, tolerance, optimization_type = extract_problem_info_from_messages(messages)
        
#         # Run Bayesian Optimization (only supports zero-shot mode)
#         if optimization_type == "zero-shot":
#             final_params, best_score = self.bo.solve(problem, task, profile, tolerance, messages)
            
#             response = str({
#                 "tool_name": _PROBLEM_TO_TOOL_NAME[problem][task],
#                 "tool_reason": "Based on Bayesian optimization with Gaussian process modeling and acquisition function maximization, I propose this parameter value as the optimal solution.",
#                 "tool_args": final_params
#             })
#         else:
#             raise NotImplementedError("BO doesn't support iterative!")
        
#         return response
