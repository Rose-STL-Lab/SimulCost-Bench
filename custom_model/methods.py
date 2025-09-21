from omegaconf import OmegaConf
from bayes_opt import BayesianOptimization
import torch, re
import numpy as np
import json
import costsci_tools.wrappers as wrappers
import pdb
NAME_TO_FOLDER = {
    "1D_heat_transfer":"heat_1d",
    "heat_1d":"heat_1d",
    "2D_heat_transfer":"heat_steady_2d",
    "1D_burgers":"burgers_1d",
    "burgers_1d":"burgers_1d",
    "euler_1d": "euler_1d",
    "ns_channel_2d": "ns_channel_2d",
    "ns_transient_2d": "ns_transient_2d",
    "epoch": "epoch"
}

def soft_success(d, epsilon):
    """计算单个 (d, epsilon) 对的 Soft Success 值"""
    r = d / epsilon
    
    if r <= 1:
        return 1.0
    
    # 参数
    alpha = 0.6
    beta = 0.43
    gamma = 1.5
    omega = 0.3
    delta = 2.2
    
    # 双组分衰减函数
    exp_component = np.exp(-beta * (r - 1)**gamma)
    logistic_component = 1 / (1 + omega * (r - 1)**delta)
    
    return alpha * exp_component + (1 - alpha) * logistic_component

def extract_problem_info_from_messages(messages, skip_profile=False):
    """Extract problem, task, profile, and tolerance from messages.

    Args:
        messages (list): List of message dictionaries
        skip_profile (bool): If True, skip profile extraction and return None for profile

    Returns:
        tuple: (problem, task, profile, qid, tolerance, type) or (problem, task, qid, tolerance, type) if skip_profile=True
    """
    import re

    prompt = messages[0]['content']
    instruction = messages[1]['content']

    # Check both system and user messages for problem identification
    all_content = prompt + " " + instruction

    if "one opportunity" in prompt:
        type = "zero-shot"
    elif "gradually refine" in prompt:
        type = "iterative"
    else:
        raise ValueError("Type not extracted!")

    if 'HIGH' in instruction:
        tolerance = "high"
    elif "MEDIUM" in instruction:
        tolerance = "medium"
    elif "LOW" in instruction:
        tolerance = "low"
    else:
        raise ValueError("Tolerance not extracted!")

    if '1D heat' in all_content or '1D Heat' in all_content:
        problem = '1D_heat_transfer'
        if 'change `cfl`' in all_content or 'only** change `cfl`' in all_content:
            task = 'cfl'
        elif 'change `n_space`' in all_content or 'only** change `n_space`' in all_content:
            task = 'n_space'
    elif "1D Euler" in prompt:
        problem = "euler_1d"
        if 'change `cfl`' in prompt:
            task = 'cfl'
        else:
            task = "n_space"
    elif "Navier-Stokes" in prompt and "2D steady" in prompt:
        problem = "ns_channel_2d"
        if 'change `mesh_x`' in prompt:
            task = 'mesh_x'
        else:
            # Default to mesh_x if no specific parameter mentioned
            task = 'mesh_x'
    elif "Navier-Stokes" in prompt and "2D transient" in prompt:
        problem = "ns_transient_2d"
        task = 'resolution'
    else:
        #pdb.set_trace()
        raise NotImplementedError("Problem not supported or not found!")

    # Extract profile only if not skipping
    if not skip_profile:
        # print(instruction)
        match = re.search(r"Profile:\s*(p\d+)", instruction)
        if match:
            profile = match.group(1)
        else:
            raise ValueError("Profile not extracted!")
    else:
        profile = None

    qid_match = re.search(r"QID:\s*(\d+)", messages[1]["content"])
    if qid_match:
        qid = f"{qid_match.group(1)}"
    else:
        raise ValueError("q_unknown")

    if skip_profile:
        return problem, task, qid, tolerance, type
    else:
        return problem, task, profile, qid, tolerance, type


def check_cost(problem, profile, params):
    runner = getattr(wrappers, f"run_sim_{NAME_TO_FOLDER[problem]}")
    if problem == "heat_1d" or problem == "1D_heat_transfer":
        cost = runner(profile, params['cfl'], int(params['n_space']))
    elif problem == "2D_heat_transfer":
        cost, steps = runner(profile, params["dx"], params["relax"], params["error_threshold"], params["t_init"]) 
    elif problem == "burgers_1d":
        # Use beta parameter but pass as w to the wrapper function
        cost = runner(profile, params["cfl"], params["k"], params["beta"], params["n_space"])
    elif problem == "euler_1d":
        cost = runner(profile, params["cfl"], params["beta"], params["k"], params["n_space"])
    elif problem == "ns_channel_2d":
        profile_to_bt = {
            "p1": "channel_flow",
            "p2": "back_stair_flow",
            "p3": "expansion_channel",
            "p4": "cube_driven_flow"
        }
        cost, steps = runner(profile, profile_to_bt[profile] , int(params["mesh_x"]), int(params["mesh_y"]), 
                     params["omega_u"], params["omega_v"], params["omega_p"],
                     params["diff_u_threshold"], params["diff_v_threshold"], 
                     params["res_iter_v_threshold"])

    return cost

def check_gt(
    problem,
    profile,
    gt,
    x,
    tolerance="medium",
    whether_soft_success=True
):
    compare_func = getattr(wrappers, f"compare_res_{NAME_TO_FOLDER[problem]}")
    
    if problem == "heat_1d" or problem == "1D_heat_transfer":
        # Map tolerance level to numerical value for heat_1d
        tolerance_map = {
            'low': 0.01,
            'medium': 0.001,
            'high': 0.0001
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.001)
        success, error = compare_func(profile, x['cfl'], int(x['n_space']),
                                      profile, gt['cfl'], int(gt['n_space']), numeric_tolerance)
    elif problem == "2D_heat_transfer":
        # Map tolerance level to numerical value for 2D_heat_transfer
        tolerance_map = {
            'low': 0.1,
            'medium': 0.01,
            'high': 0.001
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.01)
        success, error = compare_func(profile, x["dx"], x["relax"], x["error_threshold"], x["t_init"],
                                  profile, gt["dx"], gt["relax"], gt["error_threshold"], gt["t_init"], numeric_tolerance)
    elif problem == "burgers_1d":
        # Map tolerance level to numerical value for burgers_1d
        tolerance_map = {
            'low': 0.08,
            'medium': 0.04,
            'high': 0.01
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.04)
        success, _, _, error = compare_func(profile, x["cfl"], x["k"], x["beta"],
                                  profile, gt["cfl"], gt["k"], gt["beta"],
                                  numeric_tolerance,
                                  x["n_space"], gt["n_space"])
    elif problem == "euler_1d":
        # Map tolerance level to numerical value for euler_1d
        tolerance_map = {
            'low': 0.08,
            'medium': 0.02,
            'high': 0.01
        }
        numeric_tolerance = tolerance_map.get(tolerance, 0.02)
        success, success1, success2, error = compare_func(profile, x["cfl"], x["beta"], x["k"],
                                      profile, gt["cfl"], gt["beta"], gt["k"],
                                      numeric_tolerance, x["n_space"], gt["n_space"])
        
        
    if whether_soft_success:
        # Use appropriate tolerance for soft_success calculation
        return soft_success(error, numeric_tolerance), error
    else:
        return error < numeric_tolerance, error


class Verifier:
    def __init__(self,
                 problem,
                 task,
                 tolerance,
                 dummy_root="./data",
                 profile_root="./costsci_tools/run_configs",
                 iterative=False):
        self.problem = problem
        self.task = task
        self.tolerance = tolerance
        self.iterative = iterative
        
        if not iterative:
            dummy_path = f"{dummy_root}/{NAME_TO_FOLDER[problem]}/{task}/{tolerance}/zero_shot_questions.json"
        else:
            dummy_path = f"{dummy_root}/{NAME_TO_FOLDER[problem]}/{task}/{tolerance}/iterative_questions.json"
            
        #dummy_path = f"{dummy_root}/{task}/zero_shot_question.json"
        with open(dummy_path, 'r') as f:
            data = json.load(f)
        #self.best_params = {x["profile"]: self.get_best_params(x) for x in data}
        self.best_params = {str(x["QID"]): x["best_params"] for x in data}
        self.best_costs = {str(x["QID"]): x["dummy_cost"] for x in data}
        
        self.profile_root = profile_root
        
    
    def metric(self, param, profile, qid, soft_success=True, prev_cost=0):
        success, error = check_gt(self.problem,
                           profile,
                           self.best_params[qid],
                           param,
                           self.tolerance,
                           whether_soft_success=soft_success)
        cost = check_cost(self.problem, profile, param)
        if self.iterative:
            cost += prev_cost
        score = success * (self.best_costs[qid] / (1e-3 + cost))
        #efficiency = self.best_costs[profile] / (1e-3 + cost)
        return success, cost, score

def evaluate(params,
             verifier,
             profile,
             qid,
             use_soft_success=True,
             prev_cost=0):
    """
    Evaluate simulation parameters using the verifier.
    
    Args:
        params: Dictionary of parameters to evaluate
        verifier: Verifier object with metric method
        backend: Evaluation backend ("surrogate" or "ground_truth")
    
    Returns:
        float: Evaluation score (higher is better)
    """
    success, cost, score = verifier.metric(params, profile, qid, soft_success=use_soft_success, prev_cost=prev_cost)
    success = float(success)
    efficiency = verifier.best_costs[qid] / (1e-3 + cost)

    return max(0, success), efficiency, max(0, score)

class BO:
    def __init__(self, config_path):
        cfg = OmegaConf.load(config_path)
        self.cfg = cfg
        
        # Bayesian Optimization specific parameters
        self.init_points = cfg.optimization.get("init_points", 5)
        self.n_iter = cfg.optimization.get("n_iter", 25)
        self.random_state = cfg.optimization.get("random_state", 1)
        
        # General parameters
        self.log_dir = cfg.get("log_dir", ".")
        self.verbose = cfg.get("verbose", True)
        self.soft_success = cfg.get("soft_success", False)  # Use hard success for BO
        
    def vprint(self, *args, **kwargs):
        """Verbose print - only prints if verbose is enabled."""
        if self.verbose:
            print(*args, **kwargs)
    
    def create_black_box_function(self, problem, task, profile, qid, tolerance, fixed_params=None):
        """Create a black box function for Bayesian optimization.
        
        Args:
            problem (str): Problem type
            task (str): Task type  
            profile (str): Profile identifier
            tolerance (str): Tolerance level
            fixed_params (dict): Fixed parameter values
            
        Returns:
            callable: Black box function for optimization
        """
        verifier = Verifier(problem, task, tolerance)
        fixed_params = fixed_params or {}
        
        def black_box_func(**kwargs):
            # Merge optimization variables with fixed parameters
            params = {**fixed_params, **kwargs}
            
            # Convert any float parameters to int where needed
            if 'n_space' in params:
                params['n_space'] = int(params['n_space'])
            
            success_pred, cost_pred, score = evaluate(
                params,
                verifier,
                profile,
                qid,
                use_soft_success=False
            )
            self.vprint(f"Evaluated params {params}: score = {score}")
            return score
        
        
        return black_box_func
    
    def get_parameter_bounds(self, problem, task):
        """Get parameter bounds for different problem-task combinations.
        
        Args:
            problem (str): Problem type
            task (str): Task type
            
        Returns:
            dict: Parameter bounds for Bayesian optimization
        """
        bounds = {}
        
        if problem == "1D_heat_transfer":
            if task == "cfl":
                bounds = {'cfl': (0.01, 2.0)}
            elif task == "n_space":
                bounds = {'n_space': (50, 2000)}
        elif problem == "euler_1d":
            if task == "cfl":
                bounds = {'cfl': (0.01, 1.0)}
            elif task == "n_space":
                bounds = {'n_space': (64, 1024)}
        elif problem == "ns_channel_2d":
            if task == "mesh_x":
                bounds = {'mesh_x': (64, 256)}
            elif task == "mesh_y":
                bounds = {'mesh_y': (16, 64)}
            elif task in ["omega_u", "omega_v"]:
                bounds = {task: (0.1, 1.0)}
            elif task == "omega_p":
                bounds = {'omega_p': (0.1, 0.5)}
            elif task in ["diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"]:
                bounds = {task: (1e-7, 1e-3)}
        else:
            raise NotImplementedError(f"Problem {problem} not supported for BO")
        
        return bounds
    
    def get_fixed_params(self, problem, task, tolerance="medium", qid=None):
        """Get fixed parameters for different problem-task combinations.

        Args:
            problem (str): Problem type
            task (str): Task type
            tolerance (str): Tolerance level
            qid (str): Question ID to read specific config from

        Returns:
            dict: Fixed parameter values
        """
        fixed_params = {}

        if problem == "1D_heat_transfer":
            # Read fixed params from problem configuration
            dummy_path = f"./data/{NAME_TO_FOLDER[problem]}/{task}/{tolerance}/zero_shot_questions.json"
            with open(dummy_path, 'r') as f:
                data = json.load(f)
            # Find the entry with matching QID
            target_entry = None
            for entry in data:
                if str(entry["QID"]) == str(qid):
                    target_entry = entry
                    break
            if target_entry is None:
                raise ValueError(f"QID {qid} not found in {dummy_path}")

            sample_params = target_entry["best_params"]
            if task == "cfl":
                fixed_params = {'n_space': sample_params['n_space']}
            elif task == "n_space":
                fixed_params = {'cfl': sample_params['cfl']}
        elif problem == "euler_1d":
            if task == "cfl":
                fixed_params = {'beta': 1.0, 'k': -1.0, 'n_space': 256}
            elif task == "n_space":
                fixed_params = {'cfl': 0.5, 'beta': 1.0, 'k': -1.0}
        elif problem == "ns_channel_2d":
            # Fixed parameters for ns_channel_2d - use default values for non-optimized parameters
            base_params = {'mesh_x': 128, 'mesh_y': 32, 'omega_u': 0.6, 'omega_v': 0.6, 'omega_p': 0.3,
                          'diff_u_threshold': 1e-5, 'diff_v_threshold': 1e-5, 'res_iter_v_threshold': 1e-4}
            fixed_params = {k: v for k, v in base_params.items() if k != task}

        return fixed_params
    
    def solve(self, problem, task, profile, qid, tolerance, messages):
        """Main Bayesian Optimization process.
        
        Args:
            problem (str): Problem type
            task (str): Task type
            profile (str): Profile identifier
            tolerance (str): Tolerance level
            messages (list): Original messages for context (not used in BO)
            
        Returns:
            dict: Final optimized parameter set
        """
        
        self.vprint("="*50)
        self.vprint(f"Starting Bayesian Optimization")
        self.vprint(f"Problem: {problem}, Task: {task}")
        self.vprint(f"Profile: {profile}, Tolerance: {tolerance}")
        self.vprint(f"Init points: {self.init_points}, Iterations: {self.n_iter}")
        self.vprint("="*50)
        
        # Get parameter bounds and fixed parameters
        pbounds = self.get_parameter_bounds(problem, task)
        fixed_params = self.get_fixed_params(problem, task, tolerance, qid)
        
        self.vprint(f"Parameter bounds: {pbounds}")
        self.vprint(f"Fixed parameters: {fixed_params}")
        
        # Create black box function
        black_box_func = self.create_black_box_function(
            problem, task, profile, qid, tolerance, fixed_params
        )
        
        # Initialize Bayesian Optimization
        from bayes_opt import BayesianOptimization
        optimizer = BayesianOptimization(
            f=black_box_func,
            pbounds=pbounds,
            random_state=self.random_state,
            verbose=2 if self.verbose else 0
        )
        
        # Perform optimization
        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
        #pdb.set_trace()
        # Get best parameters
        best_params = optimizer.max['params']
        best_score = optimizer.max['target']
        
        # Convert float parameters to int where needed
        if 'n_space' in best_params:
            best_params['n_space'] = int(best_params['n_space'])
        
        # Combine with fixed parameters for final result
        final_params = {**fixed_params, **best_params}
        
        self.vprint("="*50)
        self.vprint(f"Optimization complete!")
        self.vprint(f"Best parameters: {final_params}")
        self.vprint(f"Best score: {best_score}")
        self.vprint("="*50)
        
        # Save results
        results_dict = {
            "method": "Bayesian Optimization",
            "problem": problem,
            "task": task,
            "profile": profile,
            "tolerance": tolerance,
            "optimization_history": [
                {
                    "iteration": i,
                    "params": {**fixed_params, **res['params']},
                    "target": res['target']
                } for i, res in enumerate(optimizer.res)
            ],
            "best_params": final_params,
            "best_score": best_score,
            "config": {
                "init_points": self.init_points,
                "n_iter": self.n_iter,
                "parameter_bounds": pbounds,
                "fixed_params": fixed_params
            }
        }
        
        return (final_params, best_score)