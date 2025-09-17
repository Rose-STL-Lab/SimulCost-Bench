from omegaconf import OmegaConf
from bayes_opt import BayesianOptimization
import torch
import numpy as np
import json
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
    elif problem == "ns_transient_2d":
        # Extract static parameters from profile configuration
        profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{NAME_TO_FOLDER[problem]}/{profile}.yaml"
        boundary_condition = extract_yaml_parameter(profile_path, 'boundary_condition', 1)
        reynolds_num = extract_yaml_parameter(profile_path, 'reynolds_num', 1000.0)
        advection_scheme = extract_yaml_parameter(profile_path, 'advection_scheme', 'cip')
        vorticity_confinement = extract_yaml_parameter(profile_path, 'vorticity_confinement', 0.0)
        total_runtime = extract_yaml_parameter(profile_path, 'total_runtime', 1.0)
        no_dye = extract_yaml_parameter(profile_path, 'no_dye', False)
        cpu = extract_yaml_parameter(profile_path, 'cpu', True)
        visualization = extract_yaml_parameter(profile_path, 'visualization', 0)
        
        cost, steps = runner(profile, boundary_condition, int(params["resolution"]), 
                             reynolds_num, params["cfl"], advection_scheme,
                             vorticity_confinement, params["relaxation_factor"], 
                             params["residual_threshold"], total_runtime,
                             no_dye, cpu, visualization)

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
    elif problem == "ns_channel_2d":
        profile_to_bt = {
            "p1": "channel_flow",
            "p2": "back_stair_flow", 
            "p3": "expansion_channel",
            "p4": "cube_driven_flow"
        }
        boundary_type = profile_to_bt[profile]
        
        # Map tolerance level to tolerance dictionary for ns_channel_2d
        tolerance_map = {
            'low': {
                'mass_tolerance': 1e-04,
                'u_rmse_tolerance': 0.11,
                'v_rmse_tolerance': 0.05,
                'p_rmse_tolerance': 0.4
            },
            'medium': {
                'mass_tolerance': 1e-06,
                'u_rmse_tolerance': 0.02,
                'v_rmse_tolerance': 0.005,
                'p_rmse_tolerance': 0.2
            },
            'high': {
                'mass_tolerance': 1e-08,
                'u_rmse_tolerance': 0.008,
                'v_rmse_tolerance': 0.001,
                'p_rmse_tolerance': 0.10
            }
        }
        tolerance_dict = tolerance_map.get(tolerance, tolerance_map['medium'])
        
        # Read length and breadth from profile
        profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{NAME_TO_FOLDER[problem]}/{profile}.yaml"
        length = extract_yaml_parameter(profile_path, 'length', 20.0)
        breadth = extract_yaml_parameter(profile_path, 'breadth', 1.0)
        
        success, rmse_u, rmse_v, rmse_p, mass_conserved1, mass_conserved2 = compare_func(
            profile1=profile,
            boundary_type1=boundary_type,
            mesh_x1=int(x["mesh_x"]),
            mesh_y1=int(x["mesh_y"]),
            omega_u1=x["omega_u"],
            omega_v1=x["omega_v"],
            omega_p1=x["omega_p"],
            diff_u_threshold1=x["diff_u_threshold"],
            diff_v_threshold1=x["diff_v_threshold"],
            res_iter_v_threshold1=x["res_iter_v_threshold"],
            profile2=profile,
            boundary_type2=boundary_type,
            mesh_x2=int(gt["mesh_x"]),
            mesh_y2=int(gt["mesh_y"]),
            omega_u2=gt["omega_u"],
            omega_v2=gt["omega_v"],
            omega_p2=gt["omega_p"],
            diff_u_threshold2=gt["diff_u_threshold"],
            diff_v_threshold2=gt["diff_v_threshold"],
            res_iter_v_threshold2=gt["res_iter_v_threshold"],
            length=length,
            breadth=breadth,
            mass_tolerance=tolerance_dict['mass_tolerance'],
            u_rmse_tolerance=tolerance_dict['u_rmse_tolerance'],
            v_rmse_tolerance=tolerance_dict['v_rmse_tolerance'],
            p_rmse_tolerance=tolerance_dict['p_rmse_tolerance']
        )
        # Calculate combined error metric (average of RMSEs)
        error = (rmse_u + rmse_v + rmse_p) / 3.0
        
    elif problem == "ns_transient_2d":
        # Map tolerance level to numerical value for ns_transient_2d
        tolerance_map = {
            'low': 0.6,
            'medium': 0.3,
            'high': 0.15
        }
        numeric_tolerance = tolerance_map.get(tolerance)
        
        # Extract static parameters from profile configuration for both x and gt
        profile_path = f"/home/ubuntu/dev/SimulCost-Bench/costsci_tools/run_configs/{NAME_TO_FOLDER[problem]}/{profile}.yaml"
        boundary_condition = extract_yaml_parameter(profile_path, 'boundary_condition', 1)
        reynolds_num = extract_yaml_parameter(profile_path, 'reynolds_num', 1000.0)
        advection_scheme = extract_yaml_parameter(profile_path, 'advection_scheme', 'cip')
        vorticity_confinement = extract_yaml_parameter(profile_path, 'vorticity_confinement', 0.0)
        total_runtime = extract_yaml_parameter(profile_path, 'total_runtime', 1.0)
        no_dye = extract_yaml_parameter(profile_path, 'no_dye', False)
        cpu = extract_yaml_parameter(profile_path, 'cpu', True)
        visualization = extract_yaml_parameter(profile_path, 'visualization', 0)
        
        success, error = compare_func(
            profile, boundary_condition, int(x["resolution"]), reynolds_num, x["cfl"], 
            advection_scheme, vorticity_confinement, x["relaxation_factor"], 
            x["residual_threshold"], total_runtime, no_dye, cpu, visualization,
            profile, boundary_condition, int(gt["resolution"]), reynolds_num, gt["cfl"], 
            advection_scheme, vorticity_confinement, gt["relaxation_factor"], 
            gt["residual_threshold"], total_runtime, no_dye, cpu, visualization,
            numeric_tolerance
        )
        
    if whether_soft_success:
        # Use appropriate tolerance for soft_success calculation
        if problem == "ns_channel_2d":
            # Use soft_success_multi for multiple RMSE values
            rmse_list = [rmse_u, rmse_v, rmse_p]
            epsilon_list = [tolerance_dict['u_rmse_tolerance'], 
                           tolerance_dict['v_rmse_tolerance'], 
                           tolerance_dict['p_rmse_tolerance']]
            
            # Handle NaN/inf values in RMSE
            valid_pairs = []
            for rmse_val, eps_val in zip(rmse_list, epsilon_list):
                if not (np.isnan(rmse_val) or np.isinf(rmse_val)):
                    valid_pairs.append((rmse_val, eps_val))
            
            if valid_pairs:
                soft_success_value = soft_success_multi([r for r, e in valid_pairs], [e for r, e in valid_pairs])
            else:
                soft_success_value = 0.0
                
            return soft_success_value, error
        else:
            return soft_success(error, numeric_tolerance), error
    else:
        # For binary success, different logic for ns_channel_2d vs others
        if problem == "ns_channel_2d":
            # For ns_channel_2d, success is already determined by the comparison function
            return success, error
        elif problem == "ns_transient_2d":
            return success, error
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
             backend: Literal["surrogate", "ground_truth"]="surrogate",
             surrogate_config=None,
             use_soft_success=True,
             prev_cost=0,
             threshold=None):
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
    
    def create_black_box_function(self, problem, task, profile, tolerance, fixed_params=None):
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
        fixed_params = fixed_params or {}
        
        def black_box_func(**kwargs):
            # Merge optimization variables with fixed parameters
            params = {**fixed_params, **kwargs}
            
            # Convert any float parameters to int where needed
            if 'n_space' in params:
                params['n_space'] = int(params['n_space'])
            
            try:
                success_pred, cost_pred, score = evaluate(
                    params,
                    verifier,
                    profile,
                    qid,
                    backend="ground_truth",
                    use_soft_success=self.soft_success
                )
                self.vprint(f"Evaluated params {params}: score = {score}")
                return score
            except Exception as e:
                self.vprint(f"Error evaluating params {params}: {e}")
                return -1e6  # Return very low score for failed evaluations
        
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
    
    def get_fixed_params(self, problem, task):
        """Get fixed parameters for different problem-task combinations.
        
        Args:
            problem (str): Problem type
            task (str): Task type
            
        Returns:
            dict: Fixed parameter values
        """
        fixed_params = {}
        
        if problem == "1D_heat_transfer":
            if task == "cfl":
                fixed_params = {'n_space': 100}
            elif task == "n_space":
                fixed_params = {'cfl': 0.25}
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
    
    def solve(self, problem, task, profile, tolerance, messages):
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
        # Extract QID for saving results
        qid_match = re.search(r"QID:\s*(\d+)", messages[1]["content"])
        qid = f"q{qid_match.group(1)}" if qid_match else f"q_unknown"
        results_json_path = create_save_path(problem, task, qid, "BO", "BO")
        
        self.vprint("="*50)
        self.vprint(f"Starting Bayesian Optimization")
        self.vprint(f"Problem: {problem}, Task: {task}")
        self.vprint(f"Profile: {profile}, Tolerance: {tolerance}")
        self.vprint(f"Init points: {self.init_points}, Iterations: {self.n_iter}")
        self.vprint("="*50)
        
        # Get parameter bounds and fixed parameters
        pbounds = self.get_parameter_bounds(problem, task)
        fixed_params = self.get_fixed_params(problem, task)
        
        self.vprint(f"Parameter bounds: {pbounds}")
        self.vprint(f"Fixed parameters: {fixed_params}")
        
        # Create black box function
        black_box_func = self.create_black_box_function(
            problem, task, profile, tolerance, fixed_params
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
        
        with open(results_json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        
        self.vprint(f"Results saved to: {results_json_path}")
        
        return (final_params, best_score)