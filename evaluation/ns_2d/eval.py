import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from costsci_tools.wrappers.ns_channel_2d import compare_res_ns_channel_2d
from inference.utils import setup_logging, NumpyEncoder

# Constants - NS 2D tolerance values by precision level
NS_2D_TOLERANCE_BY_PRECISION = {
    "low": {
        "mass_tolerance": 1e-04,
        "u_rmse_tolerance": 0.11,
        "v_rmse_tolerance": 0.05,
        "p_rmse_tolerance": 0.4,
    },
    "medium": {
        "mass_tolerance": 1e-06,
        "u_rmse_tolerance": 0.02,
        "v_rmse_tolerance": 0.005,
        "p_rmse_tolerance": 0.2,
    },
    "high": {
        "mass_tolerance": 1e-08,
        "u_rmse_tolerance": 0.008,
        "v_rmse_tolerance": 0.001,
        "p_rmse_tolerance": 0.10,
    }
}

VALID_PRECISION_LEVELS = {"low", "medium", "high"}
VALID_TASKS = {
    "mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", 
    "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"
}

def load_profile_config(profile: str) -> Dict:
    """Load configuration parameters for a specific profile"""
    import yaml
    config_path = f"costsci_tools/run_configs/ns_channel_2d/{profile}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for required parameters and raise error if missing
    required_params = ['length', 'breadth', 'boundary_condition']
    missing_params = []
    
    for param in required_params:
        if param not in config:
            missing_params.append(param)
    
    if missing_params:
        available_keys = list(config.keys())
        raise KeyError(
            f"Required parameters missing from profile '{profile}': {missing_params}. "
            f"Available keys: {available_keys}"
        )
    
    return {
        'length': config['length'],
        'breadth': config['breadth'], 
        'boundary_condition': config['boundary_condition']
    }

def soft_success(d, epsilon):
    """Calculate Soft Success value for a single (d, epsilon) pair"""
    r = d / epsilon
    
    if r <= 1:
        return 1.0
    
    # Parameters
    alpha = 0.6
    beta = 0.43
    gamma = 1.5
    omega = 0.3
    delta = 2.2
    
    # Dual-component decay function
    exp_component = np.exp(-beta * (r - 1)**gamma)
    logistic_component = 1 / (1 + omega * (r - 1)**delta)
    
    return alpha * exp_component + (1 - alpha) * logistic_component

def soft_success_multi(d_list, epsilon_list):
    """Calculate average Soft Success value for multiple (d, epsilon) pairs"""
    ss_values = []
    for d, eps in zip(d_list, epsilon_list):
        ss = soft_success(d, eps)
        ss_values.append(ss)
    
    return np.mean(ss_values)  # Arithmetic mean


def get_reference_params(dummy: Dict, task: str) -> Dict:
    """Extract reference parameters from the dummy data"""
    return dummy["best_params"]


def get_required_param(param_dict: Dict, param_name: str, alt_name: str = None, alt_name2: str = None):
    """Strict parameter fetching - no defaults, raise error if missing"""
    if param_name in param_dict:
        return param_dict[param_name]
    elif alt_name and alt_name in param_dict:
        return param_dict[alt_name]
    elif alt_name2 and alt_name2 in param_dict:
        return param_dict[alt_name2]
    else:
        available_keys = list(param_dict.keys())
        raise KeyError(
            f"Required parameter '{param_name}' not found. "
            f"Available keys: {available_keys}"
        )


def evaluate(
    dataset: str,
    task: str,
    model_name: str,
    precision_level: str = "medium",
    zero_shot: bool = False,
) -> Dict:
    """Evaluate model performance against reference solutions for ns_2d.
    
    Args:
        dataset: Dataset name (should be 'ns_2d')
        task: Task type (one of the 8 NS 2D tasks)
        model_name: Name of the model being evaluated
        precision_level: Precision level ('low', 'medium', 'high')
        zero_shot: Whether to use zero-shot evaluation
        
    Returns:
        Dictionary containing evaluation metrics
        
    Raises:
        RuntimeError: If required files are not found or data loading fails
    """
    flag = "zero_shot" if zero_shot else "iterative"
    
    # Use precision_level-based paths for ns_2d
    log_dir = f"eval_results/{dataset}/{task}/{precision_level}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name}.log"
    logger = setup_logging(log_file)
    
    # Build paths for ns_2d precision_level structure - use ns_channel_2d for data directory
    result_path = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_{model_name}.json"
    dummy_path = f"data/ns_channel_2d/{task}/{precision_level}/{flag}_questions.json"
    
    # Validate paths exist
    if not os.path.exists(result_path):
        error_msg = f"Model results file not found: {result_path}"
        logger.error(f"❌ {error_msg}")
        print(f"\n❌ Evaluation failed: {error_msg}\n")
        raise RuntimeError(error_msg)
    
    if not os.path.exists(dummy_path):
        error_msg = f"Reference data file not found: {dummy_path}"
        logger.error(f"❌ {error_msg}")
        print(f"\n❌ Evaluation failed: {error_msg}\n")
        raise RuntimeError(error_msg)
    
    # Load data files
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            result_dataset = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        error_msg = f"Error loading model results from {result_path}: {e}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
        
    try:
        with open(dummy_path, 'r', encoding='utf-8') as f:
            dummy_dataset = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        error_msg = f"Error loading reference data from {dummy_path}: {e}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    logger.info(f"✅ Loaded model results from {result_path} ({len(result_dataset)} entries)")
    logger.info(f"✅ Loaded dummy solutions from {dummy_path} ({len(dummy_dataset)} entries)")

    dummy_by_qid = {d["QID"]: d for d in dummy_dataset}

    total_model_cost = total_dummy_cost = 0.0
    success_cnt = converged_cnt = evaluated = 0
    total_rmse_u = total_rmse_v = total_rmse_p = 0.0
    total_soft_efficiency = 0.0
    total_hard_efficiency = 0.0
    total_soft_success = 0.0

    # Validate inputs
    if task not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid tasks: {sorted(VALID_TASKS)}")
    if precision_level not in VALID_PRECISION_LEVELS:
        raise ValueError(f"Invalid precision_level '{precision_level}'. Valid levels: {sorted(VALID_PRECISION_LEVELS)}")
    
    # Get precision-specific tolerances for NS 2D
    tolerances = NS_2D_TOLERANCE_BY_PRECISION[precision_level]
    mass_tol = tolerances["mass_tolerance"]
    u_rmse_tol = tolerances["u_rmse_tolerance"]
    v_rmse_tol = tolerances["v_rmse_tolerance"]
    p_rmse_tol = tolerances["p_rmse_tolerance"]

    for res in result_dataset:
        qid = res.get("QID")
        if qid is None or qid not in dummy_by_qid:
            logger.warning(f"⚠️ Skip entry without valid QID: {res}")
            continue

        dummy = dummy_by_qid[qid]
        evaluated += 1

        # Load profile-specific configuration
        profile_config = load_profile_config(dummy["profile"])
        length = profile_config['length']
        breadth = profile_config['breadth']  
        boundary_type = profile_config['boundary_condition']

        cost       = res["accumulated_cost"]
        converged  = res.get("is_converged", res.get("converged", False))
        
        # Handle entries with empty param_sequence - mark as failed instead of skipping
        if not res["param_sequence"]:
            logger.warning(f"⚠️ QID {qid}: empty param_sequence, marking as failed")
            last_iter = {}
        else:
            # Find the last valid parameter set (working backwards from the end)
            last_iter = {}
            for i in range(len(res["param_sequence"]) - 1, -1, -1):
                candidate = res["param_sequence"][i]
                if candidate and isinstance(candidate, dict):
                    # Check if this parameter set has the required keys for ns_2d
                    required_keys = ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", 
                                   "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"]
                    has_valid_params = all(
                        any(k in candidate for k in [key, f"current_{key}"]) 
                        for key in required_keys
                    )
                    
                    if has_valid_params:
                        last_iter = candidate
                        if i < len(res["param_sequence"]) - 1:
                            logger.info(f"📋 QID {qid}: Using parameter set from iteration {i+1} (last valid), skipping {len(res['param_sequence']) - 1 - i} invalid final iterations")
                        break
            
            # If no valid parameter set found, use the last one anyway (might be empty)
            if not last_iter and res["param_sequence"]:
                last_iter = res["param_sequence"][-1]
                logger.warning(f"⚠️ QID {qid}: No fully valid parameter sets found, using last attempt")
        
        # Handle entries with empty parameter dictionaries - mark as failed instead of skipping
        if not last_iter:
            logger.warning(f"⚠️ QID {qid}: empty parameter dictionary, marking as failed")
            success = False
            rmse_u = rmse_v = rmse_p = float('inf')
            
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy, task)
            
        else:
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy, task)

            try:
                success, rmse_u, rmse_v, rmse_p, mass_conserved1, mass_conserved2 = compare_res_ns_channel_2d(
                    profile1=dummy["profile"],
                    boundary_type1=boundary_type,
                    mesh_x1=get_required_param(last_iter, "mesh_x", "current_mesh_x"),
                    mesh_y1=get_required_param(last_iter, "mesh_y", "current_mesh_y"),
                    omega_u1=get_required_param(last_iter, "omega_u", "current_omega_u"),
                    omega_v1=get_required_param(last_iter, "omega_v", "current_omega_v"),
                    omega_p1=get_required_param(last_iter, "omega_p", "current_omega_p"),
                    diff_u_threshold1=get_required_param(last_iter, "diff_u_threshold", "current_diff_u_threshold"),
                    diff_v_threshold1=get_required_param(last_iter, "diff_v_threshold", "current_diff_v_threshold"),
                    res_iter_v_threshold1=get_required_param(last_iter, "res_iter_v_threshold", "current_res_iter_v_threshold"),
                    profile2=dummy["profile"],
                    boundary_type2=boundary_type,
                    mesh_x2=get_required_param(ref_iter, "mesh_x", "current_mesh_x"),
                    mesh_y2=get_required_param(ref_iter, "mesh_y", "current_mesh_y"),
                    omega_u2=get_required_param(ref_iter, "omega_u", "current_omega_u"),
                    omega_v2=get_required_param(ref_iter, "omega_v", "current_omega_v"),
                    omega_p2=get_required_param(ref_iter, "omega_p", "current_omega_p"),
                    diff_u_threshold2=get_required_param(ref_iter, "diff_u_threshold", "current_diff_u_threshold"),
                    diff_v_threshold2=get_required_param(ref_iter, "diff_v_threshold", "current_diff_v_threshold"),
                    res_iter_v_threshold2=get_required_param(ref_iter, "res_iter_v_threshold", "current_res_iter_v_threshold"),
                    length=length,
                    breadth=breadth,
                    mass_tolerance=mass_tol,
                    u_rmse_tolerance=u_rmse_tol,
                    v_rmse_tolerance=v_rmse_tol,
                    p_rmse_tolerance=p_rmse_tol,
                )
                
            except Exception as e:
                logger.warning(f"⚠️ QID {qid}: Error in success determination: {e}, marking as failed")
                success = False
                rmse_u = rmse_v = rmse_p = float('inf')

        # Calculate metrics with robust error handling
        dummy_cost = dummy["dummy_cost"]
        
        # Calculate soft success value using multi-RMSE approach
        rmse_list = [rmse_u, rmse_v, rmse_p]
        epsilon_list = [u_rmse_tol, v_rmse_tol, p_rmse_tol]
        
        # Handle NaN/inf values in RMSE
        valid_pairs = []
        for rmse_val, eps_val in zip(rmse_list, epsilon_list):
            if not (np.isnan(rmse_val) or np.isinf(rmse_val)):
                valid_pairs.append((rmse_val, eps_val))
        
        if valid_pairs:
            soft_success_value = soft_success_multi([r for r, e in valid_pairs], [e for r, e in valid_pairs])
        else:
            soft_success_value = 0.0
            logger.warning(f"⚠️ QID {qid}: All RMSE values are NaN/inf, setting soft success to 0")
            
        # Calculate efficiency metrics with division by zero protection
        if cost <= 0:
            efficiency = 0.0
            hard_efficiency = 0.0
            logger.warning(f"⚠️ QID {qid}: Cost is {cost}, setting efficiencies to 0")
        else:
            efficiency = soft_success_value * (dummy_cost / cost)
            hard_efficiency = float(success) * (dummy_cost / cost)
            
            # Handle NaN/inf values in efficiency calculations
            if np.isnan(efficiency) or np.isinf(efficiency):
                efficiency = 0.0
                logger.warning(f"⚠️ QID {qid}: Efficiency is NaN/inf, setting to 0")
                
            if np.isnan(hard_efficiency) or np.isinf(hard_efficiency):
                hard_efficiency = 0.0
                logger.warning(f"⚠️ QID {qid}: Hard efficiency is NaN/inf, setting to 0")
        
        # Accumulate metrics
        total_model_cost += cost
        total_dummy_cost += dummy_cost
        success_cnt += int(success)
        converged_cnt += int(converged)
        
        # Handle RMSE accumulation (use 0 for NaN/inf values in totals)
        rmse_u_for_total = 0.0 if (np.isnan(rmse_u) or np.isinf(rmse_u)) else rmse_u
        rmse_v_for_total = 0.0 if (np.isnan(rmse_v) or np.isinf(rmse_v)) else rmse_v
        rmse_p_for_total = 0.0 if (np.isnan(rmse_p) or np.isinf(rmse_p)) else rmse_p
        
        total_rmse_u += rmse_u_for_total
        total_rmse_v += rmse_v_for_total
        total_rmse_p += rmse_p_for_total
        
        total_soft_efficiency += efficiency
        total_hard_efficiency += hard_efficiency
        total_soft_success += soft_success_value

        logger.info(
            f"\n📊 --- Evaluation Result ---\n"
            f"🆔 QID: {qid}\n"
            f"🔄 Converged flag: {converged}\n"
            f"🎯 Success (within tolerance): {success}\n"
            f"🎯 Soft Success: {soft_success_value:.3f}\n"
            f"💰 Model Cost: {cost}\n"
            f"💰 Dummy Cost: {dummy['dummy_cost']}\n"
            f"⚡ Soft Efficiency: {efficiency:.3f}\n"
            f"⚡ Hard Efficiency: {hard_efficiency:.3f}\n"
            f"📉 RMSE U (model vs. dummy): {rmse_u:.3e}\n"
            f"📉 RMSE V (model vs. dummy): {rmse_v:.3e}\n"
            f"📉 RMSE P (model vs. dummy): {rmse_p:.3e}\n"
            f"📏 Mass Tolerance: {mass_tol:.3e}\n"
            f"📏 U RMSE Tolerance: {u_rmse_tol:.3e}\n"
            f"📏 V RMSE Tolerance: {v_rmse_tol:.3e}\n"
            f"📏 P RMSE Tolerance: {p_rmse_tol:.3e}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

    # Calculate final metrics with division by zero protection
    if evaluated == 0:
        logger.warning("⚠️ No valid evaluations performed")
        success_rate = converged_rate = mean_soft_efficiency = 0.0
        mean_hard_efficiency = mean_rmse_u = mean_rmse_v = mean_rmse_p = mean_ss = 0.0
    else:
        success_rate = success_cnt / evaluated
        converged_rate = converged_cnt / evaluated
        mean_soft_efficiency = total_soft_efficiency / evaluated
        mean_hard_efficiency = total_hard_efficiency / evaluated
        mean_rmse_u = total_rmse_u / evaluated
        mean_rmse_v = total_rmse_v / evaluated
        mean_rmse_p = total_rmse_p / evaluated
        mean_ss = total_soft_success / evaluated

    metrics = {
        "success_rate": f"{success_rate:.3f}",
        "converged_rate": f"{converged_rate:.3f}",
        "mean_soft_efficiency": f"{mean_soft_efficiency:.3f}",
        "mean_hard_efficiency": f"{mean_hard_efficiency:.3f}",
        "mean_soft_success": f"{mean_ss:.3f}",
        "mean_rmse_u": f"{mean_rmse_u:.2e}",
        "mean_rmse_v": f"{mean_rmse_v:.2e}",
        "mean_rmse_p": f"{mean_rmse_p:.2e}",
        "mass_tolerance": f"{mass_tol:.2e}",
        "u_rmse_tolerance": f"{u_rmse_tol:.2e}",
        "v_rmse_tolerance": f"{v_rmse_tol:.2e}",
        "p_rmse_tolerance": f"{p_rmse_tol:.2e}",
        "total_model_cost": total_model_cost,
        "total_dummy_cost": total_dummy_cost,
    }

    logger.info(f"🧾 Evaluation Summary for {model_name}:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="ns_2d",
                        help="Dataset name (should be 'ns_2d')")
    parser.add_argument("-t", "--task", default="mesh_x",
                        choices=list(VALID_TASKS),
                        help="Task: one of the 8 NS 2D tasks")
    parser.add_argument("-l", "--precision_level", default="medium",
                        choices=list(VALID_PRECISION_LEVELS),
                        help="Precision level for ns_2d dataset")
    parser.add_argument("-m", "--model",
                        default="anthropic.claude-3-5-haiku-20241022-v1:0")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Use zero-shot flag to locate files")
    args = parser.parse_args()

    evaluate(
        dataset=args.dataset,
        task=args.task,
        model_name=args.model,
        precision_level=args.precision_level,
        zero_shot=args.zero_shot,
    )