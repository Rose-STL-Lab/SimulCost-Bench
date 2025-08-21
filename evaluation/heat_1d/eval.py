import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from costsci_tools.wrappers.heat_1d import compare_res_heat_1d
from inference.utils import setup_logging, NumpyEncoder

# Constants
RMSE_TOLERANCE_BY_PRECISION = {
    "low": 0.01, # Relaxed convergence criteria
    "medium": 0.001, # Moderate convergence criteria
    "high": 0.0001, # Most stringent convergence criteria
}
VALID_PRECISION_LEVELS = {"low", "medium", "high"}
VALID_TASKS = {"cfl", "n_space"}

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


def get_reference_params(dummy: Dict) -> Dict:
    """Extract reference parameters from the dummy data"""
    return dummy["best_params"]


def get_required_param(param_dict: Dict, param_name: str, alt_name: str = None):
    """Strict parameter fetching - no defaults, raise error if missing"""
    if param_name in param_dict:
        return param_dict[param_name]
    elif alt_name and alt_name in param_dict:
        return param_dict[alt_name]
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
    """Evaluate model performance against reference solutions.
    
    Args:
        dataset: Dataset name (should be 'heat_1d')
        task: Task type (e.g., 'cfl', 'n_space')
        model_name: Name of the model being evaluated
        precision_level: Precision level ('low', 'medium', 'high')
        zero_shot: Whether to use zero-shot evaluation
        
    Returns:
        Dictionary containing evaluation metrics
        
    Raises:
        RuntimeError: If required files are not found or data loading fails
    """
    flag = "zero_shot" if zero_shot else "iterative"
    
    # Use precision_level-based paths for heat_1d
    log_dir = f"eval_results/{dataset}/{task}/{precision_level}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name}.log"
    logger = setup_logging(log_file)
    
    # Build paths for heat_1d precision_level structure
    result_path = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_{model_name}.json"
    dummy_path = f"data/{dataset}/{task}/{precision_level}/{flag}_questions.json"
    
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
    total_rmse = 0.0
    total_efficiency = 0.0
    total_hard_efficiency = 0.0
    total_soft_success = 0.0

    # Validate inputs
    if task not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid tasks: {sorted(VALID_TASKS)}")
    if precision_level not in VALID_PRECISION_LEVELS:
        raise ValueError(f"Invalid precision_level '{precision_level}'. Valid levels: {sorted(VALID_PRECISION_LEVELS)}")
    
    # Get precision-specific tolerance for success checking
    rmse_tol = RMSE_TOLERANCE_BY_PRECISION[precision_level]

    for res in result_dataset:
        qid = res.get("QID")
        if qid is None or qid not in dummy_by_qid:
            logger.warning(f"⚠️ Skip entry without valid QID: {res}")
            continue

        dummy = dummy_by_qid[qid]
        evaluated += 1

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
                    # Check if this parameter set has the required keys for heat_1d
                    has_cfl = any(k in candidate for k in ["cfl", "current_cfl"])
                    has_n_space = any(k in candidate for k in ["n_space", "current_n_space"])
                    has_valid_params = has_cfl and has_n_space
                    
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
            rmse = float('inf')
            
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy)
            
        else:
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy)

            try:
                success, rmse = compare_res_heat_1d(
                    profile1=dummy["profile"],
                    cfl1=get_required_param(last_iter, "cfl", "current_cfl"),
                    n_space1=get_required_param(last_iter, "n_space", "current_n_space"),
                    profile2=dummy["profile"],
                    cfl2=get_required_param(ref_iter, "cfl", "current_cfl"),
                    n_space2=get_required_param(ref_iter, "n_space", "current_n_space"),
                    tolerance=rmse_tol,
                )
                
            except Exception as e:
                logger.warning(f"⚠️ QID {qid}: Error in success determination: {e}, marking as failed")
                success = False
                rmse = float('inf')

        # Calculate metrics with robust error handling
        dummy_cost = dummy["dummy_cost"]
        
        # Calculate soft success value
        if np.isnan(rmse) or np.isinf(rmse):
            soft_success_value = 0.0
            logger.warning(f"⚠️ QID {qid}: RMSE ({rmse}) is NaN/inf, setting soft success to 0")
        else:
            soft_success_value = soft_success(rmse, rmse_tol)
            
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
        rmse_for_total = 0.0 if (np.isnan(rmse) or np.isinf(rmse)) else rmse
        total_rmse += rmse_for_total
        
        total_efficiency += efficiency
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
            f"⚡ Efficiency: {efficiency:.3f}\n"
            f"⚡ Hard Efficiency: {hard_efficiency:.3f}\n"
            f"📉 RMSE (model vs. dummy): {rmse:.3e}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

    # Calculate final metrics with division by zero protection
    if evaluated == 0:
        logger.warning("⚠️ No valid evaluations performed")
        success_rate = converged_rate = mean_efficiency = 0.0
        mean_hard_efficiency = mean_rmse = mean_ss = 0.0
    else:
        success_rate = success_cnt / evaluated
        converged_rate = converged_cnt / evaluated
        mean_efficiency = total_efficiency / evaluated
        mean_hard_efficiency = total_hard_efficiency / evaluated
        mean_rmse = total_rmse / evaluated
        mean_ss = total_soft_success / evaluated

    metrics = {
        "success_rate": f"{success_rate:.3f}",
        "converged_rate": f"{converged_rate:.3f}",
        "mean_efficiency": f"{mean_efficiency:.3f}",
        "mean_hard_efficiency": f"{mean_hard_efficiency:.3f}",
        "mean_soft_success": f"{mean_ss:.3f}",
        "mean_rmse": f"{mean_rmse:.2e}",
        "rmse_tolerance": f"{rmse_tol:.2e}",
        "total_model_cost": total_model_cost,
        "total_dummy_cost": total_dummy_cost,
    }

    logger.info(f"🧾 Evaluation Summary for {model_name}:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="heat_1d",
                        help="Dataset name (should be 'heat_1d')")
    parser.add_argument("-t", "--task", default="cfl",
                        choices=list(VALID_TASKS),
                        help="Task: cfl / n_space")
    parser.add_argument("-l", "--precision_level", default="medium",
                        choices=list(VALID_PRECISION_LEVELS),
                        help="Precision level for heat_1d dataset")
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