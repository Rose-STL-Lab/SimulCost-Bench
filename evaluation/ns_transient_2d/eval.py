import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from costsci_tools.wrappers.ns_transient_2d import compare_res_ns_transient_2d
from inference.utils import setup_logging, NumpyEncoder

# Constants - NS Transient 2D tolerance values by precision level
NS_TRANSIENT_2D_TOLERANCE_BY_PRECISION = {
    "low": {
        "norm_rmse_tolerance": 0.6,
    },
    "medium": {
        "norm_rmse_tolerance": 0.3,
    },
    "high": {
        "norm_rmse_tolerance": 0.15,
    }
}

VALID_PRECISION_LEVELS = {"low", "medium", "high"}
VALID_TASKS = {
    "resolution", "cfl", "relaxation_factor", "residual_threshold"
}

def load_profile_config(profile: str) -> Dict:
    """Load configuration parameters for a specific profile"""
    import yaml
    config_path = f"costsci_tools/run_configs/ns_transient_2d/{profile}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for required parameters and raise error if missing
    required_params = ['boundary_condition', 'reynolds_num', 'vorticity_confinement', 
                      'total_runtime', 'no_dye', 'cpu', 'visualization', 'advection_scheme']
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
        'boundary_condition': config['boundary_condition'],
        'reynolds_num': config['reynolds_num'],
        'vorticity_confinement': config['vorticity_confinement'],
        'total_runtime': config['total_runtime'],
        'no_dye': config['no_dye'],
        'cpu': config['cpu'],
        'visualization': config['visualization'],
        'advection_scheme': config['advection_scheme']
    }


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
    """Evaluate model performance against reference solutions for ns_transient_2d.

    Args:
        dataset: Dataset name ('ns_transient_2d', 'ns_transient_2d_icl', 'ns_transient_2d_icl_no_cost', or 'ns_transient_2d_icl_uniform')
        task: Task type (one of the 4 NS Transient 2D tasks)
        model_name: Name of the model being evaluated
        precision_level: Precision level ('low', 'medium', 'high')
        zero_shot: Whether to use zero-shot evaluation

    Returns:
        Dictionary containing evaluation metrics

    Raises:
        RuntimeError: If required files are not found or data loading fails
    """
    flag = "zero_shot" if zero_shot else "iterative"
    
    # Use precision_level-based paths for ns_transient_2d
    log_dir = f"eval_results/{dataset}/{task}/{precision_level}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name}.log"
    logger = setup_logging(log_file)
    
    # Build paths for ns_transient_2d precision_level structure
    result_path = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_{model_name}.json"

    # For ICL variants, dummy solutions are still from the base ns_transient_2d dataset
    # Only ICL examples differ between variants
    dummy_path = f"data/ns_transient_2d/{task}/{precision_level}/{flag}_questions.json"
    
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
    total_norm_rmse = 0.0
    total_efficiency = 0.0

    # Validate inputs
    if task not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid tasks: {sorted(VALID_TASKS)}")
    if precision_level not in VALID_PRECISION_LEVELS:
        raise ValueError(f"Invalid precision_level '{precision_level}'. Valid levels: {sorted(VALID_PRECISION_LEVELS)}")
    
    # Get precision-specific tolerances for NS Transient 2D
    tolerances = NS_TRANSIENT_2D_TOLERANCE_BY_PRECISION[precision_level]
    norm_rmse_tol = tolerances["norm_rmse_tolerance"]

    for res in result_dataset:
        qid = res.get("QID")
        if qid is None or qid not in dummy_by_qid:
            logger.warning(f"⚠️ Skip entry without valid QID: {res}")
            continue

        dummy = dummy_by_qid[qid]
        evaluated += 1

        # Load profile-specific configuration
        profile_config = load_profile_config(dummy["profile"])

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
                    # Check if this parameter set has the required keys for ns_transient_2d
                    required_keys = ["resolution", "cfl", "relaxation_factor", "residual_threshold"]
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
            norm_rmse = float('inf')
            
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy, task)
            
        else:
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy, task)

            try:
                success, norm_rmse = compare_res_ns_transient_2d(
                    profile1=dummy["profile"],
                    boundary_condition1=profile_config['boundary_condition'],
                    resolution1=get_required_param(last_iter, "resolution", "current_resolution"),
                    reynolds_num1=profile_config['reynolds_num'],
                    cfl1=get_required_param(last_iter, "cfl", "current_cfl"),
                    relaxation_factor1=get_required_param(last_iter, "relaxation_factor", "current_relaxation_factor"),
                    residual_threshold1=get_required_param(last_iter, "residual_threshold", "current_residual_threshold"),
                    total_runtime1=profile_config['total_runtime'],
                    profile2=dummy["profile"],
                    boundary_condition2=profile_config['boundary_condition'],
                    resolution2=get_required_param(ref_iter, "resolution", "current_resolution"),
                    reynolds_num2=profile_config['reynolds_num'],
                    cfl2=get_required_param(ref_iter, "cfl", "current_cfl"),
                    relaxation_factor2=get_required_param(ref_iter, "relaxation_factor", "current_relaxation_factor"),
                    residual_threshold2=get_required_param(ref_iter, "residual_threshold", "current_residual_threshold"),
                    total_runtime2=profile_config['total_runtime'],
                    norm_rmse_tolerance=norm_rmse_tol,
                )
                
            except Exception as e:
                logger.warning(f"⚠️ QID {qid}: Error in success determination: {e}, marking as failed")
                success = False
                norm_rmse = float('inf')

        # Calculate metrics with robust error handling
        dummy_cost = dummy["dummy_cost"]
        
        # Calculate efficiency metrics with division by zero protection
        if cost <= 0:
            efficiency = 0.0
            logger.warning(f"⚠️ QID {qid}: Cost is {cost}, setting efficiency to 0")
        else:
            efficiency = float(success) * (dummy_cost / cost)

            # Handle NaN/inf values in efficiency calculations
            if np.isnan(efficiency) or np.isinf(efficiency):
                efficiency = 0.0
                logger.warning(f"⚠️ QID {qid}: Efficiency is NaN/inf, setting to 0")
        
        # Accumulate metrics
        total_model_cost += cost
        total_dummy_cost += dummy_cost
        success_cnt += int(success)
        converged_cnt += int(converged)
        
        # Handle norm_rmse accumulation (use 0 for NaN/inf values in totals)
        norm_rmse_for_total = 0.0 if (np.isnan(norm_rmse) or np.isinf(norm_rmse)) else norm_rmse
        total_norm_rmse += norm_rmse_for_total
        
        total_efficiency += efficiency

        logger.info(
            f"\n📊 --- Evaluation Result ---\n"
            f"🆔 QID: {qid}\n"
            f"🔄 Converged flag: {converged}\n"
            f"🎯 Success (within tolerance): {success}\n"
            f"💰 Model Cost: {cost}\n"
            f"💰 Dummy Cost: {dummy['dummy_cost']}\n"
            f"⚡ Efficiency: {efficiency:.3f}\n"
            f"📉 Norm RMSE (model vs. dummy): {norm_rmse:.3e}\n"
            f"📏 Norm RMSE Tolerance: {norm_rmse_tol:.3e}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

    # Calculate final metrics with division by zero protection
    if evaluated == 0:
        logger.warning("⚠️ No valid evaluations performed")
        success_rate = converged_rate = mean_efficiency = 0.0
        mean_norm_rmse = 0.0
    else:
        success_rate = success_cnt / evaluated
        converged_rate = converged_cnt / evaluated
        mean_efficiency = total_efficiency / evaluated
        mean_norm_rmse = total_norm_rmse / evaluated

    metrics = {
        "success_rate": f"{success_rate:.3f}",
        "converged_rate": f"{converged_rate:.3f}",
        "mean_efficiency": f"{mean_efficiency:.3f}",
        "mean_norm_rmse": f"{mean_norm_rmse:.2e}",
        "norm_rmse_tolerance": f"{norm_rmse_tol:.2e}",
        "total_model_cost": total_model_cost,
        "total_dummy_cost": total_dummy_cost,
    }

    logger.info(f"🧾 Evaluation Summary for {model_name}:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="ns_transient_2d",
                        choices=["ns_transient_2d", "ns_transient_2d_icl_accuracy_focused", "ns_transient_2d_icl_cost_excluded", "ns_transient_2d_icl_full"],
                        help="Dataset name: ns_transient_2d, ns_transient_2d_icl_accuracy_focused, ns_transient_2d_icl_cost_excluded, or ns_transient_2d_icl_full")
    parser.add_argument("-t", "--task", default="resolution",
                        choices=list(VALID_TASKS),
                        help="Task: one of the 4 NS Transient 2D tasks")
    parser.add_argument("-l", "--precision_level", default="medium",
                        choices=list(VALID_PRECISION_LEVELS),
                        help="Precision level for ns_transient_2d dataset")
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