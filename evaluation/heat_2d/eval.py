import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pdb
import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from costsci_tools.wrappers.heat_1d import compare_res_heat_1d
from costsci_tools.wrappers.heat_steady_2d import compare_res_heat_steady_2d
from inference.utils import setup_logging, NumpyEncoder
from evaluation.validation_utils import setup_robust_evaluation, print_usage_help
from utils.param_compatibility import fetch_param

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



def evaluate(
    dataset: str,
    task: str,
    model_name: str,
    zero_shot: bool = False,
) -> Dict:
    """
    Load model attempt results and dummy solutions, calculate success rate / cost efficiency metrics,
    and write logs to eval_results/{dataset}/{task}/...

    Returns
    -------
    metrics : Dict
        Fields consistent with previous version: success_rate, converged_rate, mean_efficiency
        # model_cost_efficiency, dummy_cost_efficiency, relative_cost_efficiency
    """
    flag = "zero_shot" if zero_shot or task in {"relax", "t_init"} else "iterative"
    
    log_dir = f"eval_results/{dataset}/{task}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name}.log"
    logger = setup_logging(log_file)
    
    # Robust validation and file loading
    success, info = setup_robust_evaluation(dataset, task, model_name, "", zero_shot, logger)
    if not success:
        error_msg = info["error"]
        logger.error(f"❌ Evaluation setup failed: {error_msg}")
        
        # Print helpful information to console
        print(f"\n❌ Evaluation failed: {error_msg}\n")
        if info.get("error_type") == "invalid_configuration":
            print_usage_help()
        elif info.get("error_type") == "missing_model_results":
            print_usage_help(dataset, task)
        
        raise RuntimeError(f"Evaluation setup failed: {error_msg}")
    
    result_path = info["result_path"]
    dummy_path = info["dummy_path"]
    result_dataset = info["result_data"]
    dummy_dataset = info["dummy_data"]
    
    logger.info(f"✅ Loaded model results from {result_path} ({len(result_dataset)} entries)")
    logger.info(f"✅ Loaded dummy solutions from {dummy_path} ({len(dummy_dataset)} entries)")

    dummy_by_qid = {d["QID"]: d for d in dummy_dataset}

    # ---------- Evaluation ----------
    total_model_cost = total_dummy_cost = 0.0
    success_cnt = 0
    converged_valid = converged_cnt = evaluated = 0
    total_error = 0.0
    total_efficiency = 0.0
    total_hard_efficiency = 0.0
    total_soft_success = 0.0

    for res in result_dataset:
        qid = res.get("QID")
        if qid is None or qid not in dummy_by_qid:
            logger.warning(f"⚠️ Skip entry without valid QID: {res}")
            continue

        dummy = dummy_by_qid[qid]
        evaluated += 1

        cost = res["accumulated_cost"]
        converged = res.get("is_converged", res.get("converged", False))
        
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
                    # Check if this parameter set has the required keys for this dataset
                    has_valid_params = True
                    if dataset == "1D_heat_transfer":
                        has_cfl = any(k in candidate for k in ["cfl", "current_cfl"])
                        has_n_space = any(k in candidate for k in ["n_space", "current_n_space"]) 
                        has_valid_params = has_cfl and has_n_space
                    elif dataset == "2D_heat_transfer":
                        has_dx = any(k in candidate for k in ["dx", "current_dx"])
                        has_relax = any(k in candidate for k in ["relax", "current_relax"])
                        has_error = any(k in candidate for k in ["error_threshold", "current_error_threshold"])
                        has_t_init = any(k in candidate for k in ["t_init", "current_t_init", "T_init"])
                        has_valid_params = has_dx and has_relax and has_error and has_t_init
                    
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
            tolerance=1e-4
            error = float('inf')  # Maximum error for failed attempts
            # Reference solution - consistent with legacy code
            if task == "relax":
                optimal_val = dummy["optimal_relaxation_factor"]
                ref_iter = next(p for p in dummy["param_history"] if p["relax"] == optimal_val)
            elif task == "t_init":
                optimal_val = dummy["optimal_initial_temperature"]
                ref_iter = next(p for p in dummy["param_history"] if p["T_init"] == optimal_val)
            else:
                ref_iter = dummy["param_history"][-1]
        else:
            # Reference solution - consistent with legacy code
            if task == "relax":
                optimal_val = dummy["optimal_relaxation_factor"]
                ref_iter = next(p for p in dummy["param_history"] if p["relax"] == optimal_val)
            elif task == "t_init":
                optimal_val = dummy["optimal_initial_temperature"]
                ref_iter = next(p for p in dummy["param_history"] if p["T_init"] == optimal_val)
            else:
                ref_iter = dummy["param_history"][-1]

            # -------- Success determination --------
            try:
                if dataset == "1D_heat_transfer":
                    tolerance = 1e-4
                    # Use safe parameter fetching with reasonable defaults
                    default_cfl = fetch_param(ref_iter, "cfl", "current_cfl")  # Use reference value as fallback
                    default_n_space = fetch_param(ref_iter, "n_space", "current_n_space")
                    
                    success, error = compare_res_heat_1d(
                        profile1=dummy["profile"],
                        cfl1=fetch_param(last_iter, "cfl", "current_cfl") if last_iter.get("cfl") or last_iter.get("current_cfl") else default_cfl,
                        n_space1=fetch_param(last_iter, "n_space", "current_n_space") if last_iter.get("n_space") or last_iter.get("current_n_space") else default_n_space,
                        profile2=dummy["profile"],
                        cfl2=fetch_param(ref_iter, "cfl", "current_cfl"),
                        n_space2=fetch_param(ref_iter, "n_space", "current_n_space"),
                        tolerance=tolerance,
                    )
                elif dataset == "2D_heat_transfer":
                    tolerance = 1e-3
                    # Use safe parameter fetching with reasonable defaults
                    default_dx = fetch_param(ref_iter, "dx", "current_dx")
                    default_relax = fetch_param(ref_iter, "relax", "current_relax")
                    default_error_threshold = fetch_param(ref_iter, "error_threshold", "current_error_threshold")
                    default_t_init = fetch_param(ref_iter, "t_init", "T_init", "current_t_init")
                    
                    success, error = compare_res_heat_steady_2d(
                        profile1=dummy["profile"],
                        dx1=fetch_param(last_iter, "dx", "current_dx") if last_iter.get("dx") or last_iter.get("current_dx") else default_dx,
                        relax1=fetch_param(last_iter, "relax", "current_relax") if last_iter.get("relax") or last_iter.get("current_relax") else default_relax,
                        error_threshold1=fetch_param(last_iter, "error_threshold", "current_error_threshold") if last_iter.get("error_threshold") or last_iter.get("current_error_threshold") else default_error_threshold,
                        t_init1=fetch_param(last_iter, "t_init", "current_t_init") if last_iter.get("t_init") or last_iter.get("current_t_init") else default_t_init,
                        profile2=dummy["profile"],
                        dx2=fetch_param(ref_iter, "dx", "current_dx"),
                        relax2=fetch_param(ref_iter, "relax", "current_relax"),
                        error_threshold2=fetch_param(ref_iter, "error_threshold", "current_error_threshold"),
                        t_init2=fetch_param(ref_iter, "t_init", "T_init", "current_t_init"),
                        tolerance=tolerance,
                    )
                else:
                    raise ValueError(f"Unsupported dataset type: {dataset}")
            except Exception as e:
                logger.warning(f"⚠️ QID {qid}: Error in success determination: {e}, marking as failed")
                success = False
                error = float('inf')

        # -------- Accumulate metrics --------
        # Calculate efficiency using soft success: soft_success * (dummy_cost / model_cost)
        # Use error metric for soft success calculation
        
        # Handle NaN error values: if error is NaN or infinite, set soft success to 0
        if np.isnan(error) or np.isinf(error):
            soft_success_value = 0.0
            logger.warning(f"⚠️ QID {qid}: Error value is NaN/inf ({error}), setting soft success to 0")
        else:
            soft_success_value = soft_success(error, tolerance)
            
        efficiency = soft_success_value * (dummy["dummy_cost"] / cost) if cost > 0 else 0.0
        
        # Handle NaN values in efficiency calculation
        if np.isnan(efficiency) or np.isinf(efficiency):
            efficiency = 0.0
            logger.warning(f"⚠️ QID {qid}: Efficiency is NaN/inf, setting to 0")
        
        # Calculate hard efficiency using binary success instead of soft success
        hard_efficiency = int(success) * (dummy["dummy_cost"] / cost) if cost > 0 else 0.0
        
        # Handle NaN values in hard efficiency calculation
        if np.isnan(hard_efficiency) or np.isinf(hard_efficiency):
            hard_efficiency = 0.0
            logger.warning(f"⚠️ QID {qid}: Hard efficiency is NaN/inf, setting to 0")
        
        total_model_cost += cost
        total_dummy_cost += dummy["dummy_cost"]
        success_cnt += int(success)
        converged_valid += 1
        converged_cnt += int(converged)
        # Always add error to total (including NaN/inf) for mean_error calculation
        total_error += error
        # Only add valid efficiency and soft_success values (NaN/inf already converted to 0)
        total_efficiency += efficiency
        total_hard_efficiency += hard_efficiency
        total_soft_success += soft_success_value

        logger.info(
            f"\n📊 --- Evaluation Result ---\n"
            f"🆔 QID: {qid}\n"
            f"🔄 Converged: {converged}\n"
            f"🎯 Success: {success}\n"
            f"💰 Model Cost: {cost}\n"
            f"💰 Dummy Cost: {dummy['dummy_cost']}\n"
            f"⚡ Efficiency: {efficiency:.3f}\n"
            f"⚡ Hard Efficiency: {hard_efficiency:.3f}\n"
            f"📉 Error (model vs. dummy): {error}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

    # ---------- Summary ----------
    success_rate = success_cnt / evaluated if evaluated else 0.0
    converged_rate = converged_cnt / converged_valid if converged_valid else 0.0
    # model_cost_eff = success_cnt / total_model_cost if total_model_cost else 0.0
    # dummy_cost_eff = evaluated / total_dummy_cost if total_dummy_cost else 0.0
    # relative_eff   = model_cost_eff / dummy_cost_eff if dummy_cost_eff else 0.0
    mean_efficiency = total_efficiency / evaluated if evaluated else 0.0
    mean_hard_efficiency = total_hard_efficiency / evaluated if evaluated else 0.0
    mean_error = total_error / evaluated if evaluated else 0.0
    mean_ss = total_soft_success / evaluated if evaluated else 0.0

    metrics = {
        "converged_rate (converge does not guarantee success)": converged_rate,
        "success_rate": success_rate,
        # "model_cost_efficiency": model_cost_eff,
        # "dummy_cost_efficiency": dummy_cost_eff,
        # "relative_cost_efficiency": f"{relative_eff:.3f}",
        "mean_efficiency": f"{mean_efficiency:.3f}",
        "mean_hard_efficiency": f"{mean_hard_efficiency:.3f}",
        "mean_ss": f"{mean_ss:.3f}",
        "mean_error (model vs. dummy)": f"{mean_error:.2e}",
        "tolerance": f"{tolerance:.2e}",
    }

    # for k in ["model_cost_efficiency", "dummy_cost_efficiency"]:
    #     if k in metrics:
    #         metrics[k] = f"{metrics[k]:.2e}"

    logger.info("🧾 Evaluation summary:\n" + json.dumps(metrics, indent=2))

    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="2D_heat_transfer")
    parser.add_argument("-t", "--task",    default="dx")
    parser.add_argument("-m", "--model",   default="anthropic.claude-3-5-haiku-20241022-v1:0")
    parser.add_argument("-z", "--zero_shot", action="store_true")
    args = parser.parse_args()

    evaluate(
        dataset=args.dataset,
        task=args.task,
        model_name=args.model,
        zero_shot=args.zero_shot
    )
