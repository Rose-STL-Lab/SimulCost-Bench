import sys, os, json, logging
import numpy as np
from typing import List, Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from costsci_tools.wrappers.euler_1d import compare_res_euler_1d
from inference.utils import setup_logging, NumpyEncoder
# Note: validation_utils not needed for euler_1d precision_level structure
# from utils.param_compatibility import fetch_param  # No longer used

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


def get_reference_params(task: str, dummy: Dict) -> Dict:
    """Extract reference parameters based on task type"""
    if task == "cfl":
        return dummy["param_history"][-1]
    elif task == "k":
        best_k = dummy["best_k"]
        ref_seq = next(seq for seq in dummy["param_history"] if seq[-1]["k"] == best_k)
        return ref_seq[-1]
    elif task == "beta":
        best_beta = dummy["best_beta"]
        ref_seq = next(seq for seq in dummy["param_history"] if seq[-1]["beta"] == best_beta)
        return ref_seq[-1]
    elif task == "n_space":
        return dummy["best_params"]
    else:
        raise ValueError(f"Unsupported task: {task}")


def get_required_param(param_dict: Dict, param_name: str, alt_name: str = None):
    """Strict parameter fetching - no defaults, raise error if missing"""
    if param_name in param_dict:
        return param_dict[param_name]
    elif alt_name and alt_name in param_dict:
        return param_dict[alt_name]
    else:
        raise KeyError(f"Required parameter '{param_name}' not found in {param_dict}")


def evaluate(
    dataset: str,
    task: str,
    model_name: str,
    precision_level: str = "medium",
    zero_shot: bool = False,
) -> Dict:
    flag = "zero_shot" if zero_shot else "iterative"
    
    # Use precision_level-based paths for euler_1d
    log_dir = f"eval_results/{dataset}/{task}/{precision_level}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name}.log"
    logger = setup_logging(log_file)
    
    # Build paths for euler_1d precision_level structure
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
        with open(dummy_path, 'r', encoding='utf-8') as f:
            dummy_dataset = json.load(f)
    except Exception as e:
        error_msg = f"Error loading data files: {e}"
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

    # Fixed tolerance for success checking
    rmse_tol = 0.01

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
                    # Check if this parameter set has the required keys for euler_1d
                    has_cfl = any(k in candidate for k in ["cfl", "current_cfl"])
                    has_beta = "beta" in candidate
                    has_k = "k" in candidate
                    has_n_space = "n_space" in candidate
                    has_valid_params = has_cfl and has_beta and has_k and has_n_space
                    
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
            ref_iter = get_reference_params(task, dummy)
            
        else:
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(task, dummy)

            try:
                
                success, _, _, rmse = compare_res_euler_1d(
                    profile1=dummy["profile"],
                    cfl1=get_required_param(last_iter, "cfl", "current_cfl"),
                    beta1=get_required_param(last_iter, "beta"),
                    k1=get_required_param(last_iter, "k"),
                    profile2=dummy["profile"],
                    cfl2=get_required_param(ref_iter, "cfl", "current_cfl"),
                    beta2=get_required_param(ref_iter, "beta"),
                    k2=get_required_param(ref_iter, "k"),
                    rmse_tolerance=rmse_tol,
                    n_space1=get_required_param(last_iter, "n_space"),
                    n_space2=get_required_param(ref_iter, "n_space"),
                )
                
                # Only use rmse now, remove linf_norm concept
            except Exception as e:
                logger.warning(f"⚠️ QID {qid}: Error in success determination: {e}, marking as failed")
                success = False
                rmse = float('inf')

        # Calculate efficiency using soft success: soft_success * (dummy_cost / model_cost)
        # Use RMSE metric for soft success calculation
        
        # Handle NaN values in rmse: if rmse is NaN or infinite, set soft success to 0
        if np.isnan(rmse) or np.isinf(rmse):
            soft_success_value = 0.0
            logger.warning(f"⚠️ QID {qid}: RMSE ({rmse}) is NaN/inf, setting soft success to 0")
        else:
            soft_success_value = soft_success(rmse, rmse_tol)
            
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
        success_cnt      += int(success)
        converged_cnt    += int(converged)
        # Always add error metrics to totals (including NaN/inf) for diagnostic purposes
        total_rmse += rmse
        # Only add valid efficiency and soft_success values (NaN/inf already converted to 0)
        total_efficiency += efficiency
        total_hard_efficiency += hard_efficiency
        total_soft_success += soft_success_value

        logger.info(
            f"\n📊 --- Evaluation Result ---\n"
            f"🆔 QID: {qid}\n"
            f"🔄 Converged flag: {converged}\n"
            f"🎯 Success (within tolerance): {success}\n"
            f"💰 Model Cost: {cost}\n"
            f"💰 Dummy Cost: {dummy['dummy_cost']}\n"
            f"⚡ Efficiency: {efficiency:.3f}\n"
            f"⚡ Hard Efficiency: {hard_efficiency:.3f}\n"
            f"📉 RMSE (model vs. dummy): {rmse:.3e}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

    success_rate      = success_cnt   / evaluated        if evaluated else 0.0
    converged_rate    = converged_cnt / evaluated        if evaluated else 0.0
    # model_cost_eff    = success_cnt   / total_model_cost if total_model_cost else 0.0
    # dummy_cost_eff    = evaluated     / total_dummy_cost if total_dummy_cost else 0.0
    # relative_eff      = model_cost_eff / dummy_cost_eff  if dummy_cost_eff else 0.0
    mean_efficiency   = total_efficiency / evaluated if evaluated else 0.0
    mean_hard_efficiency = total_hard_efficiency / evaluated if evaluated else 0.0
    mean_rmse         = total_rmse / evaluated if evaluated else 0.0
    mean_ss           = total_soft_success / evaluated if evaluated else 0.0

    metrics = {
        "converged_rate (converge does not guarantee success)": converged_rate,
        "success_rate": success_rate,
        # "model_cost_efficiency": f"{model_cost_eff:.2e}",
        # "dummy_cost_efficiency": f"{dummy_cost_eff:.2e}",
        # "relative_cost_efficiency": f"{relative_eff:.3f}",
        "mean_efficiency": f"{mean_efficiency:.3f}",
        "mean_hard_efficiency": f"{mean_hard_efficiency:.3f}",
        "mean_ss": f"{mean_ss:.3f}",
        "mean_RMSE": f"{mean_rmse:.2e}",
        "tolerances": f"RMSE ≤ {rmse_tol:.2e}",
    }

    logger.info("🧾 Evaluation summary:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="euler_1d",
                        help="Dataset name, e.g. euler_1d")
    parser.add_argument("-t", "--task",    default="cfl",
                        help="Task: cfl / k / beta / n_space")
    parser.add_argument("-l", "--precision_level", default="medium",
                        choices=["low", "medium", "high"],
                        help="Precision level for euler_1d dataset")
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