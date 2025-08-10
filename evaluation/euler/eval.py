import sys, os, json, logging
import numpy as np
from typing import List, Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from costsci_tools.wrappers.euler_1d import compare_res_euler_1d
from inference.utils import setup_logging, NumpyEncoder
from evaluation.validation_utils import setup_robust_evaluation, print_usage_help
from utils.param_compatibility import fetch_param

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

def soft_success_multi(d_list, epsilon_list):
    """计算多个 (d, epsilon) 对的平均 Soft Success 值"""
    ss_values = []
    for d, eps in zip(d_list, epsilon_list):
        ss = soft_success(d, eps)
        ss_values.append(ss)
    
    return np.mean(ss_values)  # 算术平均


def evaluate(
    dataset: str,
    task: str,
    model_name: str,
    case: str = "",
    zero_shot: bool = False,
) -> Dict:
    case_prefix = f"{case}/" if case else ""
    flag = "zero_shot" if zero_shot else "iterative"
    
    log_dir = f"eval_results/{dataset}/{task}/{case or 'default'}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name}.log"
    logger = setup_logging(log_file)
    
    # Robust validation and file loading
    success, info = setup_robust_evaluation(dataset, task, model_name, case, zero_shot, logger)
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

    total_model_cost = total_dummy_cost = 0.0
    success_cnt = converged_cnt = evaluated = 0
    total_linf = total_rmse = 0.0
    total_efficiency = 0.0
    total_hard_efficiency = 0.0
    total_soft_success = 0.0

    linf_tol = 0.2     # L_infinity tolerance for euler_1d
    rmse_tol = 0.02    # L2 / RMSE tolerance for euler_1d

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
                    has_valid_params = has_cfl and has_beta and has_k
                    
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
            linf_norm = float('inf')
            rmse = float('inf')
        else:
            # --------- Select reference parameter set ---------
            if task == "cfl":
                # 1D list, take the last iteration directly
                ref_iter = dummy["param_history"][-1]
            elif task == "k":
                best_k = dummy["best_k"]
                # param_history is list[list[dict]]
                ref_seq = next(seq for seq in dummy["param_history"] if seq[-1]["k"] == best_k)
                ref_iter = ref_seq[-1]          # Last iteration under this k
            elif task == "beta":
                best_beta = dummy["best_beta"]
                # param_history is list[list[dict]]
                ref_seq = next(seq for seq in dummy["param_history"] if seq[-1]["beta"] == best_beta)
                ref_iter = ref_seq[-1]          # Last iteration under this beta
            else:
                raise ValueError(f"Unsupported task: {task}")

            try:
                # Use safe parameter fetching with reasonable defaults
                default_cfl = fetch_param(ref_iter, "cfl", "current_cfl", default=0.5)
                default_k = fetch_param(ref_iter, "k", default=0)
                default_beta = fetch_param(ref_iter, "beta", default=1.0)
                
                success, _, _, linf_norm, rmse = compare_res_euler_1d(
                    profile1=dummy["profile"],
                    cfl1=fetch_param(last_iter, "cfl", "current_cfl", default=default_cfl),
                    k1=fetch_param(last_iter, "k", default=default_k),
                    beta1=fetch_param(last_iter, "beta", default=default_beta),
                    profile2=dummy["profile"],
                    cfl2=fetch_param(ref_iter, "cfl", "current_cfl", default=default_cfl),
                    k2=fetch_param(ref_iter, "k", default=default_k),
                    beta2=fetch_param(ref_iter, "beta", default=default_beta),
                    linf_tolerance=linf_tol,
                    rmse_tolerance=rmse_tol,
                )
            except Exception as e:
                logger.warning(f"⚠️ QID {qid}: Error in success determination: {e}, marking as failed")
                success = False
                linf_norm = float('inf')
                rmse = float('inf')

        # Calculate efficiency using soft success: soft_success * (dummy_cost / model_cost)
        # Use both linf and rmse metrics for soft success calculation
        
        # Handle NaN values in linf_norm and rmse: if either is NaN or infinite, set soft success to 0
        if np.isnan(linf_norm) or np.isinf(linf_norm) or np.isnan(rmse) or np.isinf(rmse):
            soft_success_value = 0.0
            logger.warning(f"⚠️ QID {qid}: Linf ({linf_norm}) or RMSE ({rmse}) is NaN/inf, setting soft success to 0")
        else:
            soft_success_value = soft_success_multi([linf_norm, rmse], [linf_tol, rmse_tol])
            
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
        total_linf += linf_norm
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
            f"📉 Linf  (model vs. dummy): {linf_norm:.3e}\n"
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
    mean_linf         = total_linf / evaluated if evaluated else 0.0
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
        "mean_Linf": f"{mean_linf:.2e}",
        "mean_RMSE": f"{mean_rmse:.2e}",
        "tolerances": f"Linf ≤ {linf_tol:.2e}, RMSE ≤ {rmse_tol:.2e}",
    }

    logger.info("🧾 Evaluation summary:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="euler_1d",
                        help="Dataset name, e.g. euler_1d")
    parser.add_argument("-t", "--task",    default="cfl",
                        help="Task: cfl / k / beta")
    parser.add_argument("-c", "--case",    default="sod",
                        help="Initial-condition case sub-folder")
    parser.add_argument("-m", "--model",
                        default="anthropic.claude-3-5-haiku-20241022-v1:0")
    parser.add_argument("-z", "--zero_shot", action="store_true",
                        help="Use zero-shot flag to locate files")
    args = parser.parse_args()

    evaluate(
        dataset=args.dataset,
        task=args.task,
        model_name=args.model,
        case=args.case,
        zero_shot=args.zero_shot,
    )