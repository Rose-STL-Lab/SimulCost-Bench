import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from costsci_tools.wrappers.euler_1d import compare_res_euler_1d
from inference.utils import setup_logging, NumpyEncoder

# Constants
RMSE_TOLERANCE_BY_PRECISION = {
    "low": 0.08,
    "medium": 0.02,
    "high": 0.01,
}
VALID_PRECISION_LEVELS = {"low", "medium", "high"}
VALID_TASKS = {"cfl", "k", "beta", "n_space"}

# Note: validation_utils not needed for euler_1d precision_level structure

def get_reference_params(dummy: Dict) -> Dict:
    """Extract reference parameters from the dummy data"""
    return dummy["best_params"]


def get_required_param(param_dict: Dict, param_name: str, alt_name: str = None):
    """Strict parameter fetching with validation - no defaults, raise error if missing or invalid"""
    if param_name in param_dict:
        value = param_dict[param_name]
    elif alt_name and alt_name in param_dict:
        value = param_dict[alt_name]
    else:
        available_keys = list(param_dict.keys())
        raise KeyError(
            f"Required parameter '{param_name}' not found. "
            f"Available keys: {available_keys}"
        )
    
    # Validate parameter values - raise exceptions to trigger existing error handling
    if param_name in ["cfl", "current_cfl"] or (alt_name and alt_name in ["cfl", "current_cfl"]):
        if value <= 0:
            raise ValueError(f"Invalid CFL value: {value} <= 0")
    
    if param_name == "n_space" or (alt_name and alt_name == "n_space"):
        if value <= 0 or not isinstance(value, int):
            raise ValueError(f"Invalid n_space value: {value}. Must be a positive integer")
    
    return value


def evaluate(
    dataset: str,
    task: str,
    model_name: str,
    precision_level: str = "medium",
    zero_shot: bool = False,
) -> Dict:
    """Evaluate model performance against reference solutions.
    
    Args:
        dataset: Dataset name (e.g., 'euler_1d')
        task: Task type (e.g., 'cfl', 'k', 'beta', 'n_space')
        model_name: Name of the model being evaluated
        precision_level: Precision level ('low', 'medium', 'high')
        zero_shot: Whether to use zero-shot evaluation
        
    Returns:
        Dictionary containing evaluation metrics
        
    Raises:
        RuntimeError: If required files are not found or data loading fails
    """
    flag = "zero_shot" if zero_shot else "iterative"
    
    # Use precision_level-based paths for the specified dataset
    log_dir = f"eval_results/{dataset}/{task}/{precision_level}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name}.log"
    logger = setup_logging(log_file)

    # Build paths for the specified dataset with precision_level structure
    result_path = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_{model_name}.json"

    # For ICL variants, dummy solutions are still from the base euler_1d dataset
    # Only ICL examples differ between variants
    dummy_path = f"data/euler_1d/{task}/{precision_level}/{flag}_questions.json"
    
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

    # Load tool call history from Excel file (if exists)
    table_file = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_tool_call_{model_name}.xlsx"
    tool_call_df = None
    attempt_history_by_qid = {}
    if os.path.exists(table_file):
        try:
            tool_call_df = pd.read_excel(table_file)
            logger.info(f"✅ Loaded tool call history from {table_file} ({len(tool_call_df)} tool calls)")

            # Group by QID and construct attempt_history
            for qid in tool_call_df['QID'].unique():
                qid_df = tool_call_df[tool_call_df['QID'] == qid]
                attempt_list = []
                for idx, row in qid_df.iterrows():
                    attempt_dict = {
                        "attempt_number": len(attempt_list) + 1,
                        "QID": int(row['QID']),
                        "tool_name": str(row['tool_name']),
                        "tool_args": str(row['tool_args']),
                        "tool_reason": str(row['tool_reason']),
                        "RMSE": str(row['RMSE']),
                        "is_converged": str(row['is_converged']),
                        "accumulated_cost": str(row['accumulated_cost']),
                        "The cost of the solver simulating the environment": str(row['The cost of the solver simulating the environment']),
                        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": str(row['The cost of the solver verifying convergence (This will not be included in your accumulated_cost)']),
                        "metrics1": str(row['metrics1']),
                        "metrics2": str(row['metrics2'])
                    }
                    attempt_list.append(attempt_dict)
                attempt_history_by_qid[int(qid)] = attempt_list
            logger.info(f"✅ Constructed attempt_history for {len(attempt_history_by_qid)} QIDs")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load tool call history from {table_file}: {e}")
            attempt_history_by_qid = {}
    else:
        logger.warning(f"⚠️ Tool call history file not found: {table_file}")
        attempt_history_by_qid = {}

    total_model_cost = total_dummy_cost = 0.0
    success_cnt = converged_cnt = evaluated = 0
    total_rmse = 0.0
    total_efficiency = 0.0

    # Validate inputs
    if task not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid tasks: {sorted(VALID_TASKS)}")
    if precision_level not in VALID_PRECISION_LEVELS:
        raise ValueError(f"Invalid precision_level '{precision_level}'. Valid levels: {sorted(VALID_PRECISION_LEVELS)}")

    # Get precision-specific tolerance for success checking
    rmse_tol = RMSE_TOLERANCE_BY_PRECISION[precision_level]

    # Collect rows for DataFrame
    rows = []

    # Determine target and non-target parameters based on task
    # For k/beta tasks, target is k/beta + n_space
    if task == "cfl":
        target_params = ["cfl"]
        non_target_params = ["beta", "k", "n_space"]
    elif task == "k":
        target_params = ["k", "n_space"]
        non_target_params = ["cfl", "beta"]
    elif task == "beta":
        target_params = ["beta", "n_space"]
        non_target_params = ["cfl", "k"]
    elif task == "n_space":
        target_params = ["n_space"]
        non_target_params = ["cfl", "beta", "k"]
    else:
        raise ValueError(f"Unknown task: {task}")

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
            ref_iter = get_reference_params(dummy)
            
        else:
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy)

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
        
        # Handle RMSE accumulation (use 0 for NaN/inf values in totals)
        rmse_for_total = 0.0 if (np.isnan(rmse) or np.isinf(rmse)) else rmse
        total_rmse += rmse_for_total
        
        total_efficiency += efficiency

        logger.info(
            f"\n📊 --- Evaluation Result ---\n"
            f"🆔 QID: {qid}\n"
            f"🔄 Converged flag: {converged}\n"
            f"🎯 Success (within tolerance): {success}\n"
            f"💰 Model Cost: {cost}\n"
            f"💰 Dummy Cost: {dummy['dummy_cost']}\n"
            f"⚡ Efficiency: {efficiency:.3f}\n"
            f"📉 RMSE (model vs. dummy): {rmse:.3e}\n"
            f"📏 RMSE Tolerance: {rmse_tol:.3e}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

        # Extract model parameters (with None for missing values)
        try:
            model_cfl = get_required_param(last_iter, "cfl", "current_cfl") if last_iter else None
        except (KeyError, ValueError):
            model_cfl = None

        try:
            model_beta = get_required_param(last_iter, "beta") if last_iter else None
        except (KeyError, ValueError):
            model_beta = None

        try:
            model_k = get_required_param(last_iter, "k") if last_iter else None
        except (KeyError, ValueError):
            model_k = None

        try:
            model_n_space = get_required_param(last_iter, "n_space") if last_iter else None
        except (KeyError, ValueError):
            model_n_space = None

        # Extract dummy parameters (all required - no try-except)
        dummy_cfl = get_required_param(ref_iter, "cfl", "current_cfl")
        dummy_beta = get_required_param(ref_iter, "beta")
        dummy_k = get_required_param(ref_iter, "k")
        dummy_n_space = get_required_param(ref_iter, "n_space")

        # Build non_target_parameters as key-value pairs from dummy best_params
        non_target_params_dict = {}
        for param in non_target_params:
            # Map parameter names to their alternative names if needed
            param_key = param
            alt_key = "current_cfl" if param == "cfl" else None
            value = get_required_param(ref_iter, param_key, alt_key)
            non_target_params_dict[param] = value

        # Get attempt_history for this QID
        attempt_history = attempt_history_by_qid.get(qid, [])

        # Build row data
        row = {
            # Identification dimensions
            'dataset': dataset,
            'task': task,
            'precision_level': precision_level,
            'inference_mode': flag,
            'model_name': model_name,
            'qid': qid,
            'profile': json.dumps(dummy["profile"], cls=NumpyEncoder),

            # Parameter identification
            'target_parameters': ','.join(target_params),
            'non_target_parameters': json.dumps(non_target_params_dict, cls=NumpyEncoder),

            # Evaluation results
            'is_converged': converged,
            'is_successful': success,
            'model_cost': cost,
            'dummy_cost': dummy_cost,
            'rmse': rmse if not (np.isnan(rmse) or np.isinf(rmse)) else None,
            'tolerance': rmse_tol,
            'efficiency': efficiency,

            # All parameter values
            'model_cfl': model_cfl,
            'model_beta': model_beta,
            'model_k': model_k,
            'model_n_space': model_n_space,
            'dummy_cfl': dummy_cfl,
            'dummy_beta': dummy_beta,
            'dummy_k': dummy_k,
            'dummy_n_space': dummy_n_space,

            # Attempt history - complete experimental trajectory
            'attempt_history': json.dumps(attempt_history, cls=NumpyEncoder),
        }

        rows.append(row)

    # Calculate final metrics with division by zero protection
    if evaluated == 0:
        logger.warning("⚠️ No valid evaluations performed")
        success_rate = converged_rate = mean_efficiency = 0.0
        mean_rmse = 0.0
    else:
        success_rate = success_cnt / evaluated
        converged_rate = converged_cnt / evaluated
        mean_efficiency = total_efficiency / evaluated
        mean_rmse = total_rmse / evaluated

    metrics = {
        "success_rate": f"{success_rate:.3f}",
        "converged_rate": f"{converged_rate:.3f}",
        "mean_efficiency": f"{mean_efficiency:.3f}",
        "mean_rmse": f"{mean_rmse:.2e}",
        "rmse_tolerance": f"{rmse_tol:.2e}",
        "total_model_cost": total_model_cost,
        "total_dummy_cost": total_dummy_cost,
    }

    logger.info(f"🧾 Evaluation Summary for {model_name}:\n" + json.dumps(metrics, indent=2, ensure_ascii=False))

    # Create and save DataFrame
    df_new = pd.DataFrame(rows)

    if len(df_new) > 0:
        # Save as Parquet with append logic
        df_dir = f"eval_results/{dataset}/dataframes"
        os.makedirs(df_dir, exist_ok=True)
        parquet_path = f"{df_dir}/{flag}_{model_name}.parquet"

        # Check if file exists and append if it does
        if os.path.exists(parquet_path):
            df_existing = pd.read_parquet(parquet_path)

            # Remove any existing rows with the same (task, precision_level, qid) to avoid duplicates
            df_existing = df_existing[
                ~((df_existing['task'] == task) &
                  (df_existing['precision_level'] == precision_level))
            ]

            # Concatenate and save
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(parquet_path, index=False)
            logger.info(f"✅ Appended DataFrame to: {parquet_path} (new shape: {df_combined.shape}, added {len(df_new)} rows)")
        else:
            # Create new file
            df_new.to_parquet(parquet_path, index=False)
            logger.info(f"✅ Created DataFrame at: {parquet_path} (shape: {df_new.shape})")
    else:
        logger.warning("⚠️ No data to save to DataFrame")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="euler_1d",
                        choices=["euler_1d", "euler_1d_icl_accuracy_focused", "euler_1d_icl_cost_excluded", "euler_1d_icl_full"],
                        help="Dataset name: euler_1d (standard) or euler_1d_icl_* (with ICL examples)")
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