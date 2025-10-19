import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from costsci_tools.wrappers.unstruct_mpm import compare_energies_unstruct_mpm
from inference.utils import setup_logging, NumpyEncoder

# Constants - MPM 2D tolerance values by precision level
MPM_2D_TOLERANCE_BY_PRECISION = {
    "low": {
        "energy_tolerance": 0.03,
        "var_threshold": 0.05,
    },
    "medium": {
        "energy_tolerance": 0.01,
        "var_threshold": 0.02,
    },
    "high": {
        "energy_tolerance": 0.003,
        "var_threshold": 0.015,
    }
}

VALID_PRECISION_LEVELS = {"low", "medium", "high"}
VALID_TASKS = {"nx", "npart", "cfl"}


def load_profile_config(profile: str) -> Dict:
    """Load configuration parameters for a specific profile.

    Implements Fail Fast principle: strictly validates all required parameters.
    """
    import yaml
    config_path = f"costsci_tools/run_configs/unstruct_mpm/{profile}.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check for required parameters and raise error if missing
    required_params = ['case', 'envs_params']
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
        'case': config['case'],
        'envs_params': config['envs_params']
    }


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
    if param_name == "nx" or (alt_name and alt_name == "nx"):
        if value <= 0 or not isinstance(value, int):
            raise ValueError(f"Invalid nx value: {value}. Must be a positive integer")

    if param_name in ["npart", "n_part"] or (alt_name and alt_name in ["npart", "n_part"]):
        if value <= 0 or not isinstance(value, int):
            raise ValueError(f"Invalid npart value: {value}. Must be a positive integer")

    if param_name == "cfl" or (alt_name and alt_name == "cfl"):
        if value <= 0:
            raise ValueError(f"Invalid cfl value: {value} <= 0")

    return value


def evaluate(
    dataset: str,
    task: str,
    model_name: str,
    precision_level: str = "medium",
    zero_shot: bool = False,
) -> Dict:
    """Evaluate model performance against reference solutions for mpm_2d.

    Args:
        dataset: Dataset name ('mpm_2d')
        task: Task type (one of the 3 MPM 2D tasks: nx, npart, cfl)
        model_name: Name of the model being evaluated
        precision_level: Precision level ('low', 'medium', 'high')
        zero_shot: Whether to use zero-shot evaluation

    Returns:
        Dictionary containing evaluation metrics

    Raises:
        RuntimeError: If required files are not found or data loading fails
    """
    flag = "zero_shot" if zero_shot else "iterative"

    # Sanitize model name for filesystem (replace : with _)
    model_name_safe = model_name.replace(":", "_")

    # Use precision_level-based paths for mpm_2d
    log_dir = f"eval_results/{dataset}/{task}/{precision_level}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name_safe}.log"
    logger = setup_logging(log_file)

    # Build paths for mpm_2d precision_level structure
    # Try original model name first (with colon), then fallback to sanitized version
    result_path_original = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_{model_name}.json"
    result_path_safe = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_{model_name_safe}.json"

    result_path = None
    alt_task = None  # Initialize to avoid UnboundLocalError

    # Set up alt_task mapping early (before file search) so it's available throughout
    # Only handle npart ↔ n_part mapping (precise match only)
    if task == 'npart':
        alt_task = 'n_part'
    elif task == 'n_part':
        alt_task = 'npart'

    if os.path.exists(result_path_original):
        result_path = result_path_original
        logger.info(f"Using original model name in file path (with colon): {result_path}")
    elif os.path.exists(result_path_safe):
        result_path = result_path_safe
        logger.info(f"Using sanitized model name in file path (colon replaced): {result_path}")
    else:
        # Try alternative task naming if available
        if alt_task:
            result_path_alt_original = f"results_model_attempt/{dataset}/{precision_level}/{alt_task}/{flag}_{model_name}.json"
            result_path_alt_safe = f"results_model_attempt/{dataset}/{precision_level}/{alt_task}/{flag}_{model_name_safe}.json"

            if os.path.exists(result_path_alt_original):
                result_path = result_path_alt_original
                logger.info(f"Using alternative task naming '{alt_task}' (original: '{task}') with original model name: {result_path}")
            elif os.path.exists(result_path_alt_safe):
                result_path = result_path_alt_safe
                logger.info(f"Using alternative task naming '{alt_task}' (original: '{task}') with sanitized model name: {result_path}")

    if not result_path:
        # Build comprehensive error message
        tried_paths = [result_path_original, result_path_safe]
        if alt_task:
            tried_paths.extend([
                f"results_model_attempt/{dataset}/{precision_level}/{alt_task}/{flag}_{model_name}.json",
                f"results_model_attempt/{dataset}/{precision_level}/{alt_task}/{flag}_{model_name_safe}.json"
            ])
        error_msg = f"Model results file not found. Tried:\n" + "\n".join(f"  - {p}" for p in tried_paths)
        logger.error(f"❌ {error_msg}")
        print(f"\n❌ Evaluation failed: {error_msg}\n")
        raise RuntimeError(error_msg)

    # Try to find dummy/reference data file with task name fallback
    # For ICL variants, dummy solutions are still from the base mpm_2d dataset
    # Only ICL examples differ between variants
    dummy_path = f"data/mpm_2d/{task}/{precision_level}/{flag}_questions.json"

    if not os.path.exists(dummy_path) and alt_task:
        # Try alternative task naming for dummy_path
        dummy_path_alt = f"data/mpm_2d/{alt_task}/{precision_level}/{flag}_questions.json"
        if os.path.exists(dummy_path_alt):
            dummy_path = dummy_path_alt
            logger.info(f"Using alternative task naming '{alt_task}' for reference data (original: '{task}')")

    if not os.path.exists(dummy_path):
        tried_paths = [f"data/mpm_2d/{task}/{precision_level}/{flag}_questions.json"]
        if alt_task:
            tried_paths.append(f"data/mpm_2d/{alt_task}/{precision_level}/{flag}_questions.json")
        error_msg = f"Reference data file not found. Tried:\n" + "\n".join(f"  - {p}" for p in tried_paths)
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
    # Try original model name first (with colon), then fallback to sanitized version
    # Also handle alternative task naming (use the same alt_task from result_path logic)
    actual_task = task
    if result_path and alt_task:
        # If we used alt_task for result_path, use it for table_file too
        if alt_task in result_path:
            actual_task = alt_task
            logger.info(f"Using alternative task name '{alt_task}' for table file lookup (original: '{task}')")

    table_file_original = f"results_model_attempt/{dataset}/{precision_level}/{actual_task}/{flag}_tool_call_{model_name}.xlsx"
    table_file_safe = f"results_model_attempt/{dataset}/{precision_level}/{actual_task}/{flag}_tool_call_{model_name_safe}.xlsx"

    if os.path.exists(table_file_original):
        table_file = table_file_original
    elif os.path.exists(table_file_safe):
        table_file = table_file_safe
    else:
        table_file = None

    tool_call_df = None
    attempt_history_by_qid = {}
    if table_file and os.path.exists(table_file):
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
                        "refined_parameter": str(row['refined_parameter']),
                        "current_value": str(row['current_value']),
                        "refined_value": str(row['refined_value']),
                        "avg_energy_diff": str(row['avg_energy_diff']),
                        "is_converged": str(row['is_converged']),
                        "accumulated_cost": str(row['accumulated_cost']),
                        "The cost of the solver simulating the environment": str(row['The cost of the solver simulating the environment']),
                        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": str(row['The cost of the solver verifying convergence (This will not be included in your accumulated_cost)']),
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
    total_avg_energy_diff = 0.0
    total_efficiency = 0.0

    # Validate inputs
    if task not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid tasks: {sorted(VALID_TASKS)}")
    if precision_level not in VALID_PRECISION_LEVELS:
        raise ValueError(f"Invalid precision_level '{precision_level}'. Valid levels: {sorted(VALID_PRECISION_LEVELS)}")

    # Get precision-specific tolerances for MPM 2D
    tolerances = MPM_2D_TOLERANCE_BY_PRECISION[precision_level]
    energy_tol = tolerances["energy_tolerance"]
    var_tol = tolerances["var_threshold"]

    # Collect rows for DataFrame
    rows = []

    # Determine target and non-target parameters based on task
    if task == "nx":
        target_params = ["nx"]
        non_target_params = ["npart", "cfl"]
    elif task == "npart":
        target_params = ["npart"]
        non_target_params = ["nx", "cfl"]
    elif task == "cfl":
        target_params = ["cfl"]
        non_target_params = ["nx", "npart"]
    else:
        raise ValueError(f"Unknown task: {task}")

    for res in result_dataset:
        qid = res.get("QID")
        if qid is None or qid not in dummy_by_qid:
            logger.warning(f"⚠️ Skip entry without valid QID: {res}")
            continue

        dummy = dummy_by_qid[qid]
        evaluated += 1

        # Load profile-specific configuration
        profile_config = load_profile_config(dummy["profile"])
        case = profile_config['case']

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
                    # Check if this parameter set has the required keys for mpm_2d
                    required_keys = ["nx", "cfl"]
                    # Handle both "npart" and "n_part" variants
                    has_npart = "npart" in candidate or "n_part" in candidate
                    has_valid_params = all(key in candidate for key in required_keys) and has_npart

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
            avg_energy_diff = float('inf')

            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy)

        else:
            # --------- Select reference parameter set ---------
            ref_iter = get_reference_params(dummy)

            try:
                success, metrics1, metrics2, avg_energy_diff = compare_energies_unstruct_mpm(
                    profile1=dummy["profile"],
                    nx1=get_required_param(last_iter, "nx"),
                    n_part1=get_required_param(last_iter, "npart", "n_part"),
                    cfl1=get_required_param(last_iter, "cfl"),
                    profile2=dummy["profile"],
                    nx2=get_required_param(ref_iter, "nx"),
                    n_part2=get_required_param(ref_iter, "npart", "n_part"),
                    cfl2=get_required_param(ref_iter, "cfl"),
                    case1=case,
                    case2=case,
                    energy_tolerance=energy_tol,
                    var_threshold=var_tol,
                )

            except Exception as e:
                logger.warning(f"⚠️ QID {qid}: Error in success determination: {e}, marking as failed")
                success = False
                avg_energy_diff = float('inf')

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

        # Handle avg_energy_diff accumulation (use 0 for NaN/inf values in totals)
        energy_diff_for_total = 0.0 if (np.isnan(avg_energy_diff) or np.isinf(avg_energy_diff)) else avg_energy_diff
        total_avg_energy_diff += energy_diff_for_total

        total_efficiency += efficiency

        logger.info(
            f"\n📊 --- Evaluation Result ---\n"
            f"🆔 QID: {qid}\n"
            f"🔄 Converged flag: {converged}\n"
            f"🎯 Success (within tolerance): {success}\n"
            f"💰 Model Cost: {cost}\n"
            f"💰 Dummy Cost: {dummy['dummy_cost']}\n"
            f"⚡ Efficiency: {efficiency:.3f}\n"
            f"📉 Avg Energy Diff (model vs. dummy): {avg_energy_diff:.3e}\n"
            f"📏 Energy Tolerance: {energy_tol:.3e}\n"
            f"📏 Var Threshold: {var_tol:.3e}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

        # Extract model parameters (with None for missing values)
        try:
            model_nx = get_required_param(last_iter, "nx") if last_iter else None
        except (KeyError, ValueError):
            model_nx = None

        try:
            model_npart = get_required_param(last_iter, "npart", "n_part") if last_iter else None
        except (KeyError, ValueError):
            model_npart = None

        try:
            model_cfl = get_required_param(last_iter, "cfl") if last_iter else None
        except (KeyError, ValueError):
            model_cfl = None

        # Extract dummy parameters (all required - no try-except)
        dummy_nx = get_required_param(ref_iter, "nx")
        dummy_npart = get_required_param(ref_iter, "npart", "n_part")
        dummy_cfl = get_required_param(ref_iter, "cfl")

        # Build non_target_parameters as key-value pairs from dummy best_params
        non_target_params_dict = {}
        for param in non_target_params:
            # Handle npart/n_part variant
            if param == "npart":
                value = get_required_param(ref_iter, "npart", "n_part")
            else:
                value = get_required_param(ref_iter, param)
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
            'profile': dummy["profile"],
            'case': case,

            # Parameter identification
            'target_parameters': ','.join(target_params),
            'non_target_parameters': json.dumps(non_target_params_dict, cls=NumpyEncoder),

            # Evaluation results
            'is_converged': converged,
            'is_successful': success,
            'model_cost': cost,
            'dummy_cost': dummy_cost,
            'avg_energy_diff': avg_energy_diff if not (np.isnan(avg_energy_diff) or np.isinf(avg_energy_diff)) else None,
            'energy_tolerance': energy_tol,
            'var_threshold': var_tol,
            'efficiency': efficiency,

            # All parameter values
            'model_nx': model_nx,
            'model_npart': model_npart,
            'model_cfl': model_cfl,
            'dummy_nx': dummy_nx,
            'dummy_npart': dummy_npart,
            'dummy_cfl': dummy_cfl,

            # Attempt history - complete experimental trajectory
            'attempt_history': json.dumps(attempt_history, cls=NumpyEncoder),
        }

        rows.append(row)

    # Calculate final metrics with division by zero protection
    if evaluated == 0:
        logger.warning("⚠️ No valid evaluations performed")
        success_rate = converged_rate = mean_efficiency = 0.0
        mean_avg_energy_diff = 0.0
    else:
        success_rate = success_cnt / evaluated
        converged_rate = converged_cnt / evaluated
        mean_efficiency = total_efficiency / evaluated
        mean_avg_energy_diff = total_avg_energy_diff / evaluated

    metrics = {
        "success_rate": f"{success_rate:.3f}",
        "converged_rate": f"{converged_rate:.3f}",
        "mean_efficiency": f"{mean_efficiency:.3f}",
        "mean_avg_energy_diff": f"{mean_avg_energy_diff:.2e}",
        "energy_tolerance": f"{energy_tol:.2e}",
        "var_threshold": f"{var_tol:.2e}",
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
        parquet_path = f"{df_dir}/{flag}_{model_name_safe}.parquet"

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
    parser.add_argument("-d", "--dataset", default="mpm_2d",
                        choices=["mpm_2d", "mpm_2d_icl_accuracy_focused", "mpm_2d_icl_cost_excluded", "mpm_2d_icl_full"],
                        help="Dataset name: mpm_2d (standard) or mpm_2d_icl_* (with ICL examples)")
    parser.add_argument("-t", "--task", default="nx",
                        choices=list(VALID_TASKS),
                        help="Task: one of the 3 MPM 2D tasks")
    parser.add_argument("-l", "--precision_level", default="medium",
                        choices=list(VALID_PRECISION_LEVELS),
                        help="Precision level for mpm_2d dataset")
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
