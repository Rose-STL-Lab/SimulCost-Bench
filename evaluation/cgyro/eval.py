import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from costsci_tools.wrappers.cgyro import get_res_cgyro
from inference.utils import setup_logging, NumpyEncoder


# CGYRO precision tolerances are eigenvalue L∞ bounds (see docs/cgyro.md).
RMSE_TOLERANCE_BY_PRECISION = {
    "low": 1e-3,
    "medium": 1e-4,
    "high": 1e-5,
}
VALID_PRECISION_LEVELS = {"low", "medium", "high"}
VALID_TASKS = {"n_radial", "n_theta", "n_xi", "n_energy", "freq_tol", "delta_t"}

_ERROR_TOL = 1e-4  # matches dataset/cgyro/successful/tasks.json


def get_reference_params(dummy: Dict) -> Dict:
    return dummy["best_params"]


def get_required_param(param_dict: Dict, param_name: str):
    if param_name not in param_dict:
        raise KeyError(
            f"Required parameter '{param_name}' not found. "
            f"Available keys: {list(param_dict.keys())}"
        )
    value = param_dict[param_name]
    if param_name in ("n_radial", "n_theta", "n_xi", "n_energy"):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Invalid {param_name} value: {value}. Must be a positive integer")
    elif param_name in ("freq_tol", "delta_t"):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Invalid {param_name} value: {value}. Must be a positive number")
    return value


def _final_eigenvalue(profile, params):
    """Load CGYRO results and return the final-step complex eigenvalue."""
    res, _, _ = get_res_cgyro(
        profile,
        int(params["n_radial"]), int(params["n_theta"]), _ERROR_TOL,
        float(params["freq_tol"]), float(params["delta_t"]),
        int(params["n_xi"]), int(params["n_energy"]),
    )
    if not res or "eigenvalues" not in res:
        return None
    eig = np.asarray(res["eigenvalues"]).squeeze()
    if eig.size == 0:
        return None
    return eig.flat[-1]


def _compare_eigenvalues(profile, params_model, params_ref, tolerance):
    """Return (success, l_inf_error). success = both |Δω| and |Δγ| within tolerance."""
    eig_model = _final_eigenvalue(profile, params_model)
    eig_ref = _final_eigenvalue(profile, params_ref)
    if eig_model is None or eig_ref is None:
        return False, float("inf")
    diff_real = abs(float(eig_model.real) - float(eig_ref.real))
    diff_imag = abs(float(eig_model.imag) - float(eig_ref.imag))
    err = max(diff_real, diff_imag)
    return err < tolerance, err


def evaluate(dataset: str, task: str, model_name: str, precision_level: str = "medium", zero_shot: bool = False) -> Dict:
    flag = "zero_shot" if zero_shot else "iterative"
    model_name_safe = model_name.replace(":", "_")

    log_dir = f"eval_results/{dataset}/{task}/{precision_level}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name_safe}.log"
    logger = setup_logging(log_file)

    result_path_original = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_{model_name}.json"
    result_path_safe = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_{model_name_safe}.json"
    if os.path.exists(result_path_original):
        result_path = result_path_original
    elif os.path.exists(result_path_safe):
        result_path = result_path_safe
    else:
        error_msg = f"Model results file not found. Tried:\n  - {result_path_original}\n  - {result_path_safe}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    dummy_path = f"data/{dataset}/{task}/{precision_level}/{flag}_questions.json"
    if not os.path.exists(dummy_path):
        error_msg = f"Reference data file not found: {dummy_path}"
        logger.error(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    with open(result_path, 'r', encoding='utf-8') as f:
        result_dataset = json.load(f)
    with open(dummy_path, 'r', encoding='utf-8') as f:
        dummy_dataset = json.load(f)

    logger.info(f"✅ Loaded model results from {result_path} ({len(result_dataset)} entries)")
    logger.info(f"✅ Loaded dummy solutions from {dummy_path} ({len(dummy_dataset)} entries)")

    dummy_by_qid = {d["QID"]: d for d in dummy_dataset}

    table_file_original = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_tool_call_{model_name}.xlsx"
    table_file_safe = f"results_model_attempt/{dataset}/{precision_level}/{task}/{flag}_tool_call_{model_name_safe}.xlsx"
    table_file = table_file_original if os.path.exists(table_file_original) else table_file_safe if os.path.exists(table_file_safe) else None

    attempt_history_by_qid = {}
    if table_file and os.path.exists(table_file):
        try:
            tool_call_df = pd.read_excel(table_file)
            logger.info(f"✅ Loaded tool call history from {table_file} ({len(tool_call_df)} tool calls)")
            for qid in tool_call_df['QID'].unique():
                qid_df = tool_call_df[tool_call_df['QID'] == qid]
                attempt_list = []
                for _, row in qid_df.iterrows():
                    attempt_list.append({
                        "attempt_number": len(attempt_list) + 1,
                        "QID": int(row['QID']),
                        "tool_name": str(row['tool_name']),
                        "tool_args": str(row['tool_args']),
                        "tool_reason": str(row['tool_reason']),
                        "L2_error": str(row['L2_error']),
                        "is_converged": str(row['is_converged']),
                        "accumulated_cost": str(row['accumulated_cost']),
                        "The cost of the solver simulating the environment": str(row['The cost of the solver simulating the environment']),
                        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": str(row['The cost of the solver verifying convergence (This will not be included in your accumulated_cost)']),
                    })
                attempt_history_by_qid[int(qid)] = attempt_list
        except Exception as e:
            logger.warning(f"⚠️ Failed to load tool call history from {table_file}: {e}")

    if task not in VALID_TASKS:
        raise ValueError(f"Invalid task '{task}'. Valid tasks: {sorted(VALID_TASKS)}")
    if precision_level not in VALID_PRECISION_LEVELS:
        raise ValueError(f"Invalid precision_level '{precision_level}'. Valid: {sorted(VALID_PRECISION_LEVELS)}")
    rmse_tol = RMSE_TOLERANCE_BY_PRECISION[precision_level]

    total_model_cost = total_dummy_cost = 0.0
    success_cnt = converged_cnt = evaluated = 0
    total_rmse = total_efficiency = 0.0

    all_params = ["n_radial", "n_theta", "n_xi", "n_energy", "freq_tol", "delta_t"]
    target_params = [task]
    non_target_params = [p for p in all_params if p != task]

    rows = []
    for res in result_dataset:
        qid = res.get("QID")
        if qid is None or qid not in dummy_by_qid:
            logger.warning(f"⚠️ Skip entry without valid QID: {res}")
            continue

        dummy = dummy_by_qid[qid]
        evaluated += 1
        cost = res["accumulated_cost"]
        converged = res.get("is_converged", res.get("converged", False))

        last_iter = {}
        if res["param_sequence"]:
            for i in range(len(res["param_sequence"]) - 1, -1, -1):
                candidate = res["param_sequence"][i]
                if candidate and isinstance(candidate, dict) and all(p in candidate for p in all_params):
                    last_iter = candidate
                    break
            if not last_iter and res["param_sequence"]:
                last_iter = res["param_sequence"][-1]

        ref_iter = get_reference_params(dummy)

        if not last_iter:
            logger.warning(f"⚠️ QID {qid}: empty parameter dictionary, marking as failed")
            success = False
            rmse = float("inf")
        else:
            try:
                success, rmse = _compare_eigenvalues(
                    profile=dummy["profile"],
                    params_model={p: get_required_param(last_iter, p) for p in all_params},
                    params_ref={p: get_required_param(ref_iter, p) for p in all_params},
                    tolerance=rmse_tol,
                )
            except Exception as e:
                logger.warning(f"⚠️ QID {qid}: Error in success determination: {e}, marking as failed")
                success = False
                rmse = float("inf")

        dummy_cost = dummy["dummy_cost"]
        if cost <= 0:
            efficiency = 0.0
        else:
            efficiency = float(success) * (dummy_cost / cost)
            if np.isnan(efficiency) or np.isinf(efficiency):
                efficiency = 0.0

        total_model_cost += cost
        total_dummy_cost += dummy_cost
        success_cnt += int(success)
        converged_cnt += int(converged)
        rmse_for_total = 0.0 if (np.isnan(rmse) or np.isinf(rmse)) else rmse
        total_rmse += rmse_for_total
        total_efficiency += efficiency

        logger.info(
            f"\n📊 --- Evaluation Result ---\n"
            f"🆔 QID: {qid}\n"
            f"🔄 Converged flag: {converged}\n"
            f"🎯 Success (within tolerance): {success}\n"
            f"💰 Model Cost: {cost}\n"
            f"💰 Dummy Cost: {dummy_cost}\n"
            f"⚡ Efficiency: {efficiency:.3f}\n"
            f"📉 Eigenvalue L∞ error (model vs. dummy): {rmse:.3e}\n"
            f"📏 Tolerance: {rmse_tol:.3e}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

        model_params_vals = {}
        for p in all_params:
            try:
                model_params_vals[p] = get_required_param(last_iter, p) if last_iter else None
            except (KeyError, ValueError):
                model_params_vals[p] = None

        dummy_params_vals = {p: get_required_param(ref_iter, p) for p in all_params}
        non_target_params_dict = {p: dummy_params_vals[p] for p in non_target_params}

        row = {
            'dataset': dataset,
            'task': task,
            'precision_level': precision_level,
            'inference_mode': flag,
            'model_name': model_name,
            'qid': qid,
            'profile': json.dumps(dummy["profile"], cls=NumpyEncoder),
            'target_parameters': ','.join(target_params),
            'non_target_parameters': json.dumps(non_target_params_dict, cls=NumpyEncoder),
            'is_converged': converged,
            'is_successful': success,
            'model_cost': cost,
            'dummy_cost': dummy_cost,
            'rmse': rmse if not (np.isnan(rmse) or np.isinf(rmse)) else None,
            'tolerance': rmse_tol,
            'efficiency': efficiency,
            'attempt_history': json.dumps(attempt_history_by_qid.get(qid, []), cls=NumpyEncoder),
        }
        for p in all_params:
            row[f'model_{p}'] = model_params_vals[p]
            row[f'dummy_{p}'] = dummy_params_vals[p]
        rows.append(row)

    if evaluated == 0:
        success_rate = converged_rate = mean_efficiency = mean_rmse = 0.0
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

    df_new = pd.DataFrame(rows)
    if len(df_new) > 0:
        df_dir = f"eval_results/{dataset}/dataframes"
        os.makedirs(df_dir, exist_ok=True)
        parquet_path = f"{df_dir}/{flag}_{model_name_safe}.parquet"
        if os.path.exists(parquet_path):
            df_existing = pd.read_parquet(parquet_path)
            df_existing = df_existing[
                ~((df_existing['task'] == task) & (df_existing['precision_level'] == precision_level))
            ]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(parquet_path, index=False)
            logger.info(f"✅ Appended DataFrame to: {parquet_path} (new shape: {df_combined.shape}, added {len(df_new)} rows)")
        else:
            df_new.to_parquet(parquet_path, index=False)
            logger.info(f"✅ Created DataFrame at: {parquet_path} (shape: {df_new.shape})")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="cgyro", help="Dataset name (e.g. cgyro)")
    parser.add_argument("-t", "--task", default="n_radial",
                        choices=sorted(VALID_TASKS), help="CGYRO task")
    parser.add_argument("-l", "--precision_level", default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("-m", "--model", default="anthropic.claude-3-5-haiku-20241022-v1:0")
    parser.add_argument("-z", "--zero_shot", action="store_true")
    args = parser.parse_args()

    evaluate(
        dataset=args.dataset,
        task=args.task,
        model_name=args.model,
        precision_level=args.precision_level,
        zero_shot=args.zero_shot,
    )
