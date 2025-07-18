import sys, os, json, logging
from typing import List, Dict, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from costsci_tools.wrappers.euler_1d import compare_res_euler_1d
from inference.utils import setup_logging, NumpyEncoder

def _fetch_param(dic: Dict, *keys):
    """Return the value of the first existing key in the dictionary (compatible with current_xx/xx naming)"""
    for k in keys:
        if k in dic:
            return dic[k]
    raise KeyError(f"None of {keys} found in {dic}")

def evaluate(
    dataset: str,
    task: str,
    model_name: str,
    case: str = "",
    zero_shot: bool = False,
) -> Dict:
    case_prefix = f"{case}/" if case else ""
    flag = "zero_shot" if zero_shot else "iterative"

    result_path = (
        f"results_model_attempt/{dataset}/{task}/{case_prefix}{flag}_{model_name}.json"
    )
    dummy_path = f"data/{dataset}/{task}/{case_prefix}{flag}_questions.json"
    log_dir = f"eval_results/{dataset}/{task}/{case or 'default'}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{flag}_{model_name}.log"

    logger = setup_logging(log_file)
    logger.info(f"✅ Loading model results from {result_path}")
    logger.info(f"✅ Loading dummy solutions from {dummy_path}")

    try:
        with open(result_path, "r") as f:
            result_dataset: List[Dict] = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Result file not found: {result_path}")
        raise

    try:
        with open(dummy_path, "r") as f:
            dummy_dataset: List[Dict] = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ Dummy file not found: {dummy_path}")
        raise

    dummy_by_qid = {d["QID"]: d for d in dummy_dataset}

    total_model_cost = total_dummy_cost = 0.0
    success_cnt = converged_cnt = evaluated = 0
    total_linf = total_rmse = 0.0

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
        last_iter  = res["param_sequence"][-1]

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

        success, _, _, linf_norm, rmse = compare_res_euler_1d(
            profile1=dummy["profile"],
            cfl1=_fetch_param(last_iter, "cfl", "current_cfl"),
            k1=_fetch_param(last_iter, "k"),
            beta1=_fetch_param(last_iter, "beta"),
            profile2=dummy["profile"],
            cfl2=_fetch_param(ref_iter, "cfl"),
            k2=_fetch_param(ref_iter, "k"),
            beta2=_fetch_param(ref_iter, "beta"),
            linf_tolerance=linf_tol,
            rmse_tolerance=rmse_tol,
        )

        total_model_cost += cost
        total_dummy_cost += dummy["dummy_cost"]
        success_cnt      += int(success)
        converged_cnt    += int(converged)
        total_linf       += linf_norm
        total_rmse       += rmse

        logger.info(
            f"\n📊 --- Evaluation Result ---\n"
            f"🆔 QID: {qid}\n"
            f"🔄 Converged flag: {converged}\n"
            f"🎯 Success (within tolerance): {success}\n"
            f"💰 Model Cost: {cost}\n"
            f"💰 Dummy Cost: {dummy['dummy_cost']}\n"
            f"📉 Linf  (model vs. dummy): {linf_norm:.3e}\n"
            f"📉 RMSE (model vs. dummy): {rmse:.3e}\n"
            f"📌 Model Parameters:\n{json.dumps(last_iter, indent=2, cls=NumpyEncoder)}\n"
            f"📌 Dummy Parameters:\n{json.dumps(ref_iter, indent=2, cls=NumpyEncoder)}\n"
            f"------------------------------"
        )

    success_rate      = success_cnt   / evaluated        if evaluated else 0.0
    converged_rate    = converged_cnt / evaluated        if evaluated else 0.0
    model_cost_eff    = success_cnt   / total_model_cost if total_model_cost else 0.0
    dummy_cost_eff    = evaluated     / total_dummy_cost if total_dummy_cost else 0.0
    relative_eff      = model_cost_eff / dummy_cost_eff  if dummy_cost_eff else 0.0
    mean_linf         = total_linf / evaluated if evaluated else 0.0
    mean_rmse         = total_rmse / evaluated if evaluated else 0.0

    metrics = {
        "converged_rate (converge does not guarantee success)": converged_rate,
        "success_rate": success_rate,
        "model_cost_efficiency": f"{model_cost_eff:.2e}",
        "dummy_cost_efficiency": f"{dummy_cost_eff:.2e}",
        "relative_cost_efficiency": f"{relative_eff:.3f}",
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