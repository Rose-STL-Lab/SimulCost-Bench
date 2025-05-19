import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import json
import logging
from typing import List, Dict, Tuple
from costsci_tools.wrappers.heat_1d import compare_res_heat_1d
from costsci_tools.wrappers.heat_steady_2d import compare_res_heat_steady_2d
from inference.utils import setup_logging, NumpyEncoder

def _fetch_param(dic: Dict, *keys):
    """返回字典里第一个存在的键的值（用于 dx / current_dx 兼容）"""
    for k in keys:
        if k in dic:
            return dic[k]
    raise KeyError(f"None of {keys} found in {dic}")


def evaluate(
    dataset: str,
    task: str,
    model_name: str,
    zero_shot: bool = False,
) -> Dict:
    """
    读取模型尝试结果和 dummy 解，计算成功率 / 成本效率等指标，
    并把日志写入 eval_results/{dataset}/{task}/...

    Returns
    -------
    metrics : Dict
        字段与旧版一致: success_rate, converged_rate, model_cost_efficiency,
        dummy_cost_efficiency, relative_cost_efficiency
    """
    flag = "zero_shot" if zero_shot or task in {"relax", "t_init"} else "iterative"

    # ---------- 路径 ----------
    result_path = f"results_model_attempt/{dataset}/{task}/{flag}_{model_name}.json"
    dummy_path  = f"data/{dataset}/{task}/{flag}_question.json"
    log_dir     = f"eval_results/{dataset}/{task}"
    os.makedirs(log_dir, exist_ok=True)
    log_file    = f"{log_dir}/{flag}_{model_name}.log"

    logger = setup_logging(log_file)
    logger.info(f"Loading model results from {result_path}")
    logger.info(f"Loading dummy solutions from {dummy_path}")

    # ---------- 读文件 ----------
    try:
        with open(result_path, "r") as f:
            result_dataset: List[Dict] = json.load(f)
    except FileNotFoundError:
        logger.error(f"Result file not found: {result_path}")
        raise

    try:
        with open(dummy_path, "r") as f:
            dummy_dataset: List[Dict] = json.load(f)
    except FileNotFoundError:
        logger.error(f"Dummy file not found: {dummy_path}")
        raise

    dummy_by_qid = {d["QID"]: d for d in dummy_dataset}

    # ---------- 评估 ----------
    total_model_cost = total_dummy_cost = 0.0
    success_cnt = 0
    converged_valid = converged_cnt = evaluated = 0

    for res in result_dataset:
        qid = res.get("QID")
        if qid is None or qid not in dummy_by_qid:
            logger.warning(f"Skip entry without valid QID: {res}")
            continue

        dummy = dummy_by_qid[qid]
        evaluated += 1

        cost = res["accumulated_cost"]
        converged = res.get("is_converged", res.get("converged", False))
        last_iter = res["param_sequence"][-1]

        # 参考解——和旧代码保持一致
        if task == "relax":
            optimal_val = dummy["optimal_relaxation_factor"]
            ref_iter = next(p for p in dummy["param_history"] if p["relax"] == optimal_val)
        elif task == "t_init":
            optimal_val = dummy["optimal_initial_temperature"]
            ref_iter = next(p for p in dummy["param_history"] if p["T_init"] == optimal_val)
        else:
            ref_iter = dummy["param_history"][-2]

        # -------- 成功判定 --------
        if dataset == "1D_heat_transfer":
            success, _ = compare_res_heat_1d(
                profile1=dummy["profile"],
                cfl1=_fetch_param(last_iter, "cfl", "current_cfl"),
                n_space1=_fetch_param(last_iter, "n_space", "current_n_space"),
                profile2=dummy["profile"],
                cfl2=ref_iter["cfl"],
                n_space2=ref_iter["n_space"],
                tolerance=1e-4,
            )
        elif dataset == "2D_heat_transfer":
            success, _ = compare_res_heat_steady_2d(
                profile1=dummy["profile"],
                dx1=_fetch_param(last_iter, "dx", "current_dx"),
                relax1=_fetch_param(last_iter, "relax", "current_relax"),
                error_threshold1=_fetch_param(
                    last_iter, "error_threshold", "current_error_threshold"
                ),
                t_init1=_fetch_param(last_iter, "t_init", "current_t_init"),
                profile2=dummy["profile"],
                dx2=ref_iter["dx"],
                relax2=ref_iter["relax"],
                error_threshold2=ref_iter["error_threshold"],
                t_init2=ref_iter.get("t_init", ref_iter.get("T_init")),
                tolerance=1e-3,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {dataset}")

        # -------- 累积指标 --------
        total_model_cost += cost
        total_dummy_cost += dummy["dummy_cost"]
        success_cnt += int(success)
        converged_valid += 1
        converged_cnt += int(converged)

        logger.info(
            f"QID={qid} | success={success} | converged={converged} | "
            f"model_cost={cost} | dummy_cost={dummy['dummy_cost']}"
        )

    # ---------- 汇总 ----------
    success_rate = success_cnt / evaluated if evaluated else 0.0
    converged_rate = converged_cnt / converged_valid if converged_valid else 0.0
    model_cost_eff = success_cnt / total_model_cost if total_model_cost else 0.0
    dummy_cost_eff = evaluated / total_dummy_cost if total_dummy_cost else 0.0
    relative_eff   = model_cost_eff / dummy_cost_eff if dummy_cost_eff else 0.0

    metrics = {
        "converged_rate (converge does not guarantee success)": converged_rate,
        "success_rate": success_rate,
        "model_cost_efficiency": model_cost_eff,
        "dummy_cost_efficiency": dummy_cost_eff,
        "relative_cost_efficiency": round(relative_eff, 3),
    }

    for k in ["model_cost_efficiency", "dummy_cost_efficiency"]:
        if k in metrics:
            metrics[k] = f"{metrics[k]:.2e}"

    logger.info("Evaluation summary:\n" + json.dumps(metrics, indent=2))

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
