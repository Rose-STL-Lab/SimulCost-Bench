from typing import List, Dict, Tuple
from costsci_tools.wrappers.heat_1d import compare_res_heat_1d
from costsci_tools.wrappers.heat_steady_2d import compare_res_heat_steady_2d
import sys

def evaluate(
    data: str,
    task: str,
    dummy_dataset: List[Dict],
    result_dataset: List[Dict],
    agent: Dict,
) -> Dict:
    dummy_by_qid: Dict[int, Dict] = {d["QID"]: d for d in dummy_dataset}

    total_cost = total_dummy_cost = 0.0
    success_cnt = 0
    converged_valid = 0
    converged_cnt = 0
    evaluated = 0

    for res in result_dataset:
        try:
            qid = res["QID"]
        except KeyError:
            print("\033[91m[ERROR] Missing 'QID' in result: {}\033[0m".format(res), file=sys.stderr)
            continue

        dummy = dummy_by_qid[qid]
        evaluated += 1

        cost      = res["accumulated_cost"]
        converged = res["converged"]
        last_iter = res["sequence"][-1]
        if task == "relax":
            optimal_relax = dummy["optimal_relaxation_factor"]
            ref_iter = next((param for param in dummy["param_history"] if param["relax"] == optimal_relax), None)
        elif task == "t_init":
            optimal_t_init = dummy["optimal_initial_temperature"]
            ref_iter = next((param for param in dummy["param_history"] if param["T_init"] == optimal_t_init), None)
        else:
            ref_iter  = dummy["param_history"][-2]

        if data == "1D_heat_transfer":
            success, _ = compare_res_heat_1d(
                profile1=dummy["profile"],
                cfl1=last_iter["cfl"],
                n_space1=last_iter["n_space"],
                profile2=dummy["profile"],
                cfl2=ref_iter["cfl"],
                n_space2=ref_iter["n_space"],
                tolerance=1e-4,
            )
        elif data == "2D_heat_transfer":
            if last_iter["relax"] < 0.05 or last_iter["relax"] > 1.0:
                success = False
            else:
                success, _ = compare_res_heat_steady_2d(
                    profile1=dummy["profile"],
                    dx1=last_iter["dx"],  relax1=last_iter["relax"],
                    error_threshold1=last_iter["error_threshold"], t_init1=last_iter["t_init"],
                    profile2=dummy["profile"],
                    dx2=ref_iter["dx"],   relax2=ref_iter["relax"],
                    error_threshold2=ref_iter["error_threshold"], t_init2=ref_iter.get("t_init", ref_iter["T_init"]),
                    tolerance=1e-3,
                )
        else:
            raise ValueError(f"Unsupported dataset type: {data}")

        total_cost       += cost
        total_dummy_cost += dummy["dummy_cost"]
        success_cnt      += int(success)
        converged_valid  += 1
        converged_cnt    += int(converged)

    success_rate            = success_cnt / evaluated if evaluated else 0.0
    converged_rate          = converged_cnt / converged_valid if converged_valid else 0.0
    model_cost_efficiency   = success_cnt / total_cost       if total_cost else 0.0
    dummy_cost_efficiency   = evaluated   / total_dummy_cost if total_dummy_cost else 0.0
    relative_cost_efficiency = (model_cost_efficiency / dummy_cost_efficiency
                                if dummy_cost_efficiency else 0.0)

    agent.update({
        "success_rate": success_rate,
        "converged_rate": converged_rate,
        "model_cost_efficiency": model_cost_efficiency,
        "dummy_cost_efficiency": dummy_cost_efficiency,
        "relative_cost_efficiency": relative_cost_efficiency,
    })
    return agent
