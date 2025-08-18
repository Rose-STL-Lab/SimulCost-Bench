from costsci_tools.wrappers.burgers_1d import run_sim_burgers_1d, compare_res_burgers_1d
import numpy as np

def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()             # → list
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()               # → float / int
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def burgers_1d_solve(
    *, 
    accumulated_cost: int,
    profile: str,
    cfl: float,
    k: float,
    w: float,
    linf_tolerance: float,
    rmse_tolerance: float,
):
    print(f"\nRunning simulation with CFL = {cfl}")
    refine_cfl = cfl / 2

    current_cost = run_sim_burgers_1d(profile=profile, cfl=cfl, k=k, w=w)
    refine_cost  = run_sim_burgers_1d(profile=profile, cfl=refine_cfl,  k=k, w=w)

    accumulated_cost += current_cost

    (
        converged,
        metrics1,
        metrics2,
        linf_norm,
        rmse
    ) = compare_res_burgers_1d(
        profile1=profile, cfl1=cfl, k1=k, w1=w,
        profile2=profile, cfl2=refine_cfl, k2=k, w2=w,
        linf_tolerance=linf_tolerance, rmse_tolerance=rmse_tolerance
    )

    return {
        "Linf_Norm": round(linf_norm, 6),
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "metrics1": _to_jsonable(metrics1),
        "metrics2": _to_jsonable(metrics2)
    }
