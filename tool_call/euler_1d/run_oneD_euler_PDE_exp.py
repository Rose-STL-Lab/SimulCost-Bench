from costsci_tools.wrappers.euler_1d import run_sim_euler_1d, compare_res_euler_1d
import numpy as np

def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()             # list
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()               # float / int
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def euler_1d(
    *, 
    accumulated_cost: int,
    profile: str,
    current_cfl: float,
    beta: float,
    k: float,
    linf_tolerance: float,
    rmse_tolerance: float,
):
    print(f"\nRunning simulation with CFL = {current_cfl}")
    refine_cfl = current_cfl / 2

    current_cost = run_sim_euler_1d(profile=profile, cfl=current_cfl, beta=beta, k=k)
    refine_cost  = run_sim_euler_1d(profile=profile, cfl=refine_cfl,  beta=beta, k=k)

    accumulated_cost += current_cost

    (
        converged,
        metrics1,
        metrics2,
        linf_norm,
        rmse
    ) = compare_res_euler_1d(
        profile1=profile, cfl1=current_cfl, beta1=beta, k1=k,
        profile2=profile, cfl2=refine_cfl, beta2=beta, k2=k,
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