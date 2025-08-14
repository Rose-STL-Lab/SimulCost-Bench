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

def euler_1d_check_converge_cfl(
    *, 
    accumulated_cost: int,
    profile: str,
    cfl: float,
    beta: float,
    k: float,
    n_space: int,
    rmse_tolerance: float,
):
    """Check convergence by refining CFL (halving it) while keeping other parameters fixed."""
    print(f"\nRunning CFL convergence test with CFL = {cfl}")
    refine_cfl = cfl / 2

    current_cost = run_sim_euler_1d(profile=profile, cfl=cfl, beta=beta, k=k, n_space=n_space)
    refine_cost = run_sim_euler_1d(profile=profile, cfl=refine_cfl, beta=beta, k=k, n_space=n_space)

    accumulated_cost += current_cost

    (
        converged,
        metrics1,
        metrics2,
        rmse
    ) = compare_res_euler_1d(
        profile1=profile, cfl1=cfl, beta1=beta, k1=k, n_space1=n_space,
        profile2=profile, cfl2=refine_cfl, beta2=beta, k2=k, n_space2=n_space,
        rmse_tolerance=rmse_tolerance
    )

    return {
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "metrics1": _to_jsonable(metrics1),
        "metrics2": _to_jsonable(metrics2)
    }

def euler_1d_check_converge_n_space(
    *, 
    accumulated_cost: int,
    profile: str,
    cfl: float,
    beta: float,
    k: float,
    n_space: int,
    rmse_tolerance: float,
):
    """Check convergence by refining n_space (doubling it) while keeping other parameters fixed."""
    print(f"\nRunning n_space convergence test with n_space = {n_space}")
    refine_n_space = n_space * 2

    current_cost = run_sim_euler_1d(profile=profile, cfl=cfl, beta=beta, k=k, n_space=n_space)
    refine_cost = run_sim_euler_1d(profile=profile, cfl=cfl, beta=beta, k=k, n_space=refine_n_space)

    accumulated_cost += current_cost

    (
        converged,
        metrics1,
        metrics2,
        rmse
    ) = compare_res_euler_1d(
        profile1=profile, cfl1=cfl, beta1=beta, k1=k, n_space1=n_space,
        profile2=profile, cfl2=cfl, beta2=beta, k2=k, n_space2=refine_n_space,
        rmse_tolerance=rmse_tolerance
    )

    return {
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "metrics1": _to_jsonable(metrics1),
        "metrics2": _to_jsonable(metrics2)
    }

def euler_1d_check_converge_beta(
    *, 
    accumulated_cost: int,
    profile: str,
    cfl: float,
    beta: float,
    k: float,
    n_space: int,
    rmse_tolerance: float,
):
    """Check convergence by refining n_space (doubling it) for beta task while keeping other parameters fixed."""
    print(f"\nRunning beta convergence test with beta = {beta}, n_space = {n_space}")
    refine_n_space = n_space * 2

    current_cost = run_sim_euler_1d(profile=profile, cfl=cfl, beta=beta, k=k, n_space=n_space)
    refine_cost = run_sim_euler_1d(profile=profile, cfl=cfl, beta=beta, k=k, n_space=refine_n_space)

    accumulated_cost += current_cost

    (
        converged,
        metrics1,
        metrics2,
        rmse
    ) = compare_res_euler_1d(
        profile1=profile, cfl1=cfl, beta1=beta, k1=k, n_space1=n_space,
        profile2=profile, cfl2=cfl, beta2=beta, k2=k, n_space2=refine_n_space,
        rmse_tolerance=rmse_tolerance
    )

    return {
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "metrics1": _to_jsonable(metrics1),
        "metrics2": _to_jsonable(metrics2)
    }

def euler_1d_check_converge_k(
    *, 
    accumulated_cost: int,
    profile: str,
    cfl: float,
    beta: float,
    k: float,
    n_space: int,
    rmse_tolerance: float,
):
    """Check convergence by refining n_space (doubling it) for k task while keeping other parameters fixed."""
    print(f"\nRunning k convergence test with k = {k}, n_space = {n_space}")
    refine_n_space = n_space * 2

    current_cost = run_sim_euler_1d(profile=profile, cfl=cfl, beta=beta, k=k, n_space=n_space)
    refine_cost = run_sim_euler_1d(profile=profile, cfl=cfl, beta=beta, k=k, n_space=refine_n_space)

    accumulated_cost += current_cost

    (
        converged,
        metrics1,
        metrics2,
        rmse
    ) = compare_res_euler_1d(
        profile1=profile, cfl1=cfl, beta1=beta, k1=k, n_space1=n_space,
        profile2=profile, cfl2=cfl, beta2=beta, k2=k, n_space2=refine_n_space,
        rmse_tolerance=rmse_tolerance
    )

    return {
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "metrics1": _to_jsonable(metrics1),
        "metrics2": _to_jsonable(metrics2)
    }

