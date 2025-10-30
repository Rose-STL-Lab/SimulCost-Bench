from costsci_tools.wrappers.euler_2d import get_res_euler_2d, compare_res_euler_2d
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

def euler_2d_check_converge_cfl(
    *,
    accumulated_cost: int,
    profile: str,
    n_grid_x: int,
    cfl: float,
    cg_tolerance: float,
    rmse_tolerance: float,
):
    """Check convergence by refining CFL (halving it) while keeping other parameters fixed."""
    print(f"\nRunning CFL convergence test with CFL = {cfl}")
    refine_cfl = cfl * 0.5

    # get_res_euler_2d returns (results, cost) tuple
    _, current_cost = get_res_euler_2d(profile=profile, n_grid_x=n_grid_x, cfl=cfl, cg_tolerance=cg_tolerance)
    _, refine_cost = get_res_euler_2d(profile=profile, n_grid_x=n_grid_x, cfl=refine_cfl, cg_tolerance=cg_tolerance)

    accumulated_cost += current_cost

    # compare_res_euler_2d returns (converged, rmse) tuple
    converged, rmse = compare_res_euler_2d(
        profile1=profile, n_grid_x1_in=n_grid_x, cfl1=cfl, cg_tolerance1=cg_tolerance,
        profile2=profile, n_grid_x2_in=n_grid_x, cfl2=refine_cfl, cg_tolerance2=cg_tolerance,
        rmse_tolerance=rmse_tolerance
    )

    return {
        "refined_parameter": 'cfl',
        "current_value": cfl,
        "refined_value": refine_cfl,
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost
    }

def euler_2d_check_converge_n_grid_x(
    *,
    accumulated_cost: int,
    profile: str,
    n_grid_x: int,
    cfl: float,
    cg_tolerance: float,
    rmse_tolerance: float,
):
    """Check convergence by refining n_grid_x (doubling it) while keeping other parameters fixed."""
    print(f"\nRunning n_grid_x convergence test with n_grid_x = {n_grid_x}")
    refine_n_grid_x = n_grid_x * 2

    # get_res_euler_2d returns (results, cost) tuple
    _, current_cost = get_res_euler_2d(profile=profile, n_grid_x=n_grid_x, cfl=cfl, cg_tolerance=cg_tolerance)
    _, refine_cost = get_res_euler_2d(profile=profile, n_grid_x=refine_n_grid_x, cfl=cfl, cg_tolerance=cg_tolerance)

    accumulated_cost += current_cost

    # compare_res_euler_2d returns (converged, rmse) tuple
    converged, rmse = compare_res_euler_2d(
        profile1=profile, n_grid_x1_in=n_grid_x, cfl1=cfl, cg_tolerance1=cg_tolerance,
        profile2=profile, n_grid_x2_in=refine_n_grid_x, cfl2=cfl, cg_tolerance2=cg_tolerance,
        rmse_tolerance=rmse_tolerance
    )

    return {
        "refined_parameter": 'n_grid_x',
        "current_value": n_grid_x,
        "refined_value": refine_n_grid_x,
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost
    }

def euler_2d_check_converge_cg_tolerance(
    *,
    accumulated_cost: int,
    profile: str,
    n_grid_x: int,
    cfl: float,
    cg_tolerance: float,
    rmse_tolerance: float,
):
    """Check convergence by refining cg_tolerance (reducing it by 10x) while keeping other parameters fixed."""
    print(f"\nRunning cg_tolerance convergence test with cg_tolerance = {cg_tolerance}")
    refine_cg_tolerance = cg_tolerance * 0.1

    # get_res_euler_2d returns (results, cost) tuple
    _, current_cost = get_res_euler_2d(profile=profile, n_grid_x=n_grid_x, cfl=cfl, cg_tolerance=cg_tolerance)
    _, refine_cost = get_res_euler_2d(profile=profile, n_grid_x=n_grid_x, cfl=cfl, cg_tolerance=refine_cg_tolerance)

    accumulated_cost += current_cost

    # compare_res_euler_2d returns (converged, rmse) tuple
    converged, rmse = compare_res_euler_2d(
        profile1=profile, n_grid_x1_in=n_grid_x, cfl1=cfl, cg_tolerance1=cg_tolerance,
        profile2=profile, n_grid_x2_in=n_grid_x, cfl2=cfl, cg_tolerance2=refine_cg_tolerance,
        rmse_tolerance=rmse_tolerance
    )

    return {
        "refined_parameter": 'cg_tolerance',
        "current_value": cg_tolerance,
        "refined_value": refine_cg_tolerance,
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost
    }
