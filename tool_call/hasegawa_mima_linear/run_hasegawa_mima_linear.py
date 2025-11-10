from costsci_tools.wrappers.hasegawa_mima_linear import run_sim_hasegawa_mima_linear, get_error_metric
import numpy as np
import yaml
import os

def _to_jsonable(obj):
    """Convert numpy objects to JSON serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def _load_profile_config(profile):
    """Load configuration from profile YAML file."""
    config_path = f"costsci_tools/run_configs/hasegawa_mima_linear/{profile}.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return {
        'case': config['case'],
        'L': config['L'],
        'v_star': config['v_star'],
        'Dx': config['Dx'],
        'record_dt': config['record_dt'],
        'end_frame': config['end_frame'],
    }

def hasegawa_mima_linear_check_converge_parameter(
    *,
    accumulated_cost: int,
    profile: str,
    N: int,
    dt: float,
    cg_atol: float,
    tolerance_rmse: float,
    check_param: str,
):
    """
    Check convergence for hasegawa_mima_linear by comparing numerical vs analytical solution.

    NOTE: Unlike hasegawa_mima_nonlinear, this does NOT need refine functionality because we compare
    against the analytical solution (ground truth) rather than a refined simulation.

    Args:
        accumulated_cost: Current accumulated computational cost
        profile: Configuration profile name (e.g., 'p1', 'p2', etc.)
        N: Grid resolution
        dt: Time step size
        cg_atol: Conjugate gradient solver absolute tolerance
        tolerance_rmse: Convergence threshold for RMSE
        check_param: Which parameter we're checking ('N', 'dt', or 'cg_atol')

    Returns:
        dict: Contains convergence information including:
            - refined_parameter: Name of parameter being checked
            - current_value: Current value of the parameter
            - RMSE: Root mean square error vs analytical solution
            - is_converged: Boolean indicating if convergence criteria is met
            - accumulated_cost: Updated accumulated cost
            - The cost of the solver simulating the environment: Cost of current simulation
    """

    # Load profile configuration (optional - for reference)
    config = _load_profile_config(profile)

    print(f"\nRunning simulation to check {check_param} convergence:")
    print(f"Parameters: N={N}, dt={dt}, cg_atol={cg_atol:.2e}")

    # Run current numerical simulation
    # Note: max_wall_time is handled by the wrapper (defaults to config value of 120s)
    current_cost = run_sim_hasegawa_mima_linear(
        profile=profile,
        N=N,
        dt=dt,
        cg_atol=cg_atol,
        analytical=False  # Numerical method
    )

    accumulated_cost += current_cost

    # Construct simulation directory path
    sim_dir = f"sim_res/hasegawa_mima_linear/{profile}_N_{N}_dt_{dt:.2e}_cg_{cg_atol:.2e}_numerical"

    # Get error metric (RMSE vs analytical solution)
    # The get_error_metric function automatically compares with the cached analytical solution
    rmse = get_error_metric(sim_dir)

    # Check convergence
    if rmse is not None:
        is_converged = rmse <= tolerance_rmse
    else:
        is_converged = False

    if is_converged:
        rmse_str = f"{rmse:.6f}" if rmse is not None else "None"
        print(f"Convergence achieved for {check_param} (RMSE: {rmse_str} <= {tolerance_rmse})")
    else:
        rmse_str = f"{rmse:.6f}" if rmse is not None else "None"
        print(f"No convergence for {check_param} (RMSE: {rmse_str} > {tolerance_rmse})")

    # Get current parameter value
    param_values = {'N': N, 'dt': dt, 'cg_atol': cg_atol}
    current_value = param_values[check_param]

    return {
        "refined_parameter": check_param,
        "current_value": _to_jsonable(current_value),
        "RMSE": round(float(rmse), 6) if rmse is not None else None,
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
    }

# Convenience functions for each parameter
def hasegawa_mima_linear_check_converge_N(**kwargs):
    """Check convergence by testing N (grid resolution)."""
    return hasegawa_mima_linear_check_converge_parameter(check_param='N', **kwargs)

def hasegawa_mima_linear_check_converge_dt(**kwargs):
    """Check convergence by testing dt (time step size)."""
    return hasegawa_mima_linear_check_converge_parameter(check_param='dt', **kwargs)

def hasegawa_mima_linear_check_converge_cg_atol(**kwargs):
    """Check convergence by testing cg_atol (CG solver tolerance)."""
    return hasegawa_mima_linear_check_converge_parameter(check_param='cg_atol', **kwargs)
