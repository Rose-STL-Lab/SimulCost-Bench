from costsci_tools.wrappers.hasegawa_mima_nonlinear import get_results, compare_solutions
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
    config_path = f"costsci_tools/run_configs/hasegawa_mima_nonlinear/{profile}.yaml"

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

def _refine_parameters(params, refine_param):
    """Apply parameter refinement rules based on which parameter to refine."""
    refined_params = params.copy()

    # Apply hasegawa_mima_nonlinear specific refinement rules
    refinement_rules = {
        'N': lambda x: x * 2,      # Double the resolution
        'dt': lambda x: x * 0.5,   # Halve the time step
    }

    if refine_param in refinement_rules:
        refined_params[refine_param] = refinement_rules[refine_param](params[refine_param])
    else:
        raise ValueError(f"Unknown parameter to refine: {refine_param}")

    return refined_params

def hasegawa_mima_nonlinear_check_converge_parameter(
    *,
    accumulated_cost: int,
    profile: str,
    N: int,
    dt: float,
    tolerance_rmse: float,
    refine_param: str,
):
    """Generic convergence check function that can refine any parameter."""

    # Load profile configuration (optional - only needed if we use fixed params)
    config = _load_profile_config(profile)

    # Base parameters
    base_params = {
        'N': N,
        'dt': dt,
    }

    # Get refined parameters
    refined_params = _refine_parameters(base_params, refine_param)

    print(f"\nRunning simulation with {refine_param} refinement:")
    print(f"Current {refine_param} = {base_params[refine_param]}")
    print(f"Refined {refine_param} = {refined_params[refine_param]}")

    # Run current simulation
    current_cost, current_results, simulation_completed = get_results(
        profile=profile,
        N=base_params['N'],
        dt=base_params['dt']
    )

    accumulated_cost += current_cost

    # Both N and dt support iterative convergence checking
    # Run refined simulation for convergence verification (cost not included in accumulated_cost)
    is_converged, cost1, cost2, rmse_diff = compare_solutions(
        profile=profile,
        params1={'N': base_params['N'], 'dt': base_params['dt']},
        params2={'N': refined_params['N'], 'dt': refined_params['dt']},
        tolerance_rmse=tolerance_rmse
    )

    if is_converged:
        rmse_str = f"{rmse_diff:.6f}" if rmse_diff is not None else "None"
        print(f"Convergence achieved for {refine_param} refinement (RMSE: {rmse_str})")
    else:
        rmse_str = f"{rmse_diff:.6f}" if rmse_diff is not None else "None"
        print(f"No convergence for {refine_param} refinement (RMSE: {rmse_str})")

    return {
        "refined_parameter": refine_param,
        "current_value": _to_jsonable(base_params[refine_param]),
        "refined_value": _to_jsonable(refined_params[refine_param]),
        "norm_RMSE": round(float(rmse_diff), 6) if rmse_diff is not None else None,
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": cost2,
        "wall_time_exceeded": not simulation_completed,
    }

# Convenience functions for each parameter
def hasegawa_mima_nonlinear_check_converge_N(**kwargs):
    """Check convergence by refining N (*2)."""
    return hasegawa_mima_nonlinear_check_converge_parameter(refine_param='N', **kwargs)

def hasegawa_mima_nonlinear_check_converge_dt(**kwargs):
    """Check convergence by refining dt (*0.5)."""
    return hasegawa_mima_nonlinear_check_converge_parameter(refine_param='dt', **kwargs)
