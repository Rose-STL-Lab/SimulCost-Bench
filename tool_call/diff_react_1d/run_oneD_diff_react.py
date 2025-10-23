from costsci_tools.wrappers.diff_react_1d import run_sim_diff_react_1d, compare_res_diff_react_1d
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
    config_path = f"costsci_tools/run_configs/diff_react_1d/{profile}.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return {
        'reaction_type': config['reaction_type'],
        'allee_threshold': config.get('allee_threshold', None),
    }

def _refine_parameters(params, refine_param):
    """Apply parameter refinement rules based on which parameter to refine."""
    refined_params = params.copy()

    # Apply diff_react_1d specific refinement rules
    refinement_rules = {
        'cfl': lambda x: x * 0.5,
        'n_space': lambda x: x * 2,
        'tol': lambda x: x * 0.1,
    }

    if refine_param in refinement_rules:
        refined_params[refine_param] = refinement_rules[refine_param](params[refine_param])
    else:
        raise ValueError(f"Unknown parameter to refine: {refine_param}")

    return refined_params

def diff_react_1d_check_converge_parameter(
    *,
    accumulated_cost: int,
    profile: str,
    n_space: int,
    cfl: float,
    tol: float,
    rmse_tolerance: float,
    refine_param: str,
):
    """Generic convergence check function that can refine any parameter."""

    # Use default values for fixed parameters
    min_step = 1e-6
    initial_step_guess = 1.0

    # Load profile configuration
    config = _load_profile_config(profile)
    reaction_type = config['reaction_type']
    allee_threshold = config['allee_threshold']

    # Base parameters
    base_params = {
        'n_space': n_space,
        'cfl': cfl,
        'tol': tol,
        'min_step': min_step,
        'initial_step_guess': initial_step_guess,
    }

    # Get refined parameters
    refined_params = _refine_parameters(base_params, refine_param)

    print(f"\nRunning simulation with {refine_param} refinement:")
    print(f"Current {refine_param} = {base_params[refine_param]}")
    print(f"Refined {refine_param} = {refined_params[refine_param]}")

    # Run current simulation
    current_cost = run_sim_diff_react_1d(
        profile=profile,
        n_space=base_params['n_space'],
        cfl=base_params['cfl'],
        tol=base_params['tol'],
        min_step=base_params['min_step'],
        initial_step_guess=base_params['initial_step_guess'],
        reaction_type=reaction_type,
        allee_threshold=allee_threshold
    )

    accumulated_cost += current_cost

    # Run refined simulation (convergence checking does not increase cost)
    refine_cost = run_sim_diff_react_1d(
        profile=profile,
        n_space=refined_params['n_space'],
        cfl=refined_params['cfl'],
        tol=refined_params['tol'],
        min_step=refined_params['min_step'],
        initial_step_guess=refined_params['initial_step_guess'],
        reaction_type=reaction_type,
        allee_threshold=allee_threshold
    )

    # Compare results
    converged, metrics1, metrics2, rmse = compare_res_diff_react_1d(
            profile1=profile,
            n_space1=base_params['n_space'],
            cfl1=base_params['cfl'],
            tol1=base_params['tol'],
            min_step1=base_params['min_step'],
            init_step1=base_params['initial_step_guess'],
            profile2=profile,
            n_space2=refined_params['n_space'],
            cfl2=refined_params['cfl'],
            tol2=refined_params['tol'],
            min_step2=refined_params['min_step'],
            init_step2=refined_params['initial_step_guess'],
            rmse_tolerance=rmse_tolerance,
            reaction_type1=reaction_type,
            reaction_type2=reaction_type,
            allee_threshold1=allee_threshold,
            allee_threshold2=allee_threshold
        )

    if converged:
        print(f"Convergence achieved for {refine_param} refinement")
    else:
        print(f"No convergence for {refine_param} refinement")

    return {
        "refined_parameter": refine_param,
        "current_value": _to_jsonable(base_params[refine_param]),
        "refined_value": _to_jsonable(refined_params[refine_param]),
        "RMSE": round(rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }

# Convenience functions for each parameter
def diff_react_1d_check_converge_cfl(**kwargs):
    """Check convergence by refining cfl (*0.5)."""
    return diff_react_1d_check_converge_parameter(refine_param='cfl', **kwargs)

def diff_react_1d_check_converge_n_space(**kwargs):
    """Check convergence by refining n_space (*2)."""
    return diff_react_1d_check_converge_parameter(refine_param='n_space', **kwargs)

def diff_react_1d_check_converge_tol(**kwargs):
    """Check convergence by refining tol (*0.1)."""
    return diff_react_1d_check_converge_parameter(refine_param='tol', **kwargs)
