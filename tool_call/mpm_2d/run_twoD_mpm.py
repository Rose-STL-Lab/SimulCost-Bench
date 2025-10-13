from costsci_tools.wrappers.unstruct_mpm import run_sim_unstruct_mpm, compare_energies_unstruct_mpm
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
    """Load configuration from profile YAML file.

    Implements Fail Fast principle: strictly validates all required parameters.
    Raises KeyError with clear message if any required parameter is missing.
    """
    config_path = f"costsci_tools/run_configs/unstruct_mpm/{profile}.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Required keys for MPM configuration
    required_keys = ['case', 'envs_params']

    # Validate required top-level keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(
            f"Missing required configuration keys in {config_path}: {missing_keys}\n"
            f"Available keys: {list(config.keys())}"
        )

    # Return the configuration with explicit extraction (no defaults)
    return {
        'case': config['case'],
        'envs_params': config['envs_params'],
    }


def _refine_parameters(params, refine_param):
    """Apply parameter refinement rules based on which parameter to refine.

    Refinement rules for MPM:
    - nx: multiply by 2
    - npart: multiply by 2
    - cfl: multiply by 0.5
    """
    refined_params = params.copy()

    # Apply mpm_2d specific refinement rules
    refinement_rules = {
        'nx': lambda x: x * 2,
        'npart': lambda x: x * 2,
        'cfl': lambda x: x * 0.5,
    }

    if refine_param in refinement_rules:
        refined_params[refine_param] = refinement_rules[refine_param](params[refine_param])
    else:
        raise ValueError(
            f"Unknown parameter to refine: {refine_param}\n"
            f"Valid parameters: {list(refinement_rules.keys())}"
        )

    return refined_params


def mpm_2d_check_converge_parameter(
    *,
    accumulated_cost: int,
    profile: str,
    nx: int,
    npart: int,
    cfl: float,
    energy_tolerance: float,
    refine_param: str,
    var_threshold: float,
):
    """Generic convergence check function that can refine any parameter.

    Args:
        accumulated_cost: Current accumulated computational cost
        profile: Configuration profile name (p1, p2, p3)
        nx: Grid resolution parameter
        npart: Number of particles per cell
        cfl: CFL number for time stepping
        energy_tolerance: Tolerance for energy comparison
        refine_param: Which parameter to refine ('nx', 'npart', or 'cfl')
        var_threshold: Threshold for energy variation check (default: 0.01)

    Returns:
        dict: Results containing convergence status, costs, and metrics
    """

    # Load profile configuration
    config = _load_profile_config(profile)
    case = config['case']

    # Base parameters
    base_params = {
        'nx': nx,
        'npart': npart,
        'cfl': cfl,
    }

    # Get refined parameters
    refined_params = _refine_parameters(base_params, refine_param)

    print(f"\nRunning MPM simulation with {refine_param} refinement:")
    print(f"Current {refine_param} = {base_params[refine_param]}")
    print(f"Refined {refine_param} = {refined_params[refine_param]}")

    # Run current simulation
    current_cost, _ = run_sim_unstruct_mpm(
        profile=profile,
        nx=base_params['nx'],
        n_part=base_params['npart'],
        cfl=base_params['cfl'],
        case=case
    )

    accumulated_cost += current_cost

    # Run refined simulation (convergence checking does not increase cost)
    refine_cost, _ = run_sim_unstruct_mpm(
        profile=profile,
        nx=refined_params['nx'],
        n_part=refined_params['npart'],
        cfl=refined_params['cfl'],
        case=case
    )

    # Compare energy results
    converged, metrics1, metrics2, avg_energy_diff = compare_energies_unstruct_mpm(
        profile1=profile,
        nx1=base_params['nx'],
        n_part1=base_params['npart'],
        cfl1=base_params['cfl'],
        profile2=profile,
        nx2=refined_params['nx'],
        n_part2=refined_params['npart'],
        cfl2=refined_params['cfl'],
        case1=case,
        case2=case,
        energy_tolerance=energy_tolerance,
        var_threshold=var_threshold
    )

    if converged:
        print(f"Convergence achieved for {refine_param} refinement")
    else:
        print(f"No convergence for {refine_param} refinement")

    return {
        "refined_parameter": refine_param,
        "current_value": _to_jsonable(base_params[refine_param]),
        "refined_value": _to_jsonable(refined_params[refine_param]),
        "avg_energy_diff": round(avg_energy_diff, 6) if not np.isinf(avg_energy_diff) else float('inf'),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "current_energy_metrics": _to_jsonable(metrics1),
        "refine_energy_metrics": _to_jsonable(metrics2),
    }


# Convenience functions for each parameter
def mpm_2d_check_converge_nx(**kwargs):
    """Check convergence by refining nx (*2)."""
    return mpm_2d_check_converge_parameter(refine_param='nx', **kwargs)


def mpm_2d_check_converge_npart(**kwargs):
    """Check convergence by refining npart (*2)."""
    return mpm_2d_check_converge_parameter(refine_param='npart', **kwargs)


def mpm_2d_check_converge_cfl(**kwargs):
    """Check convergence by refining cfl (*0.5)."""
    return mpm_2d_check_converge_parameter(refine_param='cfl', **kwargs)
