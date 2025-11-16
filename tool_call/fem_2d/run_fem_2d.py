from costsci_tools.wrappers.fem2d import get_fem2d_data, compare_energies_fem2d
import numpy as np


def _to_jsonable(obj):
    """Convert numpy objects to JSON serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _refine_parameters(params, refine_param):
    """Apply parameter refinement rules based on which parameter to refine.

    Refinement rules for FEM2D:
    - dx: multiply by 0.5 (finer grid)
    - cfl: multiply by 0.5 (smaller time step)
    """
    refined_params = params.copy()

    # Apply fem_2d specific refinement rules
    refinement_rules = {
        'dx': lambda x: x * 0.5,
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


def fem_2d_check_converge_parameter(
    *,
    accumulated_cost: int,
    profile: str,
    dx: float,
    cfl: float,
    energy_tolerance: float,
    var_threshold: float,
    refine_param: str,
):
    """Generic convergence check function that can refine any parameter.

    Args:
        accumulated_cost: Current accumulated computational cost
        profile: Configuration profile name (p1, p2, p3, p4, p5)
        dx: Grid resolution parameter
        cfl: CFL number for time stepping
        energy_tolerance: Tolerance for L2 norm energy comparison
        var_threshold: Threshold for energy conservation check
        refine_param: Which parameter to refine ('dx' or 'cfl')

    Returns:
        dict: Results containing convergence status, costs, and metrics
    """

    # Base parameters
    base_params = {
        'dx': dx,
        'cfl': cfl,
    }

    # Get refined parameters
    refined_params = _refine_parameters(base_params, refine_param)

    print(f"\nRunning FEM2D simulation with {refine_param} refinement:")
    print(f"Current {refine_param} = {base_params[refine_param]}")
    print(f"Refined {refine_param} = {refined_params[refine_param]}")

    # Run current simulation
    _, current_cost = get_fem2d_data(
        profile=profile,
        dx=base_params['dx'],
        cfl=base_params['cfl']
    )

    accumulated_cost += current_cost

    # Run refined simulation (convergence checking does not increase cost)
    _, refine_cost = get_fem2d_data(
        profile=profile,
        dx=refined_params['dx'],
        cfl=refined_params['cfl']
    )

    # Compare energy results
    converged, metrics1, metrics2, avg_energy_diff = compare_energies_fem2d(
        profile1=profile,
        dx1=base_params['dx'],
        cfl1=base_params['cfl'],
        profile2=profile,
        dx2=refined_params['dx'],
        cfl2=refined_params['cfl'],
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
def fem_2d_check_converge_dx(**kwargs):
    """Check convergence by refining dx (*0.5)."""
    return fem_2d_check_converge_parameter(refine_param='dx', **kwargs)


def fem_2d_check_converge_cfl(**kwargs):
    """Check convergence by refining cfl (*0.5)."""
    return fem_2d_check_converge_parameter(refine_param='cfl', **kwargs)
