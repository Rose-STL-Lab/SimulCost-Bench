from costsci_tools.wrappers.ns_transient_2d import run_sim_ns_transient_2d, compare_res_ns_transient_2d
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
    config_path = f"costsci_tools/run_configs/ns_transient_2d/{profile}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return {
        'boundary_condition': config['boundary_condition'],
        'reynolds_num': config['reynolds_num'],
        'vorticity_confinement': config['vorticity_confinement'],
        'total_runtime': config['total_runtime'],
        'no_dye': config['no_dye'],
        'cpu': config['cpu'],
        'visualization': config['visualization'],
        'advection_scheme': config['advection_scheme'],
    }

def _refine_parameters(params, refine_param):
    """Apply parameter refinement rules based on which parameter to refine."""
    refined_params = params.copy()
    
    # Apply ns_transient_2d specific refinement rules
    refinement_rules = {
        'resolution': lambda x: x * 2,
        'cfl': lambda x: x * 0.5,
        'relaxation_factor': lambda x: x + 0.1,
        'residual_threshold': lambda x: x * 0.1,
    }
    
    if refine_param in refinement_rules:
        refined_params[refine_param] = refinement_rules[refine_param](params[refine_param])
    else:
        raise ValueError(f"Unknown parameter to refine: {refine_param}")
    
    return refined_params

def ns_transient_2d_check_converge_parameter(
    *,
    accumulated_cost: int,
    profile: str,
    resolution: int,
    cfl: float,
    relaxation_factor: float,
    residual_threshold: float,
    norm_rmse_tolerance: float,
    refine_param: str,
):
    """Generic convergence check function that can refine any parameter."""
    
    # Load profile configuration
    config = _load_profile_config(profile)
    boundary_condition = config['boundary_condition']
    reynolds_num = config['reynolds_num']
    vorticity_confinement = config['vorticity_confinement']
    total_runtime = config['total_runtime']
    no_dye = config['no_dye']
    cpu = config['cpu']
    visualization = config['visualization']
    advection_scheme = config['advection_scheme']
    
    # Base parameters
    base_params = {
        'resolution': resolution,
        'cfl': cfl,
        'relaxation_factor': relaxation_factor,
        'residual_threshold': residual_threshold,
    }
    
    # Get refined parameters
    refined_params = _refine_parameters(base_params, refine_param)
    
    print(f"\nRunning simulation with {refine_param} refinement:")
    print(f"Current {refine_param} = {base_params[refine_param]}")
    print(f"Refined {refine_param} = {refined_params[refine_param]}")

    # Run current simulation
    current_cost, current_num_steps = run_sim_ns_transient_2d(
        profile=profile,
        boundary_condition=boundary_condition,
        resolution=base_params['resolution'],
        reynolds_num=reynolds_num,
        cfl=base_params['cfl'],
        relaxation_factor=base_params['relaxation_factor'],
        residual_threshold=base_params['residual_threshold'],
        total_runtime=total_runtime
    )

    accumulated_cost += current_cost

    # Skip refine simulation for relaxation_factor and residual_threshold (zero-shot only)
    if refine_param in ['relaxation_factor', 'residual_threshold']:
        print(f"Skipping refine step for {refine_param} (zero-shot only)")
        converged = True
        norm_rmse = 0.0
        refine_cost = 0
        refine_num_steps = 0
    else:
        # Run refined simulation (convergence checking does not increase cost)
        refine_cost, refine_num_steps = run_sim_ns_transient_2d(
            profile=profile,
            boundary_condition=boundary_condition,
            resolution=refined_params['resolution'],
            reynolds_num=reynolds_num,
            cfl=refined_params['cfl'],
            relaxation_factor=refined_params['relaxation_factor'],
            residual_threshold=refined_params['residual_threshold'],
            total_runtime=total_runtime
        )

        # Compare results
        converged, norm_rmse = compare_res_ns_transient_2d(
            profile1=profile, 
            boundary_condition1=boundary_condition, 
            resolution1=base_params['resolution'], 
            reynolds_num1=reynolds_num, 
            cfl1=base_params['cfl'], 
            relaxation_factor1=base_params['relaxation_factor'], 
            residual_threshold1=base_params['residual_threshold'], 
            total_runtime1=total_runtime, 
            profile2=profile, 
            boundary_condition2=boundary_condition, 
            resolution2=refined_params['resolution'], 
            reynolds_num2=reynolds_num, 
            cfl2=refined_params['cfl'], 
            relaxation_factor2=refined_params['relaxation_factor'], 
            residual_threshold2=refined_params['residual_threshold'], 
            total_runtime2=total_runtime,
            norm_rmse_tolerance=norm_rmse_tolerance
        )

    if converged:
        print(f"Convergence achieved for {refine_param} refinement")
    else:
        print(f"No convergence for {refine_param} refinement")

    return {
        "refined_parameter": refine_param,
        "current_value": _to_jsonable(base_params[refine_param]),
        "refined_value": _to_jsonable(refined_params[refine_param]),
        "norm_RMSE": round(norm_rmse, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "current_num_steps": current_num_steps,
        "refine_num_steps": refine_num_steps,
    }

# Convenience functions for each parameter
def ns_transient_2d_check_converge_resolution(**kwargs):
    """Check convergence by refining resolution (*2)."""
    return ns_transient_2d_check_converge_parameter(refine_param='resolution', **kwargs)

def ns_transient_2d_check_converge_cfl(**kwargs):
    """Check convergence by refining cfl (*0.5)."""
    return ns_transient_2d_check_converge_parameter(refine_param='cfl', **kwargs)

def ns_transient_2d_check_converge_relaxation_factor(**kwargs):
    """Check convergence by refining relaxation_factor (+0.1)."""
    return ns_transient_2d_check_converge_parameter(refine_param='relaxation_factor', **kwargs)

def ns_transient_2d_check_converge_residual_threshold(**kwargs):
    """Check convergence by refining residual_threshold (*0.1)."""
    return ns_transient_2d_check_converge_parameter(refine_param='residual_threshold', **kwargs)