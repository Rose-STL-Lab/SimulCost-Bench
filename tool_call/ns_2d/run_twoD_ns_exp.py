from costsci_tools.wrappers.ns_channel_2d import run_sim_ns_channel_2d, compare_res_ns_channel_2d
import numpy as np
import yaml
import os

def _to_jsonable(obj):
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
    config_path = f"costsci_tools/run_configs/ns_channel_2d/{profile}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return {
        'boundary_type': config['boundary_condition'],
        'length': config['length'],
        'breadth': config['breadth'],
    }

def _refine_parameters(params, refine_param):
    """Apply parameter refinement rules based on which parameter to refine."""
    refined_params = params.copy()
    
    refinement_rules = {
        'mesh_x': lambda x: x * 2,
        'mesh_y': lambda x: x * 2,
        'omega_u': lambda x: x + 0.1,
        'omega_v': lambda x: x + 0.1,
        'omega_p': lambda x: x + 0.1,
        'diff_u_threshold': lambda x: x / 10,
        'diff_v_threshold': lambda x: x / 10,
        'res_iter_v_threshold': lambda x: x / 10,
    }
    
    if refine_param in refinement_rules:
        refined_params[refine_param] = refinement_rules[refine_param](params[refine_param])
    
    return refined_params

def ns_2d_check_converge_parameter(
    *,
    accumulated_cost: int,
    profile: str,
    mesh_x: int,
    mesh_y: int,
    omega_u: float,
    omega_v: float,
    omega_p: float,
    diff_u_threshold: float,
    diff_v_threshold: float,
    res_iter_v_threshold: float,
    mass_tolerance: float,
    u_rmse_tolerance: float,
    v_rmse_tolerance: float,
    p_rmse_tolerance: float,
    refine_param: str,
):
    """Generic convergence check function that can refine any parameter."""
    
    # Load profile configuration
    config = _load_profile_config(profile)
    boundary_type = config['boundary_type']
    length = config['length']
    breadth = config['breadth']
    
    # Base parameters
    base_params = {
        'mesh_x': mesh_x,
        'mesh_y': mesh_y,
        'omega_u': omega_u,
        'omega_v': omega_v,
        'omega_p': omega_p,
        'diff_u_threshold': diff_u_threshold,
        'diff_v_threshold': diff_v_threshold,
        'res_iter_v_threshold': res_iter_v_threshold,
    }
    
    # Get refined parameters
    refined_params = _refine_parameters(base_params, refine_param)
    
    print(f"\nRunning simulation with {refine_param} refinement:")
    print(f"Current {refine_param} = {base_params[refine_param]}")
    print(f"Refined {refine_param} = {refined_params[refine_param]}")

    # Run current simulation
    current_cost, current_num_steps = run_sim_ns_channel_2d(
        profile=profile,
        boundary_type=boundary_type,
        **base_params
    )
    
    # Run refined simulation (convergence checking does not increase cost)
    refine_cost, refine_num_steps = run_sim_ns_channel_2d(
        profile=profile,
        boundary_type=boundary_type,
        **refined_params
    )

    accumulated_cost += current_cost

    # Compare results
    (
        converged,
        rmse_u,
        rmse_v,
        rmse_p,
        mass_conserved1,
        mass_conserved2
    ) = compare_res_ns_channel_2d(
        profile1=profile, boundary_type1=boundary_type, 
        mesh_x1=base_params['mesh_x'], mesh_y1=base_params['mesh_y'], 
        omega_u1=base_params['omega_u'], omega_v1=base_params['omega_v'], omega_p1=base_params['omega_p'],
        diff_u_threshold1=base_params['diff_u_threshold'], diff_v_threshold1=base_params['diff_v_threshold'],
        res_iter_v_threshold1=base_params['res_iter_v_threshold'],
        profile2=profile, boundary_type2=boundary_type, 
        mesh_x2=refined_params['mesh_x'], mesh_y2=refined_params['mesh_y'], 
        omega_u2=refined_params['omega_u'], omega_v2=refined_params['omega_v'], omega_p2=refined_params['omega_p'],
        diff_u_threshold2=refined_params['diff_u_threshold'], diff_v_threshold2=refined_params['diff_v_threshold'],
        res_iter_v_threshold2=refined_params['res_iter_v_threshold'],
        length=length, breadth=breadth,
        mass_tolerance=mass_tolerance,
        u_rmse_tolerance=u_rmse_tolerance,
        v_rmse_tolerance=v_rmse_tolerance,
        p_rmse_tolerance=p_rmse_tolerance
    )

    if converged:
        print(f"Convergence achieved for {refine_param} refinement")
    else:
        print(f"No convergence for {refine_param} refinement")

    return {
        "refined_parameter": refine_param,
        "current_value": _to_jsonable(base_params[refine_param]),
        "refined_value": _to_jsonable(refined_params[refine_param]),
        "RMSE_u": round(rmse_u, 6),
        "RMSE_v": round(rmse_v, 6),
        "RMSE_p": round(rmse_p, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "current_num_steps": current_num_steps,
        "refine_num_steps": refine_num_steps,
        "mass_conserved_current": bool(mass_conserved1),
        "mass_conserved_refined": bool(mass_conserved2)
    }

# Convenience functions for each parameter
def ns_2d_check_converge_mesh_x(**kwargs):
    """Check convergence by refining mesh_x (+25)."""
    return ns_2d_check_converge_parameter(refine_param='mesh_x', **kwargs)

def ns_2d_check_converge_mesh_y(**kwargs):
    """Check convergence by refining mesh_y (+10)."""
    return ns_2d_check_converge_parameter(refine_param='mesh_y', **kwargs)

def ns_2d_check_converge_omega_u(**kwargs):
    """Check convergence by refining omega_u (+0.1)."""
    return ns_2d_check_converge_parameter(refine_param='omega_u', **kwargs)

def ns_2d_check_converge_omega_v(**kwargs):
    """Check convergence by refining omega_v (+0.1)."""
    return ns_2d_check_converge_parameter(refine_param='omega_v', **kwargs)

def ns_2d_check_converge_omega_p(**kwargs):
    """Check convergence by refining omega_p (+0.1)."""
    return ns_2d_check_converge_parameter(refine_param='omega_p', **kwargs)

def ns_2d_check_converge_diff_u_threshold(**kwargs):
    """Check convergence by refining diff_u_threshold (/10)."""
    return ns_2d_check_converge_parameter(refine_param='diff_u_threshold', **kwargs)

def ns_2d_check_converge_diff_v_threshold(**kwargs):
    """Check convergence by refining diff_v_threshold (/10)."""
    return ns_2d_check_converge_parameter(refine_param='diff_v_threshold', **kwargs)

def ns_2d_check_converge_res_iter_v_threshold(**kwargs):
    """Check convergence by refining res_iter_v_threshold (/10)."""
    return ns_2d_check_converge_parameter(refine_param='res_iter_v_threshold', **kwargs)