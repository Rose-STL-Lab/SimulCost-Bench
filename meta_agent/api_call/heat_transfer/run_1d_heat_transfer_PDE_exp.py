from typing import Dict, Any
from api_call.heat_transfer.wall_heat_transfer_solver import WallHeatTransfer, test_convergence

def run_1d_heat_transfer_PDE_exp(params: Dict[str, Any], accumulated_cost: int) -> Dict[str, Any]:
    """
    API interface for heat transfer calculations

    Args:
        params: Dictionary containing all required parameters:
            - L: Wall thickness in meters (float)
            - k: Thermal conductivity in W/(m·K) (float)
            - h: Heat transfer coefficient in W/(m²·K) (float)
            - rho: Density of wall material in kg/m³ (float)
            - cp: Specific heat capacity in J/(kg·K) (float)
            - T_inf: Ambient temperature in Celsius (float)
            - T_init: Initial temperature in Celsius (float)
            - n_space: Number of spatial points (int)
            - n_time: Number of time points (int)
            - t_final: Total simulation time in seconds (float)
            - cache_file: File path to cache results (str)
            - config_file: File path to question config (str)
            - convergence_threshold: Convergence threshold for spatial and temporal convergence
            - budget: Budget for the API call
        
        accumulated_cost: Accumulated cost of the API call

    Returns:
        Dictionary containing the following keys:
            - spatial_converged: Boolean indicating if spatial convergence was reached (bool)
            - spatial_l2_error: L2 error of the spatial solution (float)
            - temporal_converged: Boolean indicating if temporal convergence was reached (bool)
            - temporal_l2_error: L2 error of the temporal solution (float)
            - accumulated_cost: Accumulated cost of the API call
    """
    # Create solver instance
    solver = WallHeatTransfer(
        L=float(params['L']),
        k=float(params['k']),
        h=float(params['h']),
        rho=float(params['rho']),
        cp=float(params['cp']),
        T_inf=float(params['T_inf']),
        T_init=float(params['T_init']),
        n_space=int(params['n_space']),
        n_time=int(params['n_time']),
        t_final=float(params['t_final']),
        cache_file=params['cache_file'],
        config_file=params['config_file'],
    )
    
    # Call appropriate solver method
    solver.solve_pde()
    accumulated_cost += int(params['n_time']) * int(params['n_space'])

    spatial_converged, spatial_l2_error, temporal_converged, temporal_l2_error, cost = test_convergence(
        params['config_file'], params['convergence_threshold']
    )
    accumulated_cost += cost

    if spatial_converged and temporal_converged:
        is_experiment_ended = True
    else:
        is_experiment_ended = False


    return {
        "spatial_l2_error": round(spatial_l2_error, 6),
        "temporal_l2_error": round(temporal_l2_error, 6),
        "is_experiment_ended": is_experiment_ended,
        "accumulated_cost": accumulated_cost
    }
