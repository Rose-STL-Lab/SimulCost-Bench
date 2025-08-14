# Heat transfer 1D functions
from .oneD_heat_transfer.run_oneD_heat_transfer_PDE_exp import (
    heat_1d_check_converge_n_space, 
    heat_1d_check_converge_cfl
)

# Heat transfer 2D functions  
from .twoD_heat_transfer.run_twoD_heat_transfer_PDE_exp import (
    heat_2d_check_converge_dx, 
    heat_2d_check_converge_error_threshold, 
    heat_2d_check_converge_relax, 
    heat_2d_check_converge_t_init
)

# Burgers 1D functions
from .burgers_1d.run_oneD_burgers_PDE_exp import burgers_1d_solve

# Euler 1D functions
from .euler_1d.run_oneD_euler_PDE_exp import (
    euler_1d_check_converge_cfl, 
    euler_1d_check_converge_beta, 
    euler_1d_check_converge_k, 
    euler_1d_check_converge_n_space
)

# NS Channel 2D functions
from .ns_channel_2d.run_twoD_ns_channel_exp import (
    ns_2d_check_converge_mesh_x,
    ns_2d_check_converge_mesh_y,
    ns_2d_check_converge_omega_u,
    ns_2d_check_converge_omega_v,
    ns_2d_check_converge_omega_p,
    ns_2d_check_converge_diff_u_threshold,
    ns_2d_check_converge_diff_v_threshold,
    ns_2d_check_converge_res_iter_v_threshold,
    ns_2d_check_converge_parameter
)

__all__ = [
    # Heat transfer 1D
    "heat_1d_check_converge_n_space",
    "heat_1d_check_converge_cfl",
    
    # Heat transfer 2D    
    "heat_2d_check_converge_dx",
    "heat_2d_check_converge_error_threshold",
    "heat_2d_check_converge_relax",
    "heat_2d_check_converge_t_init",

    # Burgers 1D
    "burgers_1d_solve",
    
    # Euler 1D
    "euler_1d_check_converge_cfl",
    "euler_1d_check_converge_beta", 
    "euler_1d_check_converge_k",
    "euler_1d_check_converge_n_space",
    
    # NS channel 2D
    "ns_2d_check_converge_mesh_x",
    "ns_2d_check_converge_mesh_y",
    "ns_2d_check_converge_omega_u",
    "ns_2d_check_converge_omega_v",
    "ns_2d_check_converge_omega_p",
    "ns_2d_check_converge_diff_u_threshold",
    "ns_2d_check_converge_diff_v_threshold",
    "ns_2d_check_converge_res_iter_v_threshold",
    "ns_2d_check_converge_parameter",
]