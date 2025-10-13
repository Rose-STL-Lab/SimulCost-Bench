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
from .burgers_1d.run_oneD_burgers_PDE_exp import (
    burgers_1d_check_converge_cfl,
    burgers_1d_check_converge_beta,
    burgers_1d_check_converge_k,
    burgers_1d_check_converge_n_space
)

# Euler 1D functions
from .euler_1d.run_oneD_euler_PDE_exp import (
    euler_1d_check_converge_cfl,
    euler_1d_check_converge_beta,
    euler_1d_check_converge_k,
    euler_1d_check_converge_n_space
)

# EPOCH 1D functions
from .epoch_1d.run_oneD_epoch import (
    epoch_1d_check_converge_nx,
    epoch_1d_check_converge_npart,
    epoch_1d_check_converge_dt_multiplier,
    epoch_1d_check_converge_field_order,
    epoch_1d_check_converge_particle_order
)

# NS Channel 2D functions
from .ns_2d.run_twoD_ns_exp import (
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

# NS Transient 2D functions
from .ns_transient_2d.run_twoD_ns_transient import (
    ns_transient_2d_check_converge_resolution,
    ns_transient_2d_check_converge_cfl,
    ns_transient_2d_check_converge_relaxation_factor,
    ns_transient_2d_check_converge_residual_threshold,
    ns_transient_2d_check_converge_parameter
)

# MPM 2D functions
from .mpm_2d.run_twoD_mpm import (
    mpm_2d_check_converge_nx,
    mpm_2d_check_converge_npart,
    mpm_2d_check_converge_cfl,
    mpm_2d_check_converge_parameter
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
    "burgers_1d_check_converge_cfl",
    "burgers_1d_check_converge_beta",
    "burgers_1d_check_converge_k", 
    "burgers_1d_check_converge_n_space",
    
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

    # NS transient 2D
    "ns_transient_2d_check_converge_resolution",
    "ns_transient_2d_check_converge_cfl",
    "ns_transient_2d_check_converge_relaxation_factor",
    "ns_transient_2d_check_converge_residual_threshold",
    "ns_transient_2d_check_converge_parameter",

    # EPOCH 1D
    "epoch_1d_check_converge_nx",
    "epoch_1d_check_converge_npart",
    "epoch_1d_check_converge_dt_multiplier",
    "epoch_1d_check_converge_field_order",
    "epoch_1d_check_converge_particle_order",

    # MPM 2D
    "mpm_2d_check_converge_nx",
    "mpm_2d_check_converge_npart",
    "mpm_2d_check_converge_cfl",
    "mpm_2d_check_converge_parameter",
]