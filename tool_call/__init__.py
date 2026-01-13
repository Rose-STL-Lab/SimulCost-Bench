"""
Tool call functions for various PDE solvers.

This package uses lazy loading - functions are imported on-demand via
inference/utils.py:_get_tool_function() to minimize startup time and
avoid loading unnecessary dependencies (like Taichi) for solvers that
don't need them.

IMPORTANT: Do NOT add top-level imports here, as they will break lazy loading.
Functions are loaded dynamically using importlib.import_module() based on the
_TOOL_MODULE_MAP defined in inference/utils.py.
"""

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

    # Diffusion-Reaction 1D
    "diff_react_1d_check_converge_cfl",
    "diff_react_1d_check_converge_n_space",
    "diff_react_1d_check_converge_tol",
    "diff_react_1d_check_converge_parameter",

    # Euler 2D
    "euler_2d_check_converge_cfl",
    "euler_2d_check_converge_n_grid_x",
    "euler_2d_check_converge_cg_tolerance",

    # Hasegawa-Mima Nonlinear
    "hasegawa_mima_nonlinear_check_converge_N",
    "hasegawa_mima_nonlinear_check_converge_dt",

    # Hasegawa-Mima Linear
    "hasegawa_mima_linear_check_converge_N",
    "hasegawa_mima_linear_check_converge_dt",
    "hasegawa_mima_linear_check_converge_cg_atol",

    # FEM 2D
    "fem_2d_check_converge_dx",
    "fem_2d_check_converge_cfl",
    "fem_2d_check_converge_parameter",
]
