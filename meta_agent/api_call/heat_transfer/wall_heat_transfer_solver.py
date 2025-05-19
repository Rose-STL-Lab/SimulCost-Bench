import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import os
import h5py
import json


class WallHeatTransfer:
    """
    1D heat transfer in a wall using Crank–Nicolson, 
    with integer-based grids in space (n_space) and time (n_time).
    """

    def __init__(
        self, L, k, h, rho, cp, T_inf, T_init, t_final, n_space, n_time, cache_file, config_file=None
    ):
        self.L = L
        self.k = k
        self.h = h
        self.rho = rho
        self.cp = cp
        self.T_inf = T_inf
        self.T_init = T_init
        self.n_space = n_space
        self.n_time = n_time
        self.t_final = t_final
        self.cache_file = cache_file
        self._save_config(config_file)

        # Derived parameters
        self.alpha = k / (rho * cp)

        # Space step
        self.dx = L / n_space
        self.nx = n_space + 1
        self.x = np.linspace(0, L, self.nx)

        # Time step
        self.nt = n_time + 1
        self.dt = t_final / n_time
        self.t = np.linspace(0, t_final, self.nt)

    def _save_config(self, config_file):
        if config_file is not None:
            config = {
                "L": self.L,
                "k": self.k,
                "h": self.h,
                "rho": self.rho,
                "cp": self.cp,
                "T_inf": self.T_inf,
                "T_init": self.T_init,
                "t_final": self.t_final,
                "n_space": self.n_space,
                "n_time": self.n_time,
                "cache_file": self.cache_file
            }
            with open(config_file, "w") as f:
                f.write(json.dumps(config))

    def _get_cache_key(self):
        """Generate a unique key for the current parameter set"""
        return f"T_history_{self.nt}_{self.nx}"

    def _check_cache(self):
        """Check if solution exists in cache"""
        if not os.path.exists(self.cache_file):
            return None

        try:
            with h5py.File(self.cache_file, 'r') as f:
                cache_key = self._get_cache_key()
                if cache_key in f:
                    return np.array(f[cache_key])
        except:
            return None
        return None

    def _save_to_cache(self, T_history):
        """Save solution to cache"""
        with h5py.File(self.cache_file, 'a') as f:
            cache_key = self._get_cache_key()
            if cache_key in f:
                del f[cache_key]  # Remove existing dataset if it exists
            f.create_dataset(cache_key, data=T_history)

    def _calculate_and_factorize_crank_nicolson_matrix(self):
        """Calculate and factorize the Crank-Nicolson matrix."""
        r = self.alpha * self.dt / (2 * self.dx**2)
        # I - 1/2 dt / dx^2 laplacian
        main_diag = np.ones(self.nx) * (1 + 2 * r)
        off_diag_right = np.ones(self.nx - 1) * -r
        off_diag_left = np.ones(self.nx - 1) * -r

        # NOTE hard code the boundary conditions
        # left boundary, advection to T_inf
        main_diag[0] = 1 + self.h * self.dx / self.k
        off_diag_right[0] = -1
        # right boundary, adiaiabatic
        main_diag[-1] = 1
        off_diag_left[-1] = -1

        # Construct the matrix
        A = diags([main_diag, off_diag_right, off_diag_left],
                  [0, 1, -1], format="csr")

        # Perform LU factorization
        LU_pivots = splu(A.tocsc())
        return LU_pivots

    def _calculate_right_hand_side(self, T_old):
        """Calculate the right hand side of the Crank-Nicolson scheme."""
        r = self.alpha * self.dt / (2 * self.dx**2)
        # calculate the central [1:-1] part
        b = T_old.copy()
        b[1:-1] += r * (T_old[:-2] - 2 * T_old[1:-1] + T_old[2:])
        # NOTE hard code the boundary conditions
        # left boundary, advection to T_inf
        b[0] = self.h * self.dx * self.T_inf / self.k
        # right boundary, adiaiabatic
        b[-1] = 0
        return b

    def solve_pde(self):
        """
        PDE solution using Crank-Nicolson method, returning full time history
        """
        # Initialize temperature array for all time steps
        cached_solution = self._check_cache()
        if cached_solution is not None:
            return cached_solution

        T_history = np.zeros((self.nt, self.nx))
        T_history[0, :] = self.T_init * np.ones(self.nx)

        # Set up matrices for Crank-Nicolson
        LU_pivots = self._calculate_and_factorize_crank_nicolson_matrix()

        # Time stepping
        for n in range(self.n_time):
            # get right hand side
            b = self._calculate_right_hand_side(
                T_history[n, :],
            )
            # solve the linear system
            T_history[n + 1, :] = LU_pivots.solve(b)

        self._save_to_cache(T_history)

        return T_history


def solve_PDE(
    L, k, h, rho, cp, T_inf, T_init, t_final, n_space, n_time, cache_file, config_file
):
    solver = WallHeatTransfer(
        L, k, h, rho, cp, T_inf, T_init, t_final, n_space, n_time, cache_file, config_file
    )
    return solver.solve_pde()


def test_convergence(
    config_file, convergence_threshold
):
    with open(config_file, "r") as f:
        config = json.load(f)

    L = config["L"]
    k = config["k"]
    h = config["h"]
    rho = config["rho"]
    cp = config["cp"]
    T_inf = config["T_inf"]
    T_init = config["T_init"]
    t_final = config["t_final"]
    n_space = config["n_space"]
    n_time = config["n_time"]
    cache_file = config["cache_file"]
    cost = 0
    # Check for initial condition
    solver_base = WallHeatTransfer(
        L, k, h, rho, cp, T_inf, T_init, t_final, n_space, n_time, cache_file
    )
    # T_base: shape(n_time + 1, n_space + 1)
    T_base = solver_base._check_cache()
    if T_base is None:
        T_base = solver_base.solve_pde()
        cost += n_space * n_time
    x_base = solver_base.x
    t_base = solver_base.t

    # Check for Spatial Convergence
    refined_n_space = n_space * 2
    solver_spatial_refined = WallHeatTransfer(
        L, k, h, rho, cp, T_inf, T_init, t_final, refined_n_space, n_time, cache_file
    )
    # T_x_refined: shape(n_time + 1, 2 * n_space + 1)
    T_x_refined = solver_spatial_refined.solve_pde()
    cost += refined_n_space * n_time
    x_refined = solver_spatial_refined.x

    # Upsampling T_hist_base to match T_x_refined
    T_base_x_upsampled = np.zeros((n_time + 1, 2 * n_space + 1))
    for i in range(T_base.shape[0]):
        T_base_x_upsampled[i] = np.interp(x_refined, x_base, T_base[i])

    # Calculate mean L2 error across all time steps
    spatial_l2_error = np.mean(
        np.sqrt(np.mean((T_base_x_upsampled - T_x_refined)**2, axis=1)))

    # Check for Temporal Convergence
    refined_n_time = n_time * 2
    solver_temporal_refined = WallHeatTransfer(
        L, k, h, rho, cp, T_inf, T_init, t_final, n_space, refined_n_time, cache_file
    )
    # T_hist_temporal_refined: shape(2 * n_time + 1, n_space + 1)
    T_t_refined = solver_temporal_refined.solve_pde()


    cost += n_space * refined_n_time
    t_refined = solver_temporal_refined.t

    # Upsampling T_hist_base to match T_hist_temporal_refined
    T_base_t_upsampled = np.zeros((2 * n_time + 1, n_space + 1))
    for i in range(T_base.shape[1]):
        T_base_t_upsampled[:, i] = np.interp(t_refined, t_base, T_base[:, i])

    # Calculate mean L2 error across all time steps
    temporal_l2_error = np.mean(
        np.sqrt(np.mean((T_base_t_upsampled - T_t_refined)**2, axis=1)))

    if temporal_l2_error < convergence_threshold:
        temporal_converged = True
    else:
        temporal_converged = False

    if spatial_l2_error < convergence_threshold:
        spatial_converged = True
    else:
        spatial_converged = False

    return spatial_converged, spatial_l2_error, temporal_converged, temporal_l2_error, cost


def dummy_strategy(L, k, h, rho, cp, T_inf, T_init, t_final, cache_file, config_file, convergence_threshold):
    n_space = 32
    n_time = 64
    converged = False
    acc_cost = 0
    dummy_sequence = []
    while not converged:
        print("n_space: ", n_space, "n_time: ", n_time)
        dummy_sequence.append({
            "n_space": n_space,
            "n_time": n_time,
        })
        solver = WallHeatTransfer(
            L, k, h, rho, cp, T_inf, T_init, t_final, n_space, n_time, cache_file, config_file
        )
        solver.solve_pde()
        acc_cost += n_space * n_time

        spatial_converged, spatial_l2_error, temporal_converged, temporal_l2_error, cost = test_convergence(
            config_file, convergence_threshold
        )
        print("spatial_l2_error: ", spatial_l2_error, "temporal_l2_error: ", temporal_l2_error)
        acc_cost += cost
        if not spatial_converged and not temporal_converged:
            n_space *= 2
            n_time *= 2
        elif not spatial_converged:
            n_space *= 2
        elif not temporal_converged:
            n_time *= 2
        else:
            converged = True

    # just delete the cache file
    if os.path.exists(cache_file):
        os.remove(cache_file)
        
    return acc_cost, dummy_sequence