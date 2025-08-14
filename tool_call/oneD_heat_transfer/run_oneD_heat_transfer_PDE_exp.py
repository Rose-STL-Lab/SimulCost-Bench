from costsci_tools.wrappers.heat_1d import run_sim_heat_1d, compare_res_heat_1d

def heat_1d_check_converge_n_space(*, accumulated_cost: int, profile: str,
                           current_n_space: int, current_cfl: float, tolerance: float):
    # Fix the CFL for the n_space searching
    print(f"\nRunning simulation with n_space = {current_n_space}")
    refine = current_n_space * 2

    # Run simulation and load results
    accumulated_cost += run_sim_heat_1d(profile, current_cfl, current_n_space)
    _ = run_sim_heat_1d(profile, current_cfl, refine) # Convergence checking does not increase the cost

    is_converged, spatial_l2_error = compare_res_heat_1d(
        profile, current_cfl, current_n_space, profile, current_cfl, refine, tolerance
    )

    if is_converged:
        print(f"Convergence achieved between n_space {current_n_space} and {refine}")
    else:
        print(f"No convergence between n_space {current_n_space} and {refine}")

    return {
        "spatial_l2_error": round(spatial_l2_error, 6),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
    }

def heat_1d_check_converge_cfl(*, accumulated_cost: int, profile: str,
                           current_n_space: int, current_cfl: float, tolerance: float):
    # Fix the CFL for the n_space searching
    print(f"\nRunning simulation with cfl = {current_cfl}")
    refine = current_cfl / 2

    # Run simulation and load results
    accumulated_cost += run_sim_heat_1d(profile, current_cfl, current_n_space)
    _ = run_sim_heat_1d(profile, refine, current_n_space) # Convergence checking does not increase the cost

    is_converged, l2_error = compare_res_heat_1d(
        profile, current_cfl, current_n_space, profile, refine, current_n_space, tolerance
    )

    if is_converged:
        print(f"Convergence achieved between cfl {current_cfl} and {refine}")
    else:
        print(f"No convergence between cfl {current_cfl} and {refine}")

    return {
        "cfl_l2_error": round(l2_error, 6),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
    }
