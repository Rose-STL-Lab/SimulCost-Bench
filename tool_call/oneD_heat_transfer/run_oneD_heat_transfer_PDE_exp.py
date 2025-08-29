from costsci_tools.wrappers.heat_1d import run_sim_heat_1d, compare_res_heat_1d

def heat_1d_check_converge_n_space(*, accumulated_cost: int, profile: str,
                           n_space: int, cfl: float, tolerance: float):
    # Handle invalid parameter values
    if cfl <= 0:
        print(f"\nInvalid CFL value: {cfl} <= 0. Returning high error and cost.")
        return {
            "spatial_l2_error": 1e10,
            "is_converged": False,
            "accumulated_cost": accumulated_cost + int(1e9),
        }
    
    if n_space <= 0 or not isinstance(n_space, int):
        print(f"\nInvalid n_space value: {n_space}. Must be a positive integer. Returning high error and cost.")
        return {
            "spatial_l2_error": 1e10,
            "is_converged": False,
            "accumulated_cost": accumulated_cost + int(1e9),
        }
    
    # Fix the CFL for the n_space searching
    print(f"\nRunning simulation with n_space = {n_space}")
    refine = n_space * 2

    # Run simulation and load results
    accumulated_cost += run_sim_heat_1d(profile=profile, cfl=cfl, n_space=n_space)
    _ = run_sim_heat_1d(profile=profile, cfl=cfl, n_space=refine) # Convergence checking does not increase the cost

    is_converged, spatial_l2_error = compare_res_heat_1d(
        profile1=profile, cfl1=cfl, n_space1=n_space, 
        profile2=profile, cfl2=cfl, n_space2=refine, 
        tolerance=tolerance
    )

    if is_converged:
        print(f"Convergence achieved between n_space {n_space} and {refine}")
    else:
        print(f"No convergence between n_space {n_space} and {refine}")

    return {
        "spatial_l2_error": round(spatial_l2_error, 6),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
    }

def heat_1d_check_converge_cfl(*, accumulated_cost: int, profile: str,
                           n_space: int, cfl: float, tolerance: float):
    # Handle invalid parameter values
    if cfl <= 0:
        print(f"\nInvalid CFL value: {cfl} <= 0. Returning high error and cost.")
        return {
            "cfl_l2_error": 1e10,
            "is_converged": False,
            "accumulated_cost": accumulated_cost + int(1e9),
        }
    
    if n_space <= 0 or not isinstance(n_space, int):
        print(f"\nInvalid n_space value: {n_space}. Must be a positive integer. Returning high error and cost.")
        return {
            "cfl_l2_error": 1e10,
            "is_converged": False,
            "accumulated_cost": accumulated_cost + int(1e9),
        }
    
    # Fix the CFL for the n_space searching
    print(f"\nRunning simulation with cfl = {cfl}")
    refine = cfl / 2

    # Run simulation and load results
    accumulated_cost += run_sim_heat_1d(profile=profile, cfl=cfl, n_space=n_space)
    _ = run_sim_heat_1d(profile=profile, cfl=refine, n_space=n_space) # Convergence checking does not increase the cost

    is_converged, l2_error = compare_res_heat_1d(
        profile1=profile, cfl1=cfl, n_space1=n_space, 
        profile2=profile, cfl2=refine, n_space2=n_space, 
        tolerance=tolerance
    )

    if is_converged:
        print(f"Convergence achieved between cfl {cfl} and {refine}")
    else:
        print(f"No convergence between cfl {cfl} and {refine}")

    return {
        "cfl_l2_error": round(l2_error, 6),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
    }
