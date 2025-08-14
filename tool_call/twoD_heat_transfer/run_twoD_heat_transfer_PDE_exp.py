from costsci_tools.wrappers.heat_steady_2d import run_sim_heat_steady_2d, compare_res_heat_steady_2d

def heat_2d_check_converge_dx(*, accumulated_cost: int, profile: str,
                           current_dx: float, current_relax: float, 
                           current_t_init: float, current_error_threshold: float, 
                           tolerance: float):
    print(f"\nRunning simulation with dx = {current_dx}")
    refine = current_dx / 2

    # Run simulation and load results
    current_cost, current_num_steps = run_sim_heat_steady_2d(profile=profile, dx=current_dx, relax=current_relax, 
                                               t_init=current_t_init, error_threshold=current_error_threshold)
    refine_cost, refine_num_steps = run_sim_heat_steady_2d(profile=profile, dx=refine, relax=current_relax, 
                               t_init=current_t_init, error_threshold=current_error_threshold) # Convergence checking does not increase the cost

    accumulated_cost += current_cost

    is_converged, rmse = compare_res_heat_steady_2d(
        profile1=profile, dx1=current_dx, relax1=current_relax, error_threshold1=current_error_threshold, t_init1=current_t_init,
        profile2=profile, dx2=refine, relax2=current_relax, error_threshold2=current_error_threshold, t_init2=current_t_init,
        tolerance=tolerance
    )

    if is_converged:
        print(f"Convergence achieved between dx {current_dx} and {refine}")
    else:
        print(f"No convergence between dx {current_dx} and {refine}")

    return {
        "RMSE": round(rmse, 6),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
        "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": current_num_steps,
        "The number of iterations the solver will try to verify convergence based on the parameters you have given": refine_num_steps,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }

def heat_2d_check_converge_error_threshold(*, accumulated_cost: int, profile: str,
                           current_dx: float, current_relax: float, 
                           current_t_init: float, current_error_threshold: float, 
                           tolerance: float):
    print(f"\nRunning simulation with error_threshold = {current_error_threshold}")
    refine = current_error_threshold / 10

    # Run simulation and load results
    current_cost, current_num_steps = run_sim_heat_steady_2d(profile=profile, dx=current_dx, relax=current_relax, 
                                               t_init=current_t_init, error_threshold=current_error_threshold)
    refine_cost, refine_num_steps = run_sim_heat_steady_2d(profile=profile, dx=current_dx, relax=current_relax, 
                               t_init=current_t_init, error_threshold=refine)

    accumulated_cost += current_cost

    is_converged, rmse = compare_res_heat_steady_2d(
        profile1=profile, dx1=current_dx, relax1=current_relax, error_threshold1=current_error_threshold, t_init1=current_t_init,
        profile2=profile, dx2=current_dx, relax2=current_relax, error_threshold2=refine, t_init2=current_t_init,
        tolerance=tolerance
    )

    if is_converged:
        print(f"Convergence achieved between error_threshold {current_error_threshold} and {refine}")
    else:
        print(f"No convergence between error_threshold {current_error_threshold} and {refine}")

    return {
        "RMSE": round(rmse, 6),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
        "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": current_num_steps,
        "The number of iterations the solver will try to verify convergence based on the parameters you have given": refine_num_steps,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }

def heat_2d_check_converge_relax(*, accumulated_cost: int, profile: str,
                           current_dx: float, current_relax: float, 
                           current_t_init: float, current_error_threshold: float, 
                           tolerance: float):
    """Perform grid search for optimal relaxation factor."""
    print(f"\nRunning simulation with relax = {current_relax}")

    # Run simulation and load results
    current_cost, current_num_steps = run_sim_heat_steady_2d(profile=profile, dx=current_dx, relax=current_relax, 
                                               t_init=current_t_init, error_threshold=current_error_threshold)
    accumulated_cost += current_cost

    if current_relax > 1.0 or current_relax < 0.05:
        print(f"Relaxation factor {current_relax} is out of range.")
        return {
            "RMSE": 0.0,
            "is_converged": False,
            "accumulated_cost": accumulated_cost,
            "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": current_num_steps,
        }

    print(f"Convergence achieved with relax = {current_relax}")
    
    return {
        "RMSE": 0.0,
        "is_converged": True,
        "accumulated_cost": accumulated_cost,
        "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": current_num_steps,
    }

def heat_2d_check_converge_t_init(*, accumulated_cost: int, profile: str,
                           current_dx: float, current_relax: float, 
                           current_t_init: float, current_error_threshold: float, 
                           tolerance: float):
    print(f"\nRunning simulation with t_init = {current_t_init}")
    # Run simulation and load results
    current_cost, current_num_steps = run_sim_heat_steady_2d(profile=profile, dx=current_dx, relax=current_relax, 
                                               t_init=current_t_init, error_threshold=current_error_threshold)
    accumulated_cost += current_cost

    print(f"Convergence achieved with t_init = {current_t_init}")

    return {
        "RMSE": 0.0,
        "is_converged": True,
        "accumulated_cost": accumulated_cost,
        "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": current_num_steps,
    }
