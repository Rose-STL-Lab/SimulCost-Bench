from costsci_tools.wrappers.heat_steady_2d import run_sim_heat_steady_2d, compare_res_heat_steady_2d

def heat_2d_check_converge_dx(*, accumulated_cost: int, profile: str,
                           dx: float, relax: float, 
                           t_init: float, error_threshold: float, 
                           tolerance: float):
    print(f"\nRunning simulation with dx = {dx}")
    refine = dx / 2

    # Run simulation and load results
    cost, num_steps = run_sim_heat_steady_2d(profile=profile, dx=dx, relax=relax, 
                                               error_threshold=error_threshold, t_init=t_init)
    refine_cost, refine_num_steps = run_sim_heat_steady_2d(profile=profile, dx=refine, relax=relax, 
                               error_threshold=error_threshold, t_init=t_init) # Convergence checking does not increase the cost

    accumulated_cost += cost

    is_converged, metrics1, metrics2, rmse = compare_res_heat_steady_2d(
        profile1=profile, dx1=dx, relax1=relax, error_threshold1=error_threshold, t_init1=t_init,
        profile2=profile, dx2=refine, relax2=relax, error_threshold2=error_threshold, t_init2=t_init,
        rmse_tolerance=tolerance
    )

    if is_converged:
        print(f"Convergence achieved between dx {dx} and {refine}")
    else:
        print(f"No convergence between dx {dx} and {refine}")

    return {
        "RMSE": round(rmse, 6),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
        "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": num_steps,
        "The number of iterations the solver will try to verify convergence based on the parameters you have given": refine_num_steps,
        "The cost of the solver simulating the environment": cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "metrics1": metrics1,
        "metrics2": metrics2,
    }

def heat_2d_check_converge_error_threshold(*, accumulated_cost: int, profile: str,
                           dx: float, relax: float, 
                           t_init: float, error_threshold: float, 
                           tolerance: float):
    print(f"\nRunning simulation with error_threshold = {error_threshold}")
    refine = error_threshold / 10

    # Run simulation and load results
    cost, num_steps = run_sim_heat_steady_2d(profile=profile, dx=dx, relax=relax, 
                                               error_threshold=error_threshold, t_init=t_init)
    refine_cost, refine_num_steps = run_sim_heat_steady_2d(profile=profile, dx=dx, relax=relax, 
                               error_threshold=refine, t_init=t_init)

    accumulated_cost += cost

    is_converged, metrics1, metrics2, rmse = compare_res_heat_steady_2d(
        profile1=profile, dx1=dx, relax1=relax, error_threshold1=error_threshold, t_init1=t_init,
        profile2=profile, dx2=dx, relax2=relax, error_threshold2=refine, t_init2=t_init,
        rmse_tolerance=tolerance
    )

    if is_converged:
        print(f"Convergence achieved between error_threshold {error_threshold} and {refine}")
    else:
        print(f"No convergence between error_threshold {error_threshold} and {refine}")

    return {
        "RMSE": round(rmse, 6),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
        "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": num_steps,
        "The number of iterations the solver will try to verify convergence based on the parameters you have given": refine_num_steps,
        "The cost of the solver simulating the environment": cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
        "metrics1": metrics1,
        "metrics2": metrics2,
    }

def heat_2d_check_converge_relax(*, accumulated_cost: int, profile: str,
                           dx: float, relax: float, 
                           t_init: float, error_threshold: float, 
                           tolerance: float):
    """Perform grid search for optimal relaxation factor."""
    print(f"\nRunning simulation with relax = {relax}")

    # Run simulation and load results
    cost, num_steps = run_sim_heat_steady_2d(profile=profile, dx=dx, relax=relax, 
                                               error_threshold=error_threshold, t_init=t_init)
    accumulated_cost += cost

    if relax >= 2 or relax <= 0:
        print(f"Relaxation factor {relax} is out of range.")
        return {
            "RMSE": 0.0,
            "is_converged": False,
            "accumulated_cost": accumulated_cost,
            "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": num_steps,
            "The cost of the solver simulating the environment": cost,
        }

    print(f"Convergence achieved with relax = {relax}")
    
    return {
        "RMSE": 0.0,
        "is_converged": True,
        "accumulated_cost": accumulated_cost,
        "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": num_steps,
        "The cost of the solver simulating the environment": cost,
    }

def heat_2d_check_converge_t_init(*, accumulated_cost: int, profile: str,
                           dx: float, relax: float, 
                           t_init: float, error_threshold: float, 
                           tolerance: float):
    print(f"\nRunning simulation with t_init = {t_init}")
    # Run simulation and load results
    cost, num_steps = run_sim_heat_steady_2d(profile=profile, dx=dx, relax=relax, 
                                               error_threshold=error_threshold, t_init=t_init)
    accumulated_cost += cost

    print(f"Convergence achieved with t_init = {t_init}")

    return {
        "RMSE": 0.0,
        "is_converged": True,
        "accumulated_cost": accumulated_cost,
        "The number of iterations the solver will perform to simulate the environment based on the parameters you have given": num_steps,
        "The cost of the solver simulating the environment": cost,
    }
