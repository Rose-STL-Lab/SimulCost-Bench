from costsci_tools.wrappers.epoch import runEpoch, compare_res_epoch
import numpy as np

def _to_jsonable(obj):
    """Convert numpy objects to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()             # list
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()               # float / int
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def epoch_1d_check_converge_nx(
    *,
    accumulated_cost: int,
    profile: str,
    nx: int,
    dt_multiplier: float,
    npart: int,
    field_order: int,
    particle_order: int,
    tolerance: float,
):
    """Check convergence by refining nx (multiplying by 1.2) while keeping other parameters fixed."""
    print(f"\nRunning nx convergence test with nx = {nx}")
    refine_nx = int(1.2 * nx)

    current_cost = runEpoch(
        profile=profile,
        nx=nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )
    refine_cost = runEpoch(
        profile=profile,
        nx=refine_nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )

    accumulated_cost += current_cost

    (
        converged,
        l2_error
    ) = compare_res_epoch(
        profile1=profile, nx1=nx, dt_mult1=dt_multiplier, nPart1=npart, fO1=field_order, pO1=particle_order,
        profile2=profile, nx2=refine_nx, dt_mult2=dt_multiplier, nPart2=npart, fO2=field_order, pO2=particle_order,
        tolerance=tolerance
    )

    return {
        "L2_error": round(l2_error, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }

def epoch_1d_check_converge_npart(
    *,
    accumulated_cost: int,
    profile: str,
    nx: int,
    dt_multiplier: float,
    npart: int,
    field_order: int,
    particle_order: int,
    tolerance: float,
):
    """Check convergence by refining npart (multiplying by 1.2) while keeping other parameters fixed."""
    print(f"\nRunning npart convergence test with npart = {npart}")
    refine_npart = int(1.2 * npart)

    current_cost = runEpoch(
        profile=profile,
        nx=nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )
    refine_cost = runEpoch(
        profile=profile,
        nx=nx,
        dt_mult=dt_multiplier,
        nPart=refine_npart,
        field_order=field_order,
        particle_order=particle_order
    )

    accumulated_cost += current_cost

    (
        converged,
        l2_error
    ) = compare_res_epoch(
        profile1=profile, nx1=nx, dt_mult1=dt_multiplier, nPart1=npart, fO1=field_order, pO1=particle_order,
        profile2=profile, nx2=nx, dt_mult2=dt_multiplier, nPart2=refine_npart, fO2=field_order, pO2=particle_order,
        tolerance=tolerance
    )

    return {
        "L2_error": round(l2_error, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }

def epoch_1d_check_converge_dt_multiplier(
    *,
    accumulated_cost: int,
    profile: str,
    nx: int,
    dt_multiplier: float,
    npart: int,
    field_order: int,
    particle_order: int,
    tolerance: float,
):
    """Check convergence by refining nx (multiplying by 1.2) for dt_multiplier task while keeping other parameters fixed."""
    print(f"\nRunning dt_multiplier convergence test with dt_multiplier = {dt_multiplier}, nx = {nx}")
    refine_nx = int(1.2 * nx)

    current_cost = runEpoch(
        profile=profile,
        nx=nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )
    refine_cost = runEpoch(
        profile=profile,
        nx=refine_nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )

    accumulated_cost += current_cost

    (
        converged,
        l2_error
    ) = compare_res_epoch(
        profile1=profile, nx1=nx, dt_mult1=dt_multiplier, nPart1=npart, fO1=field_order, pO1=particle_order,
        profile2=profile, nx2=refine_nx, dt_mult2=dt_multiplier, nPart2=npart, fO2=field_order, pO2=particle_order,
        tolerance=tolerance
    )

    return {
        "L2_error": round(l2_error, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }

def epoch_1d_check_converge_field_order(
    *,
    accumulated_cost: int,
    profile: str,
    nx: int,
    dt_multiplier: float,
    npart: int,
    field_order: int,
    particle_order: int,
    tolerance: float,
):
    """Check convergence by refining nx (multiplying by 1.2) for field_order task while keeping other parameters fixed."""
    print(f"\nRunning field_order convergence test with field_order = {field_order}, nx = {nx}")
    refine_nx = int(1.2 * nx)

    current_cost = runEpoch(
        profile=profile,
        nx=nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )
    refine_cost = runEpoch(
        profile=profile,
        nx=refine_nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )

    accumulated_cost += current_cost

    (
        converged,
        l2_error
    ) = compare_res_epoch(
        profile1=profile, nx1=nx, dt_mult1=dt_multiplier, nPart1=npart, fO1=field_order, pO1=particle_order,
        profile2=profile, nx2=refine_nx, dt_mult2=dt_multiplier, nPart2=npart, fO2=field_order, pO2=particle_order,
        tolerance=tolerance
    )

    return {
        "L2_error": round(l2_error, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }

def epoch_1d_check_converge_particle_order(
    *,
    accumulated_cost: int,
    profile: str,
    nx: int,
    dt_multiplier: float,
    npart: int,
    field_order: int,
    particle_order: int,
    tolerance: float,
):
    """Check convergence by refining nx (multiplying by 1.2) for particle_order task while keeping other parameters fixed."""
    print(f"\nRunning particle_order convergence test with particle_order = {particle_order}, nx = {nx}")
    refine_nx = int(1.2 * nx)

    current_cost = runEpoch(
        profile=profile,
        nx=nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )
    refine_cost = runEpoch(
        profile=profile,
        nx=refine_nx,
        dt_mult=dt_multiplier,
        nPart=npart,
        field_order=field_order,
        particle_order=particle_order
    )

    accumulated_cost += current_cost

    (
        converged,
        l2_error
    ) = compare_res_epoch(
        profile1=profile, nx1=nx, dt_mult1=dt_multiplier, nPart1=npart, fO1=field_order, pO1=particle_order,
        profile2=profile, nx2=refine_nx, dt_mult2=dt_multiplier, nPart2=npart, fO2=field_order, pO2=particle_order,
        tolerance=tolerance
    )

    return {
        "L2_error": round(l2_error, 6),
        "is_converged": bool(converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }