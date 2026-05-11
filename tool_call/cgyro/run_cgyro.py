import numpy as np
from costsci_tools.wrappers.cgyro import runCgyro, get_res_cgyro

# error_tol is a fixed non-tunable parameter across the CGYRO benchmark (see
# costsci_tools/dataset/cgyro/successful/tasks.json where all tasks use 1e-4).
_ERROR_TOL = 1e-4


def _to_jsonable(obj):
    """Convert numpy and complex objects to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _refined_params(task, n_radial, n_theta, n_xi, n_energy, freq_tol, delta_t):
    """Produce the 'finer' parameter set used as the self-refined reference.

    Refinement direction follows the physics — more modes for spectral/grid
    parameters, tighter tolerance and smaller timestep for the floats.
    """
    params = {
        "n_radial": n_radial, "n_theta": n_theta, "n_xi": n_xi,
        "n_energy": n_energy, "freq_tol": freq_tol, "delta_t": delta_t,
    }
    if task == "n_radial":
        params["n_radial"] = max(int(round(2 * n_radial)), n_radial + 1)
    elif task == "n_theta":
        params["n_theta"] = max(int(round(2 * n_theta)), n_theta + 1)
    elif task == "n_xi":
        params["n_xi"] = max(int(round(2 * n_xi)), n_xi + 1)
    elif task == "n_energy":
        params["n_energy"] = max(int(round(2 * n_energy)), n_energy + 1)
    elif task == "freq_tol":
        params["freq_tol"] = 0.5 * float(freq_tol)
    elif task == "delta_t":
        params["delta_t"] = 0.5 * float(delta_t)
    else:
        raise ValueError(f"Unknown CGYRO task: {task}")
    return params


def _final_eigenvalue(res):
    """Pull the last-timestep eigenvalue out of a CGYRO result dict."""
    if not res or "eigenvalues" not in res:
        return None
    eig_series = np.asarray(res["eigenvalues"]).squeeze()
    if eig_series.size == 0:
        return None
    return eig_series.flat[-1]


def _eigenvalue_diff(profile, current, refined):
    """L∞ distance between final eigenvalues of current vs refined runs.

    CGYRO eigenvalues are complex — real part is mode frequency (ω),
    imaginary part is growth rate (γ). Both must match to claim
    cross-resolution convergence, so we use the max of the two absolute
    differences.
    """
    res_cur, _, _ = get_res_cgyro(
        profile, current["n_radial"], current["n_theta"], _ERROR_TOL,
        current["freq_tol"], current["delta_t"], current["n_xi"], current["n_energy"],
    )
    res_ref, _, _ = get_res_cgyro(
        profile, refined["n_radial"], refined["n_theta"], _ERROR_TOL,
        refined["freq_tol"], refined["delta_t"], refined["n_xi"], refined["n_energy"],
    )

    eig_cur = _final_eigenvalue(res_cur)
    eig_ref = _final_eigenvalue(res_ref)
    if eig_cur is None or eig_ref is None:
        return float("inf")

    diff_real = abs(float(eig_cur.real) - float(eig_ref.real))
    diff_imag = abs(float(eig_cur.imag) - float(eig_ref.imag))
    return max(diff_real, diff_imag)


def _check_converge(
    task,
    *,
    accumulated_cost,
    profile,
    n_radial,
    n_theta,
    n_xi,
    n_energy,
    freq_tol,
    delta_t,
    tolerance,
):
    current = {
        "n_radial": int(n_radial), "n_theta": int(n_theta),
        "n_xi": int(n_xi), "n_energy": int(n_energy),
        "freq_tol": float(freq_tol), "delta_t": float(delta_t),
    }
    refined = _refined_params(task, **current)

    print(
        f"\nRunning {task} convergence test with current={current}, refined={refined}"
    )

    current_cost, _ = runCgyro(
        profile=profile,
        n_radial=current["n_radial"], n_theta=current["n_theta"],
        error_tol=_ERROR_TOL, freq_tol=current["freq_tol"],
        delta_t=current["delta_t"], n_xi=current["n_xi"],
        n_energy=current["n_energy"],
    )
    refine_cost, _ = runCgyro(
        profile=profile,
        n_radial=refined["n_radial"], n_theta=refined["n_theta"],
        error_tol=_ERROR_TOL, freq_tol=refined["freq_tol"],
        delta_t=refined["delta_t"], n_xi=refined["n_xi"],
        n_energy=refined["n_energy"],
    )

    accumulated_cost += current_cost

    eig_err = _eigenvalue_diff(profile, current, refined)
    is_converged = bool(np.isfinite(eig_err)) and (eig_err < tolerance)

    return {
        "L2_error": round(float(eig_err), 6) if np.isfinite(eig_err) else float("inf"),
        "is_converged": bool(is_converged),
        "accumulated_cost": accumulated_cost,
        "The cost of the solver simulating the environment": current_cost,
        "The cost of the solver verifying convergence (This will not be included in your accumulated_cost)": refine_cost,
    }


def cgyro_check_converge_n_radial(
    *, accumulated_cost, profile, n_radial, n_theta, n_xi, n_energy,
    freq_tol, delta_t, tolerance,
):
    """Check convergence by doubling n_radial while keeping other parameters fixed."""
    return _check_converge(
        "n_radial",
        accumulated_cost=accumulated_cost, profile=profile,
        n_radial=n_radial, n_theta=n_theta, n_xi=n_xi, n_energy=n_energy,
        freq_tol=freq_tol, delta_t=delta_t, tolerance=tolerance,
    )


def cgyro_check_converge_n_theta(
    *, accumulated_cost, profile, n_radial, n_theta, n_xi, n_energy,
    freq_tol, delta_t, tolerance,
):
    """Check convergence by doubling n_theta while keeping other parameters fixed."""
    return _check_converge(
        "n_theta",
        accumulated_cost=accumulated_cost, profile=profile,
        n_radial=n_radial, n_theta=n_theta, n_xi=n_xi, n_energy=n_energy,
        freq_tol=freq_tol, delta_t=delta_t, tolerance=tolerance,
    )


def cgyro_check_converge_n_xi(
    *, accumulated_cost, profile, n_radial, n_theta, n_xi, n_energy,
    freq_tol, delta_t, tolerance,
):
    """Check convergence by doubling n_xi while keeping other parameters fixed."""
    return _check_converge(
        "n_xi",
        accumulated_cost=accumulated_cost, profile=profile,
        n_radial=n_radial, n_theta=n_theta, n_xi=n_xi, n_energy=n_energy,
        freq_tol=freq_tol, delta_t=delta_t, tolerance=tolerance,
    )


def cgyro_check_converge_n_energy(
    *, accumulated_cost, profile, n_radial, n_theta, n_xi, n_energy,
    freq_tol, delta_t, tolerance,
):
    """Check convergence by doubling n_energy while keeping other parameters fixed."""
    return _check_converge(
        "n_energy",
        accumulated_cost=accumulated_cost, profile=profile,
        n_radial=n_radial, n_theta=n_theta, n_xi=n_xi, n_energy=n_energy,
        freq_tol=freq_tol, delta_t=delta_t, tolerance=tolerance,
    )


def cgyro_check_converge_freq_tol(
    *, accumulated_cost, profile, n_radial, n_theta, n_xi, n_energy,
    freq_tol, delta_t, tolerance,
):
    """Check convergence by halving freq_tol (tighter convergence) with other parameters fixed."""
    return _check_converge(
        "freq_tol",
        accumulated_cost=accumulated_cost, profile=profile,
        n_radial=n_radial, n_theta=n_theta, n_xi=n_xi, n_energy=n_energy,
        freq_tol=freq_tol, delta_t=delta_t, tolerance=tolerance,
    )


def cgyro_check_converge_delta_t(
    *, accumulated_cost, profile, n_radial, n_theta, n_xi, n_energy,
    freq_tol, delta_t, tolerance,
):
    """Check convergence by halving delta_t (smaller timestep) with other parameters fixed."""
    return _check_converge(
        "delta_t",
        accumulated_cost=accumulated_cost, profile=profile,
        n_radial=n_radial, n_theta=n_theta, n_xi=n_xi, n_energy=n_energy,
        freq_tol=freq_tol, delta_t=delta_t, tolerance=tolerance,
    )
