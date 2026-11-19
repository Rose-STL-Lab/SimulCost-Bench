"""Wall-clock kill limit for CGYRO simulations.

runCgyro (costsci_tools/wrappers/cgyro.py) blocks on a shell subprocess with no
timeout. Here it runs in a spawned child process; on timeout the child and all
of its descendants (shell -> hydra runner -> mpirun -> cgyro ranks) are killed.
"""
import glob
import json
import multiprocessing
import os
import queue
import shutil
import signal
import time

import psutil
from costsci_tools.wrappers.cgyro import runCgyro, _get_sim_path

CGYRO_MAX_WALL_CLOCK_S = 24 * 3600
_TERM_GRACE_S = 30
_RESULT_WAIT_S = 60

_RUNNERS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "costsci_tools", "runners")
)
_TIMEOUT_MARKER = "timeout.json"


class CgyroTimeout(Exception):
    def __init__(self, elapsed_s, timeout_s):
        self.elapsed_s = float(elapsed_s)
        self.timeout_s = float(timeout_s)
        super().__init__(
            f"CGYRO simulation killed after {self.elapsed_s:.1f}s "
            f"(wall-clock limit {self.timeout_s:.1f}s)"
        )


def _sim_dir(profile, n_radial, n_theta, freq_tol, delta_t, n_xi, n_energy):
    # Must match the directory naming used by runCgyro / get_res_cgyro
    return _get_sim_path(
        f"sim_res/cgyro/{profile}_n_radial_{n_radial}_n_theta_{n_theta}_freq_tol_{freq_tol}_delta_t_{delta_t}_n_xi_{n_xi}_n_energy_{n_energy}"
    )


def is_cached(profile, n_radial, n_theta, freq_tol, delta_t, n_xi, n_energy):
    """True if a completed run (meta.json with cost) already exists for these params."""
    meta_file = os.path.join(
        _sim_dir(profile, n_radial, n_theta, freq_tol, delta_t, n_xi, n_energy), "meta.json"
    )
    if not os.path.exists(meta_file):
        return False
    with open(meta_file, "r") as f:
        return "cost" in json.load(f)


def _child(result_queue, kwargs):
    os.setsid()
    try:
        result_queue.put(("ok", runCgyro(**kwargs)))
    except BaseException as e:
        try:
            result_queue.put(("err", e))
        except Exception:
            result_queue.put(("err", RuntimeError(repr(e))))


def _kill_group(proc):
    """Kill the child and every descendant; MPI ranks may sit in their own process groups."""
    try:
        root = psutil.Process(proc.pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        procs = []
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=_TERM_GRACE_S)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=_TERM_GRACE_S)
    if alive:
        raise RuntimeError(f"Failed to kill CGYRO processes: {[p.pid for p in alive]}")
    proc.join()


def _cleanup_runner_dir():
    """Undo the state the CGYRO runner leaves behind when killed before its finally block."""
    input_path = os.path.join(_RUNNERS_DIR, "input.cgyro")
    backup_path = os.path.join(_RUNNERS_DIR, "input.cgyro.backup")
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, input_path)
        os.remove(backup_path)
    for pattern in ("bin.cgyro.*", "out.cgyro.*", "input.cgyro.gen"):
        for path in glob.glob(os.path.join(_RUNNERS_DIR, pattern)):
            try:
                os.remove(path)
            except OSError:
                pass


def run_cgyro_with_deadline(timeout_s, *, profile, n_radial, n_theta, error_tol,
                            freq_tol, delta_t, n_xi, n_energy):
    """runCgyro with a hard wall-clock limit. Returns (cost, converged) or raises CgyroTimeout."""
    kwargs = dict(
        profile=profile, n_radial=n_radial, n_theta=n_theta, error_tol=error_tol,
        freq_tol=freq_tol, delta_t=delta_t, n_xi=n_xi, n_energy=n_energy,
    )
    if timeout_s <= 0:
        raise CgyroTimeout(0.0, timeout_s)

    # Completed runs are served from cache by runCgyro without spawning anything
    if is_cached(profile, n_radial, n_theta, freq_tol, delta_t, n_xi, n_energy):
        return runCgyro(**kwargs)

    sim_dir = _sim_dir(profile, n_radial, n_theta, freq_tol, delta_t, n_xi, n_energy)
    marker_file = os.path.join(sim_dir, _TIMEOUT_MARKER)
    if os.path.exists(marker_file):
        with open(marker_file, "r") as f:
            marker = json.load(f)
        # Only reuse a timeout recorded with at least the budget we have now
        if marker["timeout_s"] >= timeout_s:
            print(f"Using cached CGYRO timeout from {marker_file}")
            raise CgyroTimeout(min(marker["elapsed_s"], timeout_s), timeout_s)

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(result_queue, kwargs))
    start = time.monotonic()
    proc.start()
    try:
        proc.join(timeout_s)
        if proc.is_alive():
            _kill_group(proc)
            elapsed = time.monotonic() - start
            _cleanup_runner_dir()
            os.makedirs(sim_dir, exist_ok=True)
            with open(marker_file, "w") as f:
                json.dump({"elapsed_s": elapsed, "timeout_s": timeout_s, **kwargs}, f, indent=4)
            print(f"⏱️ CGYRO run exceeded {timeout_s:.1f}s wall-clock limit and was killed ({kwargs})")
            raise CgyroTimeout(elapsed, timeout_s)
    finally:
        # Covers KeyboardInterrupt etc. while waiting
        if proc.is_alive():
            _kill_group(proc)
            _cleanup_runner_dir()

    try:
        status, payload = result_queue.get(timeout=_RESULT_WAIT_S)
    except queue.Empty:
        raise RuntimeError(
            f"CGYRO child process exited with code {proc.exitcode} without returning a result ({kwargs})"
        )
    if status == "err":
        raise payload
    return payload
