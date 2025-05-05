from typing import Dict, Any

def get_heat_transfer_exp_summary(*, converged: bool, times: int, sequence: list, qid: int, accumulated_cost: int) -> Dict[str, Any]:
    """_summary_

    Args:
        converged (bool): Whether convergence was achieved
        times (int): Number of iterations
        sequence (list): Resolution history
        accumulated_cost (int): Total cost accumulated

    Returns:
        Dict[str, Any]: Summary result
    """
    return {
        "QID": qid,
        "converged": converged,
        "times": times,
        "sequence": sequence,
        "accumulated_cost": accumulated_cost,
    }

