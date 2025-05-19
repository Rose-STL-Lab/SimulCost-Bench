from typing import Dict, Any

def get_heat_transfer_exp_summary(params: Dict[str, Any], accumulated_cost: int) -> Dict[str, Any]:
    """_summary_

    Args:
        params (Dict[str, Any]): _description_
        accumulated_cost (int): _description_
        budget (int): _description_

    Returns:
        str: _description_
    """
    converged = params["converged"]
    times = params["times"]
    sequence = params["sequence"]
    return {"converged": converged, "times": times, "sequence": sequence, "accumulated_cost": accumulated_cost}
