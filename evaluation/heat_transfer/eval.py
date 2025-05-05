from typing import List, Dict

def evaluate(base_dataset: List[Dict], result_dataset: List[Dict], agent: Dict) -> Dict:
    comparison_results = []
    total_success = 0
    total_cost = 0.0
    total_dummy_cost = 0.0  # For dummy solution

    for i in range(len(result_dataset)):
        base = base_dataset[i]
        llm_result = result_dataset[i]

        #  handling error
        format_error = False
        tool_call_error = False
        out_of_budget_but_no_final_response = False
        if "error" in llm_result:
            if "find_json" in llm_result["error"]:
                format_error = True
            elif "executing tool call" in llm_result["error"]:
                tool_call_error = True
            elif "budget exceeded" in llm_result["error"]:
                out_of_budget_but_no_final_response = True
            out_of_budget = None
            converged = None
            success = False
            cost = 0.0
        else:
            out_of_budget = True if llm_result["accumulated_cost"] > base["dummy_cost"] else False
            success = True if llm_result["converged"] and not out_of_budget else False
            converged = llm_result["converged"]
            cost = llm_result["accumulated_cost"]

        if success:
            total_success += 1
        total_cost += cost
        total_dummy_cost += base["dummy_cost"]

        comparison_results.append({
            "success": success,
            "out_of_budget": out_of_budget,
            "converged": converged,
            "format_error": format_error,
            "tool_call_error": tool_call_error,
            "out_of_budget_but_no_final_response": out_of_budget_but_no_final_response
        })
    
    success_rate = sum([llm_result["success"] for llm_result in comparison_results]) / len(comparison_results)
    # exclude the cases that out of budge are None
    out_of_budget_result = [llm_result["out_of_budget"] for llm_result in comparison_results if llm_result["out_of_budget"] is not None]
    out_of_budget_rate = sum(out_of_budget_result) / len(out_of_budget_result)
    # exclude the cases that converged are None
    converged_result = [llm_result["converged"] for llm_result in comparison_results if llm_result["converged"] is not None]
    converged_rate = sum(converged_result) / len(converged_result)
    
    # Cost efficiency
    model_cost_efficiency = total_success / total_cost if total_cost > 0 else 0.0
    dummy_cost_efficiency = len(base_dataset) / total_dummy_cost if total_dummy_cost > 0 else 0.0
    relative_cost_efficiency = model_cost_efficiency / dummy_cost_efficiency if dummy_cost_efficiency > 0 else 0.0

    # Add to agent
    agent["success_rate"] = success_rate
    agent["out_of_budget_rate"] = out_of_budget_rate
    agent["converged_rate"] = converged_rate
    agent["model_cost_efficiency"] = model_cost_efficiency
    agent["dummy_cost_efficiency"] = dummy_cost_efficiency
    agent["relative_cost_efficiency"] = relative_cost_efficiency

    return agent
