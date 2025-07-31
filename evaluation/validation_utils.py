import os
import json
from typing import Dict, List, Set, Tuple, Optional
import logging

DATASET_TASK_MAP = {
    "burgers_1d": {
        "tasks": {"cfl", "k", "w"},
        "cases": {"blast", "double_shock", "rarefaction", "sin", "sod"},
        "dummy_filename": "questions.json"  # burgers uses questions.json (plural)
    },
    "euler_1d": {
        "tasks": {"cfl", "k", "beta"},
        "cases": {"sod"},
        "dummy_filename": "questions.json"
    },
    "1D_heat_transfer": {
        "tasks": {"cfl", "n_space"},
        "cases": set(),  # No cases for heat transfer
        "dummy_filename": "question.json"  # heat transfer uses question.json (singular)
    },
    "2D_heat_transfer": {
        "tasks": {"dx", "error_threshold", "relax", "t_init"},
        "cases": set(),
        "dummy_filename": "question.json"
    }
}


def validate_dataset_task_combination(dataset: str, task: str, case: str = "") -> Tuple[bool, str]:
    """
    Validate if the given dataset-task-case combination is supported.
    
    Returns:
        (is_valid, error_message)
    """
    if dataset not in DATASET_TASK_MAP:
        available_datasets = ", ".join(DATASET_TASK_MAP.keys())
        return False, f"Unsupported dataset '{dataset}'. Available datasets: {available_datasets}"
    
    dataset_info = DATASET_TASK_MAP[dataset]
    
    if task not in dataset_info["tasks"]:
        available_tasks = ", ".join(sorted(dataset_info["tasks"]))
        return False, f"Unsupported task '{task}' for dataset '{dataset}'. Available tasks: {available_tasks}"
    
    if case and case not in dataset_info["cases"]:
        if dataset_info["cases"]:
            available_cases = ", ".join(sorted(dataset_info["cases"]))
            return False, f"Unsupported case '{case}' for dataset '{dataset}'. Available cases: {available_cases}"
        else:
            return False, f"Dataset '{dataset}' does not support cases, but case '{case}' was provided"
    
    if not case and dataset_info["cases"]:
        available_cases = ", ".join(sorted(dataset_info["cases"]))
        return False, f"Dataset '{dataset}' requires a case to be specified. Available cases: {available_cases}"
    
    return True, ""


def check_model_results_exist(dataset: str, task: str, model_name: str, case: str = "", zero_shot: bool = False) -> Tuple[bool, str, str]:
    """
    Check if model inference results exist for the given parameters.
    
    Returns:
        (exists, result_path, error_message)
    """
    case_prefix = f"{case}/" if case else ""
    flag = "zero_shot" if zero_shot else "iterative"
    result_path = f"results_model_attempt/{dataset}/{task}/{case_prefix}{flag}_{model_name}.json"
    
    if not os.path.exists(result_path):
        # Check if the directory structure exists
        result_dir = os.path.dirname(result_path)
        if not os.path.exists(result_dir):
            return False, result_path, f"Results directory does not exist: {result_dir}. Make sure you have run inference for this dataset-task combination."
        
        # List available models in this directory
        try:
            available_files = [f for f in os.listdir(result_dir) if f.endswith('.json') and flag in f]
            if available_files:
                available_models = [f.replace(f"{flag}_", "").replace(".json", "") for f in available_files]
                return False, result_path, (
                    f"Model results not found: {result_path}\n"
                    f"Available models for {dataset}/{task}/{case_prefix}{flag}: {', '.join(available_models)}\n"
                    f"Run inference for model '{model_name}' first, or choose from available models."
                )
            else:
                return False, result_path, (
                    f"No {flag} results found in {result_dir}. "
                    f"Run inference for this dataset-task combination first."
                )
        except OSError:
            return False, result_path, f"Cannot access results directory: {result_dir}"
    
    return True, result_path, ""


def check_dummy_data_exists(dataset: str, task: str, case: str = "", zero_shot: bool = False) -> Tuple[bool, str, str]:
    """
    Check if dummy/reference data exists for the given parameters.
    
    Returns:
        (exists, dummy_path, error_message)
    """
    case_prefix = f"{case}/" if case else ""
    flag = "zero_shot" if zero_shot or task in {"relax", "t_init"} else "iterative"
    
    # Use the correct filename based on dataset
    dummy_filename = DATASET_TASK_MAP.get(dataset, {}).get("dummy_filename", "question.json")
    dummy_path = f"data/{dataset}/{task}/{case_prefix}{flag}_{dummy_filename}"
    
    if not os.path.exists(dummy_path):
        return False, dummy_path, f"Reference data not found: {dummy_path}. This indicates a problem with the dataset setup."
    
    return True, dummy_path, ""


def validate_json_file(file_path: str) -> Tuple[bool, Optional[List], str]:
    """
    Validate that a JSON file exists and can be loaded.
    
    Returns:
        (is_valid, data, error_message)
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return False, None, f"Expected list format in {file_path}, got {type(data).__name__}"
        
        if len(data) == 0:
            return False, None, f"Empty dataset in {file_path}"
        
        return True, data, ""
    
    except FileNotFoundError:
        return False, None, f"File not found: {file_path}"
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON in {file_path}: {e}"
    except Exception as e:
        return False, None, f"Error reading {file_path}: {e}"


def setup_robust_evaluation(dataset: str, task: str, model_name: str, case: str = "", zero_shot: bool = False, logger: Optional[logging.Logger] = None) -> Tuple[bool, Dict]:
    """
    Perform comprehensive validation before starting evaluation.
    
    Returns:
        (success, info_dict)
        info_dict contains result_path, dummy_path, result_data, dummy_data, or error details
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Step 1: Validate dataset-task combination
    is_valid, error_msg = validate_dataset_task_combination(dataset, task, case)
    if not is_valid:
        logger.error(f"❌ Configuration validation failed: {error_msg}")
        return False, {"error": error_msg, "error_type": "invalid_configuration"}
    
    # Step 2: Check model results
    results_exist, result_path, error_msg = check_model_results_exist(dataset, task, model_name, case, zero_shot)
    if not results_exist:
        logger.error(f"❌ Model results validation failed: {error_msg}")
        return False, {"error": error_msg, "error_type": "missing_model_results", "result_path": result_path}
    
    # Step 3: Check dummy data
    dummy_exists, dummy_path, error_msg = check_dummy_data_exists(dataset, task, case, zero_shot)
    if not dummy_exists:
        logger.error(f"❌ Reference data validation failed: {error_msg}")
        return False, {"error": error_msg, "error_type": "missing_reference_data", "dummy_path": dummy_path}
    
    # Step 4: Validate JSON files
    is_valid, result_data, error_msg = validate_json_file(result_path)
    if not is_valid:
        logger.error(f"❌ Result file validation failed: {error_msg}")
        return False, {"error": error_msg, "error_type": "invalid_result_file", "result_path": result_path}
    
    is_valid, dummy_data, error_msg = validate_json_file(dummy_path)
    if not is_valid:
        logger.error(f"❌ Reference file validation failed: {error_msg}")
        return False, {"error": error_msg, "error_type": "invalid_reference_file", "dummy_path": dummy_path}
    
    logger.info("✅ All validation checks passed successfully")
    return True, {
        "result_path": result_path,
        "dummy_path": dummy_path,
        "result_data": result_data,
        "dummy_data": dummy_data
    }


def list_available_models(dataset: str, task: str, case: str = "", zero_shot: bool = False) -> List[str]:
    """List all available models for a given dataset-task-case combination."""
    case_prefix = f"{case}/" if case else ""
    flag = "zero_shot" if zero_shot else "iterative"
    result_dir = f"results_model_attempt/{dataset}/{task}/{case_prefix}"
    
    if not os.path.exists(result_dir):
        return []
    
    try:
        files = os.listdir(result_dir)
        json_files = [f for f in files if f.endswith('.json') and flag in f]
        models = [f.replace(f"{flag}_", "").replace(".json", "") for f in json_files]
        return sorted(models)
    except OSError:
        return []


def list_available_datasets() -> Dict[str, Dict]:
    """List all available datasets and their supported tasks/cases."""
    return DATASET_TASK_MAP.copy()


def print_usage_help(dataset: str = None, task: str = None):
    """Print helpful usage information."""
    print("\n" + "="*60)
    print("🔍 EVALUATION USAGE HELP")
    print("="*60)
    
    if dataset and task:
        # Specific help for dataset-task combination
        if dataset in DATASET_TASK_MAP:
            info = DATASET_TASK_MAP[dataset]
            if task in info["tasks"]:
                print(f"✅ Valid combination: {dataset} / {task}")
                if info["cases"]:
                    print(f"📁 Available cases: {', '.join(sorted(info['cases']))}")
                    models = []
                    for case in info["cases"]:
                        models.extend(list_available_models(dataset, task, case, False))
                        models.extend(list_available_models(dataset, task, case, True))
                else:
                    models = list_available_models(dataset, task, "", False)
                    models.extend(list_available_models(dataset, task, "", True))
                
                unique_models = sorted(set(models))
                if unique_models:
                    print(f"🤖 Available models: {', '.join(unique_models)}")
                else:
                    print("❌ No inference results found. Run inference first.")
    else:
        # General help
        print("📊 Available Datasets and Tasks:")
        for ds, info in DATASET_TASK_MAP.items():
            print(f"\n🔸 {ds}:")
            print(f"   Tasks: {', '.join(sorted(info['tasks']))}")
            if info["cases"]:
                print(f"   Cases: {', '.join(sorted(info['cases']))}")
        
        print(f"\n💡 Usage Examples:")
        print(f"   python eval.py -d burgers_1d -t cfl -c blast -m your_model_name")
        print(f"   python eval.py -d 1D_heat_transfer -t cfl -m your_model_name")
        print(f"   python eval.py -d 2D_heat_transfer -t dx -m your_model_name -z")
    
    print("="*60 + "\n")