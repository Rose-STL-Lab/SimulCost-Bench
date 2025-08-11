import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from inference.utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import List, Dict, Any
import logging
import argparse
import json
from evaluation.heat_transfer.eval import evaluate
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import importlib.util

def load_custom_model_config(model_name: str) -> dict:
    """
    Load specified model configuration from JSON config file
    
    Args:
        model_name (str): Model name, e.g. 'qwen3_8b'
        
    Returns:
        dict: Configuration dictionary containing custom_code, model_path, custom_class
        
    Raises:
        FileNotFoundError: If config file does not exist
        KeyError: If model name does not exist in config file
    """
    config_file = "configs/custom_models.json"
    
    # If JSON config file does not exist, fallback to .env file method
    if not os.path.exists(config_file):
        print(f"⚠️  JSON config file not found: {config_file}")
        print("⚠️  Falling back to .env file configuration...")
        return {
            "custom_code": os.getenv("custom_code"),
            "model_path": os.getenv("model_path"),
            "custom_class": os.getenv("custom_class")
        }
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        if model_name not in config_data.get("custom_models", {}):
            available_models = list(config_data.get("custom_models", {}).keys())
            print(f"❌ ERROR: Model '{model_name}' not found in JSON config!")
            print(f"📋 Available models in config: {available_models}")
            print(f"💡 Please either:")
            print(f"   1. Use one of the available models: {available_models}")
            print(f"   2. Add '{model_name}' to configs/custom_models.json")
            print(f"   3. Remove configs/custom_models.json to use .env configuration")
            print(f"")
            print(f"⚠️  REFUSING to fallback to .env to prevent testing wrong model!")
            raise ValueError(f"Model '{model_name}' not found in JSON config. Available models: {available_models}")
        
        model_config = config_data["custom_models"][model_name]
        print(f"✅ Loaded config for '{model_name}' from {config_file}")
        if "description" in model_config:
            print(f"📄 Description: {model_config['description']}")
        
        return {
            "custom_code": model_config["custom_code"],
            "model_path": model_config["model_path"],
            "custom_class": model_config["custom_class"]
        }
    
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        print("⚠️  Falling back to .env file configuration...")
        return {
            "custom_code": os.getenv("custom_code"),
            "model_path": os.getenv("model_path"),
            "custom_class": os.getenv("custom_class")
        }
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print("⚠️  Falling back to .env file configuration...")
        return {
            "custom_code": os.getenv("custom_code"),
            "model_path": os.getenv("model_path"),
            "custom_class": os.getenv("custom_class")
        }

def FORMAT_INST(request_keys):
    fields_list = request_keys.split(',') if isinstance(request_keys, str) else request_keys
    
    # Build field-specific requirements
    field_requirements = []
    field_requirements.append("- tool_reason MUST explain your choice (non-empty string)")
    field_requirements.append("- tool_name MUST be the exact function name from documentation (non-empty string)")
    field_requirements.append("- tool_args MUST be a dictionary with required parameters (non-empty dict)")
    
    if 'should_stop' in fields_list:
        field_requirements.append("- should_stop MUST be true or false (boolean)")
    
    field_requirements_text = '\n'.join(field_requirements)
    
    return f"""CRITICAL OUTPUT FORMAT REQUIREMENT:
You MUST reply with a valid JSON object containing ALL of these required keys: {str(request_keys)}

MANDATORY REQUIREMENTS:
- Each key MUST have a meaningful, non-empty value
- NO empty strings ("") are allowed for ANY field
{field_requirements_text}

DO NOT MISS ANY FIELDS. DO NOT USE EMPTY VALUES. Your response must be ONLY the JSON object, nothing else.
"""


class LLMAgentBase():
    def __init__(self, output_field: List[str], agent_name: str, logger: logging.Logger = None):
        global provider_global, model_name_global
        self.agent_name = agent_name
        self.output_field = output_field
        self.logger = logger
        if provider_global == "openai":
            self.llm = ChatOpenAI(
                model_name=model_name_global,
                seed=42,
                #temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            self.llm.bind(response_format={"type": "json_object"})

        elif provider_global == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=model_name_global,
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.llm.bind(response_format={"type": "json_object"})

        elif provider_global == "bedrock":
            if any(keyword in model_name_global for keyword in ["mistral", "llama", "jamba"]):
                model_id = model_name_global
            else:
                model_id = "us." + model_name_global
            print(model_id)

            self.llm = ChatBedrock(
                model_id=model_id,
                temperature=0,
                max_tokens=2048,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION_NAME")
            )
            self.llm.bind(response_format={"type": "json_object"})

        elif provider_global == "custom_model":
            # Load custom model parameters from JSON config (with .env fallback)
            model_config = load_custom_model_config(model_name_global)
            
            custom_code = model_config["custom_code"]
            model_path = model_config["model_path"]
            custom_class = model_config["custom_class"]

            if not all([custom_code, model_path, custom_class]):
                raise ValueError(f"Missing configuration for model '{model_name_global}'. Required: custom_code, model_path, custom_class")

            # Dynamically load user-defined code
            spec = importlib.util.spec_from_file_location("custom_inference", custom_code)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)

            # Get the user-specified custom model class
            CustomModelClass = getattr(custom_module, custom_class)

            # Instantiate the custom model class
            self.llm = CustomModelClass(model_path=model_path)
            
        else:
            raise ValueError(f"Unsupported provider: {provider_global}")
        
    def generate_output_instruction(self, instruction):
        request_keys = ",".join(self.output_field)
        return f"{instruction}\n{FORMAT_INST(request_keys)}"

    def query(self, messages: list[dict], instruction: str) -> list:
        messages[-1]["content"] += "\n" + self.generate_output_instruction(instruction)
        
        max_retries = 3
        for attempt in range(max_retries):
            if provider_global == "custom_model":
                raw_response = self.llm.invoke(messages)
                json_response = raw_response.strip()
            else:
                raw_response = self.llm.invoke(messages)
                json_response = raw_response.content.strip()
            
            # Log raw response for debugging
            if self.logger:
                attempt_text = f"attempt {attempt + 1}" if max_retries > 1 else ""
                self.logger.info(f"🤖 Raw model response {attempt_text}: {json.dumps(json_response, ensure_ascii=False)}")
            
            json_dict = find_json(json_response)

            # Check if all required fields are present and non-empty
            missing_fields = []
            for field in self.output_field:
                value = json_dict.get(field, "")
                if field == 'should_stop':
                    # should_stop can be boolean false, so check if it exists and is boolean
                    if field not in json_dict or not isinstance(value, bool):
                        missing_fields.append(field)
                else:
                    # For other fields, check if empty or whitespace
                    if not value or (isinstance(value, str) and value.strip() == ""):
                        missing_fields.append(field)
            
            if not missing_fields:
                # All fields are present and non-empty
                output_infos = []
                for field in self.output_field:
                    output_infos.append(json_dict.get(field, ""))
                return output_infos
            
            # If we have missing fields, prepare retry message
            if attempt < max_retries - 1:  # Don't add retry message on last attempt
                field_hints = []
                for field in missing_fields:
                    if field == 'tool_reason':
                        field_hints.append("- tool_reason: Provide detailed reasoning (non-empty string)")
                    elif field == 'tool_name':
                        field_hints.append("- tool_name: Use exact function name from documentation (non-empty string)")
                    elif field == 'tool_args':
                        field_hints.append("- tool_args: Provide parameter dictionary (non-empty dict)")
                    elif field == 'should_stop':
                        field_hints.append("- should_stop: Use true or false (boolean, not string)")
                
                field_hints_text = '\n'.join(field_hints)
                retry_message = f"\n\nCRITICAL ERROR: Your response failed validation. Problems found:\n- Missing or empty fields: {missing_fields}\n\nFIELD REQUIREMENTS:\n{field_hints_text}\n\nYou MUST provide ALL required fields: {self.output_field}\nNO empty strings (\"\") are allowed.\nRespond with ONLY a valid JSON object containing all required fields."
                
                # Create new messages for retry (don't modify original)
                retry_messages = messages.copy()
                retry_messages[-1] = retry_messages[-1].copy()
                retry_messages[-1]["content"] += retry_message
                messages = retry_messages

        # If all retries failed, return empty strings for missing fields but log the issue
        print(f"WARNING: Failed to get complete response after {max_retries} attempts. Missing fields: {missing_fields}")
        output_infos = []
        for field in self.output_field:
            output_infos.append(json_dict.get(field, ""))
        return output_infos

    def execute_tool(self, tool_reason: str, tool_name: str, tool_args: Dict[str, Any], tool_manager: ToolCallManager, profile: int) -> dict:
        tool_result, acc_cost = tool_manager.execute_tool_call(tool_reason, tool_name, tool_args, profile)
        
        return tool_result, acc_cost
    

class AgentSystem():
    def __init__(self, logger) -> None:
        self.logger = logger
        self.experiment_agent = None
        
    def get_experiment_agent(self, output_fields=None):
        if output_fields is None:
            output_fields = ["tool_reason", "tool_name", "tool_args"]
        
        if self.experiment_agent is None or self.experiment_agent.output_field != output_fields:
            self.experiment_agent = LLMAgentBase(output_fields, "Experiment Agent", self.logger)
        
        return self.experiment_agent

def parallel_inference(dataset: List[Dict], forward_func: str, logger: logging.Logger, provider: str, model_name: str, progress_file: str = None, resume_state: dict = None) -> tuple[List[Dict], pd.DataFrame]:
    """
    Run inference on dataset with improved resume support.
    
    Args:
        resume_state: Resume state from check_resume_state(), contains existing results and metadata
    """
    global provider_global, model_name_global
    provider_global = provider
    model_name_global = model_name
    namespace={}
    exec(forward_func, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)

    # Run inference (sequential with progress bar)
    agent = AgentSystem(logger)
    all_results = []
    all_tool_dfs = []
    start_idx = 0

    # Initialize with existing data if resuming
    if resume_state and resume_state.get('existing_results'):
        all_results = resume_state['existing_results'].copy()
        if resume_state.get('existing_tool_dfs'):
            all_tool_dfs = [pd.DataFrame(df_data) if df_data else pd.DataFrame() for df_data in resume_state['existing_tool_dfs']]
        start_idx = resume_state['start_idx']
        logger.info(f"Starting from sample {start_idx}/{len(dataset)} with {len(all_results)} existing results")

    # Process remaining samples
    for i, data in enumerate(tqdm(dataset[start_idx:], desc="Running inference", initial=start_idx, total=len(dataset))):
        try:
            result, tool_df = agent.forward(data)
            all_results.append(result)
            all_tool_dfs.append(tool_df)
            
            # Save progress after each sample
            if progress_file:
                progress_data = {
                    'results': all_results,
                    'tool_dfs': [df.to_dict('records') if not df.empty else [] for df in all_tool_dfs],
                    'completed_samples': len(all_results),
                    'total_samples': len(dataset)
                }
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            logger.error(f"Error processing sample {start_idx + i}: {e}")
            if progress_file:
                logger.info(f"Progress saved. You can resume with --resume flag.")
            raise

    final_tool_df = pd.concat(all_tool_dfs, ignore_index=True) if all_tool_dfs else pd.DataFrame()
    
    # Only clean up progress file if we completed the full requested dataset
    # Keep it for potential future extensions
    if progress_file and os.path.exists(progress_file) and len(all_results) == len(dataset):
        # Instead of deleting, mark as completed in metadata
        progress_data = {
            'results': all_results,
            'tool_dfs': [df.to_dict('records') if not df.empty else [] for df in all_tool_dfs],
            'completed_samples': len(all_results),
            'total_samples': len(dataset),
            'status': 'completed',
            'completion_time': pd.Timestamp.now().isoformat()
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        logger.info(f"All {len(dataset)} samples completed successfully. Progress file updated with completion status.")

    return all_results, final_tool_df

def ensure_file(path, default_content):
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
        print(f"[INFO] A directory has been made:{dirpath}")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_content, f, ensure_ascii=False, indent=2)
        print(f"[INFO] An empty file has been made: {path}")

def check_resume_state(progress_file: str, result_file: str, requested_samples: int, total_dataset_size: int, logger: logging.Logger) -> dict:
    """
    Check the current state for resume functionality.
    
    Returns:
        dict: {
            'action': 'skip' | 'resume' | 'restart' | 'extend',
            'completed_samples': int,
            'start_idx': int,
            'message': str,
            'existing_results': list | None,
            'existing_tool_dfs': list | None
        }
    """
    state = {
        'action': 'restart',
        'completed_samples': 0,
        'start_idx': 0,
        'message': '',
        'existing_results': None,
        'existing_tool_dfs': None
    }
    
    # Check progress file first (ongoing work)
    progress_data = None
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            completed_samples = progress_data.get('completed_samples', 0)
            total_samples_in_progress = progress_data.get('total_samples', 0)
            status = progress_data.get('status', 'in_progress')
            
            logger.info(f"Found progress file: {completed_samples}/{total_samples_in_progress} samples completed (status: {status})")
            
            # Handle completed work in progress file
            if status == 'completed':
                # This is completed work, check if we need more samples
                if completed_samples >= requested_samples:
                    state.update({
                        'action': 'skip',
                        'completed_samples': completed_samples,
                        'message': f"Already completed {completed_samples} samples (requested: {requested_samples}). No additional work needed."
                    })
                else:
                    # Need more samples - extend from completed work
                    state.update({
                        'action': 'extend',
                        'completed_samples': completed_samples,
                        'start_idx': completed_samples,
                        'message': f"Extending from {completed_samples} to {requested_samples} samples",
                        'existing_results': progress_data.get('results', []),
                        'existing_tool_dfs': progress_data.get('tool_dfs', [])
                    })
                return state
            elif completed_samples < total_samples_in_progress:
                # Incomplete work exists
                if completed_samples >= requested_samples:
                    # Already have enough samples
                    state.update({
                        'action': 'skip',
                        'completed_samples': completed_samples,
                        'message': f"Already completed {completed_samples} samples (requested: {requested_samples}). No additional work needed."
                    })
                else:
                    # Resume from where we left off
                    state.update({
                        'action': 'resume',
                        'completed_samples': completed_samples,
                        'start_idx': completed_samples,
                        'message': f"Resuming from sample {completed_samples}",
                        'existing_results': progress_data.get('results', []),
                        'existing_tool_dfs': progress_data.get('tool_dfs', [])
                    })
                return state
        except Exception as e:
            logger.warning(f"Failed to read progress file: {e}")
    
    # Check final result file (completed work)
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            completed_samples = len(existing_results)
            
            logger.info(f"Found result file with {completed_samples} completed samples")
            
            if completed_samples >= requested_samples:
                # Already have enough or more samples
                if completed_samples == requested_samples:
                    state.update({
                        'action': 'skip',
                        'completed_samples': completed_samples,
                        'message': f"Already completed exactly {requested_samples} samples. Use different -n value if you want different sample count."
                    })
                else:
                    state.update({
                        'action': 'skip',
                        'completed_samples': completed_samples,
                        'message': f"Already completed {completed_samples} samples (requested: {requested_samples}). Use -n {completed_samples + 1} or higher if you want more samples."
                    })
            else:
                # Need more samples - extend existing work
                state.update({
                    'action': 'extend',
                    'completed_samples': completed_samples,
                    'start_idx': completed_samples,
                    'message': f"Extending from {completed_samples} to {requested_samples} samples",
                    'existing_results': existing_results,
                    'existing_tool_dfs': []  # Will need to reconstruct from table file if needed
                })
            return state
        except Exception as e:
            logger.warning(f"Failed to read result file: {e}")
    
    # No existing work found
    state.update({
        'action': 'restart',
        'message': f"Starting fresh inference for {requested_samples} samples"
    })
    return state

def build_paths(dataset: str, task: str, flag: str, model_name: str,
                case: str | None = None) -> dict:
    """
    Automatically build all I/O paths based on dataset type.

    Parameters
    ----------
    dataset : "1D_heat_transfer" / "burgers_1d" / "euler_1d" / ...
    task    : e.g. "cfl"
    flag    : "zero_shot" or "iterative"
    model_name : used for log/result files
    case    : specific to burgers_1d/euler_1d; pass None for other datasets

    Returns
    -------
    dict, keys include dataset_file / archive_file / log_file /
                   result_file / table_file / result_dir / log_dir
    """
    # Common top-level directory
    case_suffix = f"/{case}" if case else ""
    result_dir = f"results_model_attempt/{dataset}/{task}{case_suffix}"
    log_dir    = f"log_model_tool_call/{dataset}/{task}{case_suffix}"
    # --------------------------------
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)

    # ==================================================
    # Determine intermediate path based on whether case exists
    # ==================================================
    if case:
        dataset_file = f"data/{dataset}/human_write/{task}_{case}_{flag}_dataset.json"
        archive_file = f"data/{dataset}/human_write/{task}_{case}_{flag}_agent.json"
    else:
        dataset_file = f"data/{dataset}/human_write/{task}_{flag}_dataset.json"
        archive_file = f"data/{dataset}/human_write/{task}_{flag}_agent.json"

    log_file    = f"{log_dir}/{flag}_{model_name}.log"
    result_file = f"{result_dir}/{flag}_{model_name}.json"
    table_file  = f"{result_dir}/{flag}_tool_call_{model_name}.xlsx"

    return dict(dataset_file=dataset_file, archive_file=archive_file,
                log_file=log_file, result_file=result_file,
                table_file=table_file, result_dir=result_dir,
                log_dir=log_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=2,
                        help="How many samples to test (ignored for burgers_1d and euler_1d)")
    parser.add_argument("-p", "--provider", default="gemini")
    parser.add_argument("-m", "--model_name", default="gemini-1.5-pro")
    parser.add_argument("-d", "--dataset", default="1D_heat_transfer",
                        help="Dataset domain")
    parser.add_argument("-t", "--task",
                        choices=["cfl", "n_space", "dx", "relax", "t_init",
                                 "error_threshold", "k", "w", "beta"],
                        default="cfl")
    parser.add_argument("-z", "--zero_shot", action="store_true")
    parser.add_argument("-c", "--case", default=None,
                        help="Case name for burgers_1d (blast, …) or euler_1d (sod, …)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous progress file")
    args = parser.parse_args()

    zero_shot = args.zero_shot or args.task in ["relax", "t_init"]
    flag = "zero_shot" if zero_shot else "iterative"

    paths = build_paths(args.dataset, args.task, flag,
                        args.model_name, case=args.case)

    ensure_file(paths["dataset_file"],  default_content=[])
    ensure_file(paths["archive_file"],  default_content=[])

    with open(paths["dataset_file"], "r") as f:
        dataset = json.load(f)
    with open(paths["archive_file"], "r") as f:
        agents = json.load(f)
    if not agents:
        raise RuntimeError("No agent found, please run dataset-generation first!")

    agent   = agents[-1]
    logger  = setup_logging(paths["log_file"], resume=args.resume)

    # Display dataset information
    logger.info(f"Dataset: {args.dataset}, Task: {args.task}, Mode: {'zero_shot' if zero_shot else 'iterative'}")
    logger.info(f"Dataset file: {paths['dataset_file']}")
    logger.info(f"Total samples available in dataset: {len(dataset)}")
    logger.info(f"Requested samples (-n): {args.num_samples}")

    # Create file paths
    progress_file = f"{paths['log_dir']}/{flag}_{args.model_name}_progress.json"
    
    # Determine requested sample count
    if args.dataset in ["burgers_1d", "euler_1d"]:
        requested_samples = len(dataset)
        logger.info(f"{args.dataset} detected — evaluating ALL {len(dataset)} samples.")
    else:
        available_samples = len(dataset)
        requested_samples = min(args.num_samples, available_samples)
        if args.num_samples > available_samples:
            logger.warning(f"Requested {args.num_samples} samples, but dataset only contains {available_samples} samples.")
            logger.warning(f"Using all available {available_samples} samples instead.")

    # Check resume state if resume flag is set
    resume_state = None
    if args.resume:
        resume_state = check_resume_state(
            progress_file, paths["result_file"], requested_samples, len(dataset), logger
        )
        
        logger.info(f"Resume check: {resume_state['message']}")
        
        # Handle different resume actions
        if resume_state['action'] == 'skip':
            logger.info("No additional work needed. Exiting.")
            return
        elif resume_state['action'] == 'extend':
            logger.info(f"Extending existing {resume_state['completed_samples']} samples to {requested_samples} samples.")
    else:
        # Fresh start - prepare dataset
        logger.info(f"Starting fresh inference for {requested_samples} samples")

    # Prepare dataset slice for processing
    if args.dataset in ["burgers_1d", "euler_1d"]:
        test_dataset = dataset
    else:
        test_dataset = dataset[:requested_samples]
        logger.info(f"Processing {len(test_dataset)} / {len(dataset)} samples from dataset.")
    
    results, tool_df = parallel_inference(
        test_dataset, agent["code"], logger,
        args.provider, args.model_name, progress_file, resume_state)

    tool_df.to_excel(paths["table_file"], index=False)
    logger.info(f"Saving results to {paths['result_file']}")
    save_result(results, paths["result_file"])

if __name__ == "__main__":
    main()