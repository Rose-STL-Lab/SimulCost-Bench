from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the JSON format that contains the following keys: {str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"


class LLMAgentBase():
    def __init__(self, output_field: List[str], agent_name: str):
        global provider_global, model_name_global
        self.agent_name = agent_name
        self.output_field = output_field
        if provider_global == "openai":
            self.llm = ChatOpenAI(
                model_name=model_name_global,
                seed=42,
                temperature=0,
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
            # print(model_name_global)
            self.llm = ChatBedrock(
                model_id="us." + model_name_global,
                temperature=0,
                max_tokens=2048,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION_NAME")
            )
            self.llm.bind(response_format={"type": "json_object"})

        elif provider_global == "custom_model":
            # Load custom model parameters from .env
            custom_code = os.getenv("custom_code")
            model_path = os.getenv("model_path")
            custom_class = os.getenv("custom_class")

            if not all([custom_code, model_path, custom_class]):
                raise ValueError("Missing one or more of custom_code, model_path, or custom_class in .env")

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
        
        if provider_global == "custom_model":
            json_response = self.llm.invoke(messages).strip()
        else:
            json_response = self.llm.invoke(messages).content.strip()
        json_dict = find_json(json_response)

        output_infos = []
        for value in json_dict.values():
            output_infos.append(value)

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
            self.experiment_agent = LLMAgentBase(output_fields, "Experiment Agent")
        
        return self.experiment_agent

def parallel_inference(dataset: List[Dict], forward_func: str, logger: logging.Logger, provider: str, model_name: str, progress_file: str = None, resume: bool = False) -> tuple[List[Dict], pd.DataFrame]:
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

    # Load progress if resuming
    if resume and progress_file and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            all_results = progress_data.get('results', [])
            all_tool_dfs_data = progress_data.get('tool_dfs', [])
            # Reconstruct DataFrames from saved data
            all_tool_dfs = [pd.DataFrame(df_data) if df_data else pd.DataFrame() for df_data in all_tool_dfs_data]
            start_idx = len(all_results)
            logger.info(f"Resuming from sample {start_idx}/{len(dataset)}")
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}. Starting from beginning.")
            start_idx = 0

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
    
    # Clean up progress file if all samples completed successfully
    if progress_file and os.path.exists(progress_file) and len(all_results) == len(dataset):
        os.remove(progress_file)
        logger.info("All samples completed successfully. Progress file removed.")

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

def build_paths(dataset: str, task: str, flag: str, model_name: str,
                case: str | None = None) -> dict:
    """
    根据数据集类型自动拼接所有 I/O 路径。

    参数
    ----
    dataset : "1D_heat_transfer" / "burgers_1d" / "euler_1d" / ...
    task    : 如 "cfl"
    flag    : "zero_shot" or "iterative"
    model_name : 用于日志/结果文件
    case    : burgers_1d/euler_1d 专用；其余数据集传 None

    返回
    ----
    dict, 键包括 dataset_file / archive_file / log_file /
                   result_file / table_file / result_dir / log_dir
    """
    # 通用顶层目录
    case_suffix = f"/{case}" if case else ""
    result_dir = f"results_model_attempt/{dataset}/{task}{case_suffix}"
    log_dir    = f"log_model_tool_call/{dataset}/{task}{case_suffix}"
    # --------------------------------
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir,    exist_ok=True)

    # ==================================================
    # 根据是否有 case 决定中间层路径
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
    logger  = setup_logging(paths["log_file"])

    # --------- 跑完整 case，否则按 -n ----------
    if args.dataset in ["burgers_1d", "euler_1d"]:
        test_dataset = dataset
        logger.info(f"{args.dataset} detected — evaluating ALL {len(dataset)} samples.")
    else:
        test_dataset = dataset[:args.num_samples]
        logger.info(f"Evaluating first {len(test_dataset)} / {len(dataset)} samples.")

    # Create progress file path
    progress_file = f"{paths['log_dir']}/{flag}_{args.model_name}_progress.json"
    
    results, tool_df = parallel_inference(
        test_dataset, agent["code"], logger,
        args.provider, args.model_name, progress_file, args.resume)

    tool_df.to_excel(paths["table_file"], index=False)
    logger.info(f"Saving results to {paths['result_file']}")
    save_result(results, paths["result_file"])

if __name__ == "__main__":
    main()