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
        elif provider_global == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=model_name_global,
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif provider_global == "claude" or provider_global == "deepseek":
            self.llm = ChatBedrock(
                model_id="us." + model_name_global,
                temperature=0,
                max_tokens=2048,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION_NAME")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider_global}")
        self.llm.bind(response_format={"type": "json_object"})
        
    def generate_output_instruction(self, instruction):
        request_keys = ",".join(self.output_field)
        return f"{instruction}\n{FORMAT_INST(request_keys)}"

    def query(self, messages: list[dict], instruction: str) -> list:
        messages[-1]["content"] += "\n" + self.generate_output_instruction(instruction)
        json_response = self.llm.invoke(messages).content.strip()
        json_dict = find_json(json_response)

        output_infos = []
        for value in json_dict.values():
            output_infos.append(value)

        return output_infos

    def execute_tool(self, tool_reason: str, tool_name: str, tool_args: Dict[str, Any], tool_manager: ToolCallManager) -> dict:
        tool_result, acc_cost = tool_manager.execute_tool_call(tool_reason, tool_name, tool_args)
        
        return tool_result, acc_cost
    

class AgentSystem():
    def __init__(self, logger) -> None:
        self.logger = logger

def parallel_inference(dataset: List[Dict], forward_func: str, logger: logging.Logger, provider: str, model_name: str) -> tuple[List[Dict], pd.DataFrame]:
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
    
    # Run inference
    agent = AgentSystem(logger)
    max_workers = 2
    all_results = []
    all_tool_dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(agent.forward, data) for data in dataset}

        for future in as_completed(futures):
            result, tool_df = future.result()
            all_results.append(result)
            all_tool_dfs.append(tool_df)

    final_tool_df = pd.concat(all_tool_dfs, ignore_index=True) if all_tool_dfs else pd.DataFrame()

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_index", type=int,
                         default=3, help="Start index")
    parser.add_argument("-p", "--provider", type=str,
                        default="gemini", help="Provider (openai/gemini)")
    parser.add_argument("-m", "--model_name", type=str,
                        default="gemini-1.5-pro", help="Model name")
    parser.add_argument("-H",  "--human_version", action="store_true",
                         default=False, help="Human-written version")
    parser.add_argument("-d", "--dataset", type=str,
                        default="heat_transfer", help="Domain of the dataset")
    args = parser.parse_args()

    result_dir = f"result/{args.dataset}/test"
    os.makedirs(result_dir, exist_ok=True)
    log_dir = f"log/{args.dataset}/test"
    os.makedirs(log_dir, exist_ok=True)

    if args.human_version:
        dataset_file = f"data/{args.dataset}/human_write/dataset.json"
        log_file = f"{log_dir}/human_write_{args.model_name}.log"
        archive_file = f"data/{args.dataset}/human_write/agent.json"
        result_file = f"{result_dir}/human_write_{args.model_name}.json"
        table_file = f"{result_dir}/tool_call_human_write_{args.model_name}.xlsx"
    else:
        dataset_file = f"data/{args.dataset}/{args.model_name}/dataset.json"
        log_file = f"{log_dir}/{args.model_name}.log"
        archive_file = f"data/{args.dataset}/{args.model_name}/agent.json"
        result_file = f"{result_dir}/{args.model_name}.json"
        table_file = f"{result_dir}/tool_call_{args.model_name}.xlsx"

    ensure_file(dataset_file, default_content=[])
    ensure_file(archive_file, default_content=[])

    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    with open(archive_file, "r") as f:
        agents = json.load(f)
        
    # use the latest agent
    agent = agents[-1]
    logger = setup_logging(log_file)

    test_dataset = dataset[args.start_index:]

    results, tool_call_df = parallel_inference(test_dataset, agent["code"], logger, args.provider, args.model_name)

    logger.info(f"Saving results to {result_file}")
    save_result(results, result_file)
    # evaluate the results
    agent = evaluate(test_dataset, results, agent)
    logger.info(f"Saving tool call to {table_file}")
    tool_call_df.to_excel(table_file, index=False)
    logger.info(f"Success rate: {agent['success_rate']}")
    logger.info(f"Converged rate: {agent['converged_rate']}")
    logger.info(f"Out of budget rate: {agent['out_of_budget_rate']}")

if __name__ == "__main__":
    main()