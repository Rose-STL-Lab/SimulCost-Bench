import argparse
import os
import json
import sys
# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from prompt import get_init_archive, query, get_init_prompt
from evaluation.heat_transfer.eval import evaluate
from inference import setup_logging, parallel_inference, save_result
from dataset_gen import HeatTransferDatasetGenerator


def get_agent(archive_file: str, provider: str, model_name: str):
    if os.path.exists(archive_file):
        with open(archive_file, 'r') as f:
            archive = json.load(f)
    else:
        archive = get_init_archive()

    agent = query(archive, provider, model_name)
    archive.append(agent)
    with open(archive_file, 'w') as f:
        json.dump(archive, f, indent=4)

    return agent["workflow"], agent["code"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--validation_dataset_number", type=str,
                        default="3", help="Number of validation dataset")
    parser.add_argument("-p", "--provider", type=str,
                        default="gemini", help="Provider (openai/gemini)")
    parser.add_argument("-m", "--model_name", type=str,
                        default="gemini-1.5-pro", help="Model name")
    parser.add_argument("-g", "--generate_times", type=int,
                        default=2, help="Number of times to generate")
    parser.add_argument("-d", "--debug_max", type=int,
                        default=2, help="Number of times to debug the code")
    args = parser.parse_args()

    dataset_dir = f"data/heat_transfer/{args.model_name}"
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_file = f"{dataset_dir}/dataset.json"
    question_file = f"data/heat_transfer/question.json"
    archive_file = f"{dataset_dir}/agent.json"

    doc_file = "tool_documentation/heat_transfer.json"
    log_dir = "log/heat_transfer/validation"
    result_dir = "result/heat_transfer/validation"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("tool_result", exist_ok=True)
    log_file = lambda generation_times: f"{log_dir}/search_{args.model_name}_g{generation_times}.log"
    table_file = lambda generation_times: f"{result_dir}/tool_call_{args.model_name}_g{generation_times}.xlsx"
    result_file = lambda generation_times: f"{result_dir}/{args.model_name}_g{generation_times}.json"
    with open(question_file, "r") as f:
        questions = json.load(f)
    
    data_gen = HeatTransferDatasetGenerator(doc_file)

    # get the initial agent archive
    if os.path.exists(archive_file):
        with open(archive_file, 'r') as f:
            archive = json.load(f)
    else:
        archive = get_init_archive()
    

    for i in range(args.generate_times):
        logger = setup_logging(log_file(i+1))
        logger.info(f"===== Generation times: {i + 1} =====")
        logger.info(f"Using model_name: {args.model_name}" )
        system_prompt, prompt = get_init_prompt(archive)
        force_output_in_json = """Output only one JSON object with strictly valid format, and no additional explanation or text. All internal line breaks must be escaped using \\n. All property names must be enclosed in double quotes ("). Do not use single quotes or Python dict syntax."""
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt + "\n\n" + force_output_in_json}
        ]
        # get the agent 
        agent = query(msg_list, args.provider, args.model_name)

        for _ in range(args.debug_max):
            try:
                # base on the agent's workflow, generate a new dataset
                dataset = data_gen.generate_dataset(agent["workflow"], questions)
                save_result(dataset, dataset_file)
                valid_dataset = dataset[:int(args.validation_dataset_number)]
                # # use the new agent for inference
                results, tool_call_df = parallel_inference(valid_dataset, agent["code"], logger, args.provider, args.model_name)
                break
            except Exception as e:
                logger.error(f"Error: {e} During generation times: {i + 1}")
                logger.error(f"Trying to debug the code for the {_ + 1} time......")
                msg_list.append({"role": "assistant", "content": str(agent)})
                msg_list.append({"role": "user", "content": "Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought."})
                agent = query(msg_list, args.provider, args.model_name)
                continue
        # Save the results
        logger.info(f"Saving results to {result_file(i+1)}")
        save_result(results, result_file(i+1))
        # evaluate the results
        agent = evaluate(valid_dataset, results, agent)
        logger.info(f"Saving tool call to {table_file(i+1)}")
        tool_call_df.to_excel(table_file(i+1), index=False)
        archive.append(agent)
        with open(archive_file, 'w') as f:
            json.dump(archive, f, indent=4)
    

if __name__ == "__main__":
    main()