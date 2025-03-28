import argparse
import json
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str,
                        default="gemini-1.5-pro", help="Model name")
    parser.add_argument("-d", "--dataset", type=str,
                        default="heat_transfer", help="Domain of the dataset")
    parser.add_argument("-v", "--validation", action="store_true", default=False, help="Use validation dataset")
    parser.add_argument("-t", "--test", action="store_true", default=False, help="Use test dataset")
    parser.add_argument("-g", "--generated_version", type=int, default=1, help="Generate version")
    args = parser.parse_args()

    # Define paths
    if args.validation:
        excel_file_path = f"log/{args.dataset}/validation/tool_call_{args.model_name}_g{args.generated_version}.xlsx"
        md_file_path = f"evaluation/{args.dataset}/validation/{args.model_name}_g{args.generated_version}.md"
        result_file_path = f"result/{args.dataset}/validation/{args.model_name}_g{args.generated_version}.json"
    elif args.test:
        excel_file_path = f"log/{args.dataset}/test/tool_call_{args.model_name}_g{args.generated_version}.xlsx"
        md_file_path = f"evaluation/{args.dataset}/test/{args.model_name}_g{args.generated_version}.md"
        result_file_path = f"result/{args.dataset}/test/{args.model_name}_g{args.generated_version}.json"
    else:
        raise ValueError("Please specify either --validation or --test")
    
    question_file_path = f"data/{args.dataset}/question.json"
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(md_file_path), exist_ok=True)

    # Load the data
    with open(question_file_path, "r") as f:
        question_dataset = json.load(f)

    with open(result_file_path, "r") as f:
        results = json.load(f)

    df = pd.read_excel(excel_file_path)
    grouped_df = df.groupby("QID")

    with open(md_file_path, "w") as md_file:
        md_file.write(f"# {args.dataset.replace('_', ' ').title()} PDE Convergence Analysis - {args.model_name}\n\n")
        
        # Iterate through each question ID
        for idx, (qid, group_df) in enumerate(grouped_df):
            # Get the corresponding data from dataset and results using the index
            question_data = question_dataset[idx]
            result_data = results[idx]
            
            # Write question header
            md_file.write(f"# Question ID: {qid}\n\n")
            
            # Write problem statement from dataset
            md_file.write("## Problem Statement\n\n")
            md_file.write(f"{question_data['question']}\n\n")
            
            # Write simulation process header
            md_file.write("## Simulation Process\n\n")
            
            # Create table header from DataFrame columns
            columns = group_df.columns.tolist()
            md_file.write("| " + " | ".join(columns) + " |\n")
            md_file.write("|" + "|".join(["---" for _ in columns]) + "|\n")
            
            # Add table rows
            for _, row in group_df.iterrows():
                row_values = []
                for col in columns:
                    # Add values as they are
                    row_values.append(str(row[col]))
                
                md_file.write("| " + " | ".join(row_values) + " |\n")
            
            # Write the final response
            md_file.write("\n## Final Response\n\n")
            md_file.write("```\n")
            md_file.write(json.dumps(result_data, indent=2))
            md_file.write("\n```\n")
            
            # Add separator between questions
            md_file.write("\n---\n\n")

if __name__ == "__main__":
    main()