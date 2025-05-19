import argparse
import json
import os
import pandas as pd


def md_escape(val):
    """
    Escape characters that would break GitHub-flavoured Markdown tables
    and normalise NaNs to empty strings.
    """
    if pd.isna(val):
        return ""
    return str(val).replace("|", r"\|").replace("\n", "<br>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str,
                        default="gemini-1.5-pro", help="Model name")
    parser.add_argument("-d", "--dataset", type=str,
                        default="heat_transfer", help="Domain of the dataset")
    parser.add_argument("-v", "--validation", action="store_true", default=False,
                        help="Use validation dataset")
    parser.add_argument("-t", "--test", action="store_true", default=False,
                        help="Use test dataset")
    parser.add_argument("-g", "--generated_version", type=int, default=None,
                        help="Generate version (optional)")
    args = parser.parse_args()

    # Decide effective model name
    if args.generated_version is None:
        args.model_name = f"human_write_{args.model_name}"

    # Check mode
    if args.validation == args.test:
        raise ValueError("Specify exactly one of --validation or --test")
    subdir = "validation" if args.validation else "test"

    # Build file paths
    suffix = f"_g{args.generated_version}" if args.generated_version else ""
    base_name = f"{args.model_name}{suffix}"
    excel_path = f"result/{args.dataset}/{subdir}/tool_call_{base_name}.xlsx"
    json_path  = f"result/{args.dataset}/{subdir}/{base_name}.json"
    md_path    = f"evaluation/{args.dataset}/{subdir}/{base_name}.md"

    # Fallback to log/ if Excel is missing
    alt_excel = f"log/{args.dataset}/{subdir}/tool_call_{base_name}.xlsx"
    if not os.path.exists(excel_path) and os.path.exists(alt_excel):
        excel_path = alt_excel

    # Ensure directories exist
    for p in (md_path, excel_path, json_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # ---------- load data ----------
    question_path = f"data/{args.dataset}/question.json"
    with open(question_path, "r") as f:
        questions = json.load(f)
    with open(json_path, "r") as f:
        results = json.load(f)

    # ---------- build lookup dicts ----------
    # *Compatibility patch* — fall back to index if no "QID" / "qid"
    # ---------------------------------------------------------------
    def get_qid(idx, item):
        """Return str(QID) when present, otherwise use list index."""
        return str(item.get("QID") or item.get("qid") or idx)

    qid_to_question = {get_qid(i, q): q for i, q in enumerate(questions)}  # <-- modified
    qid_to_result   = {get_qid(i, r): r for i, r in enumerate(results)}    # <-- modified

    # ---------- read Excel ----------
    df = pd.read_excel(excel_path)
    grouped = df.groupby("QID")

    # ---------- write Markdown ----------
    with open(md_path, "w") as md:
        md.write(f"# {args.dataset.replace('_', ' ').title()} PDE Convergence Analysis - {args.model_name}\n\n")

        for qid, group_df in grouped:
            qid_str = str(qid)                                      # <-- modified
            question_data = qid_to_question.get(qid_str)
            result_data   = qid_to_result.get(qid_str)

            # Skip group if we cannot find matching data
            if question_data is None or result_data is None:
                continue

            md.write(f"# Question ID: {qid}\n\n")
            md.write("## Problem Statement\n\n")
            md.write(f"{question_data['question']}\n\n")

            md.write("## Simulation Process\n\n")
            cols = group_df.columns.tolist()
            md.write("| " + " | ".join(cols) + " |\n")
            md.write("|" + "|".join(["---"] * len(cols)) + "|\n")

            for _, row in group_df.iterrows():
                md.write("| " + " | ".join(md_escape(row[c]) for c in cols) + " |\n")

            md.write("\n## Final Response\n\n```\n")
            md.write(json.dumps(result_data, indent=2, ensure_ascii=False))
            md.write("\n```\n\n---\n\n")


if __name__ == "__main__":
    main()
