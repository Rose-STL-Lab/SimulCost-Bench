#!/usr/bin/env python3
"""
Extract and save prompt examples from SimulCost-Bench datasets.

This script extracts prompt examples (system + user messages) from the dataset files
and saves them in human-readable format for analysis and review.

Supports heat_1d, heat_2d, euler_1d, burgers_1d, and ns_2d simulations with configurable precision levels and inference modes.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_simulation_config(simulation: str) -> Dict[str, List[str]]:
    """
    Get configuration for supported simulations.
    
    Args:
        simulation: Name of the simulation (heat_1d, heat_2d, euler_1d, burgers_1d, or ns_2d)
        
    Returns:
        Dictionary containing precision levels and task types for the simulation
        
    Raises:
        ValueError: If simulation is not supported
    """
    configs = {
        "heat_1d": {
            "precision_levels": ["high", "low", "medium"],
            "tasks": ["cfl", "n_space"]
        },
        "heat_2d": {
            "precision_levels": ["high", "low", "medium"],
            "tasks": ["dx", "relax", "t_init", "error_threshold"]
        },
        "euler_1d": {
            "precision_levels": ["high", "low", "medium"], 
            "tasks": ["beta", "cfl", "k", "n_space"]
        },
        "burgers_1d": {
            "precision_levels": ["high", "low", "medium"],
            "tasks": ["beta", "cfl", "k", "n_space"]
        },
        "ns_2d": {
            "precision_levels": ["high", "low", "medium"],
            "tasks": ["mesh_x", "mesh_y", "omega_u", "omega_v", "omega_p", 
                     "diff_u_threshold", "diff_v_threshold", "res_iter_v_threshold"]
        }
    }
    
    if simulation not in configs:
        raise ValueError(f"Unsupported simulation: {simulation}. Supported: {list(configs.keys())}")
    
    return configs[simulation]


def find_dataset_files(data_dir: Path, simulation: str) -> List[Tuple[str, str, str, Path]]:
    """
    Find all dataset files for a given simulation.
    
    Args:
        data_dir: Path to the data directory
        simulation: Name of the simulation
        
    Returns:
        List of tuples containing (precision_level, task, inference_mode, file_path)
    """
    config = get_simulation_config(simulation)
    dataset_files = []
    
    # Map user-friendly names to actual directory names
    dir_mapping = {
        "ns_2d": "ns_channel_2d"
    }
    actual_simulation = dir_mapping.get(simulation, simulation)
    simulation_dir = data_dir / actual_simulation / "human_write"
    
    if not simulation_dir.exists():
        print(f"Warning: Directory {simulation_dir} does not exist")
        return dataset_files
    
    # Iterate through all precision levels
    for precision_level in config["precision_levels"]:
        precision_dir = simulation_dir / precision_level
        
        if not precision_dir.exists():
            print(f"Warning: Directory {precision_dir} does not exist")
            continue
            
        # Look for dataset files in this precision level directory
        for file_path in precision_dir.glob("*_dataset.json"):
            # Parse filename to extract task and inference mode
            # Expected format: {task}_{inference_mode}_dataset.json
            filename = file_path.stem  # Remove .json extension
            
            if filename.endswith("_dataset"):
                # Remove "_dataset" suffix
                base_name = filename[:-8]  # Remove "_dataset"
                
                # Determine inference mode
                if base_name.endswith("_zero_shot"):
                    inference_mode = "zero_shot"
                    task = base_name[:-10]  # Remove "_zero_shot"
                elif base_name.endswith("_iterative"):
                    inference_mode = "iterative"
                    task = base_name[:-10]  # Remove "_iterative"
                else:
                    # Fallback: try to split and identify
                    parts = base_name.split("_")
                    if "zero" in base_name and "shot" in base_name:
                        inference_mode = "zero_shot"
                        # Find where zero_shot starts
                        zero_idx = None
                        for i in range(len(parts) - 1):
                            if parts[i] == "zero" and parts[i + 1] == "shot":
                                zero_idx = i
                                break
                        task = "_".join(parts[:zero_idx]) if zero_idx else parts[0]
                    elif "iterative" in base_name:
                        inference_mode = "iterative"
                        # Find where iterative starts
                        iter_idx = None
                        for i, part in enumerate(parts):
                            if part == "iterative":
                                iter_idx = i
                                break
                        task = "_".join(parts[:iter_idx]) if iter_idx else parts[0]
                    else:
                        # Fallback
                        task = parts[0]
                        inference_mode = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
                
                # Validate task is in expected tasks
                if task in config["tasks"]:
                    dataset_files.append((precision_level, task, inference_mode, file_path))
                else:
                    print(f"Warning: Unknown task '{task}' in file {file_path}")
    
    return dataset_files


def extract_prompt_from_dataset(file_path: Path, qid: int = 1) -> Optional[Dict[str, str]]:
    """
    Extract prompt messages from a dataset file for a specific QID.
    
    Args:
        file_path: Path to the dataset JSON file
        qid: Question ID to extract (default: 1)
        
    Returns:
        Dictionary with 'system' and 'user' prompts, or None if not found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Find the entry with the specified QID
        for entry in data:
            if entry.get("QID") == qid:
                messages = entry.get("messages", [])
                
                system_prompt = ""
                user_prompt = ""
                
                # Extract system and user messages
                for message in messages:
                    if message.get("role") == "system":
                        system_prompt = message.get("content", "")
                    elif message.get("role") == "user":
                        user_prompt = message.get("content", "")
                
                return {
                    "system": system_prompt,
                    "user": user_prompt,
                    "metadata": {
                        "qid": qid,
                        "profile": entry.get("profile", ""),
                        "zero_shot": entry.get("zero_shot", False),
                        "case": entry.get("case", "")
                    }
                }
        
        print(f"Warning: QID {qid} not found in {file_path}")
        return None
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def format_prompt_as_markdown(prompt_data: Dict, simulation: str, precision_level: str, 
                             task: str, inference_mode: str) -> str:
    """
    Format prompt data into markdown format with proper math rendering.
    
    Args:
        prompt_data: Dictionary containing system/user prompts and metadata
        simulation: Name of the simulation
        precision_level: Precision level (high/medium/low)
        task: Task name
        inference_mode: Inference mode (zero_shot/iterative)
        
    Returns:
        Formatted markdown string with mathematical notation preserved
    """
    metadata = prompt_data.get("metadata", {})
    
    header = f"""# Prompt Example

## Metadata
- **Simulation**: {simulation}
- **Precision Level**: {precision_level}
- **Task**: {task}
- **Inference Mode**: {inference_mode}
- **QID**: {metadata.get('qid', 'N/A')}
- **Profile**: {metadata.get('profile', 'N/A')}
- **Zero Shot**: {metadata.get('zero_shot', 'N/A')}
- **Case**: {metadata.get('case', 'N/A')}

---

"""
    
    system_section = f"""## System Prompt

{prompt_data.get('system', 'No system prompt found')}

---

"""
    
    user_section = f"""## User Prompt

{prompt_data.get('user', 'No user prompt found')}
"""
    
    return header + system_section + user_section


def save_prompt_example(output_dir: Path, simulation: str, precision_level: str, 
                       task: str, inference_mode: str, formatted_prompt: str) -> None:
    """
    Save formatted prompt to markdown file with organized directory structure.
    
    Args:
        output_dir: Base output directory
        simulation: Name of the simulation
        precision_level: Precision level (high/medium/low)
        task: Task name
        inference_mode: Inference mode (zero_shot/iterative)
        formatted_prompt: Formatted prompt text in markdown format
    """
    # Create organized directory structure
    file_dir = output_dir / simulation / precision_level
    file_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with .md extension for markdown
    filename = f"{task}_{inference_mode}_prompt_example.md"
    file_path = file_dir / filename
    
    # Write formatted prompt to markdown file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_prompt)
        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Error saving {file_path}: {e}")


def main():
    """Main function to orchestrate prompt extraction process."""
    parser = argparse.ArgumentParser(
        description="Extract prompt examples from SimulCost-Bench datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_prompts.py -d heat_1d
  python extract_prompts.py -d heat_2d
  python extract_prompts.py -d euler_1d --output ./prompts --qid 2
  python extract_prompts.py -d burgers_1d
  python extract_prompts.py -d ns_2d
        """
    )
    
    parser.add_argument(
        "-d", "--dataset", 
        required=True,
        choices=["heat_1d", "heat_2d", "euler_1d", "burgers_1d", "ns_2d"],
        help="Dataset to extract prompts from"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory (default: ./data)"
    )
    
    parser.add_argument(
        "--output",
        type=Path, 
        default=Path("extracted_prompts"),
        help="Output directory for extracted prompts (default: ./extracted_prompts)"
    )
    
    parser.add_argument(
        "--qid",
        type=int,
        default=1,
        help="Question ID to extract (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.data_dir.exists():
        print(f"Error: Data directory {args.data_dir} does not exist")
        return 1
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting prompts for simulation: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output}")
    print(f"Target QID: {args.qid}")
    print("-" * 60)
    
    # Find all dataset files for the specified simulation
    dataset_files = find_dataset_files(args.data_dir, args.dataset)
    
    if not dataset_files:
        print(f"No dataset files found for simulation: {args.dataset}")
        return 1
    
    print(f"Found {len(dataset_files)} dataset files")
    
    # Process each dataset file
    extracted_count = 0
    for precision_level, task, inference_mode, file_path in dataset_files:
        print(f"Processing: {precision_level}/{task}/{inference_mode}")
        
        # Extract prompt from dataset file
        prompt_data = extract_prompt_from_dataset(file_path, args.qid)
        
        if prompt_data:
            # Format prompt as markdown for better readability and math rendering
            formatted_prompt = format_prompt_as_markdown(
                prompt_data, args.dataset, precision_level, task, inference_mode
            )
            
            # Save formatted prompt to file
            save_prompt_example(
                args.output, args.dataset, precision_level, task, 
                inference_mode, formatted_prompt
            )
            
            extracted_count += 1
        else:
            print(f"  Warning: Could not extract prompt from {file_path}")
    
    print("-" * 60)
    print(f"Successfully extracted {extracted_count}/{len(dataset_files)} prompts")
    print(f"Output saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())