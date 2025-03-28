from typing import List, Dict
import json

def save_dataset(examples: List[Dict], filename: str):
    """Save the training examples to a JSON file"""
    # Convert numpy types to Python native types
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=4)


def load_json(filename: str) -> List[Dict]:
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
