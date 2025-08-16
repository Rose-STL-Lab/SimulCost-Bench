import json
from typing import Dict, Any, List
from abc import ABC, abstractmethod
import os
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatasetGenerator(ABC):
    """Base class for generating questions"""
    def __init__(self, doc_file: str):
        self.doc_file = doc_file

    def create_data(self, instruction, user):
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user},
        ]
    
    @abstractmethod
    def get_instruction_template(self) -> Any:
        """Get the standardized instruction template"""
        raise NotImplementedError

    def generate_user_template(self, question: str, QID: int) -> str:
        """Generate user template emphasizing accuracy and efficiency requirements"""
        with open(self.doc_file, 'r') as f:
            func_description = json.load(f)

        method_description = '\n'.join(json.dumps(
            method['description']) for method in func_description.values())

        return f"""QID: {QID} \n{question} \n\nAvailable functions: \n{method_description} """
    
    def generate_dataset(self, workflow: str, questions: List[Dict], zero_shot: bool) -> List[Dict]:
        instruction = self.get_instruction_template(zero_shot)
        instruction = instruction + "\nWorkflow:\n" + workflow # System prompt + Workflow

        dataset = []
        for q in questions:
            question = q["question"]
            idx = q["QID"]
            user_question = self.generate_user_template(question, idx)
            dataset.append({
                "QID": idx,
                "dummy_cost": q["dummy_cost"],
                "messages": self.create_data(instruction, user_question)
            })

        return dataset


    
