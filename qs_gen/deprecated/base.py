import json
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from api_call import *
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import numpy as np
import time


class QuestionGenerator(ABC):
    """Base class for generating questions"""
    def __init__(self, num_workers: int=10):
        self.num_workers = num_workers
        
    def generate_quesion_dataset(self, num_samples: int) -> List[Dict]:
        dummy_dataset = []
        dataset = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(self.generate_single_data, [self] * num_samples), 
                total=num_samples, 
                desc="Generating dummy data"
            ))

            for result in results:
                dummy_dataset.extend(result)
        
        for idx, dummy_data in enumerate(dummy_dataset):
            budget = dummy_data["budget"]
            dummy_sequence = dummy_data["dummy_sequence"]
            params = dummy_data["params"]
            question = self.generate_question(budget, params)
            dataset.append({
                "QID": idx,
                "budget": budget,
                "dummy_times": dummy_data["dummy_times"],
                "dummy_sequence": dummy_sequence,
                "question": question,
            })

        return dataset
    
    @staticmethod
    def generate_single_data(self)-> List[Dict]:
        process_id = os.getpid() % self.num_workers
        time.sleep(process_id * 2)
        np.random.seed(int(time.time()))
        budget, dummy_sequence, params = self.generate_cost_sequence()

        return[{
            "budget": budget,
            "dummy_times": len(dummy_sequence),
            "dummy_sequence": dummy_sequence,
            "params": params
        }]

    @abstractmethod
    def generate_random_parameters(self) -> Dict[str, Any]:
        """Generate physically reasonable random parameters"""
        raise NotImplementedError

    @abstractmethod
    def generate_question(self, params: Dict[str, Any], question_type: str) -> str:
        """Generate the question text"""
        raise NotImplementedError

    @abstractmethod
    def generate_cost_sequence(self) -> Any:
        """Generate the cost sequence"""
        raise NotImplementedError
