from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import requests

class Qwen3:
    def __init__(self, model_path: str):
        """
        Initialize your custom model.
        
        Args:
            model_path (str): Path to your model files/weights
        """
        self.model_path = model_path
        
        # Load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def invoke(self, messages: list[dict], profile: int = None) -> str:
        """
        Perform inference on the given messages.

        Args:
            messages (list[dict]): List of message dictionaries
                Each message should have the format:
                {"role": "system|user|assistant", "content": "message content"}
            profile (int): Profile identifier for the experiment (only used by BOAgent)

        Returns:
            str: JSON-formatted string containing the model's response
        """
        # print("================================================")
        # print("Messages:", messages)
        # print("================================================")
        # Prepare the model input        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # Parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print("thinking content:", thinking_content)
        # print("Qwen3 Response:", response)

        return response


class GPT_OSS:
    def __init__(self, model_path: str):
        """
        Initialize the GPT-OSS remote model.

        Args:
            model_path (str): URL of the inference endpoint the messages
                will be POSTed ("curled") to.
        """
        self.model_path = model_path

    def invoke(self, messages: list[dict], profile: int = None) -> str:
        """
        Perform inference by curling the messages to the remote endpoint.

        Args:
            messages (list[dict]): List of message dictionaries
                Each message should have the format:
                {"role": "system|user|assistant", "content": "message content"}
            profile (int): Profile identifier for the experiment (only used by BOAgent)

        Returns:
            str: JSON-formatted string containing the model's response
        """
        payload = {"model": "gpt-oss-120b", "messages": messages}

        response = requests.post(
            self.model_path,
            headers={"Content-Type": "application/json", "Authorization": "Bearer B05A8537-901E-497C-9DE2-CEB6BA58D4EE"},
            data=json.dumps(payload),
        )
        response.raise_for_status()

        # Parse an OpenAI-compatible chat completion response when available,
        # otherwise fall back to the raw response body.
        try:
            data = response.json()
        except ValueError:
            return response.text

        try:
            result = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            result = data if isinstance(data, str) else json.dumps(data)

        # print("GPT_OSS Response:", result)

        return result
    