# Custom Model Integration Guide

This guide explains how to integrate your own custom models into the system by implementing the required interface and configuration.

## Overview

To use a custom model, you need to:
1. Create a Python file containing your model class
2. Implement the required `invoke()` method
3. Configure the `.env` file with the appropriate paths

## Required Implementation

Your custom model class must implement the following interface:

### Class Structure
```python
class CustomModel:
    def __init__(self, model_path: str):
        """
        Initialize your custom model.
        
        Args:
            model_path (str): Path to your model files/weights
        """
        self.model_path = model_path
        # Load your model here
        # Example: self.model = load_model(model_path)
    
    def invoke(self, messages: list[dict]) -> str:
        """
        Perform inference on the given messages.
        
        Args:
            messages (list[dict]): List of message dictionaries
                Each message should have the format:
                {"role": "system|user|assistant", "content": "message content"}
        
        Returns:
            str: JSON-formatted string containing the model's response
        """
        # Your inference logic here
        # Process the messages and generate a response
        
        # Return response
        return response
```

### Message Format

The `invoke()` method receives messages in the following format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
    {"role": "user", "content": "What's the weather like?"}
]
```

## Configuration

Add the following to your `.env` file:

```ini
# Custom Model Configuration
custom_code="/path/to/custom_inference.py"   # Path to your custom inference Python code
model_path="/path/to/your/custom_model"     # Path to your custom model
custom_class="CustomModel"                  # The class name within custom_code that will handle inference
```

### Configuration Parameters

- **`custom_code`**: Absolute or relative path to your Python file containing the model class
- **`model_path`**: Path to your model files, weights, or configuration directory
- **`custom_class`**: Exact name of the class in your Python file that implements the interface

## 🧠 Run Inference
```bash
python inference/langchain_LLM.py -n 10 -p custom_model -m qwen3_8b -d 1D_heat_transfer -t cfl -z
```

## Example Implementation

Here's a complete example of `custom_inference.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    
    def invoke(self, messages: list[dict]) -> str:
        """
        Perform inference on the given messages.
        
        Args:
            messages (list[dict]): List of message dictionaries
                Each message should have the format:
                {"role": "system|user|assistant", "content": "message content"}
        
        Returns:
            str: JSON-formatted string containing the model's response
        """
        print("================================================")
        print("Messages:", messages)
        print("================================================")

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
        print("Qwen3 Response:", response)
        
        return response
```

<!-- ```python
<!-- ## Testing Your Implementation

Before using your custom model, test it locally:

```python
# test_custom_model.py
from custom_inference import CustomModel

# Initialize your model
model = CustomModel("/path/to/your/model")

# Test with sample messages
test_messages = [
    {"role": "user", "content": "Hello, test message"}
]

# Get response
response = model.invoke(test_messages)
print(response)
``` -->