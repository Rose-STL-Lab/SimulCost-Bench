# Custom Model Integration Guide

Integrate your own models into SimulCost-Bench with support for single-model or multi-model testing workflows.

## 🚀 Quick Start

### Single Model Testing
1. Implement your model class with `invoke()` method
2. Configure `.env` file with model parameters
3. Run inference: `python inference/langchain_LLM.py -p custom_model -m any_name -d 1D_heat_transfer -t cfl -z`

### Multiple Models Testing  
1. Create JSON configuration file: `configs/custom_models.json`
2. Add all your models to the JSON config
3. Run batch scripts: `bash scripts/inference_eval/inference_eval_heat_1d.sh`

## 📋 Configuration Options

**Two configuration methods available:**

### Method 1: JSON Configuration (Recommended)
**Best for:** Multiple models, production workflows, team collaboration

```json
{
  "custom_models": {
    "qwen3_8b": {
      "custom_code": "/path/to/custom_inference.py",
      "model_path": "/data/models/Qwen3-8B",
      "custom_class": "Qwen3",
      "description": "Qwen3 8B parameter model"
    },
    "llama3_7b": {
      "custom_code": "/path/to/custom_inference.py",
      "model_path": "/data/models/Llama3-7B",
      "custom_class": "Llama3",
      "description": "Llama3 7B instruction-tuned model"
    }
  }
}
```

### Method 2: Environment Variables
**Best for:** Single model testing, quick experiments

Set in your `.env` file:
```ini
custom_code="/path/to/custom_inference.py"
model_path="/path/to/your/model"
custom_class="CustomModel"
```

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
        """
        # Your inference logic here
        # Process the messages and generate a response
        
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

## 🛠️ Implementation Guide

### Configuration Parameters

- **`custom_code`**: Path to your Python file containing the model class
- **`model_path`**: Path to your model files, weights, or configuration directory  
- **`custom_class`**: Exact name of the class that implements the interface
- **`description`**: *(Optional)* Human-readable description for documentation

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

## 🧰 Utilities

### List Available Models
```bash
python scripts/list_custom_models.py
```

### Batch Testing Multiple Models
```bash
# Edit model list in scripts/inference_eval/inference_eval_heat_1d.sh
model_provider="custom_model"
models=(
 "qwen3_8b"
 "qwen3_32b"
 "qwen3_235b_a22b"
)

# Run batch test
bash scripts/inference_eval/inference_eval_heat_1d.sh
```

## ⚡ Usage Examples

### Single Model (Environment Variables)
```bash
# 1. Set up .env file
echo 'custom_code="/path/to/custom_inference.py"' >> .env
echo 'model_path="/data/models/Qwen3-8B"' >> .env  
echo 'custom_class="Qwen3"' >> .env

# 2. Run inference
python inference/langchain_LLM.py -n 100 -p custom_model -m any_name -d 1D_heat_transfer -t cfl -z
```

### Multiple Models (JSON Configuration)
```bash  
# 1. Create JSON config with multiple models
# configs/custom_models.json (see example above)

# 2. Run batch inference
bash scripts/inference_eval/inference_eval_heat_1d.sh
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

## 🔧 Advanced Features

### Configuration Priority
1. **JSON Configuration** (highest priority) - When `configs/custom_models.json` exists
2. **Environment Variables** (fallback) - When JSON config is missing or model not found

### Error Handling
- **Model not found in JSON**: Explicit error with available model list
- **JSON malformed**: Automatic fallback to environment variables  
- **Missing configuration**: Clear error messages with resolution steps

### Development Tips
- Use `description` field in JSON for team collaboration
- Group related models by naming convention (e.g., `qwen3_8b`, `qwen3_32b`)
- Test single models with `.env` before adding to JSON config

## 💡 AI-Assisted Development

Don't want to implement from scratch? Check out our [prompt template](prompt.md) for AI tools (Cursor, Claude Code) to auto-generate your `custom_inference` code!