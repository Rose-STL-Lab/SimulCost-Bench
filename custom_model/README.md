# Custom Model Integration Guide

Integrate your own models into SimulCost-Bench with support for single-model or multi-model testing workflows.

## 🚀 Quick Start

### Single Model Testing
1. Implement your model class with `invoke()` method
2. Configure `.env` file with model parameters
3. Run inference with your model name: `python inference/langchain_LLM.py -p custom_model -m your_model_name -d heat_1d -t cfl -l medium -z`

### Multiple Models Testing  
1. Create JSON configuration file: `configs/custom_models.json`
2. Add all your models to the JSON config
3. Run batch scripts: `bash scripts/inference_eval/inference_eval_heat_1d.sh`

## 📋 Configuration Options

### Method 1: JSON Configuration (Recommended)
```json
{
  "custom_models": {
    "model_name": {
      "custom_code": "/path/to/custom_inference.py",
      "model_path": "/path/to/model",
      "custom_class": "ModelClass"
    }
  }
}
```

### Method 2: Environment Variables
```ini
custom_code="/path/to/custom_inference.py"
model_path="/path/to/your/model"
custom_class="CustomModel"
```

## Required Implementation

```python
class CustomModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # Load your model here
    
    def invoke(self, messages: list[dict]) -> str:
        # Process messages and generate response
        return response
```

**Message format:**
```python
[{"role": "system|user|assistant", "content": "message content"}]
```

## 🛠️ Configuration Parameters

- **`custom_code`**: Path to Python file with model class
- **`model_path`**: Path to model files/weights
- **`custom_class`**: Class name implementing the interface
- **`description`**: *(Optional)* Model description

## Example Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen3:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
    
    def invoke(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return response.strip()
```

## 🧰 Utilities

### List Available Models
```bash
python scripts/list_custom_models.py
```

## ⚡ Usage Examples

### Single Model
```bash
# Configure .env and run
python inference/langchain_LLM.py -p custom_model -m your_model_name -d heat_1d -t cfl -l medium -z
```

### Multiple Models
```bash
# Configure JSON and run batch
bash scripts/inference_eval/inference_eval_heat_1d.sh
```

## 🔧 Notes

- JSON configuration has priority over environment variables
- Test with `.env` before adding to JSON config
- Use consistent naming conventions for related models

## 💬 Code is cheap, show me your prompt

Don't want to implement from scratch? Check out our [prompt template](prompt.md) for AI tools (Cursor, Claude Code) to auto-generate your `custom_inference` code!