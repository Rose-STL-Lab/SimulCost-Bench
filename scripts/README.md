# 🛠️ Script Usage Guide

The `scripts/` directory contains automated scripts for streamlined execution of common workflows:

## 📁 Directory Structure
```
scripts/
├── inference_eval/   # Model inference and evaluation scripts
│   ├── eval_all.sh
│   ├── inference_eval_burgers_1d.sh
│   ├── inference_eval_euler_1d.sh
│   ├── inference_eval_heat_1d.sh
│   ├── inference_eval_heat_2d.sh
│   ├── inference_eval_ns_2d.sh
│   └── *.log        # Progress tracking logs
└── list_custom_models.py  # List available custom models
```

## 🔧 Quick Start

### Run Inference + Evaluation Pipeline
```bash
# Run inference and evaluation for individual problem types
bash scripts/inference_eval/inference_eval_heat_1d.sh
bash scripts/inference_eval/inference_eval_heat_2d.sh
bash scripts/inference_eval/inference_eval_burgers_1d.sh
bash scripts/inference_eval/inference_eval_euler_1d.sh
bash scripts/inference_eval/inference_eval_ns_2d.sh
```

### Run Evaluation Only
```bash
# Run evaluation on existing inference results (no inference)
bash scripts/inference_eval/eval_all.sh
```

## ⚙️ Switching Model Providers

Scripts support multiple model providers. **Simply modify the `model_provider` variable** at the top of each script.

### Supported Providers:
- **`bedrock`**: AWS Bedrock (Claude, Llama, Mistral models)
- **`openai`**: OpenAI API (GPT models)  
- **`gemini`**: Google Gemini API
- **`custom_model`**: Your custom model implementation

### How to Switch Providers:
**Edit the provider variable** in `scripts/inference_eval/inference_eval_*.sh`:

```bash
# Example: Switch from Bedrock to OpenAI
# model_provider="bedrock"     # Comment out current provider
model_provider="openai"        # Set new provider

# Update model list accordingly
models=(
  "gpt-4o"
  "gpt-4o-mini"
)
```

**No need to modify individual `-p` parameters** - they automatically use `$model_provider`.

## 🧠 Custom Models Configuration

### Multiple Custom Models Support
Scripts now support testing multiple custom models efficiently through JSON configuration.

#### 🚀 Quick Setup
```bash
# 1. List available custom models
python scripts/list_custom_models.py

# 2. Configure models in JSON (recommended)
# Edit configs/custom_models.json

# 3. Update script model arrays
# In scripts/inference_eval/inference_eval_*.sh:
model_provider="custom_model"
models=(
 "qwen3_8b"
 "qwen3_32b" 
 "qwen3_235b_a22b"
)
```

#### 📋 Configuration Methods

**Method 1: JSON Configuration (Multi-Model)**
Create `configs/custom_models.json`:
```json
{
  "custom_models": {
    "qwen3_8b": {
      "custom_code": "/path/to/custom_inference.py",
      "model_path": "/data/models/Qwen3-8B",
      "custom_class": "Qwen3"
    },
    "llama3_7b": {
      "custom_code": "/path/to/custom_inference.py",
      "model_path": "/data/models/Llama3-7B", 
      "custom_class": "Llama3"
    }
  }
}
```

**Method 2: Environment Variables (Single Model)**
Set in `.env` file:
```ini
custom_code="/path/to/custom_inference.py"
model_path="/path/to/model"
custom_class="CustomModel"
```

## 🔄 Resume Functionality

Scripts include automatic resume capability:
- **Execution logs** track completed commands
- **Failed runs** can be resumed from the last successful command
- **Progress tracking** prevents duplicate executions

## 📊 Script Features

- **Error handling**: Scripts stop on first error
- **Logging**: Track execution progress and resume capability
- **Batch processing**: Run multiple tasks, models, and cases systematically
- **Flexible parameters**: Easy modification of models, providers, and tasks