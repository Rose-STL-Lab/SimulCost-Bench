# 🛠️ Script Usage Guide

The `scripts/` directory contains automated scripts for streamlined execution of common workflows:

## 📁 Directory Structure
```
scripts/
├── ds_gen/           # Dataset generation scripts
│   └── ds_gen_all.sh
├── inference_eval/   # Model inference and evaluation scripts
│   ├── inference_eval_all.sh
│   ├── inference_eval_burgers_1d.sh
│   ├── inference_eval_euler_1d.sh
│   ├── inference_eval_heat_1d.sh
│   └── inference_eval_heat_2d.sh
├── qs_gen/          # Question generation scripts
│   ├── qs_gen_burgers_1d.sh
│   ├── qs_gen_euler_1d.sh
│   ├── qs_gen_heat_1d.sh
│   └── qs_gen_heat_2d.sh
└── list_custom_models.py  # List available custom models
```

## 🔧 Quick Start

### 1. Generate Questions for Specific Tasks
```bash
# Generate questions for specific tasks
bash scripts/qs_gen/qs_gen_heat_1d.sh
```

### 2. Generate All Datasets
```bash
# Generate datasets for all tasks and modes
bash scripts/ds_gen/ds_gen_all.sh
```

### 2. Run Complete Inference + Evaluation Pipeline
```bash
# Execute all inference and evaluation tasks
bash scripts/inference_eval/inference_eval_all.sh

# Or run individual problem types
bash scripts/inference_eval/inference_eval_heat_1d.sh
```

## ⚙️ Modifying Provider Parameter (-p)

The `-p` parameter specifies the model provider. Currently supported providers:

- **`bedrock`**: AWS Bedrock (Claude models)
- **`openai`**: OpenAI API (GPT models)
- **`gemini`**: Google Gemini API
- **`custom_model`**: Your custom model implementation

### To Change Provider:
1. **Edit script files** in `scripts/inference_eval/`
2. **Modify the `-p` parameter** in commands like:
   ```bash
   # Current (Bedrock)
   python inference/langchain_LLM.py -p bedrock -m $model -d burgers_1d -t $task -c $case $mode
   
   # Change to OpenAI
   python inference/langchain_LLM.py -p openai -m $model -d burgers_1d -t $task -c $case $mode
   ```

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