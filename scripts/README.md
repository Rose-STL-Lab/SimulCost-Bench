# 🛠️ Script Usage Guide

The `scripts/` directory contains automated scripts for streamlined execution of common workflows:

## 📁 Directory Structure
```
scripts/
├── inference_eval/   # Model inference and evaluation scripts
│   ├── eval_all.sh             # Strict evaluation (fails on missing files)
│   ├── eval_all_partial.sh     # Partial evaluation (skips missing files)
│   ├── inference_eval_burgers_1d.sh
│   ├── inference_eval_euler_1d.sh
│   ├── inference_eval_heat_1d.sh
│   ├── inference_eval_heat_2d.sh
│   ├── inference_eval_ns_2d.sh
│   ├── inference_eval_ns_transient_2d.sh
│   └── *.log        # Progress tracking logs
└── list_custom_models.py  # List available custom models
```

## 🔧 Quick Start

### Run Inference + Evaluation
```bash
# Run inference and evaluation for individual problem types
bash scripts/inference_eval/inference_eval_heat_1d.sh
bash scripts/inference_eval/inference_eval_heat_2d.sh
bash scripts/inference_eval/inference_eval_burgers_1d.sh
bash scripts/inference_eval/inference_eval_euler_1d.sh
bash scripts/inference_eval/inference_eval_ns_2d.sh
bash scripts/inference_eval/inference_eval_ns_transient_2d.sh
```

### Run Evaluation Only
```bash
# Run evaluation on existing inference results (no inference)

# Strict mode: Fails immediately if any result files are missing
bash scripts/inference_eval/eval_all.sh -d <dataset>

# Partial mode: Skips missing result files and continues evaluation
bash scripts/inference_eval/eval_all_partial.sh -d <dataset>

# Examples:
bash scripts/inference_eval/eval_all.sh -d ns_2d                    # Strict evaluation
bash scripts/inference_eval/eval_all_partial.sh -d ns_2d           # Skip missing files
bash scripts/inference_eval/eval_all_partial.sh -d burgers_1d -d euler_1d  # Multiple datasets
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
  "gpt-5-2025-08-07"
)
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
- **Resume support**: Use `--resume` flag to continue interrupted runs
