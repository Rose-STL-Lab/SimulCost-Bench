# SimulCost-Bench

A comprehensive benchmark for evaluating Large Language Models (LLMs) on computational physics simulation parameter optimization tasks. This benchmark assesses how well LLMs can determine optimal numerical parameters for solving partial differential equations.

## 📚 Table of Contents

- [Environment Setup](#📦-environment-setup)
- [Tasks and Zero-Shot Support](#📋-tasks-and-zero-shot-support)
- [Generate Questions](#🕵️-generate-questions)
- [Generate Benchmark Datasets](#🚀-generate-benchmark-datasets)
- [Configure Model Providers](#📄-configure-model-providers)
- [Run Inference](#🧠-run-inference)
- [Resume Functionality](#🔄-resume-functionality)
- [Evaluate Performance](#📊-evaluate-models-performance)
- [Tabulate Results](#🗂️-tabulate-evaluation-results)
- [Evaluated Models](#🤖-evaluated-models)
- [Script Usage Guide](#🛠️-script-usage-guide)

## 📦 Environment Setup

### Clone repository

To clone the repository and initialize the submodule, use the following commands:

```bash
# Clone repository with submodule initialization
git clone --recursive https://github.com/Rose-STL-Lab/SimulCost-Bench.git

# If you've already cloned the repository without --recursive, run the following to initialize the submodule:
cd SimulCost-Bench
git submodule update --init --recursive
```

### Create Conda environment
```bash
# Create Conda environment
conda env create -f environment.yml
conda activate simulcost
```

### Install dependencies with Poetry
```bash
# Install dependencies with Poetry
poetry install --no-root
```

## 📋 Tasks and Zero-Shot Support

The table below summarizes the available tasks for each simulation type and indicates whether each task supports zero-shot inference.

| Simulation Type             | Task Type        | Iterative & Zero-Shot |
|--------------------------|------------------|--------------------|
| 1D Heat Transfer         | `cfl`            | ✅ Supported        |
| 1D Heat Transfer         | `n_space`        | ✅ Supported        |
| 2D Steady Heat Transfer  | `dx`             | ✅ Supported        |
| 2D Steady Heat Transfer  | `error_threshold`| ✅ Supported        |
| 2D Steady Heat Transfer  | `relax`          | ❌ Only Zero-Shot   |
| 2D Steady Heat Transfer  | `t_init`         | ❌ Only Zero-Shot   |
| 1D Burgers               | `cfl`            | ✅ Supported        |
| 1D Burgers               | `k`              | ✅ Supported (Compositional)  |
| 1D Burgers               | `beta`           | ✅ Supported (Compositional)  |
| 1D Burgers               | `n_space`        | ✅ Supported        |
| 1D Euler                 | `cfl`            | ✅ Supported        |
| 1D Euler                 | `beta`           | ✅ Supported (Compositional)  |
| 1D Euler                 | `k`              | ✅ Supported (Compositional)  |
| 1D Euler                 | `n_space`        | ✅ Supported        |
| 2D Navier-Stokes Channel| `mesh_x`         | ✅ Supported        |
| 2D Navier-Stokes Channel| `mesh_y`         | ✅ Supported        |
| 2D Navier-Stokes Channel| `omega_u`        | ❌ Only Zero-Shot   |
| 2D Navier-Stokes Channel| `omega_v`        | ❌ Only Zero-Shot   |
| 2D Navier-Stokes Channel| `omega_p`        | ❌ Only Zero-Shot   |
| 2D Navier-Stokes Channel| `diff_u_threshold` | ❌ Only Zero-Shot |
| 2D Navier-Stokes Channel| `diff_v_threshold` | ❌ Only Zero-Shot |
| 2D Navier-Stokes Channel| `res_iter_v_threshold` | ❌ Only Zero-Shot |
| 2D Navier-Stokes Transient| `resolution`   | ✅ Supported        |
| 2D Navier-Stokes Transient| `cfl`          | ✅ Supported        |
| 2D Navier-Stokes Transient| `relaxation_factor` | ❌ Only Zero-Shot |
| 2D Navier-Stokes Transient| `residual_threshold` | ❌ Only Zero-Shot |


<!-- ## 🧐 Human Written Version -->
## 🕵️ Generate Questions

Generate question templates for different physics domains and task types.
```bash
# 1D Heat Transfer
python qs_gen/1D_heat_transfer.py

# 2D Steady Heat Transfer
python qs_gen/2D_heat_transfer.py

# Burgers 1D Equation with 2nd Order Roe Method
python qs_gen/1D_burgers.py

# Euler 1D Equations with 2nd Order MUSCL-Roe Method
python qs_gen/1D_euler.py

# 2D Navier-Stokes Channel Flow with SIMPLE Algorithm
python qs_gen/2D_ns.py

# 2D Navier-Stokes Transient Flow with Taichi Framework
python qs_gen/2D_ns_transient.py
```

**Output:** Generated questions are saved to `data/{simulation}/{task}/{precision_level}/question.json`

## 🚀 Generate Benchmark Datasets

Create complete benchmark datasets with problem instances and ground truth solutions.
```bash
# 1D Heat Transfer
python dataset_gen/oneD_heat_transfer.py

# 2D Steady Heat Transfer
python dataset_gen/twoD_heat_transfer.py

# Burgers 1D Equation with 2nd Order Roe Method
python dataset_gen/oneD_burgers.py

# Euler 1D Equations with 2nd Order MUSCL-Roe Method
python dataset_gen/oneD_euler.py

# 2D Navier-Stokes Channel Flow with SIMPLE Algorithm
python dataset_gen/twoD_ns.py

# 2D Navier-Stokes Transient Flow with Taichi Framework
python dataset_gen/twoD_ns_transient.py
```

**Output:** Datasets are saved to: `data/{simulation}/{task}/{precision_level}/human_write/` directory 

## 📄 Configure Model Providers

### 🌐 Commercial API Models
Configure API keys in your `.env` file:

```ini
# OpenAI API key
OPENAI_API_KEY="your_openai_api_key"

# Google API key for Gemini
GOOGLE_API_KEY="your_google_api_key"

# AWS API credentials for Bedrock
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
AWS_REGION_NAME="your_aws_region_name"
```

### 🧠 Custom Models Configuration

**Two configuration methods available:**

#### Method 1: JSON Configuration (Recommended for Multiple Models)
Create `configs/custom_models.json` to manage multiple custom models:

```json
{
  "custom_models": {
    "qwen3_8b": {
      "custom_code": "/path/to/custom_model/custom_inference.py",
      "model_path": "/data/models/Qwen3-8B",
      "custom_class": "Qwen3"
    },
    "llama3_7b": {
      "custom_code": "/path/to/custom_model/custom_inference.py", 
      "model_path": "/data/models/Llama3-7B",
      "custom_class": "Llama3"
    }
  }
}
```

#### Method 2: Environment Variables (For Single Model)
Set in your `.env` file:

```ini
custom_code="/path/to/custom_model/custom_inference.py"
model_path="/path/to/your/custom_model"
custom_class="CustomModel"
```

**📋 List Available Models:**
```bash
python scripts/list_custom_models.py
```

For detailed implementation guide, see [Custom Model Integration Guide](custom_model/README.md).

## 🧠 Run Inference

Execute LLM inference on benchmark datasets to generate predictions.

```bash
# Commercial API Models
python inference/langchain_LLM.py -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d heat_1d -t cfl -l medium -z
python inference/langchain_LLM.py -p openai -m gpt-4o -d heat_1d -t cfl -l medium -z

# Single Custom Model
python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d heat_1d -t cfl -l medium -z

# Multiple Custom Models (use batch scripts)
bash scripts/inference_eval/inference_eval_heat_1d.sh
```

**📋 Available Datasets & Tasks:**
- **Heat 1D**: `cfl`, `n_space`
- **Heat 2D**: `dx`, `error_threshold`, `relax`, `t_init`
- **Burgers 1D**: `cfl`, `k`, `beta`, `n_space`
- **Euler 1D**: `cfl`, `beta`, `k`, `n_space`
- **2D Navier-Stokes Channel**: `mesh_x`, `mesh_y`, `omega_u`, `omega_v`, `omega_p`, `diff_u_threshold`, `diff_v_threshold`, `res_iter_v_threshold`
- **2D Navier-Stokes Transient**: `resolution`, `cfl`, `relaxation_factor`, `residual_threshold`

**Parameters:**
- `-p`: LLM provider (`openai`, `gemini`, `bedrock`, `custom_model`)
- `-m`: Model name/identifier
- `-d`: Dataset name
- `-t`: Problem task type
- `-l`: Precision level
- `-z`: Enable zero-shot mode
- `--list-combinations`: Show all valid dataset-task combinations and exit

**💡 Tip**: Use `--list-combinations` to see all available dataset-task combinations:
```bash
python inference/langchain_LLM.py --list-combinations
```

**Outputs:**
- Results: `results_model_attempt/{dataset}/{precision_level}/{task}/`
- Logs: `log_model_tool_call/{dataset}/{precision_level}/{task}/`

## 🔄 Resume Functionality

The inference system includes automatic progress tracking and resume capabilities to handle long-running experiments gracefully.

### Features
- **Automatic Progress Saving**: Progress is saved after each completed sample
- **Resume from Interruption**: Continue from where you left off using the `--resume` flag

### Usage

```bash
python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d heat_1d -t cfl -l medium --resume
```

### Progress Files
- **Location**: `log_model_tool_call/{dataset}/{precision_level}/{task}/{flag}_{model_name}_progress.json`
- **Content**: Contains completed results and intermediate data for resuming

## 📊 Evaluate Models' Performance

Compute performance metrics and accuracy scores for model predictions.
```bash
# Example (Heat 1D)
python evaluation/heat_1d/eval.py -m anthropic.claude-3-5-haiku-20241022-v1:0 -d heat_1d -t cfl -l medium -z
```
**Parameters:**
- `-m`: Model name/identifier
- `-d`: Dataset name  
- `-t`: Problem task type
- `-l`: Precision level (for heat_1d, heat_2d, burgers_1d and euler_1d: low, medium, high; default: medium)
- `-z`: Enable zero-shot mode

**Output:** Evaluation results are saved to `eval_results/{dataset}/{task}/{precision_level}/`

## 🗂️ Tabulate Evaluation Results

Generate summary tables and comparative analysis across different models and tasks.
```bash
python evaluation/tabulate.py -d heat_1d
python evaluation/tabulate.py -d heat_2d
python evaluation/tabulate.py -d burgers_1d
python evaluation/tabulate.py -d euler_1d
python evaluation/tabulate.py -d ns_2d
python evaluation/tabulate.py -d ns_transient_2d
```

**Parameters:**
- `-d`: Dataset name to tabulate results for

**Output:** Summary tables are generated in Excel/CSV format

### 📈 Generate Simulation-Level Summaries

After generating task-level results with `tabulate.py`, you can create simulation-level aggregated summaries that combine all tasks within a simulation (dataset).

```bash
# Aggregate task-level results to simulation-level summaries
python evaluation/simul_sum.py -d heat_1d
python evaluation/simul_sum.py -d heat_2d
python evaluation/simul_sum.py -d burgers_1d
python evaluation/simul_sum.py -d euler_1d
python evaluation/simul_sum.py -d ns_2d
python evaluation/simul_sum.py -d ns_transient_2d
```

**Parameters:**
- `-d`: Dataset name to aggregate results

**Output:**
- **CSV**: `eval_results/{dataset}/{dataset}_sum.csv` - Combined results with precision_level column
- **Excel**: `eval_results/{dataset}/{dataset}_sum.xlsx` - Clean, professional formatting with visual separators

**Note:** Run `tabulate.py` first to generate the required task-level CSV files before running `simul_sum.py`.

## 🤖 Evaluated Models

### Commercial Models
- **Anthropic**: 
  - `anthropic.claude-3-5-haiku-20241022-v1:0`
  - `anthropic.claude-3-5-sonnet-20240620-v1:0` 
  - `anthropic.claude-3-7-sonnet-20250219-v1:0`
- **OpenAI**: `gpt-4o`
- **Google**: `gemini-2.5-pro`
- **Mistral**: `mistral.mistral-large-2402-v1:0`
- **Meta**: `meta.llama3-70b-instruct-v1:0`
- **Amazon**: `amazon.nova-premier-v1:0`
<!-- - **DeepSeek**: `deepseek.r1-v1:0` -->
<!-- - **AI21**: `ai21.jamba-1-5-large-v1:0` -->

### Open-Source Models
- **Alibaba**:
  - `qwen3_8b`
  - `qwen3_32b`
  - `qwen3_235b_a22b`

## 🛠️ Script Usage Guide

The `scripts/` directory contains automated scripts for streamlined execution of common workflows including inference + evaluation pipelines.

For detailed usage instructions and examples, see the [Script Usage Guide](scripts/README.md).
