# SimulCost-Bench

A comprehensive benchmark for evaluating Large Language Models (LLMs) on computational physics simulation parameter optimization tasks. This benchmark assesses how well LLMs can determine optimal numerical parameters for solving partial differential equations.

## 📚 Table of Contents

- [Environment Setup](#-environment-setup)
- [Tasks and Zero-Shot Support](#-tasks-and-zero-shot-support)
- [Generate Questions](#️-generate-questions)
- [Generate Benchmark Datasets](#-generate-benchmark-datasets)
- [Configure Model Providers](#-configure-env-file-for-model-providers)
- [Run Inference](#-run-inference)
- [Resume Functionality](#-resume-functionality)
- [Evaluate Performance](#-evaluate-models-performance)
- [Tabulate Results](#️-tabulate-evaluation-results)
- [Supported Models](#-supported-models)
- [Script Usage Guide](#️-script-usage-guide)

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

The table below summarizes the available tasks for each problem type and indicates whether each task supports zero-shot inference.

| Problem Type             | Task Type        | Iterative & Zero-Shot |
|--------------------------|------------------|--------------------|
| 1D Heat Transfer         | `cfl`            | ✅ Supported        |
| 1D Heat Transfer         | `n_space`        | ✅ Supported        |
| 2D Steady Heat Transfer  | `dx`             | ✅ Supported        |
| 2D Steady Heat Transfer  | `error_threshold`| ✅ Supported        |
| 2D Steady Heat Transfer  | `relax`          | ❌ Only Zero-Shot   |
| 2D Steady Heat Transfer  | `t_init`         | ❌ Only Zero-Shot   |
| 1D Burgers               | `cfl`            | ✅ Supported        |
| 1D Burgers               | `k`              | ✅ Supported (Compositional)  |
| 1D Burgers               | `w`              | ✅ Supported (Compositional)  |
| 1D Euler                 | `cfl`            | ✅ Supported        |
| 1D Euler                 | `beta`           | ✅ Supported (Compositional)  |
| 1D Euler                 | `k`              | ✅ Supported (Compositional)  |
| 2D Navier-Stokes Channel| `mesh_x`         | ✅ Supported        |
| 2D Navier-Stokes Channel| `mesh_y`         | ✅ Supported        |
| 2D Navier-Stokes Channel| `omega_u`        | ✅ Supported        |
| 2D Navier-Stokes Channel| `omega_v`        | ✅ Supported        |
| 2D Navier-Stokes Channel| `omega_p`        | ✅ Supported        |
| 2D Navier-Stokes Channel| `diff_u_threshold` | ✅ Supported      |
| 2D Navier-Stokes Channel| `diff_v_threshold` | ✅ Supported      |
| 2D Navier-Stokes Channel| `res_iter_v_threshold` | ✅ Supported  |


<!-- ## 🧐 Human Written Version -->
## 🕵️ Generate Questions

Generate question templates for different physics domains and task types.
```bash
# 1D Heat Transfer
python qs_gen/1D_heat_transfer.py -n 100 -t cfl -z

# 2D Steady Heat Transfer
python qs_gen/2D_heat_transfer.py -n 100 -t dx -z

# Burgers 1D Equation with 2nd Order Roe Method
python qs_gen/1D_burgers.py -t cfl -z

# Euler 1D Equations with 2nd Order MUSCL-Roe Method
python qs_gen/1D_euler.py -t cfl -z

# 2D Navier-Stokes Channel Flow with SIMPLE Algorithm
python qs_gen/2D_ns.py -n 5 -t mesh_x -z
```

**Parameters:**
- `-n`: Number of examples to generate
  - **Note for 2D Navier-Stokes**: This creates N profiles for EACH boundary condition (total: 1 + 4×N profiles)
- `-t`: Problem task type (cfl, n_space, dx, mesh_x, omega_u, etc.)
- `-z`: Enable zero-shot mode

**Output:** Generated questions are saved to `data/{simulation}/{task}/question.json`

## 🚀 Generate Benchmark Datasets

Create complete benchmark datasets with problem instances and ground truth solutions.
```bash
# 1D Heat Transfer
python dataset_gen/oneD_heat_transfer.py -t cfl -z

# 2D Steady Heat Transfer
python dataset_gen/twoD_heat_transfer.py -t dx -z

# Burgers 1D Equation with 2nd Order Roe Method
python dataset_gen/oneD_burgers.py -t cfl -z

# Euler 1D Equations with 2nd Order MUSCL-Roe Method
python dataset_gen/oneD_euler.py -t cfl -z
```

**Parameters:**
- `-t`: Problem task type
- `-z`: Enable zero-shot mode

**Output:** Datasets are saved to `data/{simulation}/{task}/human_write/` directory 

## 📄 Configure `.env` File for Model Providers

To use the various model providers (OpenAI, Google, AWS, or custom models), you need to configure your `.env` file accordingly. Below is a guide on how to set up the `.env` file for each provider.

### 📜 General `.env` Configuration
In the root of your project, create or edit the `.env` file to include the following key configurations:

```ini
# .env file template

# OpenAI API key
OPENAI_API_KEY="your_openai_api_key"

# Google API key for Gemini
GOOGLE_API_KEY="your_google_api_key"

# AWS API credentials for Bedrock
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
AWS_REGION_NAME="your_aws_region_name"

# Custom Model (when using your own model)
custom_code="/path/to/custom_model/custom_inference.py"   # Path to your custom inference Python code
model_path="/path/to/your/custom_model"     # Path to your custom model
custom_class="CustomModel"                  # The class name within custom_inference.py that will handle inference
```
For detailed implementation guide and examples of using your own model, see [Custom Model Integration Guide](custom_model/README.md).

## 🧠 Run Inference

Execute LLM inference on benchmark datasets to generate predictions.
```bash
# 1D Heat Transfer
python inference/langchain_LLM.py -n 100 -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 1D_heat_transfer -t cfl -z
python inference/langchain_LLM.py -n 100 -p custom_model -m qwen3_8b -d 1D_heat_transfer -t cfl -z

# 2D Steady Heat Transfer
python inference/langchain_LLM.py -n 100 -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 2D_heat_transfer -t dx -z
python inference/langchain_LLM.py -n 100 -p custom_model -m qwen3_8b -d 2D_heat_transfer -t dx -z

# Burgers 1D Equation with 2nd Order Roe Method
python inference/langchain_LLM.py -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d burgers_1d -t cfl -c blast -z
python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d burgers_1d -t cfl -c blast -z
Cases: blast, double_shock, rarefaction, sin, sod

# Euler 1D Equations with 2nd Order MUSCL-Roe Method
python inference/langchain_LLM.py -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d euler_1d -t cfl -c sod -z
python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d euler_1d -t cfl -c sod -z
Cases: sod
```
**Parameters:**
- `-n`: Number of samples to test
- `-p`: LLM provider (`openai`, `gemini`, `bedrock`, `custom_model`)
- `-m`: Model name/identifier
- `-d`: Dataset name
- `-t`: Problem task type
- `-c`: Case name (for burgers_1d and euler_1d)
- `-z`: Enable zero-shot mode

**Outputs:**
- Results: `results_model_attempt/{dataset}/{task}/`
- Logs: `log_model_tool_call/{dataset}/{task}/`

## 🔄 Resume Functionality

The inference system includes automatic progress tracking and resume capabilities to handle long-running experiments gracefully.

### Features
- **Automatic Progress Saving**: Progress is saved after each completed sample
- **Resume from Interruption**: Continue from where you left off using the `--resume` flag
- **Error Recovery**: If inference fails at sample N, you can resume from sample N+1

### Usage

**Normal execution** (automatically saves progress):
```bash
python inference/langchain_LLM.py -n 100 -p custom_model -m qwen3_8b -d 1D_heat_transfer -t cfl
```

**Resume from interruption**:
```bash
python inference/langchain_LLM.py -n 100 -p custom_model -m qwen3_8b -d 1D_heat_transfer -t cfl --resume
```

### Progress Files
- **Location**: `log_model_tool_call/{dataset}/{task}/{flag}_{model_name}_progress.json`
- **Content**: Contains completed results and intermediate data for resuming
- **Cleanup**: Automatically deleted when all samples complete successfully

### Example Scenario
1. Start inference on 100 samples
2. Process fails at sample 90 due to format error
3. Run the same command with `--resume` flag
4. Inference continues from sample 91 

## 📊 Evaluate Models' Performance

Compute performance metrics and accuracy scores for model predictions.
```bash
# 1D Heat Transfer
python evaluation/heat_transfer/eval.py -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 1D_heat_transfer -t cfl -z

# 2D Steady Heat Transfer
python evaluation/heat_transfer/eval.py -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 2D_heat_transfer -t dx -z

# Burgers 1D Equation with 2nd Order Roe Method
python evaluation/burgers/eval.py -m anthropic.claude-3-5-haiku-20241022-v1:0 -d burgers_1d -t cfl -c blast -z
Cases: blast, double_shock, rarefaction, sin, sod

# Euler 1D Equations with 2nd Order MUSCL-Roe Method
python evaluation/euler/eval.py -m anthropic.claude-3-5-haiku-20241022-v1:0 -d euler_1d -t cfl -c sod -z
Cases: sod
```
**Parameters:**
- `-m`: Model name/identifier
- `-d`: Dataset name  
- `-t`: Problem task type
- `-c`: Case name (for burgers_1d and euler_1d)
- `-z`: Enable zero-shot mode

**Output:** Evaluation results are saved to `eval_results/{dataset}/{task}/` 

## 🗂️ Tabulate Evaluation Results

Generate summary tables and comparative analysis across different models and tasks.
```bash
python evaluation/tabulate.py -d 1D_heat_transfer
python evaluation/tabulate.py -d 2D_heat_transfer
python evaluation/tabulate.py -d burgers_1d
python evaluation/tabulate.py -d euler_1d
```

**Parameters:**
- `-d`: Dataset name to tabulate results for

**Output:** Summary tables are generated in Excel/CSV format

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

The `scripts/` directory contains automated scripts for streamlined execution of common workflows including question generation, dataset generation, and inference + evaluation pipelines.

For detailed usage instructions and examples, see the [Script Usage Guide](scripts/README.md).
