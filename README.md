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

## 📋 Tasks and Zero-Shot

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


<!-- ## 🧐 Human Written Version -->
## 🕵️ Generate Questions
```bash
# 1D Heat Transfer
python qs_gen/1D_heat_transfer.py -n 10 -t cfl -z

# 2D Steady Heat Transfer
python qs_gen/2D_heat_transfer.py -n 10 -t dx -z

# Burgers 1D Equation with 2nd Order Roe Method
python qs_gen/1D_burgers.py -t cfl -z

# Euler 1D Equations with 2nd Order MUSCL-Roe Method
python qs_gen/1D_euler.py -t cfl -z
```

### Parameters:
- n: int, number of example 
- t: problem task
- -z: zero-shot 
### Outputs:
- output_file: data/heat_transfer/question.json

## 🚀 Generate Benchmark Datasets
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

### Parameters:
- t: problem task
- -z: zero-shot 

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
custom_code="/path/to/custom_inference.py"   # Path to your custom inference Python code
model_path="/path/to/your/custom_model"     # Path to your custom model
custom_class="CustomModel"                  # The class name within custom_code that will handle inference
```
For detailed implementation guide and examples of using your own model, see [Custom Model Integration Guide](custom_model/README.md).

## 🧠 Run Inference
```bash
# 1D Heat Transfer
python inference/langchain_LLM.py -n 10 -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 1D_heat_transfer -t cfl -z
python inference/langchain_LLM.py -n 10 -p custom_model -m qwen3_8b -d 1D_heat_transfer -t cfl -z

# 2D Steady Heat Transfer
python inference/langchain_LLM.py -n 10 -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 2D_heat_transfer -t dx -z

# Burgers 1D Equation with 2nd Order Roe Method
python inference/langchain_LLM.py -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d burgers_1d -t cfl -c blast -z
Cases: blast, double_shock, rarefaction, sin, sod
```
### Parameters
- n: int, number of samples to test
- p: str, LLMs provider (openai, gemini, bedrock)
- m: str, model name
- d: str, dataset name
- t: problem task
- c: case
- -z: zero-shot 

## 📊 Evaluate Models' Performance
```bash
# 1D Heat Transfer
PYTHONPATH=$(pwd) python evaluation/heat_transfer/eval.py -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 1D_heat_transfer -t cfl -z

# 2D Steady Heat Transfer
PYTHONPATH=$(pwd) python evaluation/heat_transfer/eval.py -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 2D_heat_transfer -t dx -z

# Burgers 1D Equation with 2nd Order Roe Method
PYTHONPATH=$(pwd) python evaluation/burgers/eval.py -m anthropic.claude-3-5-haiku-20241022-v1:0 -d burgers_1d -t cfl -c blast -z
Cases: blast, double_shock, rarefaction, sin, sod
```
### Parameters
- m: str, model name
- d: str, dataset name
- t: problem task
- c: case
- -z: zero-shot 

## 🗂️ Tabulate Evaluation Results
```bash
python evaluation/tabulate.py -d 1D_heat_transfer
python evaluation/tabulate.py -d 2D_heat_transfer
python evaluation/tabulate.py -d burgers_1d
```

## 🤖 Models
1. anthropic.claude-3-5-haiku-20241022-v1:0
2. anthropic.claude-3-5-sonnet-20240620-v1:0
3. anthropic.claude-3-7-sonnet-20250219-v1:0
