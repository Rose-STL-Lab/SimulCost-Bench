# SimulCost-Bench

A comprehensive benchmark for evaluating Large Language Models (LLMs) on computational physics simulation parameter optimization tasks. This benchmark assesses how well LLMs can determine optimal numerical parameters for solving partial differential equations.

## 📚 Table of Contents

- [Environment Setup](#-environment-setup)
- [Tasks and Zero-Shot Support](#-tasks-and-zero-shot-support)
- [Generate Questions](#️-generate-questions)
- [Generate Benchmark Datasets](#-generate-benchmark-datasets)
- [Configure Model Providers](#-configure-model-providers)
- [Customize Simulation Results Directory](#-customize-simulation-results-directory)
- [Pre-cached Simulation Results](#-pre-cached-simulation-results)
- [Run Inference](#-run-inference)
- [Resume Functionality](#-resume-functionality)
- [Evaluate Performance](#-evaluate-models-performance)
- [Tabulate Results](#️-tabulate-evaluation-results)
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

**Note**: To run 1D EPOCH PIC simulations, see the [EPOCH Setup Guide](EPOCH_SETUP.md) for additional configuration requirements.

**Note**: To run 2D Euler gas dynamics simulations, see the [Euler 2D Setup Guide](EULER_2D_SETUP.md) for additional configuration requirements.

**Note**: To run 2D FEM simulations with FastIPC solver, see the [FEM 2D Setup Guide](FEM_2D_SETUP.md) for compilation and configuration requirements.

### 🐳 Docker Setup (Alternative)

A pre-built Docker image is available with all dependencies and solvers compiled. This lets you skip the Conda / Poetry / solver setup entirely.

**1. Clone the repository:**
```bash
git clone --recursive https://github.com/Rose-STL-Lab/SimulCost-Bench.git
cd SimulCost-Bench
```

**2. Pull the image:**
```bash
docker pull ghcr.io/leo-lsc/simulcost-bench:latest
docker tag ghcr.io/leo-lsc/simulcost-bench:latest simulcost-bench
```

Or build locally with `docker build -t simulcost-bench .`

**3. Run the container (Dev / hot-reload):**
This mode mounts frequently-edited code directories from the host into the container so changes take effect immediately, without rebuilding the image.  
Do NOT mount `costsci_tools/` unless you also build the solvers on the host — otherwise you may overwrite the compiled binaries shipped in the image.

```bash
docker run --rm -it \
  --env-file .env \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/inference:/app/inference \
  -v $(pwd)/evaluation:/app/evaluation \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/custom_model:/app/custom_model \
  -v $(pwd)/sim_res:/app/sim_res \
  -v $(pwd)/eval_results:/app/eval_results \
  -v $(pwd)/results_model_attempt:/app/results_model_attempt \
  -v $(pwd)/log_model_tool_call:/app/log_model_tool_call \
  -v $(pwd)/data:/app/data \
  simulcost-bench
```

- `--env-file .env` passes your API keys (e.g. OpenAI / AWS Bedrock) into the container.
- `-v` mounts persist results to the host — without them, all output is lost when the container exits.

> **Note**: The FEM 2D solver (FastIPC) is compiled with `-mavx -mavx2 -mfma`. Your CPU (both build and run host) must support AVX/AVX2/FMA instructions.

## 📋 Tasks and Zero-Shot Support

<details>
<summary><b>Click to expand the full task support table</b></summary>

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
| 1D EPOCH PIC             | `nx`           | ✅ Supported        |
| 1D EPOCH PIC             | `npart`        | ✅ Supported        |
| 1D EPOCH PIC             | `dt_multiplier` | ✅ Supported (Compositional) |
| 1D EPOCH PIC             | `field_order`  | ✅ Supported (Compositional) |
| 1D EPOCH PIC             | `particle_order` | ✅ Supported (Compositional) |
| 2D MPM                   | `nx`             | ✅ Supported        |
| 2D MPM                   | `n_part`         | ✅ Supported        |
| 2D MPM                   | `cfl`            | ✅ Supported        |
| 1D Diffusion-Reaction    | `cfl`                | ✅ Supported        |
| 1D Diffusion-Reaction    | `n_space`            | ✅ Supported        |
| 1D Diffusion-Reaction    | `tol`                | ✅ Supported        |
| 2D Euler Gas Dynamics    | `n_grid_x`       | ✅ Supported        |
| 2D Euler Gas Dynamics    | `cfl`            | ✅ Supported        |
| 2D Euler Gas Dynamics    | `cg_tolerance`   | ✅ Supported        |
| Hasegawa-Mima Nonlinear  | `N`              | ✅ Supported        |
| Hasegawa-Mima Nonlinear  | `dt`             | ✅ Supported        |
| Hasegawa-Mima Linear     | `N`              | ✅ Supported        |
| Hasegawa-Mima Linear     | `dt`             | ✅ Supported        |
| Hasegawa-Mima Linear     | `cg_atol`        | ✅ Supported        |
| 2D FEM                   | `dx`             | ✅ Supported        |
| 2D FEM                   | `cfl`            | ✅ Supported        |

</details>


<!-- ## 🧐 Human Written Version -->
## 🕵️ Generate Questions

Generate question templates for different physics domains and task types.

<details>
<summary><b>Click to view commands for all simulation types</b></summary>

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

# 1D EPOCH Particle-in-Cell Simulation
python qs_gen/1D_epoch.py

# 2D Material Point Method (MPM) Simulation
python qs_gen/2D_mpm.py

# 1D Diffusion-Reaction Equations with Newton Method
python qs_gen/1D_diff_react.py

# 2D Euler Equations with Advection-Projection Method
python qs_gen/2D_euler.py

# Hasegawa-Mima Nonlinear Equation with Pseudo-Spectral Method
python qs_gen/hasegawa_mima_nonlinear.py

# Hasegawa-Mima Linear Equation with RK4 and CG Solver
python qs_gen/hasegawa_mima_linear.py

# 2D Finite Element Method with Implicit Newton Solver
python qs_gen/2D_fem.py
```

</details>

**Output:** Generated questions are saved to `data/{simulation}/{task}/{precision_level}/question.json`

## 🚀 Generate Benchmark Datasets

Create complete benchmark datasets with problem instances and ground truth solutions.

<details>
<summary><b>Click to view commands for all simulation types</b></summary>

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

# 1D EPOCH Particle-in-Cell Simulation
python dataset_gen/oneD_epoch.py

# 2D Material Point Method (MPM) Simulation
python dataset_gen/twoD_mpm.py

# 1D Diffusion-Reaction Equations with Newton Method
python dataset_gen/oneD_diff_react.py

# 2D Euler Equations with Advection-Projection Method
python dataset_gen/twoD_euler.py

# Hasegawa-Mima Nonlinear Equation with Pseudo-Spectral Method
python dataset_gen/hasegawa_mima_nonlinear.py

# Hasegawa-Mima Linear Equation with RK4 and CG Solver
python dataset_gen/hasegawa_mima_linear.py

# 2D Finite Element Method with Implicit Newton Solver
python dataset_gen/twoD_fem.py
```

</details>

**Output:** Datasets are saved to: `data/{simulation}/{task}/{precision_level}/human_write/` directory 

## 📄 Configure Model Providers

### 🌐 Commercial API Models
Configure API keys in your `.env` file:

```ini
# OpenAI API key
OPENAI_API_KEY=your_openai_api_key

# Google API key for Gemini
GOOGLE_API_KEY=your_google_api_key

# AWS API credentials for Bedrock
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION_NAME=your_aws_region_name
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

## 📂 Customize Simulation Results Directory

By default, simulation results are saved to `sim_res/` in the current working directory. You can configure a custom absolute path for storing simulation results.

### Configuration

Add the following to your `.env` file:

```ini
# Optional: Specify custom directory for simulation results
# If not set, results will be saved to ./sim_res/ (relative path)
SIM_RES_BASE_DIR=/path/to/your/custom/directory
```

### Examples

**Use absolute path:**
```ini
SIM_RES_BASE_DIR=/data/leo_work_new
```
Results will be saved to: `/data/leo_work_new/sim_res/...`

**Use default relative path:**
```ini
# Comment out or remove the SIM_RES_BASE_DIR line
# SIM_RES_BASE_DIR=/data/leo_work_new
```
Results will be saved to: `./sim_res/...` (relative to working directory)

> **Docker users:** If you run via Docker, you don't need `SIM_RES_BASE_DIR`. Simply mount your desired host directory to the container's `sim_res/` path when running `docker run`, e.g. `-v /your/host/dir:/app/sim_res`.

## 📥 Pre-cached Simulation Results

<details>
<summary><b>⚡ Pre-cached results (optional)</b></summary>

To help you skip long simulation runtimes, **pre-computed simulation results for all baseline experiments** have been uploaded to Hugging Face:

- **Cache (≈22.5 GB):** **[LeoLai689/SimulCost-baseline-sim_res](https://huggingface.co/datasets/LeoLai689/SimulCost-baseline-sim_res)**

If you choose to use the cache, download the files you need from the dataset page and place/extract them into your simulation results directory (e.g., `./sim_res/` or your `SIM_RES_BASE_DIR`).

### Available Simulation Results

| Simulation File | Size | Simulation Type |
|----------------|------|-----------------|
| `burgers_1d.zip` | 675 MB | 1D Burgers Equation |
| `diff_react_1d.zip` | 293 MB | 1D Diffusion-Reaction |
| `epoch.zip` | 3.65 GB | 1D EPOCH PIC |
| `euler_1d.zip` | 3.86 GB | 1D Euler Equations |
| `euler_2d.zip` | 1.54 GB | 2D Euler Gas Dynamics |
| `fem_2d.zip` | 146 MB | 2D Finite Element Method |
| `hasegawa_mima_linear.zip` | 1.45 GB | Hasegawa-Mima Linear |
| `hasegawa_mima_nonlinear.zip` | 300 MB | Hasegawa-Mima Nonlinear |
| `heat_1d.zip` | 182 MB | 1D Heat Transfer |
| `heat_2d.zip` | 2.66 GB | 2D Steady Heat Transfer |
| `ns_channel_2d.zip` | 96.7 MB | 2D Navier-Stokes Channel |
| `ns_transient_2d.zip` | 6.56 GB | 2D Navier-Stokes Transient |
| `unstruct_mpm.zip` | 1.13 GB | 2D Material Point Method |

</details>

## 🧠 Run Inference

### 🚀 Simulation Speed Reference

If you're getting started, begin with the “Fast” group to avoid wasting time on very slow simulations.

- **Fast (recommended to start):** `burgers_1d`, `diff_react_1d`, `heat_1d`, `fem_2d`
- **Moderate:** `euler_1d`, `euler_2d`, `ns_transient_2d`, `unstruct_mpm`,`hasegawa_mima_linear`, `hasegawa_mima_nonlinear`, `epoch`
- **Slow:** `heat_2d`, `ns_channel_2d` (Can be **very slow**⚠️)

Execute LLM inference on benchmark datasets to generate predictions.

```bash
# Commercial API Models
python inference/langchain_LLM.py -p openai -m gpt-5-2025-08-07 -d heat_1d -t cfl -l medium -z

# Single Custom Model
python inference/langchain_LLM.py -p custom_model -m qwen3_8b -d heat_1d -t cfl -l medium -z

# Multiple Custom Models (use batch scripts)
bash scripts/inference_eval/inference_eval_heat_1d.sh
```

<!-- **📋 Available Datasets & Tasks:**
- **Heat 1D**: `cfl`, `n_space`
- **Heat 2D**: `dx`, `error_threshold`, `relax`, `t_init`
- **Burgers 1D**: `cfl`, `k`, `beta`, `n_space`
- **Euler 1D**: `cfl`, `beta`, `k`, `n_space`
- **2D Navier-Stokes Channel**: `mesh_x`, `mesh_y`, `omega_u`, `omega_v`, `omega_p`, `diff_u_threshold`, `diff_v_threshold`, `res_iter_v_threshold`
- **2D Navier-Stokes Transient**: `resolution`, `cfl`, `relaxation_factor`, `residual_threshold`
- **1D EPOCH PIC**: `dt_multiplier`, `nx`, `npart`, `field_order`, `particle_order` -->

**Parameters:**
- `-p`: LLM provider (`openai`, `gemini`, `bedrock`, `custom_model`)
- `-m`: Model name/identifier
- `-d`: Dataset name
- `-t`: Problem task type
- `-l`: Precision level
- `-z`: Enable zero-shot mode
- `--list-combinations`: Show all valid dataset-task combinations and exit

**Outputs:**
- Results: `results_model_attempt/{dataset}/{precision_level}/{task}/`
- Logs: `log_model_tool_call/{dataset}/{precision_level}/{task}/`

**💡 Tip**: Use `--list-combinations` to see all available dataset-task combinations:
```bash
python inference/langchain_LLM.py --list-combinations
```

<details>
<summary><b>OpenAI Reasoning Models: Setting reasoning_effort</b></summary>

For OpenAI reasoning models (e.g., GPT-5), you can control the reasoning effort by appending `-re-{effort}` to the model name:

- **Syntax**: `{model_name}-re-{effort}`
- **Valid effort levels**: `minimal`, `low`, `medium`, `high`

Example:
```bash
# Use GPT-5 with minimal reasoning effort
python inference/langchain_LLM.py -p openai -m gpt-5-2025-08-07-re-minimal -d heat_1d -t cfl -l medium -z
```

</details>

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

**Outputs:**
- JSON results: `eval_results/{dataset}/{task}/{precision_level}/`
- Parquet dataframes: `eval_results/{dataset}/dataframes/`

📖 **For advanced data analysis workflows and parquet usage**, see [Evaluation Documentation](evaluation/README.md).

## 🗂️ Tabulate Evaluation Results

Generate summary tables and comparative analysis across different models and tasks.
```bash
python evaluation/tabulate.py -d heat_1d
python evaluation/tabulate.py -d heat_2d
python evaluation/tabulate.py -d burgers_1d
python evaluation/tabulate.py -d euler_1d
python evaluation/tabulate.py -d ns_2d
python evaluation/tabulate.py -d ns_transient_2d
python evaluation/tabulate.py -d epoch_1d
python evaluation/tabulate.py -d mpm_2d
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
python evaluation/simul_sum.py -d epoch_1d
python evaluation/simul_sum.py -d mpm_2d
```

**Parameters:**
- `-d`: Dataset name to aggregate results

**Output:**
- **CSV**: `eval_results/{dataset}/{dataset}_sum.csv` - Combined results with precision_level column
- **Excel**: `eval_results/{dataset}/{dataset}_sum.xlsx` - Clean, professional formatting with visual separators

**Note:** Run `tabulate.py` first to generate the required task-level CSV files before running `simul_sum.py`.

<!-- ## 🤖 Evaluated Models

### Commercial Models
- **Anthropic**: 
  - `anthropic.claude-3-5-haiku-20241022-v1:0`
  - `anthropic.claude-3-5-sonnet-20240620-v1:0` 
  - `anthropic.claude-3-7-sonnet-20250219-v1:0`
- **OpenAI**: `gpt-5-2025-08-07`
- **Google**: `gemini-2.5-pro`
- **Mistral**: `mistral.mistral-large-2402-v1:0`
- **Meta**: `meta.llama3-70b-instruct-v1:0`
- **Amazon**: `amazon.nova-premier-v1:0`
- **DeepSeek**: `deepseek.r1-v1:0`
- **AI21**: `ai21.jamba-1-5-large-v1:0`

### Open-Source Models
- **Alibaba**:
  - `qwen3_8b`
  - `qwen3_32b`
  - `qwen3_235b_a22b` -->

## 🛠️ Script Usage Guide

The `scripts/` directory contains automated scripts for streamlined execution of common workflows including inference + evaluation pipelines.

For detailed usage instructions and examples, see the [Script Usage Guide](scripts/README.md).
