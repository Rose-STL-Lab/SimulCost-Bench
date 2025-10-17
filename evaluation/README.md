# 📊 Evaluation Documentation

This directory contains tools for evaluating model performance, analyzing results, and generating comprehensive statistics for SimulCost-Bench.

---

## 📑 Table of Contents

- [Quick Start](#-quick-start)
- [Evaluation Workflow](#-evaluation-workflow)
- [Data Outputs](#-data-outputs)
- [Result Merging](#-result-merging)

---

## 🚀 Quick Start

```bash
# 1. Run evaluation for a specific model and task
python evaluation/heat_1d/eval.py -m gpt-4o -d heat_1d -t cfl -l medium -z

# 2. Inspect evaluation results
python test_df.py eval_results/heat_1d/dataframes/iterative_gpt-4o.parquet --info

# 3. View attempt history for a specific question
python test_df.py eval_results/heat_1d/dataframes/iterative_gpt-4o.parquet --qid 6 --attempt-history

# 4. Merge results from multiple datasets
python evaluation/stats_utils/merge_results.py

# 5. Compare inference modes across tasks
python test_df.py eval_results/merged_results.parquet --mode-comparison

# 6. Analyze the merged data
python -c "import pandas as pd; df = pd.read_parquet('eval_results/merged_results.parquet'); print(df.head())"
```

---

## 🔄 Evaluation Workflow

### Step 1: Run Evaluation Scripts

Each dataset has its own evaluation script in the corresponding directory:

```bash
# Heat Transfer (1D)
python evaluation/heat_1d/eval.py -m <model> -d heat_1d -t <task> -l <precision> -z
```

**Common Parameters:**
- `-m, --model`: Model name/identifier
- `-d, --dataset`: Dataset name
- `-t, --task`: Task type (e.g., `cfl`, `n_space`, `beta`)
- `-l, --precision`: Precision level (`low`, `medium`, `high`) - only for certain datasets
- `-z, --zero-shot`: Enable zero-shot inference mode

### Step 2: Outputs Generated

Each evaluation run produces **two types of outputs**:

1. **JSON Files** (detailed results per question):
   - Location: `eval_results/{dataset}/{task}/{precision_level}/`
   - Contains: Full evaluation details for each question instance

2. **Parquet Files** (aggregated dataframe):
   - Location: `eval_results/{dataset}/dataframes/`
   - Filename: `{inference_mode}_{model_name}.parquet`
   - Contains: Structured data ready for analysis

---

## 📦 Data Outputs

### Parquet File Schema

The parquet files contain the following columns:

#### Common Columns (All Datasets)

| Column | Type | Description |
|--------|------|-------------|
| `dataset` | str | Dataset name (e.g., `epoch_1d`, `euler_1d`) |
| `task` | str | Task type (e.g., `cfl`, `beta`, `resolution`) |
| `precision_level` | str | Precision level (`low`, `medium`, `high`) |
| `inference_mode` | str | Inference mode (`zero_shot`, `iterative`) |
| `model_name` | str | Model identifier |
| `qid` | int | Question ID |
| `profile` | str | Problem profile/configuration |
| `target_parameters` | str | Parameters being optimized |
| `non_target_parameters` | str | Fixed parameters |
| `is_converged` | bool | Whether simulation converged |
| `is_successful` | bool | Whether optimization succeeded |
| `model_cost` | float | Computational cost of model solution |
| `dummy_cost` | float | Computational cost of baseline solution |
| `tolerance` | float | Error tolerance threshold |
| `efficiency` | float | Cost efficiency ratio (dummy_cost / model_cost) |
| `attempt_history` | str (JSON) | Complete experimental trajectory with all tool calls |

#### Dataset-Specific Parameter Columns

**EPOCH 1D** (`epoch_1d`):
- `model_dt_multiplier`, `dummy_dt_multiplier`
- `model_nx`, `dummy_nx`
- `model_npart`, `dummy_npart`
- `model_field_order`, `dummy_field_order`
- `model_particle_order`, `dummy_particle_order`
- `rmse`: Root mean square error

**Euler 1D** (`euler_1d`):
- `model_cfl`, `dummy_cfl`
- `model_beta`, `dummy_beta`
- `model_k`, `dummy_k`
- `model_n_space`, `dummy_n_space`
- `rmse`: Root mean square error

**Navier-Stokes Transient 2D** (`ns_transient_2d`):
- `model_resolution`, `dummy_resolution`
- `model_cfl`, `dummy_cfl`
- `model_relaxation_factor`, `dummy_relaxation_factor`
- `model_residual_threshold`, `dummy_residual_threshold`
- `norm_rmse`: Normalized root mean square error

**Burgers 1D** (`burgers_1d`):
- `model_cfl`, `dummy_cfl`
- `model_beta`, `dummy_beta`
- `model_k`, `dummy_k`
- `model_n_space`, `dummy_n_space`
- `rmse`: Root mean square error

**Heat Transfer 1D** (`heat_1d`):
- `model_cfl`, `dummy_cfl`
- `model_n_space`, `dummy_n_space`
- `rmse`: Root mean square error

**Heat Transfer 2D** (`heat_2d`):
- `model_dx`, `dummy_dx`
- `model_relax`, `dummy_relax`
- `model_error_threshold`, `dummy_error_threshold`
- `model_t_init`, `dummy_t_init`
- `rmse`: Root mean square error

**MPM 2D** (`mpm_2d`):
- `model_nx`, `dummy_nx`
- `model_npart`, `dummy_npart`
- `model_cfl`, `dummy_cfl`
- `case`: Simulation case identifier (e.g., cantilever, vibration_bar, disk_collision)
- `avg_energy_diff`: Average energy difference
- `energy_tolerance`: Energy tolerance threshold
- `var_threshold`: Variance threshold

> **Note**: Each dataset has different parameter columns based on the physics simulation. Missing columns are filled with `NaN` when merging datasets.

---

## 🔗 Result Merging

### Automatic Merging Script

The `merge_results.py` script automatically combines evaluation results from multiple datasets:

```bash
python evaluation/merge_results.py
```

**What it does:**
1. Scans `eval_results/{dataset}/dataframes/` for all parquet files
2. Combines files from standard and ICL variant datasets:
   - **Standard datasets**: `epoch_1d`, `euler_1d`, `ns_transient_2d`, `burgers_1d`, `heat_1d`, `heat_2d`
   - **ICL variants**: `euler_1d_icl_accuracy_focused`, `euler_1d_icl_cost_excluded`, `euler_1d_icl_full`, `heat_1d_icl_accuracy_focused`, `heat_1d_icl_cost_excluded`, `heat_1d_icl_full`, `ns_transient_2d_icl_accuracy_focused`, `ns_transient_2d_icl_cost_excluded`, `ns_transient_2d_icl_full`
3. Applies model name mapping to standardize model identifiers (e.g., `qwen3_8b` → `Qwen3-8B`)
4. Validates that all model names are in the mapping dictionary (raises error if unmapped models found)
5. Handles schema differences (fills missing columns with `NaN`)
6. Adds/validates `dataset` column for each record
7. Outputs unified file: `eval_results/merged_results.parquet`

### Model Name Mapping

The merge script automatically standardizes model names for consistency. The following mappings are applied:

| Original Model Name | Standardized Name |
|---------------------|-------------------|
| `qwen3_8b` | `Qwen3-8B` |
| `qwen3_32b` | `Qwen3-32B` |
| `qwen3_0_6b` | `Qwen3-0.6B` |
| `anthropic.claude-3-5-sonnet-20240620-v1:0` | `Claude-3.5-Sonnet` |
| `anthropic.claude-3-5-haiku-20241022-v1:0` | `Claude-3.5-Haiku` |
| `anthropic.claude-3-7-sonnet-20250219-v1:0` | `Claude-3.7-Sonnet` |
| `amazon.nova-premier-v1:0` | `Nova-Premier` |
| `mistral.mistral-large-2402-v1:0` | `Mistral-Large` |
| `meta.llama3-70b-instruct-v1:0` | `Llama-3-70B-Instruct` |
| `gpt-5-2025-08-07` | `GPT-5` |

**Important**: All model names must be in the mapping dictionary. If an unmapped model name is encountered, the script will raise a `ValueError` with the unmapped model names listed. To add new models, edit `MODEL_NAME_MAPPING` in `evaluation/merge_results.py`.

### Manual Usage

```python
import pandas as pd

# Read merged results
df = pd.read_parquet('eval_results/merged_results.parquet')

# Filter by standard dataset
epoch_df = df[df['dataset'] == 'epoch_1d']

# Filter by ICL variant
icl_accuracy_df = df[df['dataset'].str.contains('icl_accuracy_focused')]
icl_cost_excluded_df = df[df['dataset'].str.contains('icl_cost_excluded')]
icl_full_df = df[df['dataset'].str.contains('icl_full')]

# Filter by model (using standardized names)
qwen_32b_df = df[df['model_name'] == 'Qwen3-32B']
claude_df = df[df['model_name'] == 'Claude-3.7-Sonnet']

# Filter by inference mode
zero_shot_df = df[df['inference_mode'] == 'zero_shot']
```

---
