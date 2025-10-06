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

# 2. Merge results from multiple datasets
python evaluation/stats_utils/merge_results.py

# 3. Analyze the merged data
python -c "import pandas as pd; df = pd.read_parquet('eval_results/merged_results.parquet'); print(df.head())"
```

---

## 🔄 Evaluation Workflow

### Step 1: Run Evaluation Scripts

Each dataset has its own evaluation script in the corresponding directory:

```bash
# Heat Transfer (1D)
python evaluation/heat_1d/eval.py -m <model> -d heat_1d -t <task> -l <precision> -z

# Heat Transfer (2D)
python evaluation/heat_2d/eval.py -m <model> -d heat_2d -t <task> -z

# Burgers Equation (1D)
python evaluation/burgers_1d/eval.py -m <model> -d burgers_1d -t <task> -l <precision> -z

# Euler Equations (1D)
python evaluation/euler_1d/eval.py -m <model> -d euler_1d -t <task> -l <precision> -z

# Navier-Stokes (2D Channel)
python evaluation/ns_2d/eval.py -m <model> -d ns_2d -t <task> -z

# Navier-Stokes (2D Transient)
python evaluation/ns_transient_2d/eval.py -m <model> -d ns_transient_2d -t <task> -z

# EPOCH PIC (1D)
python evaluation/epoch_1d/eval.py -m <model> -d epoch_1d -t <task> -l <precision> -z
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
| `qid` | str | Question ID |
| `profile` | str | Problem profile/configuration |
| `target_parameters` | str | Parameters being optimized |
| `non_target_parameters` | str | Fixed parameters |
| `is_converged` | bool | Whether simulation converged |
| `is_successful` | bool | Whether optimization succeeded |
| `model_cost` | float | Computational cost of model solution |
| `dummy_cost` | float | Computational cost of baseline solution |
| `tolerance` | float | Error tolerance threshold |
| `efficiency` | float | Cost efficiency ratio (dummy_cost / model_cost) |

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

> **Note**: Each dataset has different parameter columns based on the physics simulation. Missing columns are filled with `NaN` when merging datasets.

---

## 🔗 Result Merging

### Automatic Merging Script

The `merge_results.py` script automatically combines evaluation results from multiple datasets:

```bash
python evaluation/stats_utils/merge_results.py
```

**What it does:**
1. Scans `eval_results/{dataset}/dataframes/` for all parquet files
2. Combines files from standard and ICL variant datasets:
   - **Standard datasets**: `epoch_1d`, `euler_1d`, `ns_transient_2d`
   - **ICL variants**: `euler_1d_icl_accuracy_focused`, `euler_1d_icl_cost_excluded`, `euler_1d_icl_full`, `ns_transient_2d_icl_accuracy_focused`, `ns_transient_2d_icl_cost_excluded`, `ns_transient_2d_icl_full`
3. Handles schema differences (fills missing columns with `NaN`)
4. Adds/validates `dataset` column for each record
5. Outputs unified file: `eval_results/merged_results.parquet`

**Output Summary:**
```
============================================================
MERGE SUMMARY REPORT
============================================================

Total rows: 4,578
Total columns: 41

Rows per dataset:
  epoch_1d: 1,638 rows
  euler_1d: 1,576 rows
  ns_transient_2d: 1,364 rows

Unique models: 4
Inference modes: ['iterative', 'zero_shot']
Precision levels: ['low', 'medium', 'high']
============================================================
```

### Understanding ICL Dataset Variants

The ICL (In-Context Learning) datasets test how different prompt configurations affect model performance:

| Dataset Variant | Description |
|----------------|-------------|
| `{dataset}_icl_accuracy_focused` | Prompts emphasize accuracy metrics only |
| `{dataset}_icl_cost_excluded` | Prompts exclude computational cost information |
| `{dataset}_icl_full` | Prompts include both accuracy and cost information |

These variants help analyze the impact of prompt engineering on model optimization strategies.

### Manual Usage

```python
import pandas as pd

# Read merged results
df = pd.read_parquet('eval_results/merged_results.parquet')

# Filter by standard dataset
epoch_df = df[df['dataset'] == 'epoch_1d']
euler_df = df[df['dataset'] == 'euler_1d']

# Filter by ICL variant
icl_full_df = df[df['dataset'].str.contains('icl_full')]
icl_accuracy_df = df[df['dataset'].str.contains('icl_accuracy_focused')]

# Filter by model
gpt4_df = df[df['model_name'] == 'gpt-4o']

# Filter by inference mode
zero_shot_df = df[df['inference_mode'] == 'zero_shot']
```

---

## 📚 Additional Resources

- **Main Documentation**: [../README.md](../README.md)
- **Custom Model Integration**: [../custom_model/README.md](../custom_model/README.md)
- **Script Automation**: [../scripts/README.md](../scripts/README.md)
