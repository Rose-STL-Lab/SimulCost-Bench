# Heat Transfer Problem System

## 📦 Environment Setup

```bash
# Clone repository
git clone git@github.com:Rose-STL-Lab/coastbench.git

# Create Conda environment
conda env create -n casebench --file environment.yml
conda activate casebench

# Install dependencies with Poetry
poetry install
```

<!-- ## 🚀 Meta Agent Usage
### Generate Questions
```bash
python qs_gen/heat_transfer.py -n 10
```
#### Parameters:
- n: int, number of example 
#### Outpts:
- output_file: data/heat_transfer/question.json

### Create Solving Agents using validation dataset
```bash
python agent_gen/heat_transfer/search.py -v 3 -p gemini -m gemini-1.5-pro -g 2
```
#### Parameters
- v: int, validation dataset number
- p: str, provider
- m: str, model name
- g: int, agent generation times

#### Outputs
- output files: 
    - data/heat_transfer/agent.json: The agent will include the code and workflow 
    - data/heat_transfer/dataset.json: The dataset will use the latest workflow in agent.json for the prompt

### Evaluate Meta Agent using test dataset
```bash
python inference/langchain_LLM.py -s 3 -p gemini -m gemini-1.5-pro -d heat_transfer
```
#### Parameters
- s: int, start idx
- p: str, provider
- m: str, model name
- d: str, dataset name

### Generate Evaluation Report 
```bash
python evaluation/create_md.py -m gemini-1.5-pro -d heat_transfer -v -g 1
```

#### Parameters
- m: str, model name
- d: str, dataset name
- v: bool, using the result from validation dataset
- t: bool, using the result from test dataset
- g: int, the version of generated agent

#### Outputs
- Output file: evaluation/heat_transfer/validation/gemini-1.5-pro_g1.md -->

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
| 1D Burgers               | `k`              | ✅ Supported (Mix)  |
| 1D Burgers               | `w`              | ✅ Supported (Mix)  |


<!-- ## 🧐 Human Written Version -->
## 🕵️ Generate Questions
```bash
# 1D Heat Transfer
python qs_gen/1D_heat_transfer.py -n 10 -t cfl -z

# 2D Steady Heat Transfer
python qs_gen/2D_heat_transfer.py -n 10 -t dx -z

# Burgers 1D Equation with 2nd Order Roe Method
python qs_gen/1D_burgers.py -t cfl -z
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
```
### Parameters:
- t: problem task
- -z: zero-shot 

## 🧠 Run Inference
```bash
# 1D Heat Transfer
python inference/langchain_LLM.py -n 10 -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 1D_heat_transfer -t cfl -z

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

## 🤖 Models
1. anthropic.claude-3-5-haiku-20241022-v1:0
2. anthropic.claude-3-5-sonnet-20240620-v1:0
3. anthropic.claude-3-7-sonnet-20250219-v1:0
