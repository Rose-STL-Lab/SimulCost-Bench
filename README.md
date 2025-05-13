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

## 🧐 Human Written Version
### Generate Questions Or Use the Previous one
```bash
python qs_gen/1D_heat_transfer.py -n 2 -t cfl
python qs_gen/2D_heat_transfer.py -n 2 -t dx
```
#### Parameters:
- n: int, number of example 
- t: problem task
- -z: zero-shot 
#### Outpts:
- output_file: data/heat_transfer/question.json

### 🕵️ Generate the Dataset for Human Written Workflow and code
```bash
python dataset_gen/oneD_heat_transfer.py -t cfl
python dataset_gen/twoD_heat_transfer.py -t dx
```
#### Parameters:
- t: problem task
- -z: zero-shot 

### 🔍 Evaluate Model using test dataset
```bash
python inference/langchain_LLM.py -n 2 -p bedrock -m anthropic.claude-3-5-haiku-20241022-v1:0 -d 1D_heat_transfer -t cfl
python inference/langchain_LLM.py -n 2 -p bedrock -m anthropic.claude-3-7-sonnet-20250219-v1:0 -d 2D_heat_transfer -t dx
```
#### Parameters
- n: int, number of samples to test
- p: str, LLM provider (openai, gemini, bedrock)
- m: str, model name
- d: str, dataset name
- t: problem task
- -z: zero-shot 
<!-- - hv: bool, human written version -->

### Models
1. anthropic.claude-3-5-haiku-20241022-v1:0
2. anthropic.claude-3-5-sonnet-20240620-v1:0
3. anthropic.claude-3-7-sonnet-20250219-v1:0

<!-- ### 📊 Generate Evaluation Report 
```bash
python evaluation/create_md.py -m gemini-1.5-pro -d heat_transfer -t
```

#### Parameters
- m: str, model name
- d: str, dataset name
- v: bool, using the result from validation dataset
- t: bool, using the result from test dataset
- g: int, the version of generated agent -->

<!-- #### Outputs
- Output file: evaluation/heat_transfer/validation/gemini-1.5-pro_g1.md -->

<!-- ## TODOs
1. Error Handling Analysis:
- Implement tracking for tool call failures
- Add metrics for incomplete experiment
2. Feedback for Current Agent:
- Use results generated from current agent to LLMs for feedback generation to the agent for further improvement
3. In-context Learning:
- Generate valid example for LLMs 
4. Span the domain of others dataset -->
