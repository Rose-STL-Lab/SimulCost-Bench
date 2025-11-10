# SimulCost-Bench Results Analysis

## Overall Results

Zero-shot performance across 9 datasets with 32 distinct tasks.

**Averaging methodology:** Two-stage to ensure fairness across imbalanced data
1. Average within each (dataset, task) combination
2. Unweighted average across all 32 task combinations

### Success Rate

![Overall Success Rate](res/overall_success_rate.png)

**Metric:** Percentage of instances meeting accuracy tolerance, task-averaged.

### Efficiency

![Overall Efficiency](res/overall_efficiency.png)

**Metric:** `success * (dummy_cost / model_cost)`, task-averaged. Higher = better cost-effectiveness.
