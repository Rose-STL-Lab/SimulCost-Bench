"""
Constants for SimulCost-Bench plotting and analysis.
"""

# Base datasets used for main results (excluding ICL variants)
BASE_DATASETS = [
    "burgers_1d",
    "diff_react_1d",
    "euler_1d",
    "heat_1d",
    "heat_2d",
    "ns_transient_2d",
    "mpm_2d",
    "epoch_1d",
    "euler_2d",
]

# Datasets that have In-Context Learning (ICL) variants
ICL_BASE_DATASETS = [
    "euler_1d",
    "heat_1d",
    "mpm_2d",
    "ns_transient_2d",
]

# ICL variant suffixes and their human-readable names
ICL_VARIANTS = {
    "": "Normal",  # Base dataset without suffix
    "_icl_full": "ICL-Full",
    "_icl_accuracy_focused": "ICL-Accuracy",
    "_icl_cost_excluded": "ICL-NoCost",
}

# Order for plotting ICL variants
ICL_VARIANT_ORDER = ["", "_icl_full", "_icl_accuracy_focused", "_icl_cost_excluded"]
