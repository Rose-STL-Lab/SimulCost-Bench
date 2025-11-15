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

# Order for plotting ICL variants
ICL_VARIANT_ORDER = ["Normal", "ICL-Full", "ICL-Accuracy", "ICL-NoCost"]

# ICL is only for Claude model
ICL_MODEL = "Claude-3.7-Sonnet"
