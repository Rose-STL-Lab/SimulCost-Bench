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

# Pairing columns for paired t-tests (columns that identify unique task configurations)
PAIRING_COLS = [
    "dataset",
    "task",
    "precision_level",
    "model_name",
    "profile",
    "target_parameters",
    "non_target_parameters",
]

# Precision level ordering for plots (high to low)
PRECISION_ORDER = ["High", "Medium", "Low"]

# Precision level markers for forest plots
PRECISION_MARKERS = {"High": "o", "Medium": "s", "Low": "^"}

# MATLAB-like default colors for subgroups
SUBGROUP_COLORS = [
    "#0072BD",  # Blue
    "#D95319",  # Orange
    "#EDB120",  # Yellow
    "#7E2F8E",  # Purple
    "#77AC30",  # Green
    "#4DBEEE",  # Cyan
    "#A2142F",  # Red
]
