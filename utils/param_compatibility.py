"""
Parameter compatibility utilities for SimulCost-Bench.

This module provides utilities to handle parameter naming inconsistencies
between different parts of the codebase, ensuring compatibility between
parameter history storage and function calling interfaces.
"""

from typing import Dict, Any, Optional, List, Union


def fetch_param(param_dict: Dict[str, Any], *param_names: str, default: Optional[Any] = None) -> Any:
    """
    Safely fetch parameter from dictionary, supporting multiple naming conventions.
    
    This function tries to find the parameter using different naming conventions:
    - Direct name (e.g., "cfl", "dx", "relax")
    - Current-prefixed name (e.g., "current_cfl", "current_dx", "current_relax")
    - Alternative names (e.g., "t_init" vs "T_init")
    
    Args:
        param_dict: Dictionary containing parameters
        *param_names: Parameter names to try in order of preference
        default: Default value to return if none of the parameter names are found
        
    Returns:
        Parameter value if found, otherwise default value
        
    Raises:
        KeyError: If none of the parameter names are found and no default is provided
        
    Examples:
        >>> params = {"cfl": 0.5, "n_space": 100}
        >>> fetch_param(params, "cfl", "current_cfl")
        0.5
        >>> fetch_param(params, "current_cfl", "cfl")
        0.5
        >>> fetch_param(params, "missing_param", default=1.0)
        1.0
    """
    for param_name in param_names:
        if param_name in param_dict:
            return param_dict[param_name]
    
    if default is not None:
        return default
        
    raise KeyError(f"None of {param_names} found in {param_dict}")


def normalize_param_dict(param_dict: Dict[str, Any], target_format: str = "current") -> Dict[str, Any]:
    """
    Normalize parameter dictionary to use consistent naming convention.
    
    Args:
        param_dict: Input parameter dictionary
        target_format: Target naming format ("current" or "direct")
            - "current": Convert to current_* format (e.g., "cfl" -> "current_cfl")
            - "direct": Convert to direct format (e.g., "current_cfl" -> "cfl")
    
    Returns:
        Dictionary with normalized parameter names
    """
    # Define parameter mapping from direct to current format
    PARAM_MAPPING = {
        "cfl": "current_cfl",
        "dx": "current_dx", 
        "relax": "current_relax",
        "error_threshold": "current_error_threshold",
        "t_init": "current_t_init",
        "T_init": "current_t_init",  # Handle T_init -> t_init -> current_t_init
        "n_space": "current_n_space",
        # Keep these without current_ prefix as they're used directly
        "k": "k",
        "w": "w", 
        "beta": "beta"
    }
    
    if target_format not in ["current", "direct"]:
        raise ValueError(f"target_format must be 'current' or 'direct', got: {target_format}")
    
    normalized = {}
    
    for key, value in param_dict.items():
        if target_format == "current":
            # Convert to current_* format
            if key in PARAM_MAPPING:
                new_key = PARAM_MAPPING[key]
            elif key.startswith("current_"):
                new_key = key  # Already in current format
            else:
                new_key = key  # Keep as is for unknown parameters
        else:  # target_format == "direct"
            # Convert to direct format
            if key.startswith("current_"):
                # Remove current_ prefix
                new_key = key.replace("current_", "")
                # Handle special case for t_init
                if new_key == "t_init":
                    # Keep as t_init (don't convert to T_init)
                    pass
            else:
                # Find reverse mapping
                reverse_mapping = {v: k for k, v in PARAM_MAPPING.items() if not k.startswith("current_")}
                new_key = reverse_mapping.get(key, key)
        
        normalized[new_key] = value
    
    return normalized


def get_parameter_alternatives(param_name: str) -> List[str]:
    """
    Get all possible alternative names for a parameter.
    
    Args:
        param_name: Base parameter name
        
    Returns:
        List of alternative parameter names to try
    """
    alternatives = {
        "cfl": ["cfl", "current_cfl"],
        "current_cfl": ["current_cfl", "cfl"],
        "dx": ["dx", "current_dx"],
        "current_dx": ["current_dx", "dx"],
        "relax": ["relax", "current_relax"],
        "current_relax": ["current_relax", "relax"],
        "error_threshold": ["error_threshold", "current_error_threshold"],
        "current_error_threshold": ["current_error_threshold", "error_threshold"],
        "t_init": ["t_init", "T_init", "current_t_init"],
        "T_init": ["T_init", "t_init", "current_t_init"],
        "current_t_init": ["current_t_init", "t_init", "T_init"],
        "n_space": ["n_space", "current_n_space"],
        "current_n_space": ["current_n_space", "n_space"],
        "k": ["k"],
        "w": ["w"], 
        "beta": ["beta"]
    }
    
    return alternatives.get(param_name, [param_name])


def validate_param_dict(param_dict: Dict[str, Any], required_params: List[str], 
                       strict: bool = False) -> tuple[bool, List[str]]:
    """
    Validate that a parameter dictionary contains all required parameters.
    
    Args:
        param_dict: Parameter dictionary to validate
        required_params: List of required parameter names
        strict: If True, require exact parameter names. If False, allow alternatives.
        
    Returns:
        Tuple of (is_valid, missing_params)
    """
    missing_params = []
    
    for param in required_params:
        if strict:
            if param not in param_dict:
                missing_params.append(param)
        else:
            # Try all alternatives
            alternatives = get_parameter_alternatives(param)
            found = any(alt in param_dict for alt in alternatives)
            if not found:
                missing_params.append(param)
    
    return len(missing_params) == 0, missing_params