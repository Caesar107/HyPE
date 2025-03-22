import os
import pytest
import numpy as np
import pickle
import torch
from pathlib import Path
import binascii

from garage.utils.common import PROJECT_ROOT


def test_examine_walker2d_transitions_file():
    """Examine the Walker2d transitions file structure and confirm its format."""
    # Define the path to the transitions file (now using v3 or v4)
    env_version = "v3"  # Change this to match your environment version
    env_name = f"Walker2d-{env_version}"
    
    # Check for different possible file paths
    possible_paths = [
        Path(PROJECT_ROOT, "experts", env_name, f"{env_name}_demos.npz"),
        Path(PROJECT_ROOT, "experts", env_name, f"transitions_{env_name}.npy"),
        Path(PROJECT_ROOT, "expert_data", f"transitions_{env_name}.npy"),
        # Add other potential paths here
    ]
    
    # Find the first existing file
    transitions_path = None
    for path in possible_paths:
        if path.exists():
            transitions_path = path
            break
    
    assert transitions_path is not None, f"No transition files found for {env_name}"
    print(f"Using file: {transitions_path}")
    print(f"File extension: {transitions_path.suffix}")
    
    try:
        # Load with torch.load directly
        transitions = torch.load(transitions_path)
        print("\n=== PYTORCH TRANSITION DATA ===")
        
        # Check if transitions has .obs attribute
        if hasattr(transitions, 'obs'):
            print(f"Number of transitions in demonstrations: {transitions.obs.shape[0]}.")
            print(f"Expert observation shape: {transitions.obs.shape}")
            
            # Print other attributes if available
            attrs_to_check = ['obs', 'actions', 'next_obs', 'rewards', 'dones', 'terminals']
            print("\n=== TRANSITIONS ATTRIBUTES ===")
            for attr in attrs_to_check:
                if hasattr(transitions, attr):
                    attr_value = getattr(transitions, attr)
                    if hasattr(attr_value, 'shape'):
                        print(f"  {attr}: shape = {attr_value.shape}, type = {type(attr_value)}")
                    else:
                        print(f"  {attr}: type = {type(attr_value)}")
            
            # Print any other attributes that might be present
            all_attrs = [attr for attr in dir(transitions) if not attr.startswith('__') and not callable(getattr(transitions, attr))]
            other_attrs = [attr for attr in all_attrs if attr not in attrs_to_check]
            if other_attrs:
                print("\n=== OTHER ATTRIBUTES ===")
                for attr in other_attrs:
                    attr_value = getattr(transitions, attr)
                    if hasattr(attr_value, 'shape'):
                        print(f"  {attr}: shape = {attr_value.shape}, type = {type(attr_value)}")
                    else:
                        print(f"  {attr}: type = {type(attr_value)}")
        
        # If transitions is a dict instead of an object with attributes
        elif isinstance(transitions, dict):
            print("\n=== TRANSITIONS DICTIONARY ===")
            for key, value in transitions.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape = {value.shape}, type = {type(value)}")
                else:
                    print(f"  {key}: type = {type(value)}")
        
        else:
            print(f"Transitions is of type {type(transitions)} without .obs attribute")
            print(f"Dir of transitions: {dir(transitions)}")
    
    except Exception as e:
        pytest.fail(f"Failed to load transitions file: {str(e)}")


if __name__ == "__main__":
    test_examine_walker2d_transitions_file()
