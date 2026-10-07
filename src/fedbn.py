"""
FedBN (Federated Learning with Local Batch/Group Normalization) implementation.
Keeps normalization affine parameters strictly local to each hospital to preserve
personalized feature normalization across non-IID imaging distributions.
Only shared convolutional and linear parameters participate in federated aggregation.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple


def is_local_param(param_name: str) -> bool:
    """
    Identifies whether a parameter is a local normalization parameter in FedBN.
    For SmallCNN, GroupNorm affine parameters (gn.weight, gn.bias) are local.
    """
    local_keywords = ["gn.weight", "gn.bias", "bn.weight", "bn.bias"]
    return any(kw in param_name for kw in local_keywords)


def extract_shared_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extracts only the shared parameters to be sent to the central server.
    All GroupNorm affine parameters are filtered out.
    """
    shared_dict = {}
    for name, param in model.state_dict().items():
        if not is_local_param(name):
            shared_dict[name] = param.detach().cpu().clone()
    return shared_dict


def extract_local_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extracts only the local personalization parameters (GroupNorm weights & biases).
    """
    local_dict = {}
    for name, param in model.state_dict().items():
        if is_local_param(name):
            local_dict[name] = param.detach().cpu().clone()
    return local_dict


def apply_shared_parameters(model: nn.Module, shared_state: Dict[str, torch.Tensor]):
    """
    Loads aggregated shared parameters into the model without overwriting
    the hospital's local normalization parameters.
    """
    current_state = model.state_dict()
    for name, param in shared_state.items():
        if name in current_state and not is_local_param(name):
            current_state[name].copy_(param.to(current_state[name].device))


def apply_local_parameters(model: nn.Module, local_state: Dict[str, torch.Tensor]):
    """
    Restores the hospital's local GroupNorm parameters.
    """
    current_state = model.state_dict()
    for name, param in local_state.items():
        if name in current_state and is_local_param(name):
            current_state[name].copy_(param.to(current_state[name].device))


def federated_average_shared(
    client_shared_weights: List[Dict[str, torch.Tensor]],
    client_sample_counts: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Performs dataset-size weighted FedAvg on the shared model parameters.
    W_global = sum( (n_k / N) * W_k )
    """
    total_samples = sum(client_sample_counts)
    weights = [count / total_samples for count in client_sample_counts]
    
    first_dict = client_shared_weights[0]
    global_shared = {}
    
    for key in first_dict.keys():
        # Weighted sum across all clients
        stacked = torch.stack([client_shared_weights[i][key].float() * weights[i] for i in range(len(client_shared_weights))])
        global_shared[key] = stacked.sum(dim=0).to(first_dict[key].dtype)
        
    return global_shared
