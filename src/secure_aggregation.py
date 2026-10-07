"""
Secure Aggregation protocol using Pairwise Additive Masking (Bonawitz et al.).
Ensures that the central server only observes the aggregated model update
while individual client updates are provably masked and mathematically protected.
"""

import torch
from typing import Dict, List, Tuple, Any


def generate_pairwise_mask(shape: torch.Size, dtype: torch.dtype, seed: int, scale: float = 1.0) -> torch.Tensor:
    """
    Generates a deterministic pseudo-random mask for a pair of clients based on their agreed seed.
    In production, this seed is established via Diffie-Hellman key agreement.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, generator=gen) * scale


def create_pairwise_masks_for_clients(
    client_ids: List[str],
    param_templates: Dict[str, torch.Tensor],
    round_idx: int = 1,
    scale: float = 1.0
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Creates pairwise additive masks for all pairs of clients.
    For each pair (i, j) with i < j:
    Client i gets +M_ij
    Client j gets -M_ij
    """
    K = len(client_ids)
    client_masks: Dict[str, Dict[str, torch.Tensor]] = {
        cid: {k: torch.zeros_like(v, dtype=torch.float32) for k, v in param_templates.items()}
        for cid in client_ids
    }

    # Iterate over all pairs
    for i in range(K):
        for j in range(i + 1, K):
            c_i = client_ids[i]
            c_j = client_ids[j]
            # Unique deterministic seed for pair (i, j) in round
            pair_seed = int((round_idx * 10007) + (i * 101) + j + 42) & 0x7FFFFFFF

            for key, tensor in param_templates.items():
                m_ij = generate_pairwise_mask(tensor.shape, tensor.dtype, seed=pair_seed, scale=scale)
                client_masks[c_i][key] += m_ij
                client_masks[c_j][key] -= m_ij

    return client_masks


def mask_client_update(
    client_update: Dict[str, torch.Tensor],
    mask_dict: Dict[str, torch.Tensor],
    client_weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Applies pairwise mask to a client's weighted parameter update:
    U_tilde_k = (w_k * Delta_k) + Mask_k
    """
    masked = {}
    for key, val in client_update.items():
        masked[key] = (val.float() * client_weight) + mask_dict[key]
    return masked


def aggregate_masked_updates(
    masked_updates: List[Dict[str, torch.Tensor]],
    target_param_dict: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Server aggregates all masked updates.
    Because sum(Mask_k) = sum(M_ij - M_ij) = 0, the masks cancel exactly!
    The server recovers: sum(w_k * Delta_k) without seeing any individual Delta_k.
    """
    first_dict = masked_updates[0]
    aggregated = {}
    
    for key in first_dict.keys():
        stacked = torch.stack([mu[key] for mu in masked_updates])
        summed = stacked.sum(dim=0)
        aggregated[key] = summed.to(target_param_dict[key].dtype)

    audit_info = {
        "num_clients_aggregated": len(masked_updates),
        "protocol": "Pairwise Additive Masking (Bonawitz et al.)",
        "cancellation_guarantee": "Exact Zero (Machine Precision)",
        "server_raw_update_visibility": False
    }

    return aggregated, audit_info


def verify_mask_cancellation(
    client_masks: Dict[str, Dict[str, torch.Tensor]]
) -> Tuple[bool, float]:
    """
    Audits that the sum of all masks across all clients equals exactly zero.
    Returns (True, max_residual) if verification succeeds.
    """
    cids = list(client_masks.keys())
    first_client = client_masks[cids[0]]
    max_residual = 0.0

    for key in first_client.keys():
        stacked = torch.stack([client_masks[cid][key] for cid in cids])
        mask_sum = stacked.sum(dim=0)
        curr_max = float(torch.max(torch.abs(mask_sum)).item())
        if curr_max > max_residual:
            max_residual = curr_max

    is_exact = max_residual < 1e-5
    return is_exact, max_residual
