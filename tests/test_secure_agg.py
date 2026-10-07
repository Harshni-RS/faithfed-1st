"""
Unit tests for Secure Aggregation and FedBN parameter isolation.
"""

import unittest
import torch
from src.model import get_model
from src.fedbn import (
    extract_shared_parameters,
    extract_local_parameters,
    apply_shared_parameters,
    apply_local_parameters,
    federated_average_shared
)
from src.secure_aggregation import (
    create_pairwise_masks_for_clients,
    mask_client_update,
    aggregate_masked_updates,
    verify_mask_cancellation
)


class TestSecureAggregation(unittest.TestCase):
    def setUp(self):
        self.client_ids = [f"hospital_{i+1}" for i in range(5)]
        self.model = get_model()
        self.shared_params = extract_shared_parameters(self.model)

    def test_pairwise_mask_cancellation(self):
        """Verifies that the sum of all pairwise masks equals 0 within float precision."""
        client_masks = create_pairwise_masks_for_clients(
            self.client_ids, self.shared_params, round_idx=1, scale=1.0
        )
        is_exact, max_residual = verify_mask_cancellation(client_masks)
        self.assertTrue(is_exact, f"Mask cancellation failed with max residual {max_residual}")
        self.assertLess(max_residual, 1e-5)

    def test_masked_aggregation_equivalence(self):
        """Verifies that server sum of masked updates equals exact weighted average of raw updates."""
        client_counts = [1000, 500, 300, 400, 600]
        total_count = sum(client_counts)
        client_weights = [c / total_count for c in client_counts]

        # Simulate 5 client shared model states
        client_updates = []
        for i in range(5):
            client_dict = {}
            for k, v in self.shared_params.items():
                client_dict[k] = v + torch.randn_like(v) * (i + 1) * 0.1
            client_updates.append(client_dict)

        # 1. Plain FedAvg (baseline unmasked)
        expected_avg = federated_average_shared(client_updates, client_counts)

        # 2. Secure Aggregation with pairwise masks
        client_masks = create_pairwise_masks_for_clients(
            self.client_ids, self.shared_params, round_idx=2, scale=2.5
        )
        masked_updates = []
        for i, cid in enumerate(self.client_ids):
            m_up = mask_client_update(client_updates[i], client_masks[cid], client_weights[i])
            masked_updates.append(m_up)

        secure_avg, audit_info = aggregate_masked_updates(masked_updates, self.shared_params)

        # 3. Check exact match
        for key in expected_avg.keys():
            diff = torch.max(torch.abs(expected_avg[key].float() - secure_avg[key].float())).item()
            self.assertLess(diff, 1e-4, f"Mismatch in layer {key}: max diff {diff}")

    def test_fedbn_isolation(self):
        """Verifies that FedBN separates GroupNorm affine params from shared params."""
        shared = extract_shared_parameters(self.model)
        local = extract_local_parameters(self.model)

        # Check disjoint keys
        shared_keys = set(shared.keys())
        local_keys = set(local.keys())
        self.assertEqual(len(shared_keys.intersection(local_keys)), 0)

        # Check total parameter count matches
        total_keys = set(self.model.state_dict().keys())
        self.assertEqual(shared_keys.union(local_keys), total_keys)

        # Check all local keys contain GroupNorm
        for k in local_keys:
            self.assertTrue("gn.weight" in k or "gn.bias" in k)


if __name__ == "__main__":
    unittest.main()
