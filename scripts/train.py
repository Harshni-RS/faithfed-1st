"""
Training script for FaithFed-XAI and Comparative Baseline Algorithms.
Supports:
- FaithFed-XAI (Explanation-Aware Adaptive DP + FedBN + Secure Aggregation)
- FedBN (Personalized Federated Learning without DP)
- FedAvg (Standard Federated Averaging without personalization/DP)
- FedAvg-FixedDP (Standard FedAvg with static uniform Differential Privacy)
- Centralized (Pooled benchmark)
"""

import os
import sys
import yaml
import argparse
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.federated_trainer import FederatedTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="FaithFed-XAI Training Script")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--algorithm", type=str, default=None, choices=["FaithFed-XAI", "FedBN", "FedAvg", "FedAvg-FixedDP", "Centralized"], help="Override federated algorithm")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of federated rounds")
    parser.add_argument("--epochs", type=int, default=None, help="Override local epochs per round")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epsilon", type=float, default=None, help="Override target privacy budget epsilon")
    parser.add_argument("--max_samples_per_client", type=int, default=None, help="Optional sample cap per client for fast testing/demo")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Overrides from CLI
    if args.algorithm:
        config["federated"]["algorithm"] = args.algorithm
    if args.rounds:
        config["federated"]["num_rounds"] = args.rounds
    if args.epochs:
        config["federated"]["local_epochs"] = args.epochs
    if args.batch_size:
        config["federated"]["batch_size"] = args.batch_size
    if args.lr:
        config["federated"]["lr"] = args.lr
    if args.epsilon:
        config["differential_privacy"]["target_epsilon"] = args.epsilon
    if args.max_samples_per_client:
        config["data"]["max_samples_per_client"] = args.max_samples_per_client

    device = None
    if args.device:
        device = torch.device(args.device)

    print("=" * 70)
    print("FaithFed-XAI: Explanation-Aware Privacy-Adaptive Federated Learning")
    print(f"Algorithm: {config['federated']['algorithm']}")
    print(f"Rounds: {config['federated']['num_rounds']} | Local Epochs: {config['federated']['local_epochs']}")
    print(f"Batch Size: {config['federated']['batch_size']} | Learning Rate: {config['federated']['lr']}")
    print(f"Target Privacy Budget (Epsilon): {config['differential_privacy']['target_epsilon']}")
    print("=" * 70)

    trainer = FederatedTrainer(config=config, device=device)
    history = trainer.train_federated()

    print("\nTraining completed successfully!")
    print(f"Artifacts saved to: {config['output']['artifacts_dir']}")


if __name__ == "__main__":
    main()
