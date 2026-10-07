"""
Smoke test: run ONE client of FaithFed-XAI through Opacus DP-SGD on synthetic data.
Run from the project root:   python scripts/test_opacus_dp.py
Expected: prints epsilon_spent (not above the target), a noise multiplier,
and confirms the weights changed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset

from src.model import get_model
from src.opacus_dp import train_client_with_opacus


class FakeFundus(Dataset):
    """Returns (image, label, id) exactly like APTOSDataset does."""
    def __init__(self, n=64, size=128):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, size, size, generator=g)
        self.y = torch.randint(0, 5, (n,), generator=g)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], int(self.y[i]), f"img_{i}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=5, pretrained=False).to(device)

    # Same freezing rule as FederatedTrainer.train_client_local
    if hasattr(model, "layer4"):
        for p in model.parameters():
            p.requires_grad = False
        for m in (model.layer4, model.fc1, model.classifier):
            for p in m.parameters():
                p.requires_grad = True

    before = [p.detach().clone() for p in model.parameters()]
    loader = DataLoader(FakeFundus(), batch_size=16, shuffle=True)

    target_eps = 4.0
    report = train_client_with_opacus(
        model=model, dataloader=loader, epochs=1, lr=1e-3,
        target_epsilon=target_eps, target_delta=1e-5, max_grad_norm=1.0,
        class_weights=torch.tensor([1.0, 1.3, 1.0, 1.6, 1.6]), device=device,
    )
    changed = any(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))

    print("Opacus report:", report)
    print("weights changed:", changed)
    assert changed, "model weights did not update"
    assert report["epsilon_spent"] <= target_eps + 1e-3, "epsilon overspent"
    print("OK: Opacus DP-SGD ran and stayed within the epsilon budget.")


if __name__ == "__main__":
    main()
