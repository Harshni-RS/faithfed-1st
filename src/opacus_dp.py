"""
Opacus-based differential privacy for FaithFed-XAI client training.

Replaces the hand-written `apply_dp_sgd_step` (batch-level clipping + manual noise)
with Opacus, which provides:
  * per-sample gradient clipping (the real DP-SGD requirement),
  * Poisson sub-sampling of batches,
  * calibrated Gaussian noise, with sigma chosen by Opacus' privacy accountant
    so that the client reaches its allocated (epsilon, delta),
  * a privacy accountant that reports the epsilon actually spent.

The AdaptivePrivacyController still decides each hospital's epsilon_k (H + A + D).
Opacus converts that epsilon_k into the noise level and enforces it.
"""

import copy
import types
from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


def _basicblock_forward(self, x):
    """torchvision BasicBlock without in-place ops (`out += identity` breaks Opacus hooks)."""
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    return self.relu(out)


def _bottleneck_forward(self, x):
    """torchvision Bottleneck without in-place ops."""
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    return self.relu(out)


def make_opacus_compatible(model: nn.Module) -> nn.Module:
    """
    Opacus attaches backward hooks to layers that have trainable parameters. PyTorch
    forbids modifying a hooked layer's output in place, which raises:
        "Output 0 of BackwardHookFunction is a view and is being modified inplace".
    So every in-place activation and in-place residual add is made out-of-place.
    The maths (and the saved weights) are unchanged.
    """
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.LeakyReLU)):
            m.inplace = False
        if isinstance(m, BasicBlock):
            m.forward = types.MethodType(_basicblock_forward, m)
        elif isinstance(m, Bottleneck):
            m.forward = types.MethodType(_bottleneck_forward, m)
    return model


def train_client_with_opacus(
    model: nn.Module,
    dataloader,
    epochs: int,
    lr: float,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    class_weights: torch.Tensor,
    device: torch.device,
    weight_decay: float = 1e-4,
    accountant: str = "rdp",
) -> Dict[str, Any]:
    """
    Trains `model` in place for `epochs` with DP-SGD via Opacus so that the
    training run satisfies (target_epsilon, target_delta)-DP for this client.

    Returns a small report: {"epsilon_spent", "noise_multiplier", "steps"}.
    """
    # Imported here so the project still runs (manual DP path) if Opacus is missing.
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator

    # Work on a copy: Opacus attaches hooks to the module it wraps, and we do not
    # want those hooks to persist across federated rounds. Frozen layers
    # (requires_grad=False) are preserved by deepcopy.
    dp_model = copy.deepcopy(model).to(device)
    make_opacus_compatible(dp_model)  # remove in-place ops that clash with Opacus hooks
    dp_model.train()

    errors = ModuleValidator.validate(dp_model, strict=False)
    if errors:
        raise RuntimeError(f"Model is not Opacus-compatible: {errors}")

    trainable = [p for p in dp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

    privacy_engine = PrivacyEngine(accountant=accountant)
    dp_model, optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=dp_model,
        optimizer=optimizer,
        data_loader=dataloader,
        target_epsilon=float(target_epsilon),
        target_delta=float(target_delta),
        epochs=int(epochs),
        max_grad_norm=float(max_grad_norm),
    )

    # Class-weighted loss written as a per-sample weighted *mean over the batch*.
    # (nn.CrossEntropyLoss(weight=...) divides by the sum of weights, which would
    # break Opacus' assumption that "mean" means sum / batch_size.)
    ce = nn.CrossEntropyLoss(reduction="none")
    class_weights = class_weights.to(device)

    steps = 0
    for _ in range(epochs):
        for batch in dp_loader:
            batch_x, batch_y = batch[0].to(device), batch[1].to(device)
            if batch_x.shape[0] == 0:  # Poisson sampling can yield empty batches
                continue
            optimizer.zero_grad()
            logits = dp_model(batch_x)
            loss = (ce(logits, batch_y) * class_weights[batch_y]).mean()
            loss.backward()
            optimizer.step()
            steps += 1

    report = {
        "epsilon_spent": float(privacy_engine.get_epsilon(float(target_delta))),
        "noise_multiplier": float(optimizer.noise_multiplier),
        "steps": steps,
    }

    # Copy the trained weights back into the original (hook-free) model.
    model.load_state_dict(dp_model._module.state_dict())
    return report