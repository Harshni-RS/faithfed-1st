"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for SmallCNN.
Extracts explainability heatmaps pinpointing lesion regions (microaneurysms, hemorrhages, exudates)
driving the model's diabetic retinopathy grading.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List


class GradCAM:
    """
    Computes Grad-CAM heatmaps for PyTorch convolutional vision models.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # If target layer is not explicitly provided, find the last Conv2d layer
        if target_layer is None:
            self.target_layer = self._find_last_conv_layer(model)
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.fwd_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.bwd_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _find_last_conv_layer(self, module: nn.Module) -> nn.Module:
        """Finds the last Conv2d instance in the model."""
        last_conv = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model.")
        return last_conv

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates a 2D Grad-CAM heatmap normalized to [0, 1].
        
        Args:
            input_tensor: [1, C, H, W] preprocessed image tensor
            target_class: Optional target class index. If None, uses model's predicted class.
            
        Returns:
            heatmap: 2D numpy array [H, W] with values in [0, 1]
            predicted_class: integer class index
            confidence: softmax confidence for the target class
        """
        self.model.zero_grad()
        
        # Ensure single sample batch
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        pred_class = int(torch.argmax(logits, dim=1).item())
        if target_class is None:
            target_class = pred_class

        target_score = logits[0, target_class]
        confidence = float(probs[0, target_class].item())

        # Backward pass to obtain gradients at the target layer
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't capture (e.g. model custom hook)
            if hasattr(self.model, "get_target_layer_activations") and hasattr(self.model, "get_target_layer_gradients"):
                self.activations = self.model.get_target_layer_activations()
                self.gradients = self.model.get_target_layer_gradients()

        grads = self.gradients[0]  # [C, H', W']
        acts = self.activations[0]  # [C, H', W']

        # Global Average Pooling on gradients to compute channel weights alpha_c
        weights = torch.mean(grads, dim=(1, 2))  # [C]

        # Linear combination of activation channels
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        # Apply ReLU to focus on features having a positive contribution
        cam = F.relu(cam)

        # Convert to numpy and normalize
        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Upsample heatmap to match input spatial dimensions
        h_orig, w_orig = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(cam_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return heatmap_resized, pred_class, confidence

    def remove_hooks(self):
        """Removes PyTorch hooks."""
        self.fwd_hook.remove()
        self.bwd_hook.remove()


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlays a normalized [0, 1] Grad-CAM heatmap on an RGB image.
    
    Args:
        image_rgb: [H, W, 3] uint8 RGB image
        heatmap: [H, W] float32 heatmap with values in [0, 1]
        alpha: heatmap transparency factor (0 = image only, 1 = heatmap only)
        colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET or cv2.COLORMAP_TURBO)
        
    Returns:
        blended_rgb: [H, W, 3] uint8 blended RGB image
    """
    h, w = image_rgb.shape[:2]
    # Ensure heatmap matches image dimensions
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    return blended
