"""
Clinical inference pipeline for Diabetic Retinopathy diagnosis and explainability.
Takes raw retinal fundus images, applies Ben Graham preprocessing, performs model inference,
generates Grad-CAM visual heatmaps, and assesses explanation faithfulness.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, Optional, Tuple, Union

from src.model import SmallCNN, get_model
from src.data_preprocessing import ben_graham_normalize, DR_CLASSES, DR_DESCRIPTIONS
from src.gradcam import GradCAM, overlay_heatmap
from src.faithfulness import compute_insertion_deletion_auc
from torchvision import transforms


CLINICAL_RISK = {
    0: {"level": "Low", "color": "#28a745", "action": "Routine annual screening recommended."},
    1: {"level": "Mild", "color": "#17a2b8", "action": "Follow-up screening in 6-12 months. Monitor blood glucose and HbA1c."},
    2: {"level": "Moderate", "color": "#ffc107", "action": "Refer to ophthalmologist within 3-6 months for detailed retinal evaluation."},
    3: {"level": "Severe", "color": "#fd7e14", "action": "Urgent ophthalmology referral within 2-4 weeks. Panretinal photocoagulation evaluation."},
    4: {"level": "Critical", "color": "#dc3545", "action": "Immediate emergency retinal specialist referral. High risk of vitrectomy or vision loss."}
}


class RetinalDiagnosticEngine:
    """
    End-to-end inference and explainability engine for ophthalmologists and clinical staff.
    """
    def __init__(
        self,
        model_path: Optional[str] = "artifacts/global_model.pth",
        img_size: int = 128,
        ben_graham_sigma: int = 10,
        device: Optional[torch.device] = None
    ):
        self.img_size = img_size
        self.ben_graham_sigma = ben_graham_sigma
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        
        model_state_to_load = None
        arch_to_use = "SmallCNN"
        if model_path and os.path.exists(model_path):
            loaded = torch.load(model_path, map_location=self.device)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                loaded = loaded["model_state_dict"]
            model_state_to_load = loaded
            # Detect architecture from checkpoint keys & tensor dimensions
            if "fc1.weight" in model_state_to_load and model_state_to_load["fc1.weight"].shape[1] == 512:
                arch_to_use = "RetinalNet"
            elif "layer4.0.conv1.weight" in model_state_to_load:
                arch_to_use = "RetinalNet"
            else:
                arch_to_use = "SmallCNN"

        self.model = get_model(arch_to_use, num_classes=5).to(self.device)
        if model_state_to_load is not None:
            curr_state = self.model.state_dict()
            safe_state = {
                k: v for k, v in model_state_to_load.items()
                if k in curr_state and v.shape == curr_state[k].shape
            }
            self.model.load_state_dict(safe_state, strict=False)

        self.model.eval()
        target_layer = getattr(self.model, "layer4", getattr(self.model, "block5", None))
        if hasattr(target_layer, "conv"):
            target_layer = target_layer.conv
        self.gradcam = GradCAM(self.model, target_layer=target_layer)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Loads and standardizes input to RGB numpy array."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise ValueError(f"Could not read image file {image_input}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            return image_input
        else:
            raise TypeError("Unsupported image input type.")

    def diagnose(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        target_class: Optional[int] = None,
        compute_faithfulness: bool = True
    ) -> Dict[str, Any]:
        """
        Runs comprehensive diagnosis on a single fundus photograph:
        1. Ben Graham illumination correction
        2. Diabetic retinopathy classification with class probabilities
        3. Grad-CAM visual lesion explanation
        4. Faithfulness check (Insertion vs Deletion AUC)
        """
        raw_rgb = self.load_image(image_input)
        
        # 1. Ben Graham normalization
        normalized_rgb = ben_graham_normalize(
            raw_rgb, img_size=self.img_size, sigma=self.ben_graham_sigma
        )

        # 2. PyTorch tensor preparation
        tensor_img = self.transform(normalized_rgb).unsqueeze(0).to(self.device)

        # 3. Model prediction
        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])

        selected_target = target_class if target_class is not None else pred_class

        # 4. Grad-CAM explanation heatmap
        heatmap, _, target_conf = self.gradcam.generate_heatmap(tensor_img, target_class=selected_target)
        
        # Overlay heatmap on both raw image and normalized image
        raw_resized = cv2.resize(raw_rgb, (self.img_size, self.img_size))
        overlay_raw = overlay_heatmap(raw_resized, heatmap, alpha=0.45)
        overlay_norm = overlay_heatmap(normalized_rgb, heatmap, alpha=0.45)

        # 5. Faithfulness evaluation
        faithfulness_results = {}
        if compute_faithfulness:
            faithfulness_results = compute_insertion_deletion_auc(
                self.model, tensor_img, heatmap, target_class=selected_target, num_steps=10, device=self.device
            )

        diagnosis_report = {
            "predicted_grade": pred_class,
            "grade_name": DR_CLASSES[pred_class],
            "description": DR_DESCRIPTIONS[pred_class],
            "confidence": round(confidence, 4),
            "class_probabilities": {DR_CLASSES[i]: round(float(probs[i]), 4) for i in range(5)},
            "clinical_risk": CLINICAL_RISK[pred_class],
            "raw_image_rgb": raw_resized,
            "normalized_image_rgb": normalized_rgb,
            "heatmap": heatmap,
            "overlay_raw": overlay_raw,
            "overlay_normalized": overlay_norm,
            "faithfulness": faithfulness_results
        }

        return diagnosis_report
