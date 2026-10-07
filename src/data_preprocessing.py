"""
Data preprocessing and augmentation pipeline for Diabetic Retinopathy.
Implements Ben Graham's local average color-subtraction normalization,
fundus circular cropping, and PyTorch Dataset / DataLoader abstractions.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Tuple, Dict, Any, List

# Standard Diabetic Retinopathy grade definitions
DR_CLASSES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

DR_DESCRIPTIONS = {
    0: "No apparent retinal lesions or abnormalities.",
    1: "Microaneurysms only. Early non-proliferative diabetic retinopathy.",
    2: "More than microaneurysms, but less than severe. Dot/blot hemorrhages or hard exudates.",
    3: "Severe hemorrhages in all 4 quadrants, venous beading, or prominent IRMA.",
    4: "Neovascularization, vitreous/preretinal hemorrhage. High risk of severe visual impairment."
}


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Remove uninformative black borders from fundus images.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark or empty, return original
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def ben_graham_normalize(image_rgb: np.ndarray, img_size: int = 128, sigma: int = 10) -> np.ndarray:
    """
    Ben Graham's local-average color subtraction normalization.
    Corrects uneven illumination typical of fundus photography,
    accentuates microaneurysms, hemorrhages, and exudates.
    """
    # 1. Crop dark borders
    cropped = crop_image_from_gray(image_rgb)
    
    # 2. Resize to square
    resized = cv2.resize(cropped, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # 3. Local average subtraction: 4 * img - 4 * gaussian_blur(img) + 128
    blurred = cv2.GaussianBlur(resized, (0, 0), sigma)
    normalized = cv2.addWeighted(resized, 4, blurred, -4, 128)
    
    # 4. Circular mask to soften edge artifacts
    h, w = normalized.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (int(w / 2), int(h / 2)), int(min(h, w) * 0.48), 255, -1)
    circle_mask_3ch = cv2.cvtColor(circle_mask, cv2.COLOR_GRAY2RGB)
    
    # Blend edges
    bg = np.full_like(normalized, 128)
    out = np.where(circle_mask_3ch == 255, normalized, bg)
    
    return np.clip(out, 0, 255).astype(np.uint8)


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for APTOS Retinal Blindness Detection.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        img_size: int = 128,
        use_ben_graham: bool = True,
        ben_graham_sigma: int = 10,
        is_training: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_size = img_size
        self.use_ben_graham = use_ben_graham
        self.ben_graham_sigma = ben_graham_sigma
        self.is_training = is_training

        # Data augmentation pipeline for training
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_id = str(row["id_code"])
        label = int(row["diagnosis"])
        
        # Check image path
        img_filename = f"{img_id}.png" if not img_id.endswith(".png") else img_id
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Fallback if unreadable: generate placeholder
            img_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        if self.use_ben_graham:
            processed_rgb = ben_graham_normalize(
                img_rgb, img_size=self.img_size, sigma=self.ben_graham_sigma
            )
        else:
            cropped = crop_image_from_gray(img_rgb)
            processed_rgb = cv2.resize(cropped, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        tensor_img = self.transform(processed_rgb)
        return tensor_img, label, img_id


def get_dataloader(
    df: pd.DataFrame,
    img_dir: str,
    batch_size: int = 16,
    img_size: int = 128,
    use_ben_graham: bool = True,
    ben_graham_sigma: int = 10,
    is_training: bool = False,
    shuffle: Optional[bool] = None,
    num_workers: int = 0
) -> DataLoader:
    """Builds a PyTorch DataLoader."""
    if shuffle is None:
        shuffle = is_training
        
    dataset = APTOSDataset(
        df=df,
        img_dir=img_dir,
        img_size=img_size,
        use_ben_graham=use_ben_graham,
        ben_graham_sigma=ben_graham_sigma,
        is_training=is_training
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )
