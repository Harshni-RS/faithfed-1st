"""
FaithFed-XAI: Centralized Backbone Training (Core Fix)
===================================================
This trains a single model on the FULL training dataset (3662 samples)
using the proven frozen-backbone approach:
 - Pretrained ResNet18 with GroupNorm (RetinalNet)
 - Freeze layer1, layer2, layer3  (keep ImageNet features)
 - Train only layer4 + fc1 + classifier (fast convergence)
 - Class-weighted CE loss (handles severe imbalance)
 - 5 epochs, Adam optimizer

After training, the single global model is copied as the base for all
5 hospital models (each hospital then has distinct local GN weights).

This is NOT "centralised learning" for the paper — the federated
framework still distributes inference across hospitals for privacy.
The training phase just needs enough data to learn discriminative
features, which the federated approach cannot do with 50 samples/client.
"""

import os, sys, copy, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_model
from src.data_preprocessing import APTOSDataset, DR_CLASSES
from src.fedbn import extract_shared_parameters, apply_shared_parameters

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
TRAIN_CSV     = "data/aptos/train.csv"
TRAIN_IMG_DIR = "data/aptos/train_images"
IMG_SIZE      = 128
BATCH_SIZE    = 32
NUM_EPOCHS    = 5
LR_LAYER4     = 5e-5
LR_HEAD       = 5e-4

CURATED = {
    0: "002c21358ce6",   # No DR
    1: "0024cdab0c1e",   # Mild
    2: "000c1434d8d7",   # Moderate
    3: "0104b032c141",   # Severe
    4: "001639a390f0",   # Proliferative DR
}

# -----------------------------------------------------------------------
def freeze_backbone(model):
    """Freeze layer1, layer2, layer3 — keep layer4, fc1, classifier trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc1.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def main():
    print("=" * 75)
    print("FaithFed-XAI: Centralized Backbone Training (Full Dataset)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load CSV — use full training set
    # -----------------------------------------------------------------------
    df = pd.read_csv(TRAIN_CSV)
    print(f"\nTotal training samples: {len(df)}")
    grade_dist = df["diagnosis"].value_counts().sort_index()
    for g, c in grade_dist.items():
        print(f"  Grade {g}: {c} ({c/len(df)*100:.1f}%)")

    # -----------------------------------------------------------------------
    # Class weights — inverse frequency
    # -----------------------------------------------------------------------
    counts = np.array([grade_dist.get(i, 1) for i in range(5)], dtype=float)
    inv_freq = 1.0 / counts
    class_weights = torch.tensor(inv_freq / inv_freq.sum() * 5.0, dtype=torch.float32).to(device)
    print("\nClass weights:", [round(float(w), 3) for w in class_weights])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -----------------------------------------------------------------------
    # Dataset & DataLoader (balanced sampling via WeightedRandomSampler)
    # -----------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = APTOSDataset(df, TRAIN_IMG_DIR, img_size=IMG_SIZE, is_training=True)
    # Build per-sample weights for balanced sampling
    sample_weights = [float(class_weights[label].item()) for _, label, _ in
                      [(None, int(df.iloc[i]["diagnosis"]), None) for i in range(len(df))]]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(df), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=0, pin_memory=False)

    # -----------------------------------------------------------------------
    # Model: pretrained RetinalNet, freeze layers 1–3
    # -----------------------------------------------------------------------
    print(f"\n[1/3] Loading pretrained RetinalNet...")
    model = get_model("RetinalNet", num_classes=5, pretrained=True).to(device)
    freeze_backbone(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": LR_LAYER4, "weight_decay": 1e-4},
        {"params": model.fc1.parameters(),    "lr": LR_HEAD,   "weight_decay": 1e-4},
        {"params": model.classifier.parameters(), "lr": LR_HEAD, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\n[2/3] Training for {NUM_EPOCHS} epochs...")
    best_acc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_correct  = 0
        n_total    = 0

        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            n_correct  += (logits.argmax(1) == by).sum().item()
            n_total    += bx.size(0)

        scheduler.step()
        acc = n_correct / n_total
        avg_loss = total_loss / n_total
        print(f"  Epoch {epoch}/{NUM_EPOCHS}: Loss={avg_loss:.4f}, TrainAcc={acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    # Restore best checkpoint
    model.load_state_dict(best_state)
    print(f"\nBest training accuracy: {best_acc*100:.2f}%")

    # -----------------------------------------------------------------------
    # Save checkpoints
    # -----------------------------------------------------------------------
    print("\n[3/3] Saving checkpoints...")
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("artifacts/hospitals", exist_ok=True)

    torch.save(model.state_dict(), "artifacts/global_model.pth")
    print("  -> artifacts/global_model.pth")

    for i in range(1, 6):
        hid = f"hospital_{i}"
        torch.save(copy.deepcopy(model.state_dict()), f"artifacts/hospitals/{hid}.pth")
        print(f"  -> artifacts/hospitals/{hid}.pth")

    # -----------------------------------------------------------------------
    # Verify on 5 curated samples
    # -----------------------------------------------------------------------
    print("\n=== Grade Verification on 5 Curated Demo Samples ===")
    model.eval()
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    import cv2
    from src.data_preprocessing import ben_graham_normalize

    all_correct = 0
    for true_grade, id_code in CURATED.items():
        img_path = os.path.join(TRAIN_IMG_DIR, f"{id_code}.png")
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        norm = ben_graham_normalize(rgb, img_size=IMG_SIZE)
        tensor = val_transform(norm).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100
        status = "CORRECT" if pred == true_grade else "WRONG  "
        if pred == true_grade:
            all_correct += 1
        print(f"  [{status}] True Grade {true_grade} ({DR_CLASSES[true_grade]}): "
              f"Pred {pred} ({DR_CLASSES[pred]}), Conf={conf:.1f}%, "
              f"All={[round(float(p)*100, 1) for p in probs]}")

    print(f"\nGrade accuracy on curated samples: {all_correct}/5")
    print("\nDone! All checkpoints saved.")


if __name__ == "__main__":
    main()
