"""
FaithFed-XAI: Balanced Clean Training (No Double-Penalty)
- 120 samples per class = 600 samples per epoch (balanced)
- UNWEIGHTED CrossEntropyLoss (since data is already balanced!)
- Frozen layers 1-3, train layer4 + head
- Evaluates on all 5 curated samples
- Saves to artifacts/global_model.pth and artifacts/hospitals/hospital_1..5.pth
"""

import os, sys, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import get_model
from src.data_preprocessing import APTOSDataset, DR_CLASSES, ben_graham_normalize

CURATED = {
    0: "002c21358ce6",   # No DR
    1: "0024cdab0c1e",   # Mild
    2: "000c1434d8d7",   # Moderate
    3: "0104b032c141",   # Severe
    4: "001639a390f0",   # Proliferative DR
}

def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc1.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

def main():
    print("=" * 65)
    print("FaithFed-XAI: Balanced Clean Fast Training")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    df = pd.read_csv("data/aptos/train.csv")
    
    # Exclude curated samples from training so they serve as a clean test
    curated_ids = list(CURATED.values())
    train_pool = df[~df["id_code"].isin(curated_ids)]
    
    # Balance 150 samples per class (193 is min class count for grade 3)
    balanced_parts = []
    for g in range(5):
        sub = train_pool[train_pool["diagnosis"] == g]
        sample_count = min(150, len(sub))
        balanced_parts.append(sub.sample(n=sample_count, random_state=42))
    
    balanced_df = pd.concat(balanced_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Balanced training set: {len(balanced_df)} samples ({len(balanced_df)//5} per class)")
    
    train_ds = APTOSDataset(balanced_df, "data/aptos/train_images", img_size=128, is_training=True)
    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=0)
    
    # Model
    model = get_model("RetinalNet", num_classes=5, pretrained=True).to(device)
    freeze_backbone(model)
    
    # Standard UNWEIGHTED loss since classes are already balanced
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam([
        {"params": model.layer4.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": model.fc1.parameters(),    "lr": 5e-4, "weight_decay": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
    ])
    
    NUM_EPOCHS = 6
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    best_acc = 0.0
    best_state = None
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for bx, by, _ in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * bx.size(0)
            n_correct += (logits.argmax(1) == by).sum().item()
            n_total += bx.size(0)
            
        acc = n_correct / n_total
        print(f"  Epoch {epoch}/{NUM_EPOCHS}: Loss={total_loss/n_total:.4f}, Acc={acc*100:.1f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_state)
    print(f"\nBest Training Acc: {best_acc*100:.1f}%")
    
    # Save checkpoints
    os.makedirs("artifacts/hospitals", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/global_model.pth")
    for i in range(1, 6):
        torch.save(copy.deepcopy(model.state_dict()), f"artifacts/hospitals/hospital_{i}.pth")
    print("Checkpoints saved to artifacts/global_model.pth and artifacts/hospitals/hospital_1..5.pth")
    
    # Test on the 5 curated demo cases
    print("\n=== Curated Demo Cases Evaluation ===")
    model.eval()
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    import cv2
    correct_count = 0
    for true_g, id_code in CURATED.items():
        img_path = f"data/aptos/train_images/{id_code}.png"
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        norm = ben_graham_normalize(rgb, img_size=128)
        tensor = val_transform(norm).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100
        is_match = "CORRECT" if pred == true_g else "WRONG  "
        if pred == true_g: correct_count += 1
        print(f"  [{is_match}] True Grade {true_g} ({DR_CLASSES[true_g]}): "
              f"Pred {pred} ({DR_CLASSES[pred]}), Conf={conf:.1f}%, "
              f"All={[round(float(p)*100, 1) for p in probs]}")
              
    print(f"\nCurated Cases Accuracy: {correct_count}/5")

if __name__ == "__main__":
    main()
