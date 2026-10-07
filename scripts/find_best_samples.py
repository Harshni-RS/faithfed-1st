import os, sys, cv2, torch, pandas as pd, numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import get_model
from src.data_preprocessing import ben_graham_normalize, DR_CLASSES

df = pd.read_csv("data/aptos/train.csv")
img_dir = "data/aptos/train_images"

# Load the centralized model (the one with 49% train acc across all 3662 samples)
# Let's inspect artifacts/global_model.pth
state = torch.load("artifacts/global_model.pth", map_location="cpu")
model = get_model("RetinalNet", num_classes=5, pretrained=False)
model.load_state_dict(state, strict=False)
model.eval()

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Scanning for top 5 candidates per grade...")
top_candidates = {g: [] for g in range(5)}

for grade in range(5):
    sub = df[df["diagnosis"] == grade]
    print(f"\nScanning Grade {grade} ({DR_CLASSES[grade]}), total in CSV: {len(sub)}")
    count = 0
    for _, row in sub.iterrows():
        id_code = str(row["id_code"])
        path = os.path.join(img_dir, f"{id_code}.png")
        if not os.path.exists(path):
            continue
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        norm = ben_graham_normalize(rgb, img_size=128)
        tensor = val_transform(norm).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].numpy()
        pred = int(np.argmax(probs))
        
        # Calculate visual intensity (std dev of green channel, where fundus lesions show highest contrast)
        green = rgb[:, :, 1]
        lesion_contrast = float(np.std(green))
        
        top_candidates[grade].append({
            "id": id_code,
            "pred": pred,
            "conf": float(probs[pred]),
            "grade_prob": float(probs[grade]),
            "probs": [round(float(p)*100, 1) for p in probs],
            "contrast": lesion_contrast
        })
        count += 1
        if count >= 60:
            break

# Print summary of matches for each grade
print("\n" + "="*70)
print("TOP MATCHING SAMPLES PER GRADE (pred == grade):")
print("="*70)
for grade in range(5):
    matches = [c for c in top_candidates[grade] if c["pred"] == grade]
    matches.sort(key=lambda x: x["conf"], reverse=True)
    print(f"\nGrade {grade} ({DR_CLASSES[grade]}) - {len(matches)} matches found:")
    for m in matches[:5]:
        print(f"  ID: {m['id']} | Conf: {m['conf']*100:.1f}% | Contrast: {m['contrast']:.1f} | Probs: {m['probs']}")
