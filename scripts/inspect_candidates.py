import os, cv2, pandas as pd, numpy as np

df = pd.read_csv("data/aptos/train.csv")
img_dir = "data/aptos/train_images"

# Let's inspect candidate images for each grade
candidates = {
    0: ["002c21358ce6", "0232dfea7547", "060e00d1e2ab", "07e827469099"],
    1: ["0024cdab0c1e", "0124dffecf29", "05a5183c92d0", "086d41d17da8", "1e8a1fdee5b9"],
    2: ["000c1434d8d7", "0161338f53cc", "0773a1c326ad", "08037e44e9ed", "0924cf2de6b0"],
    3: ["0104b032c141", "042470a92154", "05cd0178ccfe", "070f67572d03", "0c917c372572"],
    4: ["001639a390f0", "02685f13cefd", "02dda30d3acf", "080ee76c958c", "2bbcfdc477db", "f025f33b2c9b"]
}

for g in range(5):
    print(f"\n--- Grade {g} Candidates ---")
    for cid in candidates[g]:
        p = os.path.join(img_dir, f"{cid}.png")
        if os.path.exists(p):
            row = df[df["id_code"] == cid]
            csv_diag = int(row.iloc[0]["diagnosis"]) if len(row) > 0 else -1
            img = cv2.imread(p)
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"  {cid}: CSV_label={csv_diag}, shape={w}x{h}, mean_bright={np.mean(gray):.1f}, std={np.std(gray):.1f}")
        else:
            print(f"  {cid}: NOT FOUND")
