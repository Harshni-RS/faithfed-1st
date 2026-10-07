import cv2, os, pandas as pd
import numpy as np

grade3_id = '0104b032c141'
grade4_id = '001639a390f0'
img_dir = 'data/aptos/train_images'

for gid, g_name in [(grade3_id, 'Grade 3 Severe'), (grade4_id, 'Grade 4 Prolif')]:
    path = os.path.join(img_dir, f'{gid}.png')
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_bright = np.mean(gray)
        std_bright = np.std(gray)
        print(f'{g_name} ({gid}): {w}x{h}, mean={mean_bright:.1f}, std={std_bright:.1f}')

df = pd.read_csv('data/aptos/train.csv')
g4_df = df[df['diagnosis'] == 4]

print()
print('Top Grade 4 images by visual contrast (high std = more dramatic lesions):')
results = []
for _, row in g4_df.iterrows():
    id_code = str(row['id_code'])
    path = os.path.join(img_dir, f'{id_code}.png')
    if not os.path.exists(path):
        continue
    img = cv2.imread(path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results.append((id_code, float(np.mean(gray)), float(np.std(gray))))

results.sort(key=lambda x: x[2], reverse=True)
for id_c, mean, std in results[:10]:
    print(f'  {id_c}: mean={mean:.1f}, std={std:.1f}')

print()
print('Top Grade 3 images for comparison:')
g3_df = df[df['diagnosis'] == 3]
results3 = []
for _, row in g3_df.iterrows():
    id_code = str(row['id_code'])
    path = os.path.join(img_dir, f'{id_code}.png')
    if not os.path.exists(path):
        continue
    img = cv2.imread(path)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results3.append((id_code, float(np.mean(gray)), float(np.std(gray))))

results3.sort(key=lambda x: x[2], reverse=True)
for id_c, mean, std in results3[:5]:
    print(f'  {id_c}: mean={mean:.1f}, std={std:.1f}')
