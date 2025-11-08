import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = "data/Train_original"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for class_id in sorted(os.listdir(SOURCE_DIR)):
    src = os.path.join(SOURCE_DIR, class_id)
    if not os.path.isdir(src):
        continue

    images = [os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(".png")]
    train_imgs, val_imgs = train_test_split(images, test_size=0.1, random_state=42)

    train_class_dir = os.path.join(TRAIN_DIR, class_id)
    val_class_dir = os.path.join(VAL_DIR, class_id)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in train_imgs:
        shutil.copy(img, os.path.join(train_class_dir, os.path.basename(img)))
    for img in val_imgs:
        shutil.copy(img, os.path.join(val_class_dir, os.path.basename(img)))

print("✅ Dataset split complete!")
print(f"Train directory: {TRAIN_DIR}")
print(f"Validation directory: {VAL_DIR}")