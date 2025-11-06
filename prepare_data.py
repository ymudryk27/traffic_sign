import os, shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = "data/Train"
OUTPUT_TRAIN = "data/train"
OUTPUT_VAL = "data/val"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_VAL, exist_ok=True)

for class_id in os.listdir(DATASET_DIR):
    src = os.path.join(DATASET_DIR, class_id)
    if not os.path.isdir(src):
        continue

    images = [os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(".png")]
    train_imgs, val_imgs = train_test_split(images, test_size=0.1, random_state=42)

    train_class_dir = os.path.join(OUTPUT_TRAIN, class_id)
    val_class_dir = os.path.join(OUTPUT_VAL, class_id)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in train_imgs:
        dst = os.path.join(train_class_dir, os.path.basename(img))
        if os.path.abspath(img) != os.path.abspath(dst):
            shutil.copy(img, dst)

    for img in val_imgs:
        dst = os.path.join(val_class_dir, os.path.basename(img))
        if os.path.abspath(img) != os.path.abspath(dst):
            shutil.copy(img, dst)

print("✅ Dataset prepared successfully!")
print(f"Train dir: {OUTPUT_TRAIN}")
print(f"Val dir: {OUTPUT_VAL}")