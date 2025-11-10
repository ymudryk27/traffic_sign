import os, json

LABELS_PATH = os.path.join("model", "labels.json")
TRAIN_DIR = os.path.join("data", "train")

# Офіційні назви класів GTSRB (за ClassId, 0..42)
classid_to_name = {
    "0": "Speed limit (20km/h)",
    "1": "Speed limit (30km/h)",
    "2": "Speed limit (50km/h)",
    "3": "Speed limit (60km/h)",
    "4": "Speed limit (70km/h)",
    "5": "Speed limit (80km/h)",
    "6": "End of speed limit (80km/h)",
    "7": "Speed limit (100km/h)",
    "8": "Speed limit (120km/h)",
    "9": "No passing",
    "10": "No passing for vehicles over 3.5 metric tons",
    "11": "Right-of-way at the next intersection",
    "12": "Priority road",
    "13": "Yield",
    "14": "Stop",
    "15": "No vehicles",
    "16": "Vehicles over 3.5 metric tons prohibited",
    "17": "No entry",
    "18": "General caution",
    "19": "Dangerous curve to the left",
    "20": "Dangerous curve to the right",
    "21": "Double curve",
    "22": "Bumpy road",
    "23": "Slippery road",
    "24": "Road narrows on the right",
    "25": "Road work",
    "26": "Traffic signals",
    "27": "Pedestrians",
    "28": "Children crossing",
    "29": "Bicycles crossing",
    "30": "Beware of ice/snow",
    "31": "Wild animals crossing",
    "32": "End of all speed and passing limits",
    "33": "Turn right ahead",
    "34": "Turn left ahead",
    "35": "Ahead only",
    "36": "Go straight or right",
    "37": "Go straight or left",
    "38": "Keep right",
    "39": "Keep left",
    "40": "Roundabout mandatory",
    "41": "End of no passing",
    "42": "End of no passing by vehicles over 3.5 metric tons"
}

def main():
    # створюємо модельну папку
    os.makedirs("model", exist_ok=True)

    # читаємо назви папок так, як це робить Keras
    class_dirs = [
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ]
    class_dirs_sorted = sorted(class_dirs)  # ВАЖЛИВО: строкове сортування

    # будуємо labels: index (нейрон) -> людська назва
    labels = {}
    for idx, cls_dir in enumerate(class_dirs_sorted):
        class_id = cls_dir  # назва папки = ClassId як стрінга
        human_name = classid_to_name.get(class_id, f"Class {class_id}")
        labels[str(idx)] = human_name

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved index-based labels to {LABELS_PATH}")
    print("Mapping example (first few):")
    for i in range(10):
        print(i, "->", labels.get(str(i)))

if __name__ == "__main__":
    main()