import os, json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, flash

MODEL_PATH = os.path.join("model", "traffic_model.keras")
LABELS_PATH = os.path.join("model", "labels.json")

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

app = Flask(__name__)
app.secret_key = "secret-key"

def preprocess_image(file_bytes, img_size=64):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return img

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file uploaded.", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    upload_dir = os.path.join("static", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    
    with open(filepath, "rb") as f:
        img = preprocess_image(f.read(), img_size=64)
    if img is None:
        flash("Invalid image format.", "error")
        return redirect(url_for("index"))

    
    preds = model.predict(img[np.newaxis, ...], verbose=0)
    cls = int(np.argmax(preds[0]))
    prob = float(np.max(preds[0]))
    label = labels.get(str(cls), f"Class {cls}")

    if prob >= 0.6:
        message = f"{label} (p={prob:.3f})"
    elif 0.3 <= prob < 0.6:
        message = f"(LOW CONFIDENCE) {label} (p={prob:.3f})"
    else:
        message = f"Image is probably NOT a known traffic sign (max p={prob:.3f})"

    
    return render_template("index.html", result=message, image_url=url_for('static', filename=f"uploads/{file.filename}"))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))