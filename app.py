from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pathlib import Path
from datetime import datetime
import sqlite3
import os
import random

try:
    import numpy as np
    from PIL import Image
    import tensorflow as tf
except ImportError:
    np = Image = tf = None

app = Flask(__name__)
app.secret_key = "plant-disease-demo-secret"
UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Demo fallback classes. Replace with your trained model/classes for real predictions.
CLASS_INFO = {
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "symptoms": "Brown circular spots, yellowing leaves and progressive leaf drying.",
        "cause": "Commonly associated with the fungus Alternaria solani.",
        "treatment": "Remove badly infected leaves, improve air circulation, avoid overhead watering and use a locally approved fungicide according to its label.",
        "prevention": "Water at soil level, keep plants spaced, remove plant debris and rotate crops."
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "symptoms": "Dark, water-soaked lesions that can rapidly enlarge on leaves and stems.",
        "cause": "Caused by the oomycete Phytophthora infestans.",
        "treatment": "Remove severely affected plant material and follow local agricultural guidance for an approved fungicide.",
        "prevention": "Keep foliage dry, improve ventilation and remove infected debris promptly."
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "symptoms": "Target-like dark spots and yellowing on older leaves.",
        "cause": "Often associated with Alternaria solani.",
        "treatment": "Remove infected foliage, maintain balanced nutrition and use an approved fungicide if recommended locally.",
        "prevention": "Crop rotation, good spacing, sanitation and avoiding prolonged leaf wetness."
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "symptoms": "Dark irregular leaf lesions, sometimes with pale margins; stems may also be affected.",
        "cause": "Phytophthora infestans.",
        "treatment": "Remove infected material and follow local agricultural extension recommendations for disease management.",
        "prevention": "Use healthy planting material, avoid overhead irrigation and maintain good field sanitation."
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "symptoms": "No obvious disease symptoms detected.",
        "cause": "No disease detected.",
        "treatment": "No disease treatment is required. Continue normal plant care.",
        "prevention": "Maintain adequate sunlight, watering, nutrition and regular inspection."
    }
}

def init_db():
    with sqlite3.connect("predictions.db") as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )""")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(path):
    # Uses trained model if model/model.keras or model/model.h5 exists.
    model_path = Path("model/model.keras")
    if not model_path.exists():
        model_path = Path("model/model.h5")

    if tf is not None and model_path.exists():
        model = tf.keras.models.load_model(model_path)
        class_names_path = Path("model/class_names.txt")
        class_names = class_names_path.read_text().splitlines()
        img = Image.open(path).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pred = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
        idx = int(np.argmax(pred))
        label = class_names[idx]
        confidence = float(pred[idx]) * 100
        return label, confidence

    # Demo mode keeps the app runnable before model training.
    label = random.choice(list(CLASS_INFO.keys()))
    confidence = round(random.uniform(88, 97), 2)
    return label, confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("leaf_image")
    if not file or file.filename == "":
        flash("Please select a leaf image.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Only JPG, JPEG and PNG images are allowed.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    saved_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    path = UPLOAD_FOLDER / saved_name
    file.save(path)

    label, confidence = predict_image(path)
    info = CLASS_INFO.get(label, {
        "name": label.replace("___", " - ").replace("_", " "),
        "symptoms": "See the crop-specific agricultural guidance for this classification.",
        "cause": "Classification generated by the trained image model.",
        "treatment": "Use local agricultural extension guidance before applying any pesticide or fungicide.",
        "prevention": "Inspect plants regularly and maintain good field sanitation."
    })

    with sqlite3.connect("predictions.db") as con:
        con.execute(
            "INSERT INTO predictions(filename,disease,confidence,created_at) VALUES (?,?,?,?)",
            (saved_name, info["name"], confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

    return render_template("result.html", image=url_for("static", filename=f"uploads/{saved_name}"),
                           disease=info["name"], confidence=confidence, info=info)

@app.route("/history")
def history():
    with sqlite3.connect("predictions.db") as con:
        rows = con.execute("SELECT filename,disease,confidence,created_at FROM predictions ORDER BY id DESC").fetchall()
    return render_template("history.html", rows=rows)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
