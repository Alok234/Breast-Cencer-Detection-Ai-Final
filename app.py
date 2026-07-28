import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # GitHub Pages থেকে রিকোয়েস্ট আসার অনুমতি দেওয়ার জন্য

MODEL_PATH = "BreastCancer_HighAccuracy_HybridModel.keras"

# মডেল লোড করা
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Model Loading Failed: {e}")

IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Breast Cancer Detection API is Live!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    try:
        img = preprocess_image(file.read())
        pred = model.predict(img)[0][0]
        
        result = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
        confidence = float(pred if pred > 0.5 else 1 - pred) * 100

        return jsonify({
            "result": result,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
