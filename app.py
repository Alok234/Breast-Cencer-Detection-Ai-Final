import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Allow all origins and explicitly permit preflight OPTIONS requests
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_FILENAME = "BreastCancer_HighAccuracy_HybridModel.keras"
HF_REPO_ID = "alo234/Breast_Cancer_MOdel"

# 1. Download Model securely with fallback
def load_keras_model():
    try:
        if not os.path.exists(MODEL_FILENAME):
            print("⏳ Downloading model from Hugging Face Hub...")
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir="."
            )
            print("✅ Model Downloaded Successfully!")
        else:
            model_path = MODEL_FILENAME

        model = tf.keras.models.load_model(model_path)
        print("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        print(f"⚠️ Model Loading Failed: {e}")
        return None

model = load_keras_model()

IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image_bytes):
    # Decode byte array to OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid or corrupted image file.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalization (0 to 1 scaling)
    img = img.astype(np.float32) / 255.0  
    
    # Expand dims for batch size -> shape: (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Breast Cancer Detection API is Live!"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle CORS preflight request explicitly
    if request.method == "OPTIONS":
        return "", 200

    if model is None:
        return jsonify({"error": "Model is not loaded on the server."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img = preprocess_image(file.read())
        
        # Make prediction
        raw_pred = model.predict(img)
        pred = float(raw_pred[0][0])
        
        # Calculate result and confidence score
        if pred > 0.5:
            result = CLASS_NAMES[1]
            confidence = pred * 100
        else:
            result = CLASS_NAMES[0]
            confidence = (1 - pred) * 100

        return jsonify({
            "status": "success",
            "result": result,
            "confidence": f"{confidence:.2f}%"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
