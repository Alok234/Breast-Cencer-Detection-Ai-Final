import os
import gc
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_FILENAME = "BreastCancer_HighAccuracy_HybridModel.keras"
HF_REPO_ID = "alo234/Breast_Cancer_MOdel"

# Global model variable initialized to None
model = None

def get_model():
    global model
    if model is None:
        print("⏳ Downloading/Loading model on demand...")
        if not os.path.exists(MODEL_FILENAME):
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir="."
            )
        else:
            model_path = MODEL_FILENAME
        
        # Load model with lightweight configuration
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully into memory!")
    return model

IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image file uploaded.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0  
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Breast Cancer Detection API is Live!"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load model only when endpoint is triggered
        loaded_model = get_model()
        
        img = preprocess_image(file.read())
        
        # Inference using model call instead of predict to save RAM
        raw_pred = loaded_model(img, training=False)
        pred = float(raw_pred.numpy()[0][0])
        
        # Cleanup
        gc.collect()

        result = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
        confidence = float(pred if pred > 0.5 else 1.0 - pred) * 100.0

        return jsonify({
            "status": "success",
            "result": result,
            "confidence": f"{confidence:.2f}%"
        }), 200

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
