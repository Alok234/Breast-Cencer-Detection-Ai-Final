import os
import gc
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Allow cross-origin requests from any domain
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_FILENAME = "BreastCancer_HighAccuracy_HybridModel.keras"
HF_REPO_ID = "alo234/Breast_Cancer_MOdel"

def load_keras_model():
    """Download and load the Keras model with error handling."""
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

        loaded_model = tf.keras.models.load_model(model_path)
        print("✅ Model Loaded Successfully!")
        return loaded_model
    except Exception as e:
        print(f"⚠️ Model Loading Failed: {e}")
        return None

# Load model globally on startup
model = load_keras_model()

IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image_bytes):
    """Decode byte stream and preprocess image for model input."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid or corrupted image file uploaded.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values to [0, 1] range
    img = img.astype(np.float32) / 255.0  
    
    # Add batch dimension: shape (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Breast Cancer Detection API is Live!"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle CORS preflight requests
    if request.method == "OPTIONS":
        return "", 200

    # Ensure model is available
    if model is None:
        return jsonify({"error": "Model failed to load on server launch."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Preprocess input image
        img = preprocess_image(file.read())
        
        # Perform inference with memory-efficient call
        raw_pred = model(img, training=False)
        pred = float(raw_pred.numpy()[0][0])
        
        # Free memory post-inference to stay under RAM limits
        tf.keras.backend.clear_session()
        gc.collect()

        # Generate output labels
        result = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
        confidence = float(pred if pred > 0.5 else 1.0 - pred) * 100.0

        return jsonify({
            "status": "success",
            "result": result,
            "confidence": f"{confidence:.2f}%"
        }), 200

    except Exception as e:
        print(f"❌ Prediction Error Log: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
