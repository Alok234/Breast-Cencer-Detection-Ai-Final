import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import onnxruntime as ort

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_FILENAME = "model.onnx"
HF_REPO_ID = "alo234/Breast_Cancer_MOdel"

def load_onnx_model():
    """Download and load lightweight ONNX model."""
    try:
        if not os.path.exists(MODEL_FILENAME):
            print("⏳ Downloading model.onnx from Hugging Face...")
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir="."
            )
        else:
            model_path = MODEL_FILENAME

        # Initialize ONNX Runtime Inference Session
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print("✅ ONNX Model Loaded Successfully!")
        return session, None
    except Exception as e:
        print(f"❌ ONNX Load Error: {e}")
        return None, str(e)

# Load ONNX session on startup
session, load_error = load_onnx_model()

if session:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image_bytes):
    """Decode and normalize image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image file uploaded.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0  # Normalize [0, 1]
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    if session:
        return jsonify({"status": "Breast Cancer Detection API is Live!", "model_status": "Model Active"})
    else:
        return jsonify({"status": "Breast Cancer Detection API is Live!", "model_status": f"Model Failed: {load_error}"})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200

    if session is None:
        return jsonify({"error": f"Model failed to load: {load_error}"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img = preprocess_image(file.read())
        
        # ONNX Model Inference
        outputs = session.run([output_name], {input_name: img})
        output_data = outputs[0]
        
        pred = float(output_data.flatten()[0])

        result = CLASS_NAMES[1] if pred > 0.5 else CLASS_NAMES[0]
        confidence = float(pred if pred > 0.5 else 1.0 - pred) * 100.0

        return jsonify({
            "status": "success",
            "result": result,
            "confidence": f"{confidence:.2f}%"
        }), 200

    except Exception as e:
        print(f"❌ Prediction Runtime Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
