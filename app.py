import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import onnxruntime as ort

app = Flask(__name__)
# Enable CORS for all domains
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_FILENAME = "model.onnx"
HF_REPO_ID = "alo234/Breast_Cancer_MOdel"

def load_onnx_model():
    """Download and load lightweight ONNX model with RAM optimization."""
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

        # Limit CPU threads to prevent RAM memory spikes on Render free tier
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
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
# Standard Binary Indexing: Index 0 = Benign, Index 1 = Malignant
CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image_bytes):
    """
    Decode, resize and normalize for Keras/TensorFlow NHWC layout (1, 224, 224, 3)
    with Standard ImageNet Normalization.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image file uploaded.")
        
    # 1. Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Resize to 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # 3. Standard Scale [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # 4. ImageNet Mean and Standard Deviation Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # 5. Add Batch Dimension -> Shape: (1, 224, 224, 3) EXACT MATCH FOR MODEL
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    if session:
        return jsonify({
            "status": "Breast Cancer Detection API is Live!", 
            "model_status": "Model Active"
        })
    else:
        return jsonify({
            "status": "Breast Cancer Detection API is Live!", 
            "model_status": f"Model Failed: {load_error}"
        })

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
        # 1. Preprocess uploaded image
        img = preprocess_image(file.read())
        
        # 2. Run ONNX Model Inference
        outputs = session.run([output_name], {input_name: img})
        output_data = outputs[0]
        
        raw_score = float(output_data.flatten()[0])
        print(f"🔍 [DEBUG] Model Output Score: {raw_score}")

        # 3. Classification Logic
        if raw_score > 0.5:
            result = CLASS_NAMES[1]  # Malignant
            confidence = raw_score * 100.0
        else:
            result = CLASS_NAMES[0]  # Benign
            confidence = (1.0 - raw_score) * 100.0

        return jsonify({
            "status": "success",
            "result": result,
            "confidence": f"{confidence:.2f}%",
            "raw_score": raw_score
        }), 200

    except Exception as e:
        print(f"❌ Prediction Runtime Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
