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
    """Download and load ONNX model."""
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

session, load_error = load_onnx_model()

if session:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

IMG_SIZE = 224
# Class Mapping: Index 0 = Benign, Index 1 = Malignant
CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image_bytes):
    """Decode, resize and normalize image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image file uploaded.")
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image to 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Scale pixels to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension -> (1, 224, 224, 3)
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
        # 1. Preprocess Image
        img = preprocess_image(file.read())
        
        # 2. Run Model Inference
        outputs = session.run([output_name], {input_name: img})
        output_data = outputs[0]
        
        print(f"🔍 [DEBUG] Full Raw Model Output: {output_data}")

        # 3. Smart Handling for Softmax vs Sigmoid Outputs
        flat_output = output_data.flatten()
        
        if len(flat_output) >= 2: # Softmax / Multi-class Output [Benign_prob, Malignant_prob]
            predicted_class_idx = int(np.argmax(flat_output))
            result = CLASS_NAMES[predicted_class_idx]
            confidence = float(flat_output[predicted_class_idx]) * 100.0
            raw_score = float(flat_output[1]) # Malignant probability
        else: # Sigmoid Output (1 single score)
            raw_score = float(flat_output[0])
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
