import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
# Enable CORS for all incoming origins and routes
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_FILENAME = "model.tflite"
HF_REPO_ID = "alo234/Breast_Cancer_MOdel"

def load_tflite_model():
    """Download and initialize lightweight TFLite model from Hugging Face."""
    try:
        if not os.path.exists(MODEL_FILENAME):
            print("⏳ Downloading model.tflite from Hugging Face...")
            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir="."
            )
        else:
            model_path = MODEL_FILENAME

        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("✅ TFLite Interpreter Loaded Successfully!")
        return interpreter
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        return None

# Load model interpreter on startup
interpreter = load_tflite_model()

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

IMG_SIZE = 224
CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image_bytes):
    """Decode byte array and preprocess image for model inference."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image file uploaded.")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET"])
def home():
    status_msg = "Model Active" if interpreter else "Model Failed to Load"
    return jsonify({"status": "Breast Cancer Detection API is Live!", "model_status": status_msg})

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200

    if interpreter is None:
        return jsonify({"error": "Model failed to initialize on server launch."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in request."}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        img = preprocess_image(file.read())
        
        # Execute TFLite model inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
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
