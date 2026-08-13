import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import onnxruntime as ort

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILENAME = "model.onnx"
HF_REPO_ID = "alo234/Breast_Cancer_MOdel"

IMG_SIZE = 224

# IMPORTANT:
# Change this only if your model was trained with the opposite
# class order.
#
# For a single sigmoid output:
# 0 = Benign
# 1 = Malignant
#
CLASS_0 = "Benign"
CLASS_1 = "Malignant"


# ============================================================
# LOAD ONNX MODEL
# ============================================================

def load_onnx_model():

    try:

        if not os.path.exists(MODEL_FILENAME):

            print("Downloading model.onnx from Hugging Face...")

            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME,
                local_dir="."
            )

        else:

            model_path = MODEL_FILENAME

        # RAM optimized settings for Render
        opts = ort.SessionOptions()

        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )

        print("========================================")
        print("ONNX MODEL LOADED SUCCESSFULLY")
        print("========================================")

        # Model input information
        input_info = session.get_inputs()[0]

        print("Input Name   :", input_info.name)
        print("Input Shape  :", input_info.shape)
        print("Input Type   :", input_info.type)

        # Model output information
        for i, output in enumerate(session.get_outputs()):

            print(f"Output {i} Name  :", output.name)
            print(f"Output {i} Shape :", output.shape)
            print(f"Output {i} Type  :", output.type)

        print("========================================")

        return session, None

    except Exception as e:

        print("ONNX MODEL LOAD ERROR:", str(e))

        return None, str(e)


# Load model at startup
session, load_error = load_onnx_model()


if session:

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes):

    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:

        raise ValueError(
            "Invalid image file uploaded."
        )

    # BGR -> RGB
    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Convert to float32
    img = img.astype(np.float32)

    # Normalize [0,255] -> [0,1]
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(
        img,
        axis=0
    )

    print(
        "Preprocessed Image Shape:",
        img.shape
    )

    print(
        "Image Min:",
        float(img.min()),
        "Image Max:",
        float(img.max())
    )

    return img


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def interpret_prediction(output_data):

    """
    Handles different ONNX output formats.

    Supported examples:

    1. Single sigmoid:
       [[0.73]]

    2. Single value:
       [0.73]

    3. Two-class probability:
       [[0.20, 0.80]]

    """

    output = np.asarray(output_data)

    print("========================================")
    print("RAW MODEL OUTPUT:", output)
    print("OUTPUT SHAPE:", output.shape)
    print("========================================")

    # --------------------------------------------------------
    # CASE 1: TWO CLASS OUTPUT
    # Example:
    # [[0.20, 0.80]]
    # --------------------------------------------------------

    if output.size == 2:

        probabilities = output.flatten().astype(float)

        # If values are logits rather than probabilities,
        # convert them using softmax.
        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(
                np.sum(probabilities),
                1.0,
                atol=0.01
            )
        ):

            exp_values = np.exp(
                probabilities - np.max(probabilities)
            )

            probabilities = (
                exp_values /
                np.sum(exp_values)
            )

        class_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_index]
        )

        if class_index == 0:

            result = CLASS_0

        else:

            result = CLASS_1

        return result, confidence, probabilities.tolist()


    # --------------------------------------------------------
    # CASE 2: SINGLE OUTPUT
    # Example:
    # [[0.73]]
    # --------------------------------------------------------

    raw_score = float(
        output.flatten()[0]
    )

    # If output looks like a logit instead of probability,
    # apply sigmoid.
    if raw_score < 0 or raw_score > 1:

        raw_score = 1.0 / (
            1.0 + np.exp(-raw_score)
        )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # For a standard binary sigmoid model:
    #
    # score < 0.5 -> class 0
    # score >= 0.5 -> class 1
    #
    # Here:
    # class 0 = Benign
    # class 1 = Malignant
    # --------------------------------------------------------

    if raw_score >= 0.5:

        result = CLASS_1
        confidence = raw_score

    else:

        result = CLASS_0
        confidence = 1.0 - raw_score

    return result, confidence, raw_score


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    if session is not None:

        return jsonify({

            "status":
                "Breast Cancer Detection API is Live!",

            "model_status":
                "Model Active",

            "model":
                "ONNX",

            "input_size":
                "224x224",

            "endpoint":
                "/predict"

        })

    else:

        return jsonify({

            "status":
                "Breast Cancer Detection API is Live!",

            "model_status":
                "Model Failed",

            "error":
                load_error

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    if session is not None:

        return jsonify({

            "status": "healthy",
            "model": "loaded"

        })

    return jsonify({

        "status": "unhealthy",
        "model": "not loaded",
        "error": load_error

    }), 500


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST", "OPTIONS"]
)
def predict():

    # CORS preflight
    if request.method == "OPTIONS":

        return "", 200

    # Check model
    if session is None:

        return jsonify({

            "status": "error",

            "error":
                f"Model failed to load: {load_error}"

        }), 500

    # Check file
    if "file" not in request.files:

        return jsonify({

            "status": "error",

            "error":
                "No file uploaded."

        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({

            "status": "error",

            "error":
                "No file selected."

        }), 400

    try:

        # ----------------------------------------------------
        # 1. READ IMAGE
        # ----------------------------------------------------

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({

                "status": "error",

                "error":
                    "Uploaded file is empty."

            }), 400


        # ----------------------------------------------------
        # 2. PREPROCESS
        # ----------------------------------------------------

        img = preprocess_image(
            image_bytes
        )


        # ----------------------------------------------------
        # 3. MODEL INFERENCE
        # ----------------------------------------------------

        outputs = session.run(
            output_names,
            {
                input_name: img
            }
        )


        output_data = outputs[0]


        # ----------------------------------------------------
        # 4. INTERPRET MODEL OUTPUT
        # ----------------------------------------------------

        result, confidence, raw_output = (
            interpret_prediction(
                output_data
            )
        )


        # ----------------------------------------------------
        # 5. FORMAT CONFIDENCE
        # ----------------------------------------------------

        confidence_percent = (
            float(confidence) * 100.0
        )


        # ----------------------------------------------------
        # 6. RESPONSE
        # ----------------------------------------------------

        response = {

            "status":
                "success",

            "result":
                result,

            "confidence":
                f"{confidence_percent:.2f}%",

            "raw_score":
                raw_output

        }

        print("========================================")
        print("FINAL RESULT:", result)
        print(
            "CONFIDENCE:",
            f"{confidence_percent:.2f}%"
        )
        print("========================================")

        return jsonify(response), 200


    except Exception as e:

        print(
            "Prediction Runtime Error:",
            str(e)
        )

        return jsonify({

            "status": "error",

            "error":
                str(e)

        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
