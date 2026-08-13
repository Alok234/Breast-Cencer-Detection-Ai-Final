import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
import onnxruntime as ort

app = Flask(__name__)

# ============================================================
# CORS
# ============================================================

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILENAME = "model.onnx"
HF_REPO_ID = "alo234/Breast_Cancer_MOdel"

IMG_SIZE = 224

# IMPORTANT:
# Based on your testing:
#
# HIGH SCORE  -> MALIGNANT
# LOW SCORE   -> BENIGN
#
# Example:
# 0.9672 -> Malignant
# 0.20   -> Benign


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

        # Render Free Tier RAM optimization
        opts = ort.SessionOptions()

        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = (
            ort.ExecutionMode.ORT_SEQUENTIAL
        )

        # Load ONNX model
        session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=[
                "CPUExecutionProvider"
            ]
        )

        print("")
        print("========================================")
        print("       ONNX MODEL LOADED")
        print("========================================")

        # Input information
        input_info = session.get_inputs()[0]

        print(
            "Input Name  :",
            input_info.name
        )

        print(
            "Input Shape :",
            input_info.shape
        )

        print(
            "Input Type  :",
            input_info.type
        )

        # Output information
        for i, output in enumerate(
            session.get_outputs()
        ):

            print(
                f"Output {i} Name  :",
                output.name
            )

            print(
                f"Output {i} Shape :",
                output.shape
            )

            print(
                f"Output {i} Type  :",
                output.type
            )

        print("========================================")
        print("")

        return session, None

    except Exception as e:

        print(
            "ONNX Model Load Error:",
            str(e)
        )

        return None, str(e)


# ============================================================
# INITIALIZE MODEL
# ============================================================

session, load_error = load_onnx_model()


if session is not None:

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    output_names = [
        output.name
        for output in session.get_outputs()
    ]

else:

    input_name = None
    output_names = []


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_bytes):

    # Convert uploaded bytes to numpy
    nparr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    # Decode image
    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:

        raise ValueError(
            "Invalid image file uploaded."
        )

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Resize to model input
    # --------------------------------------------------------

    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    img = img.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Normalize [0,255] -> [0,1]
    #
    # This matches your previous preprocessing.
    # --------------------------------------------------------

    img = img / 255.0

    # --------------------------------------------------------
    # Add batch dimension
    #
    # (224,224,3)
    #        ↓
    # (1,224,224,3)
    # --------------------------------------------------------

    img = np.expand_dims(
        img,
        axis=0
    )

    print(
        "Preprocessed Shape:",
        img.shape
    )

    print(
        "Pixel Range:",
        float(img.min()),
        "to",
        float(img.max())
    )

    return img


# ============================================================
# MODEL OUTPUT INTERPRETATION
# ============================================================

def interpret_prediction(output_data):

    output = np.asarray(
        output_data
    )

    print("")
    print("========================================")
    print("RAW MODEL OUTPUT:")
    print(output)

    print(
        "OUTPUT SHAPE:",
        output.shape
    )

    print("========================================")

    # ========================================================
    # CASE 1:
    # TWO CLASS OUTPUT
    #
    # Example:
    # [[0.10, 0.90]]
    # ========================================================

    if output.size == 2:

        probabilities = (
            output
            .flatten()
            .astype(float)
        )

        # ----------------------------------------------------
        # If values are logits, convert using softmax
        # ----------------------------------------------------

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
                probabilities
                - np.max(probabilities)
            )

            probabilities = (
                exp_values
                / np.sum(exp_values)
            )

        class_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_index]
        )

        # ----------------------------------------------------
        # IMPORTANT CLASS MAPPING
        #
        # Based on your model testing:
        #
        # index 0 -> Malignant
        # index 1 -> Benign
        # ----------------------------------------------------

        if class_index == 0:

            result = "Malignant"

        else:

            result = "Benign"

        print(
            "Class Index:",
            class_index
        )

        print(
            "Prediction:",
            result
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )

        return (
            result,
            confidence,
            probabilities.tolist()
        )


    # ========================================================
    # CASE 2:
    # SINGLE OUTPUT
    #
    # Example:
    # [[0.9672]]
    #
    # YOUR MODEL:
    #
    # HIGH SCORE -> MALIGNANT
    # LOW SCORE  -> BENIGN
    # ========================================================

    raw_score = float(
        output
        .flatten()[0]
    )

    # --------------------------------------------------------
    # If output is a logit instead of probability
    # --------------------------------------------------------

    if (
        raw_score < 0
        or raw_score > 1
    ):

        raw_score = 1.0 / (
            1.0
            + np.exp(-raw_score)
        )

    # --------------------------------------------------------
    # YOUR MODEL MAPPING
    #
    # score >= 0.5 -> Malignant
    # score <  0.5 -> Benign
    # --------------------------------------------------------

    if raw_score >= 0.5:

        result = "Malignant"

        confidence = raw_score

    else:

        result = "Benign"

        confidence = 1.0 - raw_score

    print(
        "Score:",
        raw_score
    )

    print(
        "Prediction:",
        result
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    print("========================================")
    print("")

    return (
        result,
        confidence,
        raw_score
    )


# ============================================================
# HOME ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
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

            "prediction_mapping":
                "High score = Malignant",

            "endpoint":
                "/predict"

        })

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

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    if session is not None:

        return jsonify({

            "status": "healthy",

            "model":
                "loaded"

        })

    return jsonify({

        "status":
            "unhealthy",

        "model":
            "not loaded",

        "error":
            load_error

    }), 500


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST", "OPTIONS"]
)
def predict():

    # --------------------------------------------------------
    # CORS preflight
    # --------------------------------------------------------

    if request.method == "OPTIONS":

        return "", 200

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if session is None:

        return jsonify({

            "status":
                "error",

            "error":
                f"Model failed to load: {load_error}"

        }), 500

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if "file" not in request.files:

        return jsonify({

            "status":
                "error",

            "error":
                "No file uploaded."

        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({

            "status":
                "error",

            "error":
                "No file selected."

        }), 400

    try:

        # ====================================================
        # STEP 1: READ IMAGE
        # ====================================================

        image_bytes = file.read()

        if not image_bytes:

            return jsonify({

                "status":
                    "error",

                "error":
                    "Uploaded file is empty."

            }), 400


        # ====================================================
        # STEP 2: PREPROCESS IMAGE
        # ====================================================

        img = preprocess_image(
            image_bytes
        )


        # ====================================================
        # STEP 3: RUN ONNX MODEL
        # ====================================================

        outputs = session.run(
            output_names,
            {
                input_name: img
            }
        )


        # Get first output
        output_data = outputs[0]


        # ====================================================
        # STEP 4: INTERPRET RESULT
        # ====================================================

        (
            result,
            confidence,
            raw_output
        ) = interpret_prediction(
            output_data
        )


        # ====================================================
        # STEP 5: CONFIDENCE
        # ====================================================

        confidence_percent = (
            confidence * 100.0
        )


        # ====================================================
        # STEP 6: RESPONSE
        # ====================================================

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


        print("")
        print("========================================")
        print("FINAL API RESPONSE")
        print("========================================")

        print(
            "Result:",
            result
        )

        print(
            "Confidence:",
            f"{confidence_percent:.2f}%"
        )

        print(
            "Raw Score:",
            raw_output
        )

        print("========================================")
        print("")


        return jsonify(
            response
        ), 200


    except Exception as e:

        print("")
        print(
            "Prediction Runtime Error:",
            str(e)
        )
        print("")

        return jsonify({

            "status":
                "error",

            "error":
                str(e)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print("")
    print("========================================")
    print("     BREAST CANCER DETECTION API")
    print("========================================")
    print(
        "Server running on port:",
        port
    )
    print("========================================")
    print("")

    app.run(
        host="0.0.0.0",
        port=port
    )
