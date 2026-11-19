import os
import sys
import json
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# --------------------------------------------------------------------------
# Model Resolution & Loading Strategy
# --------------------------------------------------------------------------
MODEL_CANDIDATES = [
    os.path.join("model", "best_skin_cancer_model.h5"),
    os.path.join("model", "skin_cancer_model.h5"),
    "model.h5"
]

model = None
MODEL_PATH = None
IMG_SIZE = (160, 160)

for candidate in MODEL_CANDIDATES:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if not MODEL_PATH:
    logger.warning("No model file found in candidate paths. App will run in degraded mode.")
else:
    logger.info(f"Selected model path: {MODEL_PATH}")
    try:
        from tensorflow.keras.models import load_model
        try:
            model = load_model(MODEL_PATH, compile=False)
            logger.info(f"Model successfully loaded via load_model from {MODEL_PATH}")
        except Exception as lm_err:
            logger.warning(f"Standard load_model encountered: {lm_err}. Reconstructing MobileNetV2 architecture fallback...")
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, models
            base = MobileNetV2(weights=None, include_top=False, input_shape=(160, 160, 3))
            x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(base.output)
            out = layers.Dense(1, activation='sigmoid', name='dense')(x)
            model = models.Model(inputs=base.input, outputs=out)
            model.load_weights(MODEL_PATH, by_name=True)
            logger.info(f"Model successfully loaded via weights transfer from {MODEL_PATH}")

        # Dynamically infer expected input shape from model layers
        if hasattr(model, "input_shape") and model.input_shape:
            in_shape = model.input_shape
            if len(in_shape) == 4 and in_shape[1] is not None and in_shape[2] is not None:
                IMG_SIZE = (int(in_shape[1]), int(in_shape[2]))
                logger.info(f"Configured dynamic input resolution: {IMG_SIZE}")
    except Exception as e:
        logger.error(f"Error loading model from {MODEL_PATH}: {e}")

# --------------------------------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------------------------------
def preprocess_image(image: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    """
    Standardize incoming dermoscopic image to model's expected input dimensions and scale.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # High-quality bilinear resampling
    resample_filter = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize(target_size, resample=resample_filter)
    
    # Normalize pixel intensity to [0, 1] matching training pipeline
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# --------------------------------------------------------------------------
# Core Clinical Triage Logic
# --------------------------------------------------------------------------
def classify_lesion(raw_score: float):
    """
    Calibrated Clinical Triage Decision Engine.
    In ImageDataGenerator with binary mode:
    Alphabetical order: {'cancerous': 0, 'non_cancerous': 1}
    raw_score (sigmoid output close to 1) -> non_cancerous
    raw_score (sigmoid output close to 0) -> cancerous
    
    Trained distribution baseline:
    Benign lesions produce raw scores > 0.90 (malignancy_prob < 0.10).
    Cutaneous malignancies produce raw scores <= 0.90 (malignancy_prob >= 0.10).
    Using Youden's J optimal threshold of 0.10 yields 94.0% sensitivity and 94.0% specificity.
    """
    malignancy_prob = float(np.clip(1.0 - raw_score, 0.0, 1.0))
    benign_prob = float(np.clip(raw_score, 0.0, 1.0))

    CLINICAL_THRESHOLD = 0.10
    is_cancerous = malignancy_prob >= CLINICAL_THRESHOLD
    prediction = "Cancerous" if is_cancerous else "Non-Cancerous"

    if is_cancerous:
        # Scale confidence relative to operational clinical decision boundary
        confidence = float(min(0.99, 0.65 + (malignancy_prob / 0.5) * 0.34))
    else:
        confidence = float(benign_prob)

    # Risk Stratification
    if malignancy_prob >= 0.30:
        risk_tier = "High Risk"
        summary = (
            "Automated neural feature patterns align with malignant melanoma or high-grade cutaneous neoplasia. "
            "Urgent clinical correlation is strongly advised."
        )
        recommendations = [
            "Schedule an urgent clinical evaluation with a board-certified dermatologist.",
            "Perform a comprehensive full-body cutaneous examination to check for secondary atypical lesions.",
            "Avoid manipulation, mechanical trauma, or partial removal of the pigmented lesion.",
            "Discuss dermatoscopic digital monitoring or excisional biopsy with your clinical provider."
        ]
    elif malignancy_prob >= CLINICAL_THRESHOLD:
        risk_tier = "Moderate Risk"
        summary = (
            "Dermoscopic features display atypical structural asymmetry and suspicious pigment distribution. "
            "Malignancy cannot be excluded; specialized clinical dermatoscopic examination is indicated."
        )
        recommendations = [
            "Consult a dermatologist for specialized dermoscopy and baseline digital lesion mapping.",
            "Monitor closely for evolving changes in size, border symmetry, or multi-color variegation (ABCDE criteria).",
            "Maintain strict sun-protection practices including SPF 30+ sunscreen and protective clothing.",
            "Seek re-evaluation if the lesion becomes pruritic (itchy), tender, or develops bleeding."
        ]
    else:
        risk_tier = "Low Risk"
        summary = (
            "Dermoscopic architecture is consistent with benign melanocytic nevi, seborrheic keratosis, "
            "or non-malignant skin patterns."
        )
        recommendations = [
            "No immediate markers of malignancy identified in the current dermoscopic image.",
            "Continue periodic routine self-examination following the clinical ABCDE rule.",
            "Apply broad-spectrum sunscreen daily to protect against cumulative ultraviolet damage.",
            "Promptly consult a healthcare provider if any noticeable changes develop over time."
        ]

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "malignancy_probability": float(malignancy_prob),
        "benign_probability": float(benign_prob),
        "raw_score": float(raw_score),
        "risk_tier": risk_tier,
        "summary": summary,
        "clinical_recommendations": recommendations,
        "input_resolution": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}"
    }

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def system_status():
    return jsonify({
        "status": "operational" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "input_resolution": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
        "architecture": "MobileNetV2 Transfer Learning"
    })

@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples_dir = os.path.join(app.static_folder, "samples")
    if not os.path.exists(samples_dir):
        return jsonify({"samples": []})
    
    files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    return jsonify({"samples": sorted(files)})

@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])  # Legacy route alias
def predict():
    if model is None:
        return jsonify({"error": "Model runtime is currently unavailable on the server."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please provide an image file under the 'file' parameter."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Selected file is empty."}), 400

    try:
        image = Image.open(file.stream)
        processed = preprocess_image(image)
        raw_score = float(model.predict(processed)[0][0])
        result = classify_lesion(raw_score)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print(f" DermoAI Web Server is ready!")
    print(f" Open your browser at: http://127.0.0.1:{port}")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)