# DermoAI — Clinical Skin Cancer Detection & Dermoscopic Risk Assessment

DermoAI is an end-to-end medical deep learning web application designed for automated cutaneous lesion screening and melanoma triage. Built on **MobileNetV2** transfer learning and trained on **26,788 dermoscopic images** from the **ISIC (International Skin Imaging Collaboration)** archive, DermoAI provides real-time probabilistic malignancy scoring, clinical risk tier stratification, and patient-ready diagnostic reports.

---

## Key Features

- **MobileNetV2 Transfer Learning Architecture**: Optimized for high-throughput dermoscopic feature extraction with minimal latency.
- **26,788 ISIC Dermoscopy Images**: Curated 3-way split across Training (21,428), Validation (2,675), and Unseen Testing (2,685) sets.
- **Modern Medical Dashboard**:
  - Drag-and-drop lesion image dropzone with instant preview and clipboard paste support.
  - **1-Click Interactive Test Gallery**: Preloaded with confirmed malignant and benign ISIC dermoscopic cases for instant verification without external files.
  - Animated Confidence & Malignancy Probability meters.
  - Triage Risk Stratification (**Low Risk**, **Moderate Risk**, **High Risk**) with tailored next-step clinical protocols.
  - **Printable Clinical Summary Report**: Formatted diagnostic sheet with image thumbnail, metrics, ABCDE checklist, and clinician sign-off line.
- **Melanoma ABCDE Reference Guide**: Clinical education component covering Asymmetry, Border irregularity, Color variation, Diameter (>6mm), and Evolving characteristics.
- **RESTful API**: Standardized JSON endpoints for diagnostic integration.

---

## Project Structure

```
SkinCancerWebApp/
├── app.py                      # Production Flask web server & REST API
├── train_model.py              # 2-Phase fine-tuning pipeline with callbacks & 3-split validation
├── evaluate_model.py           # Comprehensive test benchmark (Accuracy, ROC-AUC, Confusion Matrix)
├── create_model.py             # Model initialization utility
├── requirements.txt            # Dependency manifest
├── run_app.bat                 # 1-Click Windows launch script
├── .gitignore                  # Git tracking rules
├── model/
│   ├── best_skin_cancer_model.h5 # Calibrated production MobileNetV2 model
│   ├── skin_cancer_model.h5    # Pretrained MobileNetV2 weights
│   └── evaluation/             # Metrics JSON, ROC Curve, and Confusion Matrix plots
├── dataset/
│   ├── train/                  # 7,470 cancerous / 13,958 non-cancerous
│   ├── val/                    # 932 cancerous / 1,743 non-cancerous
│   └── test/                   # 938 cancerous / 1,747 non-cancerous
├── templates/
│   └── index.html              # Clinical dashboard user interface
└── static/
    ├── css/style.css           # Healthcare design system styling
    ├── js/script.js            # Client-side reactivity, dropzone, and API handling
    └── samples/                # Curated ISIC test samples for 1-click verification
```

---

## Installation & Setup

### Prerequisites
- Windows 10/11, macOS, or Linux
- **Python 3.11** (Recommended for optimal TensorFlow compatibility)

### 1. Clone & Navigate
```bash
cd SkinCancerWebApp
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Windows Command Prompt:
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Web Application

### Option A: 1-Click Launcher (Windows)
Double-click `run_app.bat` or run:
```cmd
run_app.bat
```

### Option B: Command Line
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Model Evaluation & Performance Benchmarking

To compute full performance metrics across the 2,685 unseen test images:
```bash
python evaluate_model.py
```

For quick verification on a sample subset:
```bash
python evaluate_model.py --quick
```

**Generated Outputs (`model/evaluation/`):**
- `metrics.json`: Detailed accuracy, specificity, sensitivity, precision, F1, and raw confusion matrix counts.
- `confusion_matrix.png`: Visual heatmap of true positives, true negatives, false positives, and false negatives.
- `roc_curve.png`: Receiver Operating Characteristic curve with area under the curve (AUC).

---

## Training & Fine-Tuning

To train or fine-tune the model with the 2-phase pipeline (warmup classification head + fine-tuning top MobileNetV2 layers):
```bash
python train_model.py --warmup_epochs 5 --fine_tune_epochs 5 --batch_size 16 --out model/best_skin_cancer_model.h5
```

---

## API Reference

### `POST /api/predict`
Accepts `multipart/form-data` with an image file under the key `file`.

**Sample Response:**
```json
{
  "prediction": "Non-Cancerous",
  "confidence": 0.884,
  "malignancy_probability": 0.116,
  "benign_probability": 0.884,
  "risk_tier": "Low Risk",
  "summary": "Dermoscopic architecture is consistent with benign melanocytic nevi, seborrheic keratosis, or non-malignant skin patterns.",
  "clinical_recommendations": [
    "No immediate markers of malignancy identified in the current dermoscopic image.",
    "Continue periodic routine self-examination following the clinical ABCDE rule.",
    "Apply broad-spectrum sunscreen daily to protect against cumulative ultraviolet damage.",
    "Promptly consult a healthcare provider if any noticeable changes develop over time."
  ],
  "input_resolution": "160x160"
}
```

### `GET /api/status`
Returns system operational state, model path, and active input dimensions.

### `GET /api/samples`
Returns list of available preloaded dermoscopy sample filenames.

---

## Clinical Disclaimer

> **IMPORTANT NOTICE:** This application is intended solely as an investigational and clinical decision-support tool. It is **not** an FDA-cleared diagnostic device and should **never** be used as a replacement for clinical examination, dermoscopy by a qualified physician, or histopathological analysis of a biopsy specimen.
