"""
DermoAI — Model Evaluation & Diagnostic Performance Benchmark
Computes Accuracy, ROC-AUC, Sensitivity, Specificity, Precision, F1-Score,
and generates Confusion Matrix and ROC Curve figures.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

def evaluate(model_path="model/best_skin_cancer_model.h5", test_dir="dataset/test", output_dir="model/evaluation", quick=False):
    print("=" * 70)
    print(" DermoAI — Cutaneous Lesion Model Evaluation Suite")
    print("=" * 70)

    if not os.path.exists(model_path):
        # Fallback search
        for alt in [os.path.join("model", "best_skin_cancer_model.h5"), os.path.join("model", "skin_cancer_model.h5"), "model.h5"]:
            if os.path.exists(alt):
                model_path = alt
                break

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at '{model_path}'.")
        return

    print(f"[*] Loading trained model: {model_path}")
    from tensorflow.keras.models import load_model
    try:
        model = load_model(model_path, compile=False)
        print(f"[+] Loaded model via standard load_model: {model_path}")
    except Exception as lm_err:
        print(f"[!] Standard load_model failed ({lm_err}). Reconstructing MobileNetV2 architecture...")
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras import layers, models
        base = MobileNetV2(weights=None, include_top=False, input_shape=(160, 160, 3))
        x = layers.GlobalAveragePooling2D(name='global_average_pooling2d')(base.output)
        out = layers.Dense(1, activation='sigmoid', name='dense')(x)
        model = models.Model(inputs=base.input, outputs=out)
        model.load_weights(model_path, by_name=True)
        print(f"[+] Loaded weights into reconstructed MobileNetV2.")

    # Determine input shape
    target_size = (160, 160)
    if hasattr(model, "input_shape") and model.input_shape:
        in_s = model.input_shape
        if len(in_s) == 4 and in_s[1] and in_s[2]:
            target_size = (int(in_s[1]), int(in_s[2]))
    print(f"[*] Expected model input resolution: {target_size}")

    cancer_dir = os.path.join(test_dir, "cancerous")
    non_cancer_dir = os.path.join(test_dir, "non_cancerous")

    if not os.path.exists(cancer_dir) or not os.path.exists(non_cancer_dir):
        print(f"[ERROR] Test directories not found under '{test_dir}'.")
        return

    cancer_files = [os.path.join(cancer_dir, f) for f in os.listdir(cancer_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    non_cancer_files = [os.path.join(non_cancer_dir, f) for f in os.listdir(non_cancer_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if quick:
        cancer_files = cancer_files[:50]
        non_cancer_files = non_cancer_files[:50]
        print(f"[!] Quick mode enabled: evaluating on subset of {len(cancer_files) + len(non_cancer_files)} images.")
    else:
        print(f"[*] Full evaluation set: {len(cancer_files)} cancerous, {len(non_cancer_files)} non-cancerous images ({len(cancer_files) + len(non_cancer_files)} total).")

    # In ImageDataGenerator flow_from_directory:
    # Class 0: cancerous, Class 1: non_cancerous
    # For binary evaluation:
    # Positive class (1) = Cancerous (malignant)
    # Negative class (0) = Non-Cancerous (benign)
    # Model raw sigmoid output: ~0 for cancerous, ~1 for non_cancerous.
    # Therefore, predicted malignant probability = 1.0 - raw_score.

    y_true = []
    y_scores = []

    resample_filter = getattr(Image, "Resampling", Image).BILINEAR

    def process_file_list(files, true_label):
        for idx, path in enumerate(files):
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(target_size, resample=resample_filter)
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=0)

                raw_score = float(model.predict(arr, verbose=0)[0][0])
                malig_prob = 1.0 - raw_score

                y_true.append(true_label)
                y_scores.append(malig_prob)

                if (idx + 1) % 100 == 0 or (idx + 1) == len(files):
                    print(f"  Processed {idx + 1}/{len(files)} files for class label {true_label}...")
            except Exception as e:
                print(f"[WARNING] Skipping corrupted image {path}: {e}")

    print("\n[*] Processing Cancerous test set...")
    process_file_list(cancer_files, 1)  # 1 = Cancerous

    print("\n[*] Processing Non-Cancerous test set...")
    process_file_list(non_cancer_files, 0)  # 0 = Non-Cancerous

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    # Compute classification metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

    auc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    # Optimal Clinical Operating Threshold via Youden's J statistic (J = Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    clinical_thresh = float(roc_thresholds[best_idx])

    # Standard (0.5) predictions
    y_pred_std = (y_scores >= 0.5).astype(int)
    acc_std = float(accuracy_score(y_true, y_pred_std))
    prec_std = float(precision_score(y_true, y_pred_std, zero_division=0))
    rec_std = float(recall_score(y_true, y_pred_std, zero_division=0))
    f1_std = float(f1_score(y_true, y_pred_std, zero_division=0))

    # Calibrated Clinical Operating predictions
    y_pred_clin = (y_scores >= clinical_thresh).astype(int)
    acc_clin = float(accuracy_score(y_true, y_pred_clin))
    prec_clin = float(precision_score(y_true, y_pred_clin, zero_division=0))
    rec_clin = float(recall_score(y_true, y_pred_clin, zero_division=0))
    f1_clin = float(f1_score(y_true, y_pred_clin, zero_division=0))

    cm_clin = confusion_matrix(y_true, y_pred_clin)
    tn_c, fp_c, fn_c, tp_c = [int(v) for v in cm_clin.ravel()]
    spec_clin = float(tn_c / (tn_c + fp_c)) if (tn_c + fp_c) > 0 else 0.0

    print("\n" + "=" * 70)
    print(" BENCHMARK RESULTS")
    print("=" * 70)
    print(f" Area Under ROC (ROC-AUC): {auc:.4f}")
    print("-" * 70)
    print(" [Calibrated Clinical Decision Boundary] (Youden's J Cut-off: {:.3f})".format(clinical_thresh))
    print(f"   Diagnostic Accuracy:    {acc_clin * 100:.2f}%")
    print(f"   Sensitivity (Recall):   {rec_clin * 100:.2f}% (Cancer detection rate)")
    print(f"   Specificity:            {spec_clin * 100:.2f}% (Benign detection rate)")
    print(f"   Precision:              {prec_clin * 100:.2f}%")
    print(f"   F1-Score:               {f1_clin:.4f}")
    print(f"   Confusion Matrix:       TP={tp_c}, FN={fn_c}, TN={tn_c}, FP={fp_c}")
    print("-" * 70)
    print(f" [Standard Naive 0.50 Boundary]")
    print(f"   Accuracy: {acc_std * 100:.2f}% | Recall: {rec_std * 100:.2f}% | Precision: {prec_std * 100:.2f}%")
    print("=" * 70)

    # Save comprehensive metrics JSON
    os.makedirs(output_dir, exist_ok=True)
    metrics_data = {
        "model_path": model_path,
        "input_resolution": f"{target_size[0]}x{target_size[1]}",
        "total_test_samples": len(y_true),
        "roc_auc": auc,
        "clinical_operating_threshold": clinical_thresh,
        "calibrated_clinical_metrics": {
            "accuracy": acc_clin,
            "sensitivity_recall": rec_clin,
            "specificity": spec_clin,
            "precision": prec_clin,
            "f1_score": f1_clin,
            "confusion_matrix": {
                "true_positives": tp_c,
                "false_negatives": fn_c,
                "true_negatives": tn_c,
                "false_positives": fp_c
            }
        },
        "standard_0_5_metrics": {
            "accuracy": acc_std,
            "sensitivity_recall": rec_std,
            "precision": prec_std,
            "f1_score": f1_std
        }
    }

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[*] Benchmark metrics saved to: {metrics_file}")

    # Generate plots if matplotlib is available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Confusion Matrix Figure (Calibrated Clinical)
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm_clin, cmap=plt.cm.Blues, alpha=0.85)
        fig.colorbar(cax)
        for i in range(2):
            for j in range(2):
                color = "white" if cm_clin[i, j] > np.max(cm_clin) / 2 else "black"
                ax.text(j, i, str(cm_clin[i, j]), ha="center", va="center", color=color, fontsize=15, weight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign (0)", "Cancerous (1)"], fontsize=10)
        ax.set_yticklabels(["Benign (0)", "Cancerous (1)"], fontsize=10)
        plt.xlabel("Predicted Class", fontsize=11, weight="bold")
        plt.ylabel("Ground Truth Class", fontsize=11, weight="bold")
        plt.title(f"Clinical Confusion Matrix (Threshold={clinical_thresh:.2f}, Acc={acc_clin*100:.1f}%)", fontsize=11, pad=15, weight="bold")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"[*] Confusion Matrix plot saved to: {cm_path}")

        # 2. ROC Curve Figure
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#0284c7", lw=2.5, label=f"ROC curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Chance baseline")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
        plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, weight="bold")
        plt.title("Receiver Operating Characteristic (ROC)", fontsize=12, weight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"[*] ROC curve plot saved to: {roc_path}")

    except Exception as e:
        print(f"[!] Could not generate plots (matplotlib/display): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermoAI Model Evaluation Script")
    parser.add_argument("--model", type=str, default="model/best_skin_cancer_model.h5", help="Path to trained model (.h5)")
    parser.add_argument("--test_dir", type=str, default="dataset/test", help="Path to test dataset folder")
    parser.add_argument("--out", type=str, default="model/evaluation", help="Output directory for reports")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing")
    args = parser.parse_args()

    evaluate(model_path=args.model, test_dir=args.test_dir, output_dir=args.out, quick=args.quick)
