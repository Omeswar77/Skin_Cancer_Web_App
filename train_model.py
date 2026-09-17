"""
DermoAI — Deep Learning Training Pipeline
Fine-tuned Transfer Learning with MobileNetV2 for Dermoscopic Skin Lesion Classification.
Includes Data Augmentation, 3-Split Validation, Class Balancing, and Callbacks.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

def train(
    train_dir="dataset/train",
    val_dir="dataset/val",
    test_dir="dataset/test",
    img_size=(160, 160),
    batch_size=16,
    warmup_epochs=5,
    fine_tune_epochs=5,
    output_model="model/best_skin_cancer_model.h5"
):
    print("=" * 70)
    print(" DermoAI — Cutaneous Lesion Deep Learning Training Pipeline")
    print("=" * 70)
    print(f" Input Resolution:     {img_size}")
    print(f" Batch Size:           {batch_size}")
    print(f" Warmup Epochs:        {warmup_epochs}")
    print(f" Fine-tune Epochs:     {fine_tune_epochs}")
    print(f" Target Model Save:    {output_model}")
    print("=" * 70)

    os.makedirs(os.path.dirname(output_model), exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Data Generators & Augmentation
    # --------------------------------------------------------------------------
    print("\n[*] Initializing Data Generators...")
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    val_test_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True
    )

    val_data = val_test_gen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    print(f"[*] Class Mappings: {train_data.class_indices}")

    # Compute class weights to handle imbalance (approx 1:2 ratio)
    classes = train_data.classes
    class_weights_vals = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_vals)}
    print(f"[*] Computed Balanced Class Weights: {class_weight_dict}")

    # --------------------------------------------------------------------------
    # 2. Model Architecture Setup
    # --------------------------------------------------------------------------
    print("\n[*] Constructing MobileNetV2 Architecture...")
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze base model initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # --------------------------------------------------------------------------
    # 3. Callbacks
    # --------------------------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            output_model,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------------------------
    # 4. Phase 1: Classification Head Warmup
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(" PHASE 1: Training Classification Head (Base Frozen)")
    print("=" * 50)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=warmup_epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # --------------------------------------------------------------------------
    # 5. Phase 2: Fine-Tuning Top Layers of MobileNetV2
    # --------------------------------------------------------------------------
    if fine_tune_epochs > 0:
        print("\n" + "=" * 50)
        print(" PHASE 2: Fine-Tuning Top Layers")
        print("=" * 50)

        # Unfreeze the top 30 layers
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 30
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        print(f"[*] Unfrozen top {len(base_model.layers) - fine_tune_at} layers for fine-tuning.")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Low LR for fine-tuning
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

        model.fit(
            train_data,
            validation_data=val_data,
            epochs=warmup_epochs + fine_tune_epochs,
            initial_epoch=warmup_epochs,
            class_weight=class_weight_dict,
            callbacks=callbacks
        )

    # Also save a copy as model.h5 in root for convenience
    try:
        model.save("model.h5")
        print(f"[*] Synced root model: model.h5")
    except Exception as e:
        print(f"[!] Note on root model save: {e}")

    # --------------------------------------------------------------------------
    # 6. Final Test Set Evaluation
    # --------------------------------------------------------------------------
    if os.path.exists(test_dir):
        print("\n" + "=" * 50)
        print(" FINAL EVALUATION ON UNSEEN TEST SPLIT")
        print("=" * 50)
        test_data = val_test_gen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False
        )
        test_results = model.evaluate(test_data)
        print(f"Test Loss:     {test_results[0]:.4f}")
        print(f"Test Accuracy: {test_results[1] * 100:.2f}%")
        print(f"Test ROC-AUC:  {test_results[2]:.4f}")

    print("\n Training Complete! Best model saved to:", output_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DermoAI Model Training Pipeline")
    parser.add_argument("--train_dir", type=str, default="dataset/train")
    parser.add_argument("--val_dir", type=str, default="dataset/val")
    parser.add_argument("--test_dir", type=str, default="dataset/test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--out", type=str, default="model/best_skin_cancer_model.h5")
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        output_model=args.out
    )