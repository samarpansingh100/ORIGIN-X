"""
Training and Evaluation Script for Baseline Models.
Extracts statistical/spectral feature vectors and trains Logistic Regression
and Random Forest baselines on the train split, evaluating on val, test, and untouched_eval splits.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, roc_auc_score

from originx.config import (
    SPLITS_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    LABEL_TO_ID,
    RANDOM_SEED,
)
from originx.data.dataset import load_manifest, ImageRecord
from originx.data.transforms import extract_statistical_feature_vector
from originx.models.baseline import StatisticalBaselineModel


def extract_split_features(records, base_dir=PROCESSED_DATA_DIR):
    """Extracts features X and labels y for a list of ImageRecords."""
    X_list = []
    y_list = []
    meta_list = []
    for r in records:
        img_path = base_dir / r.relative_path
        with Image.open(img_path) as img:
            feat = extract_statistical_feature_vector(img)
            X_list.append(feat)
            y_list.append(LABEL_TO_ID[r.label])
            meta_list.append(r)
    return np.array(X_list), np.array(y_list), meta_list


def evaluate_model(model, X, y, split_name):
    """Computes comprehensive evaluation metrics."""
    preds = model.predict(X)
    probas = model.predict_proba(X)
    prob_ai = probas[:, 1]

    acc = float(np.mean(preds == y))
    brier = float(brier_score_loss(y, prob_ai))
    try:
        auc = float(roc_auc_score(y, prob_ai))
    except Exception:
        auc = 0.5

    report = classification_report(y, preds, target_names=["HUMAN_CREATED", "AI_GENERATED"], output_dict=True)
    cm = confusion_matrix(y, preds).tolist()

    return {
        "split": split_name,
        "sample_count": len(y),
        "accuracy": acc,
        "roc_auc": auc,
        "brier_score": brier,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def run_baseline_training():
    print("=== Training ORIGIN X Baseline Models ===")
    train_records = load_manifest(SPLITS_DIR / "train.json")
    val_records = load_manifest(SPLITS_DIR / "val.json")
    test_records = load_manifest(SPLITS_DIR / "test.json")
    untouched_records = load_manifest(SPLITS_DIR / "untouched_eval.json")

    print(f"Extracting features for {len(train_records)} train images...")
    X_train, y_train, _ = extract_split_features(train_records)
    print(f"Extracting features for {len(val_records)} val images...")
    X_val, y_val, _ = extract_split_features(val_records)
    print(f"Extracting features for {len(test_records)} test images...")
    X_test, y_test, _ = extract_split_features(test_records)
    print(f"Extracting features for {len(untouched_records)} untouched eval images...")
    X_untouched, y_untouched, _ = extract_split_features(untouched_records)

    # 1. Logistic Regression Baseline
    print("\n--- Training Logistic Regression Baseline ---")
    lr_baseline = StatisticalBaselineModel(classifier_type="logistic_regression", random_state=RANDOM_SEED)
    lr_baseline.fit(X_train, y_train)

    lr_val_metrics = evaluate_model(lr_baseline, X_val, y_val, "val")
    lr_test_metrics = evaluate_model(lr_baseline, X_test, y_test, "test")
    lr_untouched_metrics = evaluate_model(lr_baseline, X_untouched, y_untouched, "untouched_eval")

    print(f"Logistic Regression Validation Accuracy: {lr_val_metrics['accuracy']:.4f} | ROC-AUC: {lr_val_metrics['roc_auc']:.4f}")
    print(f"Logistic Regression Held-out Test Accuracy: {lr_test_metrics['accuracy']:.4f} | ROC-AUC: {lr_test_metrics['roc_auc']:.4f}")
    print(f"Logistic Regression Untouched Eval Accuracy: {lr_untouched_metrics['accuracy']:.4f} | ROC-AUC: {lr_untouched_metrics['roc_auc']:.4f}")

    lr_checkpoint_path = MODELS_DIR / "baseline_logistic_regression.pkl"
    lr_baseline.save(lr_checkpoint_path)
    print(f"Saved baseline checkpoint: {lr_checkpoint_path}")

    # 2. Random Forest Baseline
    print("\n--- Training Random Forest Baseline ---")
    rf_baseline = StatisticalBaselineModel(classifier_type="random_forest", random_state=RANDOM_SEED)
    rf_baseline.fit(X_train, y_train)

    rf_val_metrics = evaluate_model(rf_baseline, X_val, y_val, "val")
    rf_test_metrics = evaluate_model(rf_baseline, X_test, y_test, "test")
    rf_untouched_metrics = evaluate_model(rf_baseline, X_untouched, y_untouched, "untouched_eval")

    print(f"Random Forest Validation Accuracy: {rf_val_metrics['accuracy']:.4f} | ROC-AUC: {rf_val_metrics['roc_auc']:.4f}")
    print(f"Random Forest Held-out Test Accuracy: {rf_test_metrics['accuracy']:.4f} | ROC-AUC: {rf_test_metrics['roc_auc']:.4f}")
    print(f"Random Forest Untouched Eval Accuracy: {rf_untouched_metrics['accuracy']:.4f} | ROC-AUC: {rf_untouched_metrics['roc_auc']:.4f}")

    rf_checkpoint_path = MODELS_DIR / "baseline_random_forest.pkl"
    rf_baseline.save(rf_checkpoint_path)
    print(f"Saved baseline checkpoint: {rf_checkpoint_path}")

    metrics_output = {
        "model_type": "statistical_baselines",
        "features": "22-dim spectral, noise, gradient, and color covariance",
        "logistic_regression": {
            "val": lr_val_metrics,
            "test": lr_test_metrics,
            "untouched_eval": lr_untouched_metrics,
        },
        "random_forest": {
            "val": rf_val_metrics,
            "test": rf_test_metrics,
            "untouched_eval": rf_untouched_metrics,
        }
    }

    metrics_path = METRICS_DIR / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Saved baseline metrics to {metrics_path}")


if __name__ == "__main__":
    run_baseline_training()
