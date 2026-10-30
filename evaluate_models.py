"""
Comprehensive Model Evaluation and Comparison Harness for ORIGIN X.
Evaluates baselines and neural networks across validation, held-out test,
untouched evaluation sets, robustness perturbations, and shortcut stress tests.
"""

import sys
import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image
import torch
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, roc_auc_score

from originx.config import (
    SPLITS_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    LABEL_TO_ID,
    RANDOM_SEED,
)
from originx.data.dataset import load_manifest
from originx.data.transforms import (
    get_eval_transforms,
    extract_statistical_feature_vector,
    apply_jpeg_compression,
    apply_gaussian_blur,
    apply_resampling_cycle,
    apply_watermark_shortcut,
)
from originx.models.baseline import StatisticalBaselineModel
from originx.models.neural_net import OriginX_ConvNet, OriginX_TransferNet
from originx.models.calibration import TemperatureScaler, compute_expected_calibration_error, evaluate_decision_and_abstention


def load_all_models(device="cpu"):
    """Loads all trained model checkpoints and scalers."""
    # Baselines
    lr_baseline = StatisticalBaselineModel.load(MODELS_DIR / "baseline_logistic_regression.pkl")
    rf_baseline = StatisticalBaselineModel.load(MODELS_DIR / "baseline_random_forest.pkl")

    # ConvNet
    conv_ckpt = torch.load(MODELS_DIR / "convnet_multiscale_checkpoint.pt", map_location=device)
    conv_net = OriginX_ConvNet().to(device)
    conv_net.load_state_dict(conv_ckpt["model_state_dict"])
    conv_net.eval()
    conv_scaler = TemperatureScaler(init_temperature=conv_ckpt["temperature"]).to(device)

    # TransferNet
    transfer_ckpt = torch.load(MODELS_DIR / "transfer_resnet_checkpoint.pt", map_location=device)
    transfer_net = OriginX_TransferNet(pretrained=False).to(device)
    transfer_net.load_state_dict(transfer_ckpt["model_state_dict"])
    transfer_net.eval()
    transfer_scaler = TemperatureScaler(init_temperature=transfer_ckpt["temperature"]).to(device)

    return {
        "baseline_logistic_regression": {"type": "baseline", "model": lr_baseline},
        "baseline_random_forest": {"type": "baseline", "model": rf_baseline},
        "convnet_multiscale": {"type": "neural", "model": conv_net, "scaler": conv_scaler},
        "transfer_resnet": {"type": "neural", "model": transfer_net, "scaler": transfer_scaler},
    }


def predict_single_image(model_info, img: Image.Image, device="cpu"):
    """Computes calibrated P(AI) and latency for an image."""
    t0 = time.perf_counter()
    if model_info["type"] == "baseline":
        res = model_info["model"].predict_image(img)
        prob_ai = res["probability_ai"]
    else:
        transform = get_eval_transforms()
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_logits = model_info["model"](tensor)
            calibrated_logits = model_info["scaler"](raw_logits)
            probs = torch.softmax(calibrated_logits, dim=1).cpu().numpy()[0]
            prob_ai = float(probs[1])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return prob_ai, latency_ms


def evaluate_split(model_name, model_info, records, split_name, device="cpu"):
    """Evaluates a model over an entire split of records."""
    y_true = []
    y_prob_ai = []
    latencies = []

    for r in records:
        img_path = PROCESSED_DATA_DIR / r.relative_path
        with Image.open(img_path) as img:
            prob, lat = predict_single_image(model_info, img, device=device)
            y_prob_ai.append(prob)
            y_true.append(LABEL_TO_ID[r.label])
            latencies.append(lat)

    y_true = np.array(y_true)
    y_prob_ai = np.array(y_prob_ai)
    y_pred = (y_prob_ai >= 0.5).astype(int)

    acc = float(np.mean(y_pred == y_true))
    brier = float(brier_score_loss(y_true, y_prob_ai))
    try:
        auc = float(roc_auc_score(y_true, y_prob_ai))
    except Exception:
        auc = 0.5

    ece_res = compute_expected_calibration_error(y_prob_ai, y_true)
    report = classification_report(y_true, y_pred, target_names=["HUMAN_CREATED", "AI_GENERATED"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Abstention Analysis
    abstention_count = 0
    correct_when_decided = 0
    total_decided = 0
    for prob, true_lbl in zip(y_prob_ai, y_true):
        dec = evaluate_decision_and_abstention(prob)
        if dec["is_inconclusive"]:
            abstention_count += 1
        else:
            total_decided += 1
            pred_lbl = LABEL_TO_ID[dec["verdict"]]
            if pred_lbl == true_lbl:
                correct_when_decided += 1

    decided_accuracy = float(correct_when_decided / total_decided) if total_decided > 0 else 0.0

    return {
        "model": model_name,
        "split": split_name,
        "sample_count": len(records),
        "accuracy": acc,
        "roc_auc": auc,
        "brier_score": brier,
        "ece": ece_res["ece"],
        "confusion_matrix": cm,
        "classification_report": report,
        "avg_latency_ms": float(np.mean(latencies)),
        "abstention_count": abstention_count,
        "abstention_rate": float(abstention_count / len(records)),
        "decided_accuracy": decided_accuracy,
    }


def run_robustness_and_shortcut_tests(model_name, model_info, records, device="cpu"):
    """
    Evaluates robustness under perturbations and checks for shortcut reliance.
    """
    perturbation_fns = {
        "clean": lambda img: img,
        "jpeg_q90": lambda img: apply_jpeg_compression(img, 90),
        "jpeg_q50": lambda img: apply_jpeg_compression(img, 50),
        "blur": lambda img: apply_gaussian_blur(img, 1.5),
        "resample": lambda img: apply_resampling_cycle(img, 0.5),
        "watermark_shortcut": lambda img: apply_watermark_shortcut(img),
    }

    results = {}
    for p_name, p_fn in perturbation_fns.items():
        correct = 0
        probs = []
        for r in records:
            img_path = PROCESSED_DATA_DIR / r.relative_path
            with Image.open(img_path) as img:
                perturbed = p_fn(img)
                prob, _ = predict_single_image(model_info, perturbed, device=device)
                pred = int(prob >= 0.5)
                true_lbl = LABEL_TO_ID[r.label]
                if pred == true_lbl:
                    correct += 1
                probs.append(prob)
        results[p_name] = {
            "accuracy": float(correct / len(records)),
            "mean_prob_ai": float(np.mean(probs)),
            "std_prob_ai": float(np.std(probs)),
        }
    return results


def run_full_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading trained models on {device}...")
    models = load_all_models(device=device)

    splits = {
        "val": load_manifest(SPLITS_DIR / "val.json"),
        "test": load_manifest(SPLITS_DIR / "test.json"),
        "untouched_eval": load_manifest(SPLITS_DIR / "untouched_eval.json"),
    }

    full_results = {"models": {}, "robustness": {}}

    for model_name, model_info in models.items():
        print(f"\nEvaluating model: {model_name}")
        full_results["models"][model_name] = {}
        for split_name, records in splits.items():
            metrics = evaluate_split(model_name, model_info, records, split_name, device=device)
            full_results["models"][model_name][split_name] = metrics
            print(f"  [{split_name:14s}] Acc: {metrics['accuracy']:.4f} | "
                  f"AUC: {metrics['roc_auc']:.4f} | ECE: {metrics['ece']:.4f} | "
                  f"Abstain: {metrics['abstention_rate']*100:.1f}% | "
                  f"Decided Acc: {metrics['decided_accuracy']:.4f}")

        # Robustness & Shortcut tests on held-out test records
        robustness_metrics = run_robustness_and_shortcut_tests(
            model_name, model_info, splits["test"], device=device
        )
        full_results["robustness"][model_name] = robustness_metrics
        print(f"  [Robustness on Test]")
        for p_name, p_res in robustness_metrics.items():
            print(f"    - {p_name:18s}: Acc={p_res['accuracy']:.4f}")

    output_path = METRICS_DIR / "full_model_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nFull evaluation comparison saved to {output_path}")


if __name__ == "__main__":
    run_full_evaluation()
