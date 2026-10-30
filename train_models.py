"""
Model Training and Calibration Pipeline for ORIGIN X.
Trains Artificial Neural Network (OriginX_ConvNet) and Transfer Learning (OriginX_TransferNet),
applies post-hoc temperature scaling calibration, and records training metrics.
"""

import sys
import os
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from originx.config import (
    SPLITS_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    RANDOM_SEED,
)
from originx.data.dataset import load_manifest, OriginXDataset
from originx.data.transforms import get_train_transforms, get_eval_transforms
from originx.models.neural_net import OriginX_ConvNet, OriginX_TransferNet
from originx.models.calibration import TemperatureScaler, compute_expected_calibration_error


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_nn(model, loader, device, temperature_scaler=None):
    model.eval()
    all_logits = []
    all_labels = []
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    if temperature_scaler is not None:
        calibrated_logits = temperature_scaler(logits_tensor)
    else:
        calibrated_logits = logits_tensor

    probs = torch.softmax(calibrated_logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    labels_np = labels_tensor.numpy()

    acc = float(np.mean(preds == labels_np))
    prob_ai = probs[:, 1]
    ece_data = compute_expected_calibration_error(prob_ai, labels_np)

    return {
        "accuracy": acc,
        "ece": ece_data["ece"],
        "probabilities": probs,
        "logits": logits_tensor,
        "labels": labels_tensor,
    }


def train_and_calibrate_model(
    model_cls,
    model_name: str,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu"
):
    print(f"\n==========================================")
    print(f"Training {model_name} on {device}")
    print(f"==========================================")
    set_seed(RANDOM_SEED)

    train_records = load_manifest(SPLITS_DIR / "train.json")
    val_records = load_manifest(SPLITS_DIR / "val.json")

    train_ds = OriginXDataset(train_records, transform=get_train_transforms())
    val_ds = OriginXDataset(val_records, transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = model_cls().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        val_eval = evaluate_nn(model, val_loader, device)
        val_acc = val_eval["accuracy"]

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} ECE: {val_eval['ece']:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Fit Temperature Scaler on Validation Logits
    val_eval = evaluate_nn(model, val_loader, device)
    temp_scaler = TemperatureScaler()
    optimal_temp = temp_scaler.fit(val_eval["logits"], val_eval["labels"])
    print(f"Fitted Optimal Temperature T={optimal_temp:.4f}")

    # Evaluate post-calibration
    post_cal_eval = evaluate_nn(model, val_loader, device, temperature_scaler=temp_scaler)
    print(f"Post-calibration Val ECE: {post_cal_eval['ece']:.4f} (Pre: {val_eval['ece']:.4f})")

    # Save Checkpoint
    checkpoint_path = MODELS_DIR / f"{model_name}_checkpoint.pt"
    torch.save({
        "model_name": model_name,
        "model_state_dict": best_weights,
        "temperature": optimal_temp,
        "random_seed": RANDOM_SEED,
        "best_val_acc": best_val_acc,
        "post_cal_ece": post_cal_eval["ece"],
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    return model, temp_scaler, checkpoint_path


def run_training_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # 1. Train Multi-Scale ConvNet
    convnet_model, convnet_scaler, _ = train_and_calibrate_model(
        OriginX_ConvNet,
        model_name="convnet_multiscale",
        epochs=12,
        lr=1e-3,
        device=device
    )

    # 2. Train Transfer Learning Net (ResNet18 backbone)
    transfer_model, transfer_scaler, _ = train_and_calibrate_model(
        lambda: OriginX_TransferNet(pretrained=False),  # Pretrained=False for reliable offline deterministic builds
        model_name="transfer_resnet",
        epochs=12,
        lr=1e-3,
        device=device
    )

    print("\nTraining completed successfully for all neural architectures.")


if __name__ == "__main__":
    run_training_pipeline()
