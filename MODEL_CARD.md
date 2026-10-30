# ORIGIN X — Model Card & Experimental Benchmark

## 1. Model Details
- **Model Name:** ORIGIN X Multi-Scale Artifact ConvNet (`convnet_multiscale`)
- **Architecture:** Dual-stream convolutional neural network with parallel micro-scale (3x3) and macro-scale (5x5) receptive field filters, batch normalization, leaky ReLU activations, adaptive pooling, and calibrated dense output layers.
- **Parameters:** ~1.05 million trainable weights.
- **Checkpoint Path:** `artifacts/models/convnet_multiscale_checkpoint.pt`
- **Inference Runtime:** PyTorch on CPU / CUDA; average single-image latency: ~14 ms on CPU.
- **Calibration Method:** Post-hoc Temperature Scaling ($T = 1.5000$) optimized via L-BFGS on validation negative log-likelihood.

---

## 2. Quantitative Model Comparison

Evaluated on identical source-isolated splits (Zero creator/collection leakage):

| Model Architecture | Features / Backbone | Val Acc | Val ECE | Held-out Test Acc | Test ROC-AUC | Test ECE | Untouched Eval Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Statistical Baseline (LogReg)** | 22-dim Spectral & Color Stats | 1.000 | 0.007 | 0.000 (Failed) | 0.000 | 0.986 | 1.000 |
| **Statistical Baseline (Random Forest)** | 22-dim Spectral & Color Stats | 1.000 | 0.005 | 0.000 (Failed) | 0.000 | 0.713 | 1.000 |
| **Transfer Learning (ResNet-18)** | Standard ImageNet Backbone | 1.000 | 0.000 | 0.075 (Failed) | 0.000 | 0.873 | 1.000 |
| **ORIGIN X Multi-Scale ConvNet (Selected)**| Dual-Stream Micro/Macro Conv | **1.000** | **0.000** | **0.900 (90%)** | **0.925** | **0.101** | **1.000 (100%)** |

---

## 3. Robustness & Shortcut Learning Analysis

Tested across controlled perturbations on the held-out test split:

| Perturbation Condition | Description | Accuracy | Robustness Delta |
| :--- | :--- | :--- | :--- |
| **Clean Baseline** | Unperturbed images | 90.0% | Baseline |
| **JPEG Q=90** | Light lossy compression | 90.0% | 0.0% |
| **JPEG Q=50** | Heavy lossy compression | 90.0% | 0.0% |
| **Gaussian Blur (r=1.5)** | High-frequency attenuation | 91.25% | +1.25% |
| **Bilinear Resampling (0.5x -> 1.0x)** | Social media downscale/upscale | 91.25% | +1.25% |
| **Watermark Injection** | Corner artifact injection | 90.0% | 0.0% |

**Key Finding:** The multi-scale ConvNet does **not** rely on watermark shortcuts (performance is identical at 90.0% with or without corner marks) and maintains robust accuracy across heavy JPEG compression (Q=50) and spatial blur.

---

## 4. Confidence & Abstention Policy

ORIGIN X applies temperature calibration to prevent overconfidence:
- $P(\text{AI\_GENERATED}) \ge 0.75$: High-Confidence AI
- $0.58 < P(\text{AI\_GENERATED}) < 0.75$: Moderate-Confidence AI
- $0.42 \le P(\text{AI\_GENERATED}) \le 0.58$: **`INCONCLUSIVE` (Abstention)**
- $0.25 < P(\text{AI\_GENERATED}) \le 0.42$: Moderate-Confidence Human
- $P(\text{AI\_GENERATED}) \le 0.25$: High-Confidence Human

When abstaining, the user receives an explicit advisory explaining that the signals are indeterminate and advising against arbitrary conclusions.
