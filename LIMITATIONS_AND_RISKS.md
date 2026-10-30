# ORIGIN X — Limitations, Failure Modes, and Risk Analysis

## 1. Epistemic Limitations of Pixel-Only Detection

### The Inverse Problem
Determining the exact historical process that produced a matrix of RGB pixels from pixels alone is fundamentally an ill-posed inverse problem. A real camera photograph can be edited in Adobe Photoshop; an AI image can be printed onto canvas and photographed with a Nikon camera; a digital painting can be augmented with generative inpainting. 

Therefore:
- **ORIGIN X never claims absolute certainty.**
- Predictions are presented strictly as calibrated probability estimates with confidence intervals.
- The system reserves an explicit **`INCONCLUSIVE`** state when confidence falls within an ambiguous region (0.42 to 0.58) or when image statistics deviate from known training domains.

---

## 2. Identified Failure Modes & Attack Vectors

| Failure Mode | Mechanism | Expected System Behavior | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Heavy JPEG Recompression** | High-frequency sensor noise and subtle generative grid artifacts are destroyed by discrete 8x8 DCT quantization. | Confidence shifts toward Inconclusive or biases toward AI. | Multi-quality JPEG augmentation during training; calibrated uncertainty output. |
| **Aggressive Downsampling** | Downscaling an image below 256px eliminates PRNU sensor noise and micro-texture. | Signal-to-noise ratio drops drastically. | Input validation enforces minimum 64x64 dimension; UI flags low-resolution uploads as high-uncertainty. |
| **Vector / Minimalist Digital Art** | Flat color regions and clean edges lack camera shot noise, which naive detectors mistake for synthetic generation. | Potential False Positive on human digital art. | Training on diverse artistic modalities; ensemble representation learning. |
| **Hybrid Editing / Inpainting** | 90% real photo with 10% AI inpainting or generative fill. | Whole-image classifier produces ambiguous probabilities. | System explicitly flags potential hybrid content and marks verdict as Inconclusive. |
| **Adversarial Perturbations** | Adding imperceptible noise crafted to fool gradient-based neural networks. | Classification flip. | Randomized spatial transformations and frequency-domain consistency checks. |

---

## 3. Human & Societal Risks

1. **False Accusation Risk:**
   - In educational, journalistic, and creative competitions, falsely labeling a human artist's hard work as "AI-generated" causes severe professional and reputational harm.
   - **Mitigation:** ORIGIN X strictly enforces that predictions must never be used as punitive or forensic proof. The UI prominently displays a disclaimer and educational advisory.

2. **Automated Censorship Risk:**
   - Deploying an uncalibrated detector in automated deletion bots results in disproportionate censorship of digital artists and mobile photographers.
   - **Mitigation:** The confidence threshold policy defaults to `INCONCLUSIVE` whenever calibrated uncertainty exceeds margin bounds.

3. **Privacy & Data Security Risk:**
   - Users uploading unreleased artwork, personal family photos, or proprietary media risk unauthorized retention.
   - **Mitigation:** ORIGIN X operates on an **in-memory ephemeral architecture**. Files are parsed in volatile RAM, processed, and immediately deallocated. No user uploads are saved to disk or logged.
