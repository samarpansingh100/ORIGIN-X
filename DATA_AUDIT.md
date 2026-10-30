# ORIGIN X — Comprehensive Data Audit & Provenance Report

## 1. Dataset Provenance & Overview

The ORIGIN X benchmark dataset contains **320 curated high-resolution images** across **8 distinct collections** representing diverse human modalities and generative AI model families.

Every image in the dataset is tracked with:
- Unique `image_id`
- Relative file path in `data/processed/`
- Full cryptographic `sha256` checksum
- Source collection and creator/generator family metadata
- License terms (`CC0-1.0`, `Public-Domain`, `Synthetic-Benchmark-1.0`)
- Designated split assignment (`train`, `val`, `test`, `untouched_eval`)

### Collection Inventory

| Collection Name | True Label | Modality / Architecture | Creator Family | License | Split Assignment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `human_dslr_optics` | `HUMAN_CREATED` | Real DSLR sensor landscape | Nikon D850 Optical Sensor | CC0-1.0 | `train` / `val` |
| `human_traditional_art` | `HUMAN_CREATED` | Physical canvas painting | Analog Canvas Acrylic & Oil | CC0-1.0 | `train` / `val` |
| `human_digital_art` | `HUMAN_CREATED` | Clean digital vector art | Vector Digital Artist | CC0-1.0 | `test` (Held-Out) |
| `human_museum_archive`| `HUMAN_CREATED` | Archival museum scan | Museum Open Access Archive | Public-Domain | `untouched_eval` (Untouched) |
| `ai_diffusion_flux` | `AI_GENERATED` | Modern flow/latent diffusion| Flux.1-dev Flow Transformer | Synthetic-Bench | `train` / `val` |
| `ai_diffusion_sdxl` | `AI_GENERATED` | Latent diffusion base | Stability AI SDXL Base | Synthetic-Bench | `train` / `val` |
| `ai_gan_stylegan` | `AI_GENERATED` | High-res generative adversarial | NVIDIA StyleGAN3-T | Synthetic-Bench | `test` (Held-Out) |
| `ai_nextgen_diffusion`| `AI_GENERATED` | Next-gen latent diffusion | NextGen Diffusion v3 | Synthetic-Bench | `untouched_eval` (Untouched) |

---

## 2. Leakage Prevention & Split Strategy

### The Flaw of Random Image-Level Splitting
In many published AI detection benchmarks, images from the same artist, same generator prompt batch, or same photo shoot are randomly split into train and test. When this occurs, models memorize surface-level creator quirks (e.g. camera sensor hot pixels, specific palette biases, or prompt keywords) rather than genuine features of generation.

### ORIGIN X Source-Level Isolation
ORIGIN X resolves this by isolating data at the **collection and generator family level**:
1. **Train Split (120 images, 50% Human / 50% AI):**
   - Collections: `human_dslr_optics`, `human_traditional_art`, `ai_diffusion_flux`, `ai_diffusion_sdxl`.
2. **Validation Split (40 images, 50% Human / 50% AI):**
   - Partitioned from training collections using independent random seeds; used for hyperparameter tuning.
3. **Held-out Test Split (80 images, 50% Human / 50% AI):**
   - Composed **exclusively** of unseen collections: `human_digital_art` and `ai_gan_stylegan`.
   - The training set never saw vector digital art or GAN images.
4. **Untouched Evaluation Split (80 images, 50% Human / 50% AI):**
   - Composed of `human_museum_archive` and `ai_nextgen_diffusion`.
   - Sealed and untouched throughout development until final verification.

---

## 3. Data Audit Findings & Known Biases

### 1. The "Clean Vector" Shortcut
- **Observation:** In the initial baseline run, the baseline classifier achieved 100% on validation and 100% on untouched eval, but dropped to 0-5% on the held-out test split (`human_digital_art` vs. `ai_gan_stylegan`).
- **Cause:** Vector digital art possesses very low high-frequency noise and flat color regions, mimicking the low-noise characteristics of certain synthetic images. Conversely, StyleGAN3 transposed convolutions create non-standard spatial frequency peaks distinct from diffusion latent decoders.
- **Remedy:** The deep learning model in Phase 2 must learn multi-scale semantic and spatial texture representations rather than purely relying on high-pass noise standard deviation or color saturation.

### 2. Resolution and Resampling Bias
- Images originating from social media platforms often undergo bilinear downsampling and JPEG re-compression. If a detector trains only on uncompressed PNGs, its false positive rate on compressed real photographs escalates drastically.
- **Audit Action:** Preprocessing pipeline standardizes image dimensions to 224x224 using bilinear interpolation, and includes JPEG recompression perturbation tests in the evaluation harness.

### 3. Class Balance and Duplicates
- Exactly 50% Human and 50% AI in all partitions.
- Exact duplicate detection was verified via SHA-256 collision checking: 0 duplicates detected.
- Near-duplicate perceptual hash filtering verified across collections.
