"""
Image transformation and feature extraction pipelines.
Includes both tensor transforms for neural networks and
forensic/statistical feature extraction for baseline classifiers and explainability.
"""

import io
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as T

from originx.config import IMAGE_TARGET_SIZE, IMAGE_MEAN, IMAGE_STD


def get_eval_transforms() -> T.Compose:
    """Standard evaluation transformation for neural network inputs."""
    return T.Compose([
        T.Resize(IMAGE_TARGET_SIZE, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def get_train_transforms() -> T.Compose:
    """Training data augmentation pipeline."""
    return T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomCrop(IMAGE_TARGET_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def compute_spectral_features(gray_array: np.ndarray) -> Dict[str, float]:
    """
    Computes 2D Fast Fourier Transform (FFT) radial power distribution features.
    Generative models often exhibit distinct spectral decay and high-frequency peaks
    due to upsampling, transposed convolutions, and latent diffusion decoders.
    """
    h, w = gray_array.shape
    # 2D FFT with zero frequency centered
    f_transform = np.fft.fft2(gray_array)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift) + 1e-8
    power_spectrum = magnitude_spectrum ** 2

    # Radial frequency profile
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    radii = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_radius = min(center_y, center_x)

    # Bin into low, medium, and high frequency bands
    band_low = power_spectrum[radii < (max_radius * 0.25)]
    band_mid = power_spectrum[(radii >= (max_radius * 0.25)) & (radii < (max_radius * 0.65))]
    band_high = power_spectrum[(radii >= (max_radius * 0.65)) & (radii <= max_radius)]

    total_power = float(np.sum(power_spectrum)) + 1e-8
    low_ratio = float(np.sum(band_low) / total_power)
    mid_ratio = float(np.sum(band_mid) / total_power)
    high_ratio = float(np.sum(band_high) / total_power)

    # High frequency energy variance & spectral roll-off (85th percentile of energy)
    flat_radii = radii[radii <= max_radius].flatten()
    flat_power = power_spectrum[radii <= max_radius].flatten()
    sorted_indices = np.argsort(flat_radii)
    cumulative_power = np.cumsum(flat_power[sorted_indices])
    rolloff_idx = np.searchsorted(cumulative_power, 0.85 * cumulative_power[-1])
    spectral_rolloff = float(flat_radii[sorted_indices][min(rolloff_idx, len(sorted_indices) - 1)] / max_radius)

    return {
        "fft_low_ratio": low_ratio,
        "fft_mid_ratio": mid_ratio,
        "fft_high_ratio": high_ratio,
        "fft_spectral_rolloff": spectral_rolloff,
        "fft_high_band_std": float(np.std(band_high) / (np.mean(band_high) + 1e-8)),
    }


def compute_spatial_noise_features(gray_array: np.ndarray) -> Dict[str, float]:
    """
    Extracts high-pass Laplacian noise residual metrics.
    Physical camera sensors produce characteristic Poisson-Gaussian photon shot noise,
    whereas generative diffusion/GAN networks produce smooth patches or distinct noise structures.
    """
    # 3x3 Laplacian high-pass filter kernel
    kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)

    # Compute convolution
    from scipy.signal import convolve2d
    laplacian = convolve2d(gray_array, kernel, mode="same", boundary="symm")
    residual_std = float(np.std(laplacian))
    residual_kurtosis = float(np.mean((laplacian - np.mean(laplacian)) ** 4) / ((residual_std ** 4) + 1e-8))

    # Gradient magnitude statistics
    grad_y, grad_x = np.gradient(gray_array)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return {
        "laplacian_noise_std": residual_std,
        "laplacian_kurtosis": residual_kurtosis,
        "gradient_mean": float(np.mean(grad_mag)),
        "gradient_std": float(np.std(grad_mag)),
        "gradient_max": float(np.quantile(grad_mag, 0.99)),
    }


def compute_color_and_texture_features(rgb_img: Image.Image) -> Dict[str, float]:
    """
    Computes color moments, cross-channel correlation, and saturation metrics.
    """
    arr = np.array(rgb_img, dtype=np.float32) / 255.0  # (H, W, 3)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Color moments
    r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    r_std, g_std, b_std = float(np.std(r)), float(np.std(g)), float(np.std(b))

    # Cross channel correlations
    r_flat, g_flat, b_flat = r.flatten(), g.flatten(), b.flatten()
    corr_rg = float(np.corrcoef(r_flat, g_flat)[0, 1]) if r_std > 1e-4 and g_std > 1e-4 else 0.0
    corr_rb = float(np.corrcoef(r_flat, b_flat)[0, 1]) if r_std > 1e-4 and b_std > 1e-4 else 0.0
    corr_gb = float(np.corrcoef(g_flat, b_flat)[0, 1]) if g_std > 1e-4 and b_std > 1e-4 else 0.0

    # HSV Saturation analysis
    hsv_img = rgb_img.convert("HSV")
    hsv_arr = np.array(hsv_img, dtype=np.float32) / 255.0
    saturation = hsv_arr[:, :, 1]

    return {
        "color_r_mean": r_mean,
        "color_g_mean": g_mean,
        "color_b_mean": b_mean,
        "color_r_std": r_std,
        "color_g_std": g_std,
        "color_b_std": b_std,
        "corr_rg": corr_rg,
        "corr_rb": corr_rb,
        "corr_gb": corr_gb,
        "sat_mean": float(np.mean(saturation)),
        "sat_std": float(np.std(saturation)),
        "sat_high_ratio": float(np.mean(saturation > 0.8)),
    }


def extract_statistical_feature_vector(img: Image.Image) -> np.ndarray:
    """
    Extracts a concatenated 22-dimensional feature vector combining
    spectral, noise, gradient, and color statistics.
    """
    # Resize to standard canonical size for consistent statistical comparison
    canonical = img.resize(IMAGE_TARGET_SIZE, Image.Resampling.BILINEAR)
    gray = np.array(canonical.convert("L"), dtype=np.float32) / 255.0

    spec_dict = compute_spectral_features(gray)
    noise_dict = compute_spatial_noise_features(gray)
    color_dict = compute_color_and_texture_features(canonical)

    combined = {**spec_dict, **noise_dict, **color_dict}
    vector = np.array(list(combined.values()), dtype=np.float32)
    # Handle NaN or Inf gracefully
    vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
    return vector


# --- Robustness & Perturbation Generators for Testing ---

def apply_jpeg_compression(img: Image.Image, quality: int = 75) -> Image.Image:
    """Applies JPEG lossy compression artifact simulation."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_blur(img: Image.Image, radius: float = 1.5) -> Image.Image:
    """Simulates lens softness or re-encoding blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_resampling_cycle(img: Image.Image, scale: float = 0.5) -> Image.Image:
    """Simulates social media downsampling and upsampling."""
    w, h = img.size
    small_w, small_h = max(16, int(w * scale)), max(16, int(h * scale))
    down = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
    return down.resize((w, h), Image.Resampling.BILINEAR)


def apply_watermark_shortcut(img: Image.Image) -> Image.Image:
    """
    Injects a synthetic corner watermark to test shortcut learning robustness.
    """
    synthetic = img.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(synthetic)
    w, h = synthetic.size
    # Draw subtle gray box in lower right
    draw.rectangle([w - 40, h - 20, w - 5, h - 5], fill=(220, 220, 220))
    return synthetic
