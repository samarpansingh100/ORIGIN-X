"""
Inference and Explainability Service for ORIGIN X Web Product.
Processes in-memory images, executes calibrated neural network inference,
and extracts multi-signal diagnostic metrics without disk persistence.
"""

import io
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import torch

from originx.config import (
    MODELS_DIR,
    IMAGE_TARGET_SIZE,
    LABEL_HUMAN,
    LABEL_AI,
    LABEL_INCONCLUSIVE,
)
from originx.data.validator import validate_image_bytes, sanitize_to_rgb
from originx.data.transforms import (
    get_eval_transforms,
    compute_spectral_features,
    compute_spatial_noise_features,
    compute_color_and_texture_features,
)
from originx.models.neural_net import OriginX_ConvNet
from originx.models.calibration import TemperatureScaler, evaluate_decision_and_abstention


class InferenceService:
    """
    Singleton service managing model checkpoints, execution,
    and diagnostic signal extraction.
    """
    _instance: Optional["InferenceService"] = None

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.model_metadata = {}
        self._load_model()

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        ckpt_path = MODELS_DIR / "convnet_multiscale_checkpoint.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Model checkpoint not found at {ckpt_path}. Run training first.")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = OriginX_ConvNet().to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        temp = float(ckpt.get("temperature", 1.5))
        self.scaler = TemperatureScaler(init_temperature=temp).to(self.device)
        self.scaler.eval()

        self.model_metadata = {
            "model_name": ckpt.get("model_name", "convnet_multiscale"),
            "temperature": temp,
            "device": self.device,
            "post_cal_ece": ckpt.get("post_cal_ece", 0.0),
        }

    def analyze_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Validates, sanitizes, and evaluates an image in memory.
        Guarantees zero disk writes for privacy.
        """
        # 1. Validate magic bytes, limits, and integrity
        validation_meta = validate_image_bytes(image_bytes)

        # 2. Open and standardize to RGB in memory
        with Image.open(io.BytesIO(image_bytes)) as raw_img:
            rgb_img = sanitize_to_rgb(raw_img)

        # 3. Neural Network Inference
        transform = get_eval_transforms()
        tensor = transform(rgb_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_logits = self.model(tensor)
            calibrated_logits = self.scaler(raw_logits)
            probs = torch.softmax(calibrated_logits, dim=1).cpu().numpy()[0]
            prob_ai = float(probs[1])

        # 4. Apply Abstention and Calibration Policy
        decision = evaluate_decision_and_abstention(prob_ai)

        # 5. Extract Explainability & Forensic Signals
        canonical = rgb_img.resize(IMAGE_TARGET_SIZE, Image.Resampling.BILINEAR)
        gray = np.array(canonical.convert("L"), dtype=np.float32) / 255.0

        spectral_stats = compute_spectral_features(gray)
        noise_stats = compute_spatial_noise_features(gray)
        color_stats = compute_color_and_texture_features(canonical)

        # Synthesize interpretable diagnostics
        diagnostics = {
            "spectral_profile": {
                "high_frequency_ratio": round(spectral_stats["fft_high_ratio"], 4),
                "spectral_rolloff": round(spectral_stats["fft_spectral_rolloff"], 4),
                "description": (
                    "Elevated high-frequency radial harmonics typical of latent upsampling"
                    if spectral_stats["fft_high_ratio"] > 0.35
                    else "Natural optical frequency rolloff characteristic of camera optics"
                ),
            },
            "noise_consistency": {
                "residual_std": round(noise_stats["laplacian_noise_std"], 4),
                "residual_kurtosis": round(noise_stats["laplacian_kurtosis"], 2),
                "description": (
                    "Smooth patches with localized denoiser residual"
                    if noise_stats["laplacian_kurtosis"] > 8.0
                    else "Uniform Poisson-Gaussian photon shot noise"
                ),
            },
            "chromatic_profile": {
                "rg_correlation": round(color_stats["corr_rg"], 4),
                "saturation_mean": round(color_stats["sat_mean"], 4),
                "high_saturation_ratio": round(color_stats["sat_high_ratio"], 4),
                "description": (
                    "High saturation peaks observed in chromatic channels"
                    if color_stats["sat_high_ratio"] > 0.15
                    else "Natural organic color distribution"
                ),
            },
        }

        return {
            "prediction": {
                "verdict": decision["verdict"],
                "is_inconclusive": decision["is_inconclusive"],
                "confidence_tier": decision["confidence_tier"],
                "probability_ai": decision["probability_ai"],
                "probability_human": decision["probability_human"],
                "confidence_interval": decision["confidence_interval"],
                "rationale": decision["rationale"],
            },
            "image_metadata": {
                "width": validation_meta["width"],
                "height": validation_meta["height"],
                "mime_type": validation_meta["mime_type"],
                "size_bytes": validation_meta["size_bytes"],
            },
            "diagnostics": diagnostics,
            "disclaimer": (
                "ORIGIN X provides probabilistic statistical estimates based on pixel patterns. "
                "These results should be considered informative and not interpreted as absolute or legal forensic proof."
            ),
            "privacy_notice": "Analyzed entirely in volatile memory. No image data was persisted to disk.",
        }
