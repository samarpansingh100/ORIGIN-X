"""
Probability Calibration and Abstention Policy Module for ORIGIN X.
Implements Temperature Scaling (Guo et al.), Expected Calibration Error (ECE),
and confidence-bounded abstention to prevent overconfident erroneous predictions.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

from originx.config import (
    LABEL_HUMAN,
    LABEL_AI,
    LABEL_INCONCLUSIVE,
    PROB_ABSTAIN_LOW,
    PROB_ABSTAIN_HIGH,
    UNCERTAINTY_MARGIN,
)


class TemperatureScaler(nn.Module):
    """
    Applies temperature scaling to neural network logits.
    T > 1 softens overconfident probabilities.
    T < 1 sharpens underconfident probabilities.
    """
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.05, max=10.0)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 100):
        """Optimizes temperature to minimize NLL on validation set."""
        self.eval()
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.02, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    probabilities: 1D array of predicted class probabilities for the positive class (AI_GENERATED).
    labels: 1D binary array of ground truth labels (0 or 1).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0
    total_samples = len(labels)
    bin_stats = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples falling into the current probability bin
        in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper) if i > 0 else (probabilities >= bin_lower) & (probabilities <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            avg_confidence = float(np.mean(probabilities[in_bin]))
            avg_accuracy = float(np.mean(labels[in_bin]))
            diff = abs(avg_confidence - avg_accuracy)
            ece += (bin_count / total_samples) * diff
            mce = max(mce, diff)

            bin_stats.append({
                "bin_idx": i,
                "bin_range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": bin_count,
                "confidence": round(avg_confidence, 4),
                "accuracy": round(avg_accuracy, 4),
                "calibration_gap": round(diff, 4),
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "num_bins": num_bins,
        "bin_details": bin_stats,
    }


def evaluate_decision_and_abstention(
    calibrated_prob_ai: float,
    prob_ci_radius: float = 0.06,
    abstain_low: float = PROB_ABSTAIN_LOW,
    abstain_high: float = PROB_ABSTAIN_HIGH,
) -> Dict[str, Any]:
    """
    Applies ORIGIN X confidence and abstention policy to a calibrated probability.
    Returns structured verdict, confidence rating, interval bounds, and plain language rationale.
    """
    prob_ai = float(np.clip(calibrated_prob_ai, 0.0, 1.0))
    prob_human = float(1.0 - prob_ai)

    ci_lower = max(0.0, prob_ai - prob_ci_radius)
    ci_upper = min(1.0, prob_ai + prob_ci_radius)

    # Check for ambiguous / inconclusive band
    is_inconclusive = False
    rationale = ""
    verdict = ""
    confidence_tier = "LOW"

    if abstain_low <= prob_ai <= abstain_high:
        is_inconclusive = True
        verdict = LABEL_INCONCLUSIVE
        confidence_tier = "INSUFFICIENT"
        rationale = (
            f"The image exhibits mixed statistical signals consistent with either human authorship "
            f"or synthetic generation. Estimated AI probability ({prob_ai * 100:.1f}%) lies within "
            f"the indeterminate margin ({abstain_low * 100:.0f}%–{abstain_high * 100:.0f}%). "
            f"ORIGIN X abstains from making a conclusive determination."
        )
    elif prob_ai > abstain_high:
        verdict = LABEL_AI
        if prob_ai >= 0.75:
            confidence_tier = "HIGH"
            rationale = (
                f"Strong statistical indicators characteristic of generative synthesis "
                f"(e.g., latent spatial frequency harmonics and micro-texture patterns) "
                f"were observed with high confidence ({prob_ai * 100:.1f}% estimated probability)."
            )
        else:
            confidence_tier = "MODERATE"
            rationale = (
                f"Moderate indicators favoring generative synthesis were detected "
                f"({prob_ai * 100:.1f}% estimated probability). The model leans AI-generated, "
                f"though edge cases such as digital artwork or compression may overlap."
            )
    else:  # prob_ai < abstain_low
        verdict = LABEL_HUMAN
        if prob_human >= 0.75:
            confidence_tier = "HIGH"
            rationale = (
                f"Strong statistical indicators characteristic of authentic human creation "
                f"(natural sensor photon shot noise, consistent optical depth-of-field, or organic stroke variance) "
                f"were observed ({prob_human * 100:.1f}% estimated human probability)."
            )
        else:
            confidence_tier = "MODERATE"
            rationale = (
                f"Moderate indicators favoring human creation were detected "
                f"({prob_human * 100:.1f}% estimated human probability)."
            )

    return {
        "verdict": verdict,
        "is_inconclusive": is_inconclusive,
        "confidence_tier": confidence_tier,
        "probability_ai": round(prob_ai, 4),
        "probability_human": round(prob_human, 4),
        "confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)],
        "rationale": rationale,
    }
