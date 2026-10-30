"""
Unit tests for temperature scaling, expected calibration error, and abstention policy.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from originx.config import LABEL_HUMAN, LABEL_AI, LABEL_INCONCLUSIVE
from originx.models.calibration import (
    TemperatureScaler,
    compute_expected_calibration_error,
    evaluate_decision_and_abstention,
)


class TestCalibration:
    def test_temperature_scaler_scaling(self):
        scaler = TemperatureScaler(init_temperature=2.0)
        logits = torch.tensor([[4.0, -4.0]])
        scaled = scaler(logits)
        assert torch.allclose(scaled, torch.tensor([[2.0, -2.0]]))

    def test_ece_computation(self):
        # Perfect calibration scenario
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        res = compute_expected_calibration_error(probs, labels, num_bins=5)
        assert "ece" in res
        assert "mce" in res
        assert res["ece"] >= 0.0

    def test_abstention_policy_inconclusive_band(self):
        # Probability 0.50 should trigger abstention
        res = evaluate_decision_and_abstention(0.50)
        assert res["is_inconclusive"] is True
        assert res["verdict"] == LABEL_INCONCLUSIVE
        assert res["confidence_tier"] == "INSUFFICIENT"

    def test_abstention_policy_high_confidence_ai(self):
        res = evaluate_decision_and_abstention(0.88)
        assert res["is_inconclusive"] is False
        assert res["verdict"] == LABEL_AI
        assert res["confidence_tier"] == "HIGH"

    def test_abstention_policy_high_confidence_human(self):
        res = evaluate_decision_and_abstention(0.12)
        assert res["is_inconclusive"] is False
        assert res["verdict"] == LABEL_HUMAN
        assert res["confidence_tier"] == "HIGH"
