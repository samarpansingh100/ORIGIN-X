"""
Unit tests for model architectures, forward pass shapes, checkpoints, and inference.
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from originx.config import MODELS_DIR, IMAGE_TARGET_SIZE
from originx.models.neural_net import OriginX_ConvNet, OriginX_TransferNet
from originx.models.baseline import StatisticalBaselineModel
from originx.data.transforms import get_eval_transforms


class TestModelArchitectures:
    def test_convnet_forward_pass_shape(self):
        model = OriginX_ConvNet()
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, 2)

    def test_transfer_net_forward_pass_shape(self):
        model = OriginX_TransferNet(pretrained=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, 2)

    def test_convnet_checkpoint_loadable(self):
        ckpt_path = MODELS_DIR / "convnet_multiscale_checkpoint.pt"
        assert ckpt_path.exists(), "Checkpoint file must exist"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "temperature" in ckpt

        model = OriginX_ConvNet()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy)
        assert logits.shape == (1, 2)

    def test_baseline_model_predict_image(self):
        baseline_path = MODELS_DIR / "baseline_logistic_regression.pkl"
        assert baseline_path.exists()
        model = StatisticalBaselineModel.load(baseline_path)
        img = Image.new("RGB", (256, 256), (100, 150, 200))
        res = model.predict_image(img)
        assert "predicted_label" in res
        assert "probability_ai" in res
        assert 0.0 <= res["probability_ai"] <= 1.0
