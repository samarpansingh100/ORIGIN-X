"""
API integration and security tests for ORIGIN X FastAPI application.
"""

import io
import sys
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from originx.web.app import app
from originx.config import API_PREFIX, MAX_FILE_SIZE_BYTES


@pytest.fixture
def client():
    return TestClient(app)


def create_test_image_bytes(format="JPEG", size=(256, 256), color=(100, 150, 200)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestAPIEndpoints:
    def test_health_check(self, client):
        resp = client.get(f"{API_PREFIX}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_ready"] is True

    def test_model_info(self, client):
        resp = client.get(f"{API_PREFIX}/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "calibration_method" in data
        assert "confidence_thresholds" in data

    def test_serve_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "ORIGIN" in resp.text
        assert "Image Origin Estimation" in resp.text

    def test_security_headers_present(self, client):
        resp = client.get(f"{API_PREFIX}/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"

    def test_analyze_valid_jpeg(self, client):
        img_bytes = create_test_image_bytes("JPEG")
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        resp = client.post(f"{API_PREFIX}/analyze", files=files)
        assert resp.status_code == 200
        data = resp.json()

        assert "prediction" in data
        assert "verdict" in data["prediction"]
        assert "probability_ai" in data["prediction"]
        assert "confidence_interval" in data["prediction"]
        assert "diagnostics" in data
        assert "spectral_profile" in data["diagnostics"]
        assert "privacy_notice" in data

    def test_analyze_unsupported_format_raises_422(self, client):
        fake_bmp = b"BM" + b"\x00" * 100
        files = {"file": ("test.bmp", fake_bmp, "image/bmp")}
        resp = client.post(f"{API_PREFIX}/analyze", files=files)
        assert resp.status_code == 422
        assert "Unsupported" in resp.json()["detail"]

    def test_analyze_corrupt_data_raises_422(self, client):
        corrupt = b"\xff\xd8\xff\xe0" + b"garbage_data" * 20
        files = {"file": ("corrupt.jpg", corrupt, "image/jpeg")}
        resp = client.post(f"{API_PREFIX}/analyze", files=files)
        assert resp.status_code == 422

    def test_analyze_oversized_payload_raises_413(self, client):
        # Create bytes larger than MAX_FILE_SIZE_BYTES
        oversized = b"\x00" * (MAX_FILE_SIZE_BYTES + 2048)
        files = {"file": ("huge.jpg", oversized, "image/jpeg")}
        resp = client.post(f"{API_PREFIX}/analyze", files=files)
        assert resp.status_code == 413
        assert "exceeds maximum" in resp.json()["detail"]
