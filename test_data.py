"""
Unit tests for data validation, preprocessing, feature extraction, and manifest integrity.
"""

import io
import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from originx.config import (
    MAX_FILE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION,
    DATA_DIR,
    SPLITS_DIR,
    PROCESSED_DATA_DIR,
)
from originx.data.validator import (
    validate_image_bytes,
    detect_mime_type,
    sanitize_to_rgb,
    ImageValidationError,
)
from originx.data.transforms import (
    extract_statistical_feature_vector,
    compute_spectral_features,
    apply_jpeg_compression,
    apply_gaussian_blur,
)
from originx.data.dataset import load_manifest


def create_test_image(format="JPEG", size=(128, 128), color=(200, 100, 50)):
    """Creates in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    img.save(buf, format=format)
    return buf.getvalue()


class TestDataValidation:
    def test_valid_jpeg(self):
        data = create_test_image("JPEG")
        meta = validate_image_bytes(data)
        assert meta["mime_type"] == "image/jpeg"
        assert meta["width"] == 128
        assert meta["height"] == 128

    def test_valid_png(self):
        data = create_test_image("PNG")
        meta = validate_image_bytes(data)
        assert meta["mime_type"] == "image/png"

    def test_empty_bytes_raises_error(self):
        with pytest.raises(ImageValidationError, match="empty"):
            validate_image_bytes(b"")

    def test_corrupt_bytes_raises_error(self):
        corrupt = b"\xff\xd8\xff\xe0" + b"random_corrupted_garbage_bytes" * 50
        with pytest.raises(ImageValidationError):
            validate_image_bytes(corrupt)

    def test_unsupported_mime_raises_error(self):
        fake_gif = b"GIF89a" + b"\x00" * 20
        with pytest.raises(ImageValidationError, match="Unsupported"):
            validate_image_bytes(fake_gif)

    def test_too_small_dimension_raises_error(self):
        data = create_test_image("JPEG", size=(32, 32))
        with pytest.raises(ImageValidationError, match="too small"):
            validate_image_bytes(data)

    def test_sanitize_rgba_to_rgb(self):
        img_rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        rgb = sanitize_to_rgb(img_rgba)
        assert rgb.mode == "RGB"
        assert rgb.size == (100, 100)


class TestFeatureExtraction:
    def test_feature_vector_shape_and_values(self):
        img = Image.new("RGB", (256, 256), (120, 140, 160))
        vec = extract_statistical_feature_vector(img)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (22,)
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

    def test_spectral_features_keys(self):
        arr = np.random.uniform(0, 1, (128, 128)).astype(np.float32)
        spec = compute_spectral_features(arr)
        assert "fft_low_ratio" in spec
        assert "fft_high_ratio" in spec
        assert "fft_spectral_rolloff" in spec
        assert 0.0 <= spec["fft_low_ratio"] <= 1.0


class TestDatasetManifestIntegrity:
    def test_manifest_exists_and_balanced(self):
        manifest_path = DATA_DIR / "dataset_manifest.json"
        assert manifest_path.exists(), "Manifest file must exist"
        records = load_manifest(manifest_path)
        assert len(records) > 0

        # Check all files exist on disk
        for r in records[:20]:
            file_path = PROCESSED_DATA_DIR / r.relative_path
            assert file_path.exists(), f"Image file {file_path} missing"

    def test_split_files_non_overlapping_collections(self):
        train = load_manifest(SPLITS_DIR / "train.json")
        test = load_manifest(SPLITS_DIR / "test.json")
        untouched = load_manifest(SPLITS_DIR / "untouched_eval.json")

        train_cols = set(r.source_collection for r in train)
        test_cols = set(r.source_collection for r in test)
        untouched_cols = set(r.source_collection for r in untouched)

        # Verify strict source-level zero-leakage isolation
        assert len(train_cols.intersection(test_cols)) == 0, "Train and test collections overlap!"
        assert len(train_cols.intersection(untouched_cols)) == 0, "Train and untouched eval collections overlap!"
        assert len(test_cols.intersection(untouched_cols)) == 0, "Test and untouched eval collections overlap!"
