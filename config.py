"""
ORIGIN X Configuration Module
Defines system-wide constants, prediction labels, data parameters,
model architectures, and inference constraints.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, ARTIFACTS_DIR, MODELS_DIR, METRICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Prediction Labels
LABEL_HUMAN = "HUMAN_CREATED"
LABEL_AI = "AI_GENERATED"
LABEL_INCONCLUSIVE = "INCONCLUSIVE"

CLASS_LABELS: List[str] = [LABEL_HUMAN, LABEL_AI]
LABEL_TO_ID: Dict[str, int] = {LABEL_HUMAN: 0, LABEL_AI: 1}
ID_TO_LABEL: Dict[int, str] = {0: LABEL_HUMAN, 1: LABEL_AI}

# Decision & Abstention Policy Thresholds
# Calibrated probability P(AI_GENERATED) thresholds:
# P(AI) >= HIGH_CONFIDENCE_AI -> High confidence AI
# P(AI) <= HIGH_CONFIDENCE_HUMAN -> High confidence Human
# Between LOW_CONFIDENCE_BOUND and HIGH_CONFIDENCE_BOUND -> Abstain with INCONCLUSIVE
PROB_ABSTAIN_LOW: float = 0.42
PROB_ABSTAIN_HIGH: float = 0.58
UNCERTAINTY_MARGIN: float = 0.08  # Bootstrap CI radius limit for conclusive verdict

# Supported Image Ingestion Constraints
SUPPORTED_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/webp"
}
SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
MIN_IMAGE_DIMENSION: int = 64
MAX_IMAGE_DIMENSION: int = 8192

# Standard Preprocessing Parameters
IMAGE_TARGET_SIZE: Tuple[int, int] = (224, 224)
IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Random Seeds for Reproducibility
RANDOM_SEED: int = 42

# API Server Configuration
HOST: str = "127.0.0.1"
PORT: int = 8000
API_PREFIX: str = "/api/v1"
