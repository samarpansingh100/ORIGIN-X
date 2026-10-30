"""
Baseline statistical and frequency-domain models for ORIGIN X.
Uses interpretable spectral, noise, and color features to establish
a rigorous non-deep-learning classification benchmark.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from originx.config import MODELS_DIR, LABEL_HUMAN, LABEL_AI, LABEL_TO_ID, ID_TO_LABEL
from originx.data.transforms import extract_statistical_feature_vector


class StatisticalBaselineModel:
    """
    Interpretable baseline classifier based on 22 hand-crafted physical
    and spectral features (FFT harmonics, Laplacian noise residual, color covariance).
    """
    def __init__(self, classifier_type: str = "logistic_regression", random_state: int = 42):
        self.classifier_type = classifier_type
        self.random_state = random_state
        if classifier_type == "logistic_regression":
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        elif classifier_type == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the pipeline on feature vectors X and integer labels y."""
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [P(HUMAN), P(AI)]."""
        if not self.is_fitted:
            raise RuntimeError("Baseline model is not fitted yet.")
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class ID (0 for HUMAN, 1 for AI)."""
        if not self.is_fitted:
            raise RuntimeError("Baseline model is not fitted yet.")
        return self.pipeline.predict(X)

    def predict_image(self, img: Image.Image) -> Dict[str, Any]:
        """Extracts features from a single image and computes prediction and class probabilities."""
        feat = extract_statistical_feature_vector(img).reshape(1, -1)
        proba = self.predict_proba(feat)[0]
        prob_human, prob_ai = float(proba[0]), float(proba[1])
        pred_id = int(np.argmax(proba))
        return {
            "predicted_label": ID_TO_LABEL[pred_id],
            "probability_human": prob_human,
            "probability_ai": prob_ai,
            "raw_features": feat.flatten().tolist()
        }

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "classifier_type": self.classifier_type,
                "random_state": self.random_state,
                "pipeline": self.pipeline,
                "is_fitted": self.is_fitted,
            }, f)

    @classmethod
    def load(cls, filepath: Path) -> "StatisticalBaselineModel":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls(classifier_type=data["classifier_type"], random_state=data["random_state"])
        instance.pipeline = data["pipeline"]
        instance.is_fitted = data["is_fitted"]
        return instance
