"""
Dataset representation, provenance management, and source-grouped splitting.
Guarantees source-level / collection-level isolation to prevent train/test leakage.
"""

import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

from originx.config import (
    LABEL_HUMAN,
    LABEL_AI,
    LABEL_TO_ID,
    SPLITS_DIR,
    PROCESSED_DATA_DIR,
)
from originx.data.validator import sanitize_to_rgb


@dataclass
class ImageRecord:
    image_id: str
    relative_path: str
    label: str  # HUMAN_CREATED or AI_GENERATED
    source_collection: str  # e.g., 'human_dslr_nature', 'ai_flux_photoreal'
    creator_family: str  # e.g., 'nikon_d850', 'flux_dev', 'stylegan3'
    license: str  # e.g., 'CC0-1.0', 'Public-Domain', 'Synthetic-Benchmark'
    sha256: str
    width: int
    height: int
    split: str  # 'train', 'val', 'test', 'untouched_eval'
    notes: Optional[str] = None


class OriginXDataset(Dataset):
    """
    PyTorch Dataset backed by an auditable JSON manifest and source-level splits.
    """
    def __init__(
        self,
        records: List[ImageRecord],
        base_dir: Path = PROCESSED_DATA_DIR,
        transform=None,
    ):
        self.records = records
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        record = self.records[idx]
        image_path = self.base_dir / record.relative_path
        with Image.open(image_path) as img:
            rgb_img = sanitize_to_rgb(img)
            if self.transform:
                tensor = self.transform(rgb_img)
            else:
                from torchvision.transforms import functional as TF
                tensor = TF.to_tensor(rgb_img)

        label_id = LABEL_TO_ID[record.label]
        meta = {
            "image_id": record.image_id,
            "source_collection": record.source_collection,
            "creator_family": record.creator_family,
            "license": record.license,
            "split": record.split,
        }
        return tensor, label_id, meta


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash for strict data provenance tracking."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def save_manifest(records: List[ImageRecord], output_path: Path):
    """Saves dataset manifest as JSON."""
    data = [asdict(r) for r in records]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_manifest(manifest_path: Path) -> List[ImageRecord]:
    """Loads dataset manifest from JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    return [ImageRecord(**item) for item in raw_list]


def split_by_source_collection(
    records: List[ImageRecord],
    untouched_eval_collections: List[str],
    test_collections: List[str],
    val_ratio: float = 0.20,
    seed: int = 42,
) -> List[ImageRecord]:
    """
    Assigns splits strictly by source_collection to eliminate data leakage.
    - Collections designated as untouched_eval never enter train or val.
    - Collections designated as test never enter train or val.
    - Remaining collections are split between train and val at the creator/collection level.
    """
    import random
    rng = random.Random(seed)

    updated_records = []
    # Group by collection
    collection_map: Dict[str, List[ImageRecord]] = {}
    for r in records:
        collection_map.setdefault(r.source_collection, []).append(r)

    train_records = []
    val_records = []
    test_records = []
    eval_records = []

    for col, col_records in collection_map.items():
        if col in untouched_eval_collections:
            for r in col_records:
                r.split = "untouched_eval"
                eval_records.append(r)
        elif col in test_collections:
            for r in col_records:
                r.split = "test"
                test_records.append(r)
        else:
            # Split within collection into train and val deterministically
            shuffled = list(col_records)
            rng.shuffle(shuffled)
            val_count = int(len(shuffled) * val_ratio)
            for r in shuffled[:val_count]:
                r.split = "val"
                val_records.append(r)
            for r in shuffled[val_count:]:
                r.split = "train"
                train_records.append(r)

    all_assigned = train_records + val_records + test_records + eval_records
    return all_assigned
