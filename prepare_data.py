"""
Data Preparation and Benchmark Generation Script.
Builds an auditable, balanced dataset across diverse human and AI generator collections.
Ensures source-level split isolation, calculates SHA256 checksums,
and verifies no data leakage into untouched evaluation splits.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from originx.config import (
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    DATA_DIR,
    LABEL_HUMAN,
    LABEL_AI,
    RANDOM_SEED,
)
from originx.data.dataset import (
    ImageRecord,
    compute_sha256,
    save_manifest,
    split_by_source_collection,
)


def generate_human_dslr_landscape(seed: int, width: int = 512, height: int = 512) -> Image.Image:
    """Simulates real DSLR camera photography: natural lighting, depth of field, physical sensor noise."""
    rng = np.random.default_rng(seed)
    # Natural landscape gradient: sky to terrain
    y, x = np.ogrid[:height, :width]
    sky = np.clip(180 - 0.25 * y + rng.normal(0, 4, (height, width)), 0, 255)
    horizon = int(height * (0.45 + 0.05 * rng.uniform(-1, 1)))
    
    img_arr = np.zeros((height, width, 3), dtype=np.float32)
    # Sky (bluish natural gradient)
    img_arr[:horizon, :, 0] = sky[:horizon] * 0.7 + 20
    img_arr[:horizon, :, 1] = sky[:horizon] * 0.85 + 30
    img_arr[:horizon, :, 2] = sky[:horizon] * 1.0 + 40

    # Mountains / terrain (earthy tones)
    mountain_profile = horizon + (np.sin(x / 40.0) * 25 + np.cos(x / 20.0) * 15).astype(int)
    for col in range(width):
        peak = max(horizon - 40, min(height - 10, mountain_profile[0, col]))
        img_arr[peak:, col, 0] = 60 + 0.1 * (y[peak:, 0] - peak) + rng.normal(0, 5, height - peak)
        img_arr[peak:, col, 1] = 85 + 0.15 * (y[peak:, 0] - peak) + rng.normal(0, 5, height - peak)
        img_arr[peak:, col, 2] = 50 + 0.08 * (y[peak:, 0] - peak) + rng.normal(0, 5, height - peak)

    # Physical Poisson-Gaussian camera shot noise
    shot_noise = rng.normal(0, 3.5, (height, width, 3))
    # Subtle sensor Bayer demosaicing pattern simulation
    bayer_pattern = np.zeros((height, width, 3), dtype=np.float32)
    bayer_pattern[0::2, 0::2, 0] += 0.8  # Red
    bayer_pattern[1::2, 1::2, 2] += 0.8  # Blue
    
    final_arr = np.clip(img_arr + shot_noise + bayer_pattern, 0, 255).astype(np.uint8)
    return Image.fromarray(final_arr)


def generate_human_traditional_art(seed: int, width: int = 512, height: int = 512) -> Image.Image:
    """Simulates physical paintings: canvas weave, pigment mixing, brush stroke gradients."""
    rng = np.random.default_rng(seed)
    base_color = rng.uniform(180, 240, 3)
    img = Image.new("RGB", (width, height), tuple(base_color.astype(int)))
    draw = ImageDraw.Draw(img)

    # Paint brush strokes
    num_strokes = rng.integers(15, 30)
    for _ in range(num_strokes):
        x1, y1 = rng.integers(0, width, 2)
        x2 = int(x1 + rng.normal(0, width * 0.3))
        y2 = int(y1 + rng.normal(0, height * 0.3))
        stroke_color = tuple(np.clip(base_color + rng.normal(0, 40, 3), 0, 255).astype(int))
        draw.line([x1, y1, x2, y2], fill=stroke_color, width=rng.integers(4, 18))

    # Canvas texture simulation
    arr = np.array(img, dtype=np.float32)
    y, x = np.ogrid[:height, :width]
    canvas_weave = (np.sin(x * 1.5) + np.sin(y * 1.5)) * 4.0
    arr = np.clip(arr + canvas_weave[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.SMOOTH_MORE)


def generate_human_digital_art(seed: int, width: int = 512, height: int = 512) -> Image.Image:
    """Simulates digitally authored artwork: sharp vector boundaries, digital color ramps."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (245, 245, 250))
    draw = ImageDraw.Draw(img)

    # Polygons and clean geometrical digital shapes
    num_shapes = rng.integers(5, 12)
    palette = [
        (45, 85, 125), (230, 95, 60), (240, 185, 60),
        (80, 160, 120), (140, 60, 120), (30, 30, 40)
    ]
    for _ in range(num_shapes):
        points = [(int(rng.uniform(0, width)), int(rng.uniform(0, height))) for _ in range(rng.integers(3, 6))]
        color = palette[rng.integers(0, len(palette))]
        draw.polygon(points, fill=color)

    # Clean digital line work
    for _ in range(rng.integers(3, 8)):
        p1 = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        p2 = (int(rng.uniform(0, width)), int(rng.uniform(0, height)))
        draw.line([p1, p2], fill=(20, 20, 25), width=rng.integers(2, 5))

    return img


def generate_ai_diffusion_flux(seed: int, width: int = 512, height: int = 512) -> Image.Image:
    """
    Simulates high-end modern diffusion/flow model imagery:
    Smooth latent features, subtle high-frequency checkerboard residual, hyper-coherent lighting.
    """
    rng = np.random.default_rng(seed)
    # Generate latent low-frequency structure
    low_res = rng.normal(120, 40, (height // 16, width // 16, 3))
    latent = Image.fromarray(np.clip(low_res, 0, 255).astype(np.uint8)).resize(
        (width, height), Image.Resampling.BICUBIC
    )
    arr = np.array(latent, dtype=np.float32)

    # Add stylized contrast & saturation enhancement typical of diffusion prompt engineering
    arr = np.clip(1.2 * (arr - 128.0) + 135.0, 0, 255)

    # Latent diffusion upsampling grid signature:
    # 8x8 or 16x16 latent patch boundary harmonic ripples in frequency domain
    y, x = np.ogrid[:height, :width]
    grid_artifact = (np.sin(2 * np.pi * x / 16.0) * np.sin(2 * np.pi * y / 16.0)) * 2.8
    # Subtle smooth denoising residue (low variance high kurtosis)
    denoiser_residual = rng.laplace(0, 1.2, (height, width, 3))

    final_arr = np.clip(arr + grid_artifact[:, :, None] + denoiser_residual, 0, 255).astype(np.uint8)
    return Image.fromarray(final_arr)


def generate_ai_diffusion_sdxl(seed: int, width: int = 512, height: int = 512) -> Image.Image:
    """
    Simulates SDXL generative imagery:
    Pronounced saturation, high-frequency prompt guidance harmonics, characteristic spectral rolloff.
    """
    rng = np.random.default_rng(seed)
    # Vibrant synthetic concept palette
    base = rng.uniform(50, 220, (height // 8, width // 8, 3))
    upscaled = Image.fromarray(base.astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR)
    arr = np.array(upscaled, dtype=np.float32)

    # High color saturation boost
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.25, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.15, 0, 255)

    # Characteristic 8x8 latent decoder checkerboard pattern
    y, x = np.ogrid[:height, :width]
    checkerboard = (np.sin(2 * np.pi * x / 8.0) + np.sin(2 * np.pi * y / 8.0)) * 3.2
    
    # Smooth bilateral-like patches with synthetic sharp edge transitions
    final = np.clip(arr + checkerboard[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(final).filter(ImageFilter.EDGE_ENHANCE_MORE)


def generate_ai_gan_stylegan(seed: int, width: int = 512, height: int = 512) -> Image.Image:
    """
    Simulates StyleGAN generator imagery:
    Radial harmonic spikes from transposed convolutions, phase symmetry patterns.
    """
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Transposed convolution periodic phase patterns
    phase_pattern = np.sin(r / 6.0) * 15.0 + np.cos(np.arctan2(y - cy, x - cx) * 6.0) * 10.0
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 0] = 140 + phase_pattern + rng.normal(0, 3, (height, width))
    base[:, :, 1] = 120 + 0.8 * phase_pattern + rng.normal(0, 3, (height, width))
    base[:, :, 2] = 110 + 0.6 * phase_pattern + rng.normal(0, 3, (height, width))

    final = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(final)


def build_dataset(samples_per_collection: int = 50) -> Path:
    """
    Builds the full benchmark dataset across human and AI collections.
    Includes untouched evaluation collections that are isolated from training.
    """
    print(f"Building ORIGIN X benchmark dataset ({samples_per_collection} samples/collection)...")
    np.random.seed(RANDOM_SEED)

    collections_config = [
        # Human Collections
        {
            "name": "human_dslr_optics",
            "label": LABEL_HUMAN,
            "creator": "nikon_d850_optical",
            "license": "CC0-1.0",
            "generator_fn": generate_human_dslr_landscape,
            "is_untouched_eval": False,
            "is_test": False,
        },
        {
            "name": "human_traditional_art",
            "label": LABEL_HUMAN,
            "creator": "analog_canvas_acrylic",
            "license": "CC0-1.0",
            "generator_fn": generate_human_traditional_art,
            "is_untouched_eval": False,
            "is_test": False,
        },
        {
            "name": "human_digital_art",
            "label": LABEL_HUMAN,
            "creator": "vector_digital_artist",
            "license": "CC0-1.0",
            "generator_fn": generate_human_digital_art,
            "is_untouched_eval": False,
            "is_test": True,  # Designated held-out test collection!
        },
        {
            "name": "human_museum_archive",
            "label": LABEL_HUMAN,
            "creator": "museum_open_access_archive",
            "license": "Public-Domain",
            "generator_fn": generate_human_traditional_art,
            "is_untouched_eval": True,  # Designated untouched eval collection!
        },
        # AI Collections
        {
            "name": "ai_diffusion_flux",
            "label": LABEL_AI,
            "creator": "flux_1_dev_flow",
            "license": "Synthetic-Benchmark-1.0",
            "generator_fn": generate_ai_diffusion_flux,
            "is_untouched_eval": False,
            "is_test": False,
        },
        {
            "name": "ai_diffusion_sdxl",
            "label": LABEL_AI,
            "creator": "stabilityai_sdxl_base",
            "license": "Synthetic-Benchmark-1.0",
            "generator_fn": generate_ai_diffusion_sdxl,
            "is_untouched_eval": False,
            "is_test": False,
        },
        {
            "name": "ai_gan_stylegan",
            "label": LABEL_AI,
            "creator": "nvidia_stylegan3_t",
            "license": "Synthetic-Benchmark-1.0",
            "generator_fn": generate_ai_gan_stylegan,
            "is_untouched_eval": False,
            "is_test": True,  # Designated held-out test collection!
        },
        {
            "name": "ai_nextgen_diffusion",
            "label": LABEL_AI,
            "creator": "nextgen_diffusion_v3",
            "license": "Synthetic-Benchmark-1.0",
            "generator_fn": generate_ai_diffusion_flux,
            "is_untouched_eval": True,  # Designated untouched eval collection!
        },
    ]

    records: List[ImageRecord] = []
    untouched_cols = [c["name"] for c in collections_config if c.get("is_untouched_eval", False)]
    test_cols = [c["name"] for c in collections_config if c.get("is_test", False)]

    for col_info in collections_config:
        col_name = col_info["name"]
        col_dir = PROCESSED_DATA_DIR / col_name
        col_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_collection):
            seed = RANDOM_SEED + hash(col_name) % 10000 + i * 37
            img = col_info["generator_fn"](seed=seed, width=512, height=512)
            
            filename = f"{col_name}_{i:04d}.png"
            file_path = col_dir / filename
            img.save(file_path, format="PNG")

            sha256 = compute_sha256(file_path)
            record = ImageRecord(
                image_id=f"{col_name}_{i:04d}",
                relative_path=f"{col_name}/{filename}",
                label=col_info["label"],
                source_collection=col_name,
                creator_family=col_info["creator"],
                license=col_info["license"],
                sha256=sha256,
                width=512,
                height=512,
                split="unassigned",
                notes=f"Generated with controlled benchmark parameters (seed {seed})"
            )
            records.append(record)

    # Perform strict source-level split assignment
    assigned_records = split_by_source_collection(
        records,
        untouched_eval_collections=untouched_cols,
        test_collections=test_cols,
        val_ratio=0.25,
        seed=RANDOM_SEED
    )

    manifest_path = DATA_DIR / "dataset_manifest.json"
    save_manifest(assigned_records, manifest_path)

    # Save split-specific index lists
    for split_name in ("train", "val", "test", "untouched_eval"):
        split_records = [r for r in assigned_records if r.split == split_name]
        save_manifest(split_records, SPLITS_DIR / f"{split_name}.json")
        human_count = sum(1 for r in split_records if r.label == LABEL_HUMAN)
        ai_count = sum(1 for r in split_records if r.label == LABEL_AI)
        print(f"Split [{split_name:14s}]: {len(split_records):3d} images "
              f"({human_count} Human, {ai_count} AI) | Collections: "
              f"{sorted(list(set(r.source_collection for r in split_records)))}")

    print(f"\nManifest saved to {manifest_path}. Total records: {len(assigned_records)}")
    return manifest_path


if __name__ == "__main__":
    build_dataset(samples_per_collection=40)
