"""
Image validation and security sanitization module.
Checks magic bytes, file bounds, color profiles, and detects corrupted uploads.
"""

import io
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageOps, ImageFile

from originx.config import (
    MAX_FILE_SIZE_BYTES,
    MIN_IMAGE_DIMENSION,
    MAX_IMAGE_DIMENSION,
    SUPPORTED_MIME_TYPES,
)

# Prevent decompression bomb DOS attacks while allowing reasonable resolutions
Image.MAX_IMAGE_PIXELS = 100_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Known magic bytes signatures
MAGIC_BYTES_MAP = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # RIFF....WEBP
}


class ImageValidationError(ValueError):
    """Raised when an uploaded file fails validation checks."""
    pass


def detect_mime_type(data: bytes) -> Optional[str]:
    """Inspects file header magic bytes to detect true MIME type."""
    if len(data) < 12:
        return None
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image_bytes(data: bytes) -> Dict[str, Any]:
    """
    Validates binary image data against security, integrity, and dimensional bounds.
    Returns sanitized metadata dictionary or raises ImageValidationError.
    """
    size_bytes = len(data)
    if size_bytes == 0:
        raise ImageValidationError("Uploaded image file is empty (0 bytes).")
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError(
            f"File size {size_bytes / (1024 * 1024):.2f} MB exceeds limit of "
            f"{MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB."
        )

    mime_type = detect_mime_type(data)
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ImageValidationError(
            f"Unsupported or unrecognized file format. Only JPEG, PNG, and WebP are supported."
        )

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception as exc:
        raise ImageValidationError(f"Image integrity check failed: file appears corrupted ({exc}).")

    # Re-open after verify() to inspect properties safely
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format
    except Exception as exc:
        raise ImageValidationError(f"Failed to read image dimensions or header: {exc}")

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are too small. "
            f"Minimum allowed is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}."
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed "
            f"{MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}."
        )

    return {
        "mime_type": mime_type,
        "format": format_name,
        "width": width,
        "height": height,
        "size_bytes": size_bytes,
        "original_mode": mode,
    }


def sanitize_to_rgb(img: Image.Image) -> Image.Image:
    """
    Converts any PIL image (RGBA, Grayscale, CMYK, Palette) into a clean,
    standardized 8-bit RGB image, stripping EXIF and applying orientation.
    """
    # Auto-orient based on EXIF before stripping
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode == "RGB":
        return img.copy()

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        # Composite transparent pixels over a clean neutral white background
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, rgba)
        return composite.convert("RGB")

    return img.convert("RGB")
