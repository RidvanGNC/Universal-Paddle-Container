import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.api.errors import InvalidImageError


def decode_image_to_bgr(image_bytes: bytes) -> np.ndarray:
    """PIL-decodes arbitrary image bytes to an RGB numpy array, then reverses channel order to
    BGR - PaddleOCR/PaddleX/OpenCV-based pipelines expect BGR. Raises InvalidImageError on
    failure."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            array = np.array(rgb)
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Could not decode uploaded file as an image: {exc}") from exc

    return array[:, :, ::-1]


def encode_bgr_to_png(image: np.ndarray) -> bytes:
    """Inverse of decode_image_to_bgr: reverses BGR back to RGB and PNG-encodes it. Used by
    endpoints (e.g. document unwarping) whose PaddleX model returns a rectified image instead of
    structured data."""
    rgb = image[:, :, ::-1]
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()
