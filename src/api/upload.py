from fastapi import UploadFile

from src.api.errors import InvalidImageError
from src.config import Settings


async def read_validated_image(file: UploadFile, settings: Settings) -> bytes:
    if file.content_type not in settings.allowed_content_types:
        raise InvalidImageError(
            f"Unsupported content type {file.content_type!r}; allowed: {settings.allowed_content_types}"
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    image_bytes = await file.read(max_bytes + 1)
    if len(image_bytes) > max_bytes:
        raise InvalidImageError(f"Upload exceeds max_upload_size_mb={settings.max_upload_size_mb}.")
    if not image_bytes:
        raise InvalidImageError("Uploaded file is empty.")

    return image_bytes
