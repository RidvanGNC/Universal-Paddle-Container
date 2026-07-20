import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Empty model_files dir + no configured model names -> engine.load() degrades gracefully
    # instead of trying to import/construct a real PaddleOCR pipeline.
    monkeypatch.setenv("MODEL_FILES_DIR", str(tmp_path))
    monkeypatch.setenv("STRICT_MODEL_LOADING", "false")
    monkeypatch.setenv("DEVICE_PREFERENCE", "cpu")

    from src.config import get_settings

    get_settings.cache_clear()

    from src.main import app

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()
