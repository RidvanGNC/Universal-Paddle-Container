from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "paddle-ocr-api"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    model_files_dir: str = "model_files"

    device_preference: Literal["auto", "cpu", "gpu"] = "auto"
    fail_fast_on_gpu_mismatch: bool = True
    strict_model_loading: bool = False

    det_model_dir_name: str | None = None
    det_model_name: str | None = None
    rec_model_dir_name: str | None = None
    rec_model_name: str | None = None
    doc_orientation_model_dir_name: str | None = None
    doc_orientation_model_name: str | None = None

    table_cell_wired_model_dir_name: str | None = None
    table_cell_wired_model_name: str | None = None
    table_cell_wireless_model_dir_name: str | None = None
    table_cell_wireless_model_name: str | None = None
    table_cell_detection_threshold: float = 0.3

    inference_queue_max_size: int = 50
    inference_timeout_seconds: float = 30.0

    max_upload_size_mb: int = 10
    allowed_content_types: list[str] = ["image/png", "image/jpeg", "image/webp", "image/bmp"]

    security_mode: Literal["none", "api_key"] = "none"
    api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
