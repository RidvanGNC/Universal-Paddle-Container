import io
import contextlib
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from src.config import Settings


class HardwareError(RuntimeError):
    """Raised when the requested device (cpu/gpu) cannot actually be satisfied."""


@dataclass
class HardwareInfo:
    compiled_with_cuda: bool
    cuda_device_count: int
    device_name: str | None
    using_device: Literal["cpu", "gpu"]
    paddle_device_string: str  # "cpu" or "gpu:0" - fed directly into PaddleOCR(device=...)


def _is_compiled_with_cuda() -> bool:
    import paddle

    try:
        return bool(paddle.device.is_compiled_with_cuda())
    except Exception as exc:  # pragma: no cover - defensive, paddle should always expose this
        logger.warning(f"paddle.device.is_compiled_with_cuda() raised: {exc!r}")
        return False


def _cuda_device_count() -> int:
    import paddle

    try:
        return int(paddle.device.cuda.device_count())
    except Exception as exc:
        logger.warning(f"paddle.device.cuda.device_count() raised: {exc!r}")
        return 0


def run_paddle_check() -> bool:
    """Runs paddle.utils.run_check(), capturing its stdout into the log instead of letting
    it print raw to the console. Returns True if it completed without raising."""
    import paddle

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            paddle.utils.run_check()
        for line in buffer.getvalue().splitlines():
            if line.strip():
                logger.info(f"[paddle.utils.run_check] {line}")
        return True
    except Exception as exc:
        for line in buffer.getvalue().splitlines():
            if line.strip():
                logger.warning(f"[paddle.utils.run_check] {line}")
        logger.warning(f"paddle.utils.run_check() failed: {exc!r}")
        return False


def resolve_device(settings: Settings) -> HardwareInfo:
    """Determines which device PaddleOCR should run on, based on settings.device_preference
    and what's actually available. Logs explicit diagnostics either way so a CUDA-detection
    mismatch is visible in container logs immediately rather than manifesting as silently
    slow/incorrect inference (see PaddlePaddle/Paddle#67982, #69218)."""
    compiled_with_cuda = _is_compiled_with_cuda()
    device_count = _cuda_device_count() if compiled_with_cuda else 0

    logger.info(
        f"Hardware probe: compiled_with_cuda={compiled_with_cuda}, "
        f"cuda_device_count={device_count}, device_preference={settings.device_preference!r}"
    )

    gpu_available = compiled_with_cuda and device_count > 0

    if settings.device_preference == "cpu":
        using_device: Literal["cpu", "gpu"] = "cpu"
    elif settings.device_preference == "gpu":
        if not gpu_available:
            message = (
                "device_preference='gpu' was requested but no usable GPU was found "
                f"(compiled_with_cuda={compiled_with_cuda}, cuda_device_count={device_count})."
            )
            if settings.fail_fast_on_gpu_mismatch:
                logger.error(message + " Failing fast (fail_fast_on_gpu_mismatch=true).")
                raise HardwareError(message)
            logger.warning(message + " Falling back to CPU (fail_fast_on_gpu_mismatch=false).")
            using_device = "cpu"
        else:
            using_device = "gpu"
    else:  # "auto"
        using_device = "gpu" if gpu_available else "cpu"

    if using_device == "gpu":
        run_paddle_check()

    device_name = None
    if using_device == "gpu":
        try:
            import paddle

            device_name = paddle.device.cuda.get_device_name(0)
        except Exception as exc:
            logger.warning(f"Could not resolve GPU device name: {exc!r}")

    info = HardwareInfo(
        compiled_with_cuda=compiled_with_cuda,
        cuda_device_count=device_count,
        device_name=device_name,
        using_device=using_device,
        paddle_device_string="gpu:0" if using_device == "gpu" else "cpu",
    )
    logger.info(f"Resolved hardware: {info}")
    return info
