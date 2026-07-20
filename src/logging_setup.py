import sys

from loguru import logger

from src.config import Settings

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(sys.stdout, level=settings.log_level, format=_LOG_FORMAT, backtrace=False, diagnose=False)
