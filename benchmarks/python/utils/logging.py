import logging
import os
import sys

from loguru import logger

# Suppress passlib bcrypt version warnings
logging.getLogger("passlib").setLevel(logging.ERROR)


def configure_logging():
    """
    Configure Loguru with industry-standard settings.
    """

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    JSON_LOGS = os.getenv("JSON_LOGS", "false") == "true"

    # Configure logger format
    string_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Remove all existing handlers first
    logger.remove()

    # Add console handler (enqueue=True for multiprocess safety)
    if JSON_LOGS:
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            serialize=True,
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )
    else:
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            format=string_format,
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

    # Add file handler for errors and above
    if JSON_LOGS:
        logger.add(
            "logs/error.log",
            level="ERROR",
            serialize=True,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )
    else:
        logger.add(
            "logs/error.log",
            level="ERROR",
            format=string_format,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )

    # Add file handler for all logs
    if JSON_LOGS:
        logger.add(
            "logs/info.log",
            level=LOG_LEVEL,
            serialize=True,
            rotation="100 MB",
            retention="1 week",
            compression="zip",
            enqueue=True,
        )
    else:
        logger.add(
            "logs/info.log",
            level=LOG_LEVEL,
            format=string_format,
            rotation="100 MB",
            retention="1 week",
            compression="zip",
            enqueue=True,
        )


# Initialise and configure logger
configure_logging()


def get_logger(name: str | None = None, level: str | None = None):
    """
    Returns a configured Loguru logger instance.

    Args:
        name (str | None): Optional name to bind to the logger context.
        level (str | None): Optional level to set for this specific logger.

    Returns:
        Configured Loguru logger instance.
    """
    if name:
        bound_logger = logger.bind(name=name)
        return bound_logger.level(level) if level else bound_logger
    return logger
