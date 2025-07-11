import logging
import os
from logging.handlers import RotatingFileHandler


def configure_root_logger():
    """Configure root logger with rotating file handler."""
    base_path = os.getenv("BASE_PATH", ".")
    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")
    if not logging.getLogger().handlers:
        handler = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name."""
    configure_root_logger()
    return logging.getLogger(name)
