import logging
import sys
import yaml
from pathlib import Path

# Load config to get log level/file
config_path = Path(__file__).parent.parent / "config" / "config.yaml"
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        log_cfg = config.get("logging", {})
        log_level = getattr(logging, log_cfg.get("level", "INFO"))
        log_format = log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
except Exception:
    log_level = logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicating handlers
    if not logger.handlers:
        formatter = logging.Formatter(log_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
