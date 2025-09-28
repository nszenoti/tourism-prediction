import logging

def get_logger(name):
    """Get logger with consistent formatting"""
    logger = logging.getLogger(name)

    if not logger.handlers:  # Avoid adding handlers multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
