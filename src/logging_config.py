import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging settings.

    Args:
        level: Logging level to use.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
