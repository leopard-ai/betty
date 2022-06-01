import sys
import abc
import logging


_logger = None

def get_logger():
    """Get betty logger."""
    global _logger
    if _logger:
        return _logger
    logger = logging.getLogger('betty')
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    logger.propagate = False
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    _logger = logger
    return _logger


class LoggerBase:
    @abc.abstractmethod
    def log(self, stats, tag=None, step=None):
        raise NotImplementedError

    @staticmethod
    def debug(msg, *args, **kwargs):
        get_logger().debug(msg, *args, **kwargs)

    @staticmethod
    def info(msg, *args, **kwargs):
        get_logger().info(msg, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        get_logger().warning(msg, *args, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        get_logger().error(msg, *args, **kwargs)
