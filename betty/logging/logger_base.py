import sys
import abc
import logging


_logger = None


def get_logger():
    """
    Get global logger.
    """
    global _logger
    if _logger:
        return _logger
    logger = logging.getLogger("betty")
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
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
    def log(self, stats, tag=None, step=None):
        """
        Log metrics/stats to a visualization logger (e.g. tensorboard, wandb)

        :param stats: Dictoinary of values and their names to be recorded
        :type stats: dict
        :param tag:  Data identifier
        :type tag: str, optional
        :param step: step value associated with ``stats`` to record
        :type step: int, optional
        """
        return

    @staticmethod
    def debug(msg, *args, **kwargs):
        """
        Logs a message with level DEBUG on the global logger

        :param msg: debugg message
        :type msg: str
        """
        get_logger().debug(msg, *args, **kwargs)

    @staticmethod
    def info(msg, *args, **kwargs):
        """
        Logs a message with level INFO on the global logger

        :param msg: info message
        :type msg: str
        """
        get_logger().info(msg, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        """
        Logs a message with level WARNING on the global logger

        :param msg: warning message
        :type msg: str
        """
        get_logger().warning(msg, *args, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        """
        Logs a message with level ERROR on the global logger

        :param msg: error message
        :type msg: str
        """
        get_logger().error(msg, *args, **kwargs)
