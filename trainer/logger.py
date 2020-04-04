import logging
import sys


def get_std_out_logger(logging_level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(filename)s:%(lineno)d]'
    ))
    logger = logging.Logger('stdout_logger')
    logger.addHandler(handler)
    return logger

def get_logger():
    return get_std_out_logger(logging_level=logging.INFO)