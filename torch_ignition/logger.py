#!/usr/bin/env python

import logging
import colorlog


def get_logger(name: str, level=logging.INFO):
    log_format = '%(log_color)s[%(levelname)s] %(name)s:%(message)s'
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(log_format))

    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
