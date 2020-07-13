#!/usr/bin/env python

import logging
from torch_ignition.logger import get_logger


class Config(object):

    INPUT_SHAPE: list = [3, 224, 224]

    BATCH_SIZE: int = 1

    EPOCHS: int = 200
    

    
    def __init__(self):
        self.__logger = get_logger(self.__class__.__name__)
    
    def display(self):
        """Display configuration values
        Prints all configuration values to the cout.
        """
        self.__logger.info("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                self.__logger.info("{:30} {}".format(a, getattr(self, a)))
        self.__logger.info("\n")

c = Config()
import IPython
IPython.embed()
