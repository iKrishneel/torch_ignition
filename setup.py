#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import setup
from setuptools import find_packages


# write all package dependencies here
install_requires = [
    'colorlog',
    'pillow',
    'numpy',
    'matplotlib',
    'pytorch-ignite',
    'torch',
    'torchvision',
    'tqdm',
    'tensorboard'
]

setup(
    name='torch_ignition',
    version='0.0.0',
    url='https://github.com/iKrishneel/torch_ignition.git',
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests'
)

