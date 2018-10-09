#!/usr/bin/env python3

"""
setup.py

Metadata of simplegcn package. Defines version info and dependencies.

Will Badart <badart_william (at) bah (dot) com>
created: OCT 2018
"""

from setuptools import find_packages, setup
from yaml import load as load_yml


PATH_CONDA_ENV = './.conda.env.yml'

with open(PATH_CONDA_ENV) as fs:
    REQUIREMENTS = load_yml(fs)['dependencies']

setup(name='simplegcn',
      version='0.0.1-alpha',
      packages=find_packages(),

      package_data={'': ['*.md']},
      install_requires=REQUIREMENTS)
