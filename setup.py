#!/usr/bin/env python3

"""
setup.py

Metadata of simplegcn package. Defines version info and dependencies.

Will Badart <badart_william (at) bah (dot) com>
created: OCT 2018
"""

from setuptools import find_packages, setup


setup(name='simplegcn',
      version='0.0.1-alpha',
      packages=find_packages(),
      package_data={'': ['*.md']})
