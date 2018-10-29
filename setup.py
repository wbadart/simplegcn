#!/usr/bin/env python3

"""
setup.py

Package config for siplegcn.

Will Badart
created: OCT 2018
"""

from setuptools import find_packages, setup


setup(name='simplegcn',
      version='0.0.1a',
      license='BSD3',
      author='Will Badart',
      packages=find_packages(),

      install_requires=[
          'numpy',
      ],
     )
