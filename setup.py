#! /usr/bin/env python

from setuptools import setup

setup(name='vecstack',
      version='0.1',
      description='Python package for stacking (machine learning technique)',
      url='https://github.com/vecxoz/vecstack',
      author='vecxoz',
      author_email='vecxoz@gmail.com',
      license='MIT',
      packages=['vecstack'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn'
      ],
      zip_safe=False)
