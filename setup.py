#! /usr/bin/env python

from setuptools import setup

setup(name='vecstack',
      version='0.2',
      description='Python package for stacking (machine learning technique)',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
      ],
      keywords='machine learning, ensemble, ensembling, bagging, stacking, blending',
      url='https://github.com/vecxoz/vecstack',
      author='Igor Ivanov',
      author_email='vecxoz@gmail.com',
      license='MIT',
      packages=['vecstack'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn>=0.18'
      ],
      zip_safe=False)
