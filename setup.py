#! /usr/bin/env python

from setuptools import setup

setup(name='vecstack',
      version='0.3.0',
      description='Python package for stacking (machine learning technique)',
      long_description='Convenient way to automate OOF computation, prediction and bagging using any number of models',
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
      ],
      keywords=['stacking', 'blending', 'bagging', 'ensemble', 'ensembling', 'machine learning'],
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
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
