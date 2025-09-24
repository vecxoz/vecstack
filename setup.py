#! /usr/bin/env python

from setuptools import setup

long_desc = '''
Python package for stacking (stacked generalization) featuring lightweight functional API and fully compatible scikit-learn API.
Convenient way to automate OOF computation, prediction and bagging using any number of models.
All details, FAQ, and tutorials: https://github.com/vecxoz/vecstack
'''

setup(name='vecstack',
      version='0.5.0',
      description='Python package for stacking (machine learning technique)',
      long_description=long_desc,
      classifiers=[
          'License :: OSI Approved :: MIT License',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
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
      extras_require={
          'test': [
              'pytest',
              'pytest-cov',
              'pandas',
              'pyarrow'
          ]
      },
      zip_safe=False)
