# -*- coding: utf-8 -*-
"""Setup file for PyConTurb

See README.md for how to use this file.
"""
from os import path
from setuptools import setup


def get_version():
    """Get version number from text file"""
    lines = open('./pyconturb/_version.py').readlines()
    version = [l for l in lines if '__version__' in l][0].split()[-1]
    return version.strip('"').strip("'")


def load_readme():
    """Load readme to put into pypi long description"""
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


setup(name='pyconturb',
      version=get_version(),
      description='An open-source constrained turbulence generator',
      long_description=load_readme(),
      long_description_content_type='text/markdown',
      url='https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb',
      author='Jenni Rinker',
      author_email='rink@dtu.dk',
      license='MIT',
      packages=['pyconturb',  # top-level package
                ],
      install_requires=['h5py',  # load coherence array from file
                        'numpy',  # numberic arrays
                        'pandas',  # column-labelled arrays
                        'scipy',  # interpolating profile functions
                        ],
      zip_safe=False)
