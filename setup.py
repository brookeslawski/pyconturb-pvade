# -*- coding: utf-8 -*-
"""Setup file for ksec3d package

To install, cd to directory with the setup.py file and then
run following command in Anaconda prompt/terminal:
    pip install -e .
"""


from setuptools import setup

setup(name='ksec3d',
      version='0.1dev',
      description='Kaimal spectrum with 3D exponential coherence',
      url='https://gitlab.windenergy.dtu.dk/rink/ksec3d',
      author='Jenni Rinker',
      author_email='rink@dtu.dk',
      license='GNU GPL',
      packages=['ksec3d'],
      zip_safe=False)
