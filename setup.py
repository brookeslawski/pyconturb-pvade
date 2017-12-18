# -*- coding: utf-8 -*-
"""Setup file for PyConTurb

To install, cd to directory with the setup.py file and then
run following command in Anaconda prompt/terminal:
    pip install -e .
"""


from setuptools import setup

setup(name='pyconturb',
      version='0.1',
      description='An open-source constrained turbulence generator',
      url='https://gitlab.windenergy.dtu.dk/rink/pyconturb',
      author='Jenni Rinker',
      author_email='rink@dtu.dk',
      license='MIT',
      packages=['pyconturb'],
      zip_safe=False)
