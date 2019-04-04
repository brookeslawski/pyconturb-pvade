# -*- coding: utf-8 -*-
"""Setup file for PyConTurb

Non-developer install v1.0. Run following command in Anaconda prompt/terminal:
    pip install git+https://gitlab.windenergy.dtu.dk/rink/pyconturb.git@v1.0

Developer install most updated version (editable installation).
    git clone https://gitlab.windenergy.dtu.dk/rink/pyconturb.git
    pip install pytest
    cd pyconturb
    pip install -e .
"""


from setuptools import setup

setup(name='pyconturb',
      version='1.0',
      description='An open-source constrained turbulence generator',
      url='https://gitlab.windenergy.dtu.dk/rink/pyconturb',
      author='Jenni Rinker',
      author_email='rink@dtu.dk',
      license='MIT',
      packages=['pyconturb',  # top-level package
                'pyconturb.core',  # main functions
                'pyconturb.io',  # file io
                ],
      install_requires=['numpy',
                        'pandas',
                        ],
      zip_safe=False)
