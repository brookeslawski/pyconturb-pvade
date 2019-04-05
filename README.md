[![pipeline status](https://gitlab.windenergy.dtu.dk/rink/pyconturb/badges/master/pipeline.svg)](https://gitlab.windenergy.dtu.dk/rink/pyconturb/commits/master)
[![coverage report](https://gitlab.windenergy.dtu.dk/rink/pyconturb/badges/master/coverage.svg)](https://gitlab.windenergy.dtu.dk/rink/pyconturb/commits/master)


# PyConTurb: Constrained Stochastic Turbulence for Wind Energy Applications

This Python package uses a novel method to generate stochastic turbulence boxes
that are constrained by one or more measured time series. Details on the theory
can be found in [this paper from Torque 2016](https://iopscience.iop.org/article/10.1088/1742-6596/1037/6/062032/meta).

Despite the package's name, the main function, `gen_turb` can be used with or
without constraining time series. Without the constraining time series, it is
the Veers simulation method.

## Something wrong/missing?

If you find an issue with the code or you'd like a new feature, please submit an
issue through our [issue tracker](https://gitlab.windenergy.dtu.dk/rink/pyconturb/issues). 
**NOTE!** Please use the Bug template for the issue when submitting bug reports.

## Installation

Please note the code only runs on Python 3.6 or higher.

Simple installation of v1.0 (non-editable source code, not most updated version):  
   `pip install git+https://gitlab.windenergy.dtu.dk/rink/pyconturb.git@v1.0`  
If you want to run the examples, you will also need Jupyter and matplotlib:  
    `pip install jupyter matplotlib`

Developer install:  
```
git clone https://gitlab.windenergy.dtu.dk/rink/pyconturb.git
pip install pytest pytest-cov jupyter matplotlib
cd pyconturb
pip install -e .

```
 
## Examples

There is an example Jupyter notebook in the examples folder. To run the
notebook, do the following:
1. Open an Anaconda prompt.
2. Navigate to the folder with the notebook.
3. Enter `jupyter notebook` into the prompt.
4. When your internet browser opens up, click on the notebook. It should open
in a new tab.