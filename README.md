[![pipeline status](https://gitlab.windenergy.dtu.dk/rink/pyconturb/badges/master/pipeline.svg)](https://gitlab.windenergy.dtu.dk/rink/pyconturb/commits/master)
[![coverage report](https://gitlab.windenergy.dtu.dk/rink/pyconturb/badges/master/coverage.svg)](https://gitlab.windenergy.dtu.dk/rink/pyconturb/commits/master)


# PyConTurb: Constrained Stochastic Turbulence for Wind Energy Applications

This Python package uses a novel method to generate stochastic turbulence boxes
that are constrained by one or more measured time series. A paper detailing
the simulation methodology will be submitted to Wind Energy.

## To install the code

Please note the code only runs on Python 3.6 or higher.

Simple installation of v1.0 (non-editable source code):  
   `pip install git+https://gitlab.windenergy.dtu.dk/rink/pyconturb.git@v1.0`

Developer install:  
```
git clone https://gitlab.windenergy.dtu.dk/rink/pyconturb.git
pip install pytest pytest-cov
cd pyconturb
pip install -e .

```


## Constrained or unconstrained turbulence

The main function, `gen_turb` can be used with or without constraining time
series.
 
## Examples

There is an example Jupyter notebook in the examples folder.

To run the notebook, do the following:
1. Open an Anaconda prompt.
2. Navigate to the folder with the notebook.
3. Enter `jupyter notebook` into the prompt.
4. When your internet browser opens up, click on the notebook. It should open
in a new tab.