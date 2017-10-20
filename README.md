# Constrained Turbulence Simulations based on the KSEC Method

This repository contains a package to simulate constrained stochastic
atmospheric turbulence for wind energy applications with any desired
three-dimensional (3D) spectral or coherence model.

## To install the code

Please note the code runs on Python 3.6.

Follow these instructions to install the package:
1. Clone the respository to your local machine
2. Navigate to the newly cloned folder in your terminal/Anaconda Prompt
3. `pip install -e . `


## Background

There are two main contributions of the code in this repository:
1. Easy and rapid simulation of constrained turbulence
2. Ability to simulate with custom, 3D spectral and coherence models.

### Constrained turbulence

The main function, `gen_turb` can be used with or without constraining time
series.

### 3D coherence

The method prescribed in IEC 61400-1 Ed. 3 recommends only coherence in the
longitudinal direction, which is of course not accurate. This code presents
a way to expand that to include coherence between the lateral and vertical 
directions and also between different components.
 
## Examples

There is an example Jupyter notebook in the examples folder.

To run the notebook, do the following:
1. Open an Anaconda prompt.
2. Navigate to the folder with the notebook.
3. Enter `jupyter notebook` into the prompt.
4. When your internet browser opens up, click on the notebook. It should open
in a new tab.