# Kaimal Spectrum with Exponential Coherence in 3D

This repository contains a package to simulate stochastic atmospheric 
turbulence for wind energy applications with a three-dimensional coherence
model.

## Background

The method prescribed in IEC 61400-1 Ed. 3 recommends only coherence in the
longitudinal direction, which is of course not accurate. This code presents
a way to expand that to include coherence between the lateral and vertical 
directions and also between different components.

## To install the code

Please note the code runs on Python 3.6.

Follow these instructions to install the package:
1. Clone the respository to your local machine
2. Navigate to the newly cloned folder in your terminal/Anaconda Prompt
3. `pip install -e .`