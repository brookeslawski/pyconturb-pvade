.. _getting_started:

Getting started
===========================

What is PyConturb?
--------------------------------

PyConTurb is an open-source Python package intended to simulate turbulence
boxes constrained to provided time series. The theoretical background is
provided in :ref:`research`. Note that PyConturb can be used without
constraints, in which case it is a standard turbulence simulator that uses
the Kaimal spectrum with exponential coherence.

The package is developed and maintained by Jenni Rinker, Ph.D. (DTU Wind
Energy).


Your first simulations
--------------------------------

I recommend that you read this documentation thoroughly to familiarize 
yourself with how the code is intended to work. Be especially sure to focus
on the provided examples.

PyConTurb can be used in three main ways:  

1. Unconstrained simulation with default IEC 61400-1 parameters.  
2. Unconstrained simulation with custom functions for the mean wind speed,
   turbulence standard deviation and/or turbulence spectrum as function of
   spatial location and/or turbulence component.  
3. Simulation constrained on measured time series (e.g., met mast data).

Regardless of which method you use, you will first need to specify the
spatial locations and turbulence components of the box you want to simulate.
Look at :ref:`examples` to get ideas on how to do this.

If you are constraining against time series, you will need to transform your
data into the correct format for PyConTurb before you can feed it to
``gen_turb``. Look at the constraining data in :ref:`examples` to check the
format. Currently, PyConTurb only supports constraining time series in the u,
v and/or w directions (i.e., no line-of-sight lidar measurements...yet).

**Note**: Simulating a turbulence component where you do not have a
constraint can have some unintended side effects. See below for more discussion.

If you want to model a custom spatial variation of the mean wind speed, 
turbulence standard deviation and/or turbulence spectrum (e.g., model a wake
deficit in the y-z plane), you will need to define the appropriate
``wsp_func``, ``sig_func`` and/or ``spec_func`` functions. Look at the related
:ref:`common_functions` sections (namely, :ref:`profiles`, :ref:`sig_models` and
:ref:`spectral_models`).
