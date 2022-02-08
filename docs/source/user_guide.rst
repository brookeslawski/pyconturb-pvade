.. _user_guide:


User guide
==============================

Here is some information on how to improve your experience with
PyConTurb.


Enhancing performance
------------------------------

There are a few ways to reduce the memory impact and increase
the speed of your computations.

To reduce memory when calling ``gen_turb``, you could try:  

  1. Using ``np.float32`` instead of the default 64-bit float.  
  2. Reducing ``nf_chunk``.  
  3. Simulating the u, v, and w components separately (if using
     the exponential coherence model from IEC 61400-1).  

To increase speed when calling ``gen_turb``, you could try:  

  1. Simulating the u, v, and w components separately (if using
     the exponential coherence model from IEC 61400-1).  
  2. Pre-generating the coherence files and loading them on
     simulation (if simulating the same geometry multiple times,
     e.g., with different constraining time series or different
     random seeds).  
  3. Increasing ``nf_chunk``.  


Common mistakes
--------------------------------

Here is a non-exhaustive list of common issues when simulating with PyConTurb.

* **Simulating component where there is no constraint**.
  Because there is no correlation between the three turbulence components
  (i.e., u is not correlated to v or w), PyConTurb will default to IEC
  parameters for the components without a constraint. You can get around
  this default behaviour by defining a custom ``sig_func`` (see examples).

* **Mismatching time steps in constraint and turbulence box**.
  Due to the constrained-turbulence methodology, PyConTurb requires the time
  steps of the desired simulated turbulence to match the time steps of the
  constraining time series.

* **Using `interp_data` without understanding interpolator**.
  PyConTurb has the option to attempt to interpolate and of the three
  profile functions from the constraining time series. This option is provided
  for convenience, but using it without understanding how the interpolator
  works can produce unexpected results. In general, I recommend fitting
  custom profile functions to your data manually instead of using the
  interpolator.

* **Trying to simulate very large boxes**.
  It is easy to run out of memory if you request a box that is very large. To
  save memory, you can try setting ``nf_chunks`` to 1 (will slow down simulation)
  or ``dtype=np.float32``.


.. _issues:

Bugs and feature requests
------------------------------

This package is a labor of love, and it has by no means achieved perfection.
I have written many tests to ensure that the code is acting as expected, but
bugs may still be hiding! I am relying on you, the user, to report and/or
fix any bugs that you find in the code. Suggestions for improvement are also
welcome.

**Please do not email me directly** for bug reports or feature requests. 
Instead, use GitLab's
`issue tracker <https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb/issues>`_.
Note that I am unfortunately unable to provide support for issues related to
incorrect usage.

**Bugs**

If PyConTurb is not behaving as you think it should, 
please submit an issue using GitLab's
`issue tracker <https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb/issues>`_.
**NOTE!** Please be sure to use the "Bug" template!


**Feature requests**

If you would like PyConTurb to have a specific feature that it currently
doesn't have, you have two options:

  1. Submit a Feature Request via GitLab's
     `issue tracker <https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb/issues>`_.
     Be sure to use the "Feature request" template!
  2. Implement the feature in a new branch, and submit a merge request.
     (See instructions in the :ref:`Developer corner <developer_corner>`.)

Because PyConTurb is not under heavy development, priority will be given to
merge requests. A new feature request is only likely to be implemented if it is
needed in a separate project. Note that if you author a merge request and it is
accepted, you will be added to the list of contributors (unless you wish
otherwise).
