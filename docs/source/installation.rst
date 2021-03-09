.. _installation:

Installation
===========================


Before you can install the package, you need to have a suitable Python environment
installed. If you don't have Python, follow these instructions before attempting
to install PyConTurb.

.. toctree::
    :maxdepth: 2

    install_python
    

Requirements
--------------------------------

PyConTurb requires Python 3.6+. All other dependencies are installed during
installation.

Optional packages to, e.g., run the example notebooks or develop the package
can also be installed. See instructions below.


Normal user
--------------------------------

We generally recommend installing PyConTurb into its own environment. Instructions
are provided :ref:`above <install_python>`.

* Install the most recent, stable version of the code::
  
    pip install pyconturb

* Update an installation to the most recent version::

    pip install --upgrade pyconturb

* Install a specific version on PyPI::

   pip install pyconturb==2.6.3

* (Optional) Install packages to run the notebook examples on your machine::

    pip install jupyter matplotlib


Advanced user
--------------------------------

Please see the installation instructions in the
:ref:`Developer corner <developer_corner>`.

