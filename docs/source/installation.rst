.. _installation:


Installation
===========================

Before you can install the package, you need to have a suitable Python package
installed. If you don't have Python, follow these instructions before attempting
to install PyConTurb.

.. toctree::
    :maxdepth: 2

    install_python
    

Requirements
--------------------------------

PyConTurb requires Python 3.6+, so be sure that you are installing it into
a suitable environment (see above). For simple users, all dependencies will be
installed with PyConTurb, except for two optional packages for running the
examples and plotting. For developers, use the ``dev_reqs.txt`` requirements file
to install the optional dependencies related to testing and building documentation.


Normal user
--------------------------------

* To run the notebook examples on your machine (optional)::

    pip install jupyter matplotlib

* Install most recent official release::
  
    pip install git+https://gitlab.windenergy.dtu.dk/rink/pyconturb.git@latest

* Install most recent unofficial version::
  
    pip install git+https://gitlab.windenergy.dtu.dk/rink/pyconturb.git

* Pull any new changes to the unofficial version::

    pip install --upgrade git+https://gitlab.windenergy.dtu.dk/rink/pyconturb.git


Developer
------------------------------

We highly recommend developers install PyConTurb into its own environment
(instructions above). The commands to clone and install PyConTurb with developer
options into the current active environment in an Anaconda Prompt are as
follows::

   git clone https://gitlab.windenergy.dtu.dk/rink/pyconturb.git
   cd PyConTurb
   pip install -r dev_reqs.txt
   pip install -e .

