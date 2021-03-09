.. _developer_corner:


Developer corner
==============================

Interested in adding new functions or making PyConTurb even better? Great to
have you! This page is a resource to get you coding and contributing in no
time.


Developer install
------------------

We highly recommend developers install PyConTurb into its own environment
to prevent installation conflicts. Instructions are provided
:ref:`here <install_python>`. The commands to clone and install
PyConTurb with developer options into the current active environment in an
Anaconda Prompt are as follows::

   git clone https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git
   cd PyConTurb
   pip install -r dev_reqs.txt
   pip install -e .

If you plan on writing/running tests, running Jupyter notebooks, or 


Versioning
------------------------------

This project follows `Semantic Versioning (2.0) <https://semver.org/spec/v2.0.0.html>`_.


Submitting merge requests
-------------------------------

PyConTurb is an open-source, community-based project. Therefore, merge requests
with bug fixes or implemented feature requests are welcome! To ensure that
your merge request is reviewed and accepted as quickly as possible, please
be sure to write a test for your bug fix and be sure to adhere to PEP8
conventions. Submitting a merge request from a branch in your own fork of
the repo is preferred, but you may also work in a branch on this repo.

Here is a quick, step-by-step guide for fixing a bug:

#. If the bug is not reported in the `issue tracker <https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb/issues>`_,
#. Make sure your master branch is updated (``git pull origin master``).  
#. Switch to a new branch (``git checkout -b branch_name``).  
#. Write a test for the bug (it should fail) in the relevant
   ``pyconturb/tests/`` file(s). See other test files to figure out how
   to do this.  
#. Write code to fix the bug. Be sure to follow PEP8 conventions!  
#. Run the tests locally to make sure they all pass (from repo,
   ``python -m pytest pyconturb/tests``).  
#. Add and commit your related files using git, if you haven't already.  
#. Push your feature branch to GitLab (``git push origin branch_name``).  
#. Create a merge request in GitLab from your feature branch and assign to
   ``@rink``.  
#. Make any suggested changes based on the code review, then push those
   changes.  
#. After the merge request is accepted, pull the new changes to master
   (``git checkout master``, then ``git pull origin master``).

