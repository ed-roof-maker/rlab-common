
********
RLab Common
********
A module containing modules that humanizes output, and that abstracts common tasks.

.. contents:: 
	rlab_common.py
	rlab_common_numpy_pandas.py
	rlab_common_scipy_ml.py


Installation
============
Install to local user bin under ~/.local/lib/python3.6, ~/.local/bin

.. code:: bash
	 user:work/]$ pip3 wheel --wheel-dir=wheel $(realpath src)

.. code:: bash
	 user:work/]$ pip3 install --user --no-index --find-links=wheel rlab_common
	 user:work/]$ pip3 --list



Usage
=====
See examples/*
No examples yet. TBD.
	


Use Pip Without Internet
========================
.. code:: bash
	$ pip3 install --no-index --find-links=<Custom Package Dir> <Package Name>


Setup environment variables in bashrc so you don't have to provide extra command line arguments.
.. code:: bash
	# Save these 2 variables in your profile 
	$ export PIP_NO_INDEX=true
	$ export PIP_FIND_LINKS=<Custom Package Dir>
	
	# Then run pip as usual
	$ pip install <package-name>



