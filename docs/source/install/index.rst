Installation
============

Requirements
------------

Acoular runs under 64bit Windows and Linux (OS X untested).
In order to install Acoular you need a Python 2.7 environment running on your machine. 
Furthermore, the Acoular module requires the following Python Packages to be installed: NumPy, SciPy, Traits, scikit-learn, 
pytables, and Chaco. In order to run the examples you will also need the matplotlib and Mayavi Packages.

Option 1: Anaconda (Windows and Linux)
--------------------------------------

This option assumes that you have the `Anaconda <http://continuum.io/downloads>`_ Python-distribution installed on your computer. If this is not the case you may `download <http://continuum.io/downloads>`_ and install it (higly recommended). 

Once Anaconda Python is up and running you should use the command line ("Anaconda command line" on Windows, Terminal on Linux) to install acoular by typing

>>> conda install -c https://conda.binstar.org/acoular acoular

This will install Acoular in your Anaconda Python enviroment and make the Acoular library available from Python. In addition this will install all dependencies (those other packages mentioned above), if they are not already present on your system.
Depending on your type of Anaconda installation (single user or system-wide), you may be asked for admin privileges in order to start the installation process.

Option 2: Windows with other Python distributions
-------------------------------------------------
Download and run the `windows installer <https://pypi.python.org/pypi/acoular>`_. You will have to take care by yourself that all required packages are installed, otherwise you will not be able to use Acoular.

Option 3: Source install
------------------------
You may download the `source tarball <https://pypi.python.org/pypi/acoular>`_. Unzip it and change into the "acoular" directory, then type

>>> python setup.py install

to compile and install the software. This requires a properly set up system with all installed dependecies and a compiler.  
Another option to get the source is to clone or fork from `Github <https://github.com/acoular/acoular>`_.

    


