Installation
============

Depending on your operating system and python distribution, there are different options (see below) how to install Acoular.

Requirements
------------

Acoular runs under 64bit Windows and Linux (it should also run under OS X, but this untested).
In order to install Acoular you need a Python 2.7 environment running on your machine. 
Furthermore, Acoular requires the following Python Packages to be installed: 
    * `NumPy <http://www.numpy.org/>`_, 
    * `SciPy <http://www.scipy.org/scipylib/index.html>`_, 
    * `Traits <http://code.enthought.com/projects/traits/>`_, 
    * `scikit-learn <http://scikit-learn.org/stable/>`_ (aka sklearn),
    * `pytables <http://www.pytables.org/>`_ (aka tables),
    * `Chaco <http://code.enthought.com/projects/chaco/>`_. 
If you do not chose option 1 below for installation, you may have to take care that these are installed on your machine before you can install Acoular.

In order to run the demo and examples you will also need the `matplotlib <http://matplotlib.org>`_ package and for one example the `Mayavi <http://docs.enthought.com/mayavi/mayavi/>`_ package.


Option 1 (recommended): Anaconda (Windows and Linux)
--------------------------------------

This option assumes that you have the `Anaconda <http://continuum.io/downloads>`_ Python-distribution installed on your computer. If this is not the case you may `download <http://continuum.io/downloads>`_ and install it (higly recommended). You may install Anaconda alongside any other Python installation on your system, which is an advantage especially with 

Once Anaconda Python is up and running, start a console, i.e. either "cmd" or the "Anaconda command prompt" on Windows, Terminal on Linux.
In the command line, type

>>> conda install -c https://conda.binstar.org/acoular acoular

This will install Acoular in your Anaconda Python enviroment and make the Acoular library available from Python. In addition, this will install all dependencies (those other packages mentioned above) if they are not already present on your system.
Depending on your type of Anaconda installation (single user or system-wide), you may be asked for admin privileges in order to start the installation process.

If the installation has finished, you may try out Acoular by typing

>>> acoular_demo.py

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).

Option 2: Windows with other Python distributions
-------------------------------------------------
Download and run the `windows installer <https://pypi.python.org/pypi/acoular>`_. You will have to take care by yourself that all required packages are installed, otherwise you will not be able to use Acoular.

If the installation has finished, you may try out Acoular by typing

>>> acoular_demo.py

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).

Option 3: pip
-------------
If you have pip installed on your system, you may type

>>> pip install acoular

to install acoular. Note that the required packages are not installed this way and you have to do it by yourself.

If the installation has finished, you may try out Acoular by typing

>>> acoular_demo.py

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).


Option 4: Source install
------------------------
You may download the `source tarball <https://pypi.python.org/pypi/acoular>`_. Unzip it and change into the "acoular" directory, then type

>>> python setup.py install

to compile and install the software. This requires a properly set up system with all installed dependencies and a compiler.  
Another option to get the source is to clone or fork from `Github <https://github.com/acoular/acoular>`_.

If the installation has finished, you may try out Acoular by typing

>>> acoular_demo.py

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).
    


