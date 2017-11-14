Installation
============

Depending on your operating system and python distribution, there are different options (see below) how to install Acoular.

Requirements
------------

Acoular runs under 64bit Windows and Linux (it should also run under OS X, but this is untested).
In order to install Acoular you need a Python 2.7 or Python 3 environment running on your machine. 
Furthermore, Acoular requires the following Python Packages: 
    * `NumPy <http://www.numpy.org/>`_, 
    * `SciPy <http://www.scipy.org/scipylib/index.html>`_, 
    * `Traits <http://code.enthought.com/projects/traits/>`_, 
    * `scikit-learn <http://scikit-learn.org/stable/>`_ (aka sklearn),
    * `pytables <http://www.pytables.org/>`_ (aka tables),
    * `numba <https://numba.pydata.org/>`_, 
    * `PyQt <https://www.riverbankcomputing.com/software/pyqt/intro>`_, 
    * `traitsui <http://docs.enthought.com/traitsui/>`_,  
    * `libpython <https://anaconda.org/anaconda/libpython>`_ (only for Windows),
    * `configparser <https://docs.python.org/2/library/configparser.html>`_ (only for Python 2.7), 
    * `six <https://pythonhosted.org/six/>`_ (only for Python 3). 

If you do not chose option 1 below for installation, you may have to take care that these are all installed on your machine before you can install Acoular.

In order to run the demo and examples you will also need: 

    * `matplotlib <http://matplotlib.org>`_.

and for the example 3:

    * `Mayavi <http://docs.enthought.com/mayavi/mayavi/>`_.


Option 1 (recommended): Anaconda (Windows and Linux)
----------------------------------------------------

This option assumes that you have the `Anaconda <https://www.anaconda.com/download/>`_ Python-distribution installed on your computer. If this is not the case you may `download <https://www.anaconda.com/download/>`_ and install it (higly recommended). **You may install Anaconda alongside any other Python installation on your system**, without the need to interfere the other Python installation.

Once Anaconda Python is properly installed and works, start a console, i.e. either "cmd" or the "Anaconda command prompt" on Windows, Terminal on Linux.
In the command line, type

.. code-block:: python

    $ conda install -c acoular acoular

This will install Acoular in your Anaconda Python enviroment and make the Acoular library available from Python. In addition, this will install all dependencies (those other packages mentioned above) if they are not already present on your system.
Depending on your type of Anaconda installation (single user or system-wide), you may be asked for admin privileges in order to start the installation process.

If the installation has finished, you may try out Acoular by typing

.. code-block:: python

    $ acoular_demo.py

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).

Option 2: Windows with other Python distributions
-------------------------------------------------
Download and run the `windows installer <https://pypi.python.org/pypi/acoular>`_. You will have to take care by yourself that all required packages are installed, otherwise you will not be able to use Acoular.

If the installation has finished, you may try out Acoular by typing

.. code-block:: python

    $ acoular_demo.py

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).

Option 3: pip
-------------
If you have pip installed on your system, you may type

.. code-block:: python

	$ pip install acoular

to install acoular. Note that the required packages are not installed this way and you have to do it by yourself.

If the installation has finished, you may try out Acoular by typing

.. code-block:: python

    $ acoular_demo.py

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).


Option 4: Source install
------------------------
You may download the `source tarball <https://pypi.python.org/pypi/acoular>`_. Unzip it and change into the "acoular" directory, then type

.. code-block:: python

    $ python setup.py install

to compile and install the software. This requires a properly set up system with all installed dependencies and a compiler.  
Another option to get the source is to clone or fork from `Github <https://github.com/acoular/acoular>`_.

If the installation has finished, you may try out Acoular by typing

.. code-block:: python

    $ acoular_demo.py

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).
    


