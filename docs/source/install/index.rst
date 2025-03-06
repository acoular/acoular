Installation
============

Depending on your operating system and python distribution, there are different options (see below) how to install Acoular.

Requirements
------------

Acoular runs under 64bit Windows, Linux and MacOS, and needs Python 3.10, 3.11, 3.12 or 3.13

Upon installation using options 1 or 2 below, all necessary dependencies will also be installed.

Optional dependencies are only required in some circumsstances and must be installed seperately when needed:
    1. In order to run the demo and examples you will also need  `matplotlib <http://matplotlib.org>`_.
    2. If you want to use input from a soundcard hardware, you will also need to install `sounddevice <https://python-sounddevice.readthedocs.io/en/0.3.12/installation.html>`_.
    3. Some solvers for the CMF method need `Pylops <https://pylops.readthedocs.io/en/stable/installation.html>`_.


Install option 1: pip
----------------------------------------------------
If you have pip installed on your system, you may type

.. code-block:: console

	$ pip install acoular

to install Acoular without optional dependencies. If you wish to include the optional dependencies, you may type

.. code-block:: console

    $ pip install acoular[full]

Install option 2: Anaconda 
----------------------------------------------------

This option assumes that you have the `Anaconda <https://www.anaconda.com/download/>`_ Python-distribution installed on your computer.

Once Anaconda Python is properly installed and works, start a console, i.e. either "cmd" or the "Anaconda command prompt" on Windows, Terminal on Linux.
In the command line, type

.. code-block:: console

    $ conda install -c acoular acoular

This will install Acoular in your Anaconda Python enviroment and make the Acoular library available from Python. In addition, this will install all dependencies (those other packages mentioned above) if they are not already present on your system.
Depending on your type of Anaconda installation (single user or system-wide), you may be asked for admin privileges in order to start the installation process.

Alternatively, you may choose to install into a dedicated conda environment. 


Install option 3: source install
--------------------------------
You may download the `source tarball <https://pypi.python.org/pypi/acoular>`_. Unzip it and change into the "acoular" directory, then type

.. code-block:: console

    $ pip install .

to install the software.   
Another option to get the source is to clone or fork from `Github <https://github.com/acoular/acoular>`_.

Known Issues
------------

Recently, we have encountered some issues with the installation of Acoular via conda using the defaults channel.
On Linux OS and for Python version 3.13, the following error may occur after importing acoular in Python:

.. code-block:: python

    ImportError: traits/ctraits.cpython-313-x86_64-linux-gnu.so: undefined symbol: Py_TRASHCAN_SAFE_BEGIN

We therefore recommend installing Acoular via pip or to use a different Python version (3.10, 3.11, 3.12) when installing via conda.
Alternatively, you can update the traits package from the conda-forge channel:

.. code-block:: console

    $ conda update -c conda-forge traits

Verify Acoular installation
---------------------------

If the installation has finished, you may try out Acoular by importing the module and running the Acoular demo script in a Python console. 
The Acoular demo script needs `matplotlib <http://matplotlib.org>`_ installed to render the graphical output.

.. code-block:: console

    > import acoular
    > acoular.demo.acoular_demo.run()

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).


