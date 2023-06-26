Installation
============

Depending on your operating system and python distribution, there are different options (see below) how to install Acoular.

Requirements
------------

Acoular runs under 64bit Windows, Linux and MacOS, and needs Python 3.7, 3.8, 3.9, 3.10 or 3.11

Upon installation using options 1 or 2 below, all necessary dependencies will also be installed.

Optional dependencies are only required in some circumsstances and must be installed seperately when needed:
    1. In order to run the demo and examples you will also need  `matplotlib <http://matplotlib.org>`_.
    2. If you want to use input from a soundcard hardware, you will also need to install `sounddevice <https://python-sounddevice.readthedocs.io/en/0.3.12/installation.html>`_.
    3. Some solvers for the CMF method need `Pylops <https://pylops.readthedocs.io/en/stable/installation.html>`_.


Install option 1: Anaconda 
----------------------------------------------------

This option assumes that you have the `Anaconda <https://www.anaconda.com/download/>`_ Python-distribution installed on your computer.

Once Anaconda Python is properly installed and works, start a console, i.e. either "cmd" or the "Anaconda command prompt" on Windows, Terminal on Linux.
In the command line, type

.. code-block:: console

    $ conda install -c acoular acoular

This will install Acoular in your Anaconda Python enviroment and make the Acoular library available from Python. In addition, this will install all dependencies (those other packages mentioned above) if they are not already present on your system.
Depending on your type of Anaconda installation (single user or system-wide), you may be asked for admin privileges in order to start the installation process.

Alternatively, you may choose to install into a dedicated conda environment. 

Install option 2: pip
----------------------------------------------------
If you have pip installed on your system, you may type

.. code-block:: console

	$ pip install acoular

to install acoular.

Install option 3: source install
--------------------------------
You may download the `source tarball <https://pypi.python.org/pypi/acoular>`_. Unzip it and change into the "acoular" directory, then type

.. code-block:: console

    $ pip install .

to install the software.   
Another option to get the source is to clone or fork from `Github <https://github.com/acoular/acoular>`_.

Verify Acoular installation
---------------------------

If the installation has finished, you may try out Acoular by importing the module and running the Acoular demo script in a Python console. The acoular demo script needs `matplotlib <http://matplotlib.org>`_ installed to render the graphical output.

.. code-block:: console

    > import acoular
    > acoular.demo.acoular_demo.run()

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).


