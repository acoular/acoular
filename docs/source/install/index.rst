.. _installation of uv: https://docs.astral.sh/uv/#installation
.. _installation of pip: https://pip.pypa.io/en/stable/installation
.. _installation of conda: https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html
.. _installation of mamba: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

Install Acoular
===============
Acoular runs under 64-bit Linux, MacOS and Windows. It currently supports Python 3.10, 3.11, 3.12 and 3.13.

Tools
-----
There are many different tools for Python environment management and package installation.
Select your preferred method by clicking on one of the tabs below.

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        `"An extremely fast Python package and project manager, written in Rust."`

        This method only requires an `installation of uv`_ itself.
        Choose this for a beginner-friendly and streamlined experience.

    .. tab-item:: ``pip``
        :sync: pip

        `"The PyPA recommended tool for installing Python packages."`

        This method requires Python and an `installation of pip`_.
        Choose this for a traditional Python experience.

    .. tab-item:: ``mamba``
        :sync: mamba

        `"The fast cross-platform package manager."`

        This method only requires an `installation of mamba`_ itself.
        Choose this if you rely on Anaconda and want a fast experience.

    .. tab-item:: ``conda``
        :sync: conda

        `"OS-agnostic, system-level binary package and environment manager."`

        This method only requires an `installation of conda`_ itself.
        Choose this if you rely on Anaconda.

Virtual environment
-------------------
We strongly encourage the use of virtual environments. An environment can be created with:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv venv

        .. note::
           ``uv`` will handle environment activation implicitly.

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ python3 -m venv my-env

        and activate the environment with:

        .. code-block:: console

            $ source my-env/bin/activate

    .. tab-item:: ``mamba``
        :sync: mamba

        .. code-block:: console

            $ mamba create -n my-env

        and activate the environment with:

        .. code-block:: console

            $ mamba activate my-env

    .. tab-item:: ``conda``
        :sync: conda

        .. code-block:: console

            $ conda create -n my-env

        and activate the environment with:

        .. code-block:: console

            $ conda activate my-env

Installation
------------
Then, install Acoular with:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv pip install acoular

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ pip3 install -U acoular

    .. tab-item:: ``mamba``
        :sync: mamba

        .. code-block:: console

            $ mamba install -c acoular acoular

    .. tab-item:: ```conda``
        :sync: conda

        .. code-block:: console

            $ conda install -c acoular acoular

Dependencies
------------
Acoular depends on the following packages which will be installed automatically (unless they are already installed):

============================================= ========
Package                                       Function
============================================= ========
`Numba <https://numba.pydata.org>`_           Automatic JIT-compilation of performance critical computations.
`NumPy <https://numpy.org>`_                  Array data types for efficient block processing.
`scikit-learn <https://scikit-learn.org>`_    L1-optimization algorithms.
`SciPy <https://scipy.org>`_                  Optimization and linear algebra algorithms.
`PyTables <https://www.pytables.org>`_        HDF5-support for caching and file I/O.
`Traits <https://docs.enthought.com/traits>`_ Type-checking and trait-change notifications for lazy-evaluation.
============================================= ========

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^
Optional dependencies are only required in some circumstances. They must be installed separately when needed.

================================================================= ========
Package                                                           Needed for
================================================================= ========
`Matplotlib <https:/matplotlib.org>`_                             Running the examples and to generate output in the demo.
`python-sounddevice <https://python-sounddevice.readthedocs.io>`_ Using input from physical soundcard hardware.
`PyLops <https://pylops.readthedocs.io>`_                         Some solvers of the CMF method.
================================================================= ========

When installing Acoular from PyPI, Acoular and all optional dependencies can be installed with the ``acoular[full]`` extra. Otherwise, the dependencies need to be installed manually:

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv pip install 'acoular[full]'

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ pip3 install -U 'acoular[full]'

    .. tab-item:: ``mamba``
        :sync: mamba

        .. code-block:: console

            $ mamba install -c acoular acoular matplotlib pylops python-sounddevice

    .. tab-item:: ``conda``
        :sync: conda

        .. code-block:: console

            $ conda install -c acoular acoular matplotlib pylops python-sounddevice

Verify your installation
------------------------
After installation, you may verify your Acoular installation by importing the module and running the Acoular demo script in a Python console.
In order to render the graphical output, the Acoular demo script requires ``matplotlib``.

.. tab-set::
    :sync-group: tool

    .. tab-item:: ``uv``
        :sync: uv

        .. code-block:: console

            $ uv run python -c "import acoular; acoular.demo.acoular_demo.run()"

    .. tab-item:: ``pip``
        :sync: pip

        .. code-block:: console

            $ python -c "import acoular; acoular.demo.acoular_demo.run()"

    .. tab-item:: ``mamba``
        :sync: mamba

        .. code-block:: console

            $ python -c "import acoular; acoular.demo.acoular_demo.run()"

    .. tab-item:: ``conda``
        :sync: conda

        .. code-block:: console

            $ python -c "import acoular; acoular.demo.acoular_demo.run()"

After a couple of seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources). If ``matplotlib`` is not found, the beamforming map will be printed in ASCII format.

Known Issues
------------

Using ``conda`` with the ``defaults`` channel on Linux (Python 3.13)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Recently, we have encountered some issues with the installation of Acoular via ``conda`` using the ``defaults`` channel.
On Linux and for Python version 3.13, the following error may occur after importing Acoular:

.. code-block:: python

    ImportError: traits/ctraits.cpython-313-x86_64-linux-gnu.so: undefined symbol: Py_TRASHCAN_SAFE_BEGIN

Therefore, we recommend either installing Acoular with another tool, or using a different Python version (3.10, 3.11, 3.12) when installing via ``conda``.
Alternatively, install the ``traits`` package from the ``conda-forge`` channel to fix the error with:

.. code-block:: console

    $ conda update -c conda-forge traits

Verify Acoular installation
---------------------------

If the installation has finished, you may try out Acoular by importing the module and running the Acoular demo script in a Python console.
The Acoular demo script needs `matplotlib <http://matplotlib.org>`_ installed to render the graphical output.

.. code-block:: python

    >>> import acoular
    >>> acoular.demo.run()

After some seconds, this should produce two pictures (a 64 microphone arrangement and a beamforming map with three sources).

Missing PortAudio on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``sounddevice`` library depends on PortAudio which might not be installed on Linux. On Ubuntu, for example, you can install PortAudio with:

.. code-block:: console

    $ sudo apt install libportaudio2
