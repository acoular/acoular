Install Acoular
===============
Acoular runs under 64-bit Linux, MacOS and Windows. It currently supports Python 3.10, 3.11, 3.12 and 3.13.

Tools
-----
There are many different tools for Python environment management and package installation.
Select your preferred method by clicking on one of the tabs below.

.. include:: tools.rst

Virtual environment
-------------------
We strongly encourage the use of virtual environments. An environment ``my-env`` can be created with:
        
.. include:: environment-commands.rst

Installation
------------
Then, install Acoular with:

.. include:: install-commands.rst

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
`PyLops <https://pylops.readthedocs.io>`_                         Some solvers of :class:`~acoular.fbeamform.BeamformerCMF`.
================================================================= ========

When installing Acoular from PyPI, Acoular and all optional dependencies can be installed with the ``full`` extra:

.. literalinclude:: /../../pyproject.toml
    :start-at: [project.optional-dependencies]
    :end-before: [dependency-groups]
                 
Otherwise, the dependencies need to be installed manually:

.. include:: full-install-commands.rst

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
