
Setting up a development environment
====================================


Prerequisites
-------------

Before you can start contributing to Acoular

* Create a fork of the `Acoular repository <https://github.com/acoular/acoular>`_ on GitHub
* Clone your fork to your local machine via 

.. code-block:: bash

    git clone <https://your_fork/acoular>
    cd acoular


If you don't already have an existing virtual environment, you have the choice to create one with a package distribution of your choice.
The Acoular development team had good experience with the following distributions: 

    * `conda <https://docs.conda.io/en/latest/>`_ 
    * `mamba <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_
    * `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_ 
    * `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ 


Create a fresh environment
--------------------------

We recommend using the latest compatible version of Python, which is currently 3.13.
If you are using the anaconda distribution, you can create a new environment, for example named `dev`, with

.. code-block:: bash

    conda create -n dev python=3.13
    conda activate dev

The same command can be used with mamba or micromamba by simply replacing `conda` with `mamba` or `micromamba`.


Equally, you can create a new environment with virtualenv:

.. code-block:: bash

    virtualenv -p python3.13 dev
    source dev/bin/activate



Install dependencies
--------------------

Then, make an editable installation of Acoular with:

.. code-block:: bash

    pip install -e ".[dev]"

The editable installation allows you to make changes to the source code and see the effects immediately without having to reinstall the package.

The ``".[dev]"`` argument is responsible for installing the right dependencies as defined in ``pyproject.toml``:

.. code-block:: toml

    dev = [
        "acoular[docs,full,tests]",
        "hatch",
        "ruff==0.8.1",
    ]

If not included, the dependencies have to be installed manually.

System dependencies
-------------------

Some of the Acoular dependencies require system packages to be installed. On Ubuntu, we frequently encounter that the portaudio library is missing, which is required by the sounddevice package. In addition, the graphviz package is required for the documentation. 
You can install both with:

.. code-block:: bash

    sudo apt install graphviz libportaudio2


