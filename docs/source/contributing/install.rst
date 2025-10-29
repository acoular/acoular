
Setting up a development environment
====================================


Prerequisites
-------------

Before you can start contributing to Acoular

* Create a fork of the `Acoular repository <https://github.com/acoular/acoular>`_ on GitHub
* Clone your fork to your local machine via 

.. code-block:: bash

    git clone <https://your_fork>
    cd acoular

We encourage using the latest compatible Python version, which is currently 3.13.

Creating a development environment
----------------------------------

For development, a virtual environment is needed. Create one with your preferred tool:

.. include:: ../install/environment-commands.rst

Editable installation
---------------------

Next, we need to make an editable installation of Acoular with:

.. include:: ../install/editable-install-commands.rst

An editable installation allows you to make changes to the source code and see the effects immediately without having to reinstall the package.


The ``".[dev]"`` argument is responsible for installing the right dependencies as defined in ``pyproject.toml``:

.. literalinclude:: /../../pyproject.toml
    :start-at: [dependency-groups]
    :end-before: [project.urls]

If not included, the dependencies have to be installed manually.

System dependencies
-------------------

Some of the Acoular dependencies require system packages to be installed. On Ubuntu, we frequently encounter that the portaudio library is missing, which is required by the sounddevice package. In addition, the graphviz package is required for the documentation. 
You can install both with:

.. code-block:: bash

    sudo apt install graphviz libportaudio2


