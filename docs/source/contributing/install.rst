Developer Installation
======================

Prerequisites
-------------

Before you can start contributing to Acoular

* Create a fork of the `Acoular repository <https://github.com/acoular/acoular>`_ on GitHub
* Clone your fork to your local machine via 

.. code-block:: console

    $ git clone <https://your_fork>
    $ cd acoular

We encourage using the latest compatible Python version, which is currently 3.13.

Creating a development environment
----------------------------------

For development, a virtual environment is needed. Create one with your preferred tool:

.. include:: ../commands/environment.rst

Editable installation
---------------------

Next, we need to make an editable installation of Acoular with:

.. include:: ../commands/editable-install.rst

An editable installation allows you to make changes to the source code and see the effects immediately without having to reinstall the package.

The ``--group dev`` argument is responsible for installing the corresponding dependency group as defined in:

.. dropdown:: ``pyproject.toml`` (click to expand)

  .. literalinclude:: /../../pyproject.toml
      :start-at: [dependency-groups]
      :end-before: [project.urls]

If omitted, the dependencies have to be installed manually.

System dependencies
-------------------

Some of the Acoular dependencies require system packages to be installed. On Ubuntu, we frequently encounter that PortAudio is missing, which is required by the ``sounddevice`` library. In addition, Graphviz is required for the Acoular documentation.
You can install both with:

.. code-block:: console

    $ sudo apt install graphviz libportaudio2
