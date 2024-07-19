.. _Code Quality Standards:

Code Quality Standards
----------------------

Acoular requires a high standard of code quality, and we test if these requirements are met via Continuous Integration (CI) with `GitHub Actions <https://github.com/acoular/acoular/actions>`_. 


Code Style
~~~~~~~~~~

Acoular follows the coding style of `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.
To verify that the code meets the PEP8 standard, we use the `ruff <https://pypi.org/project/ruff/>`_ code linter and formatter. Configurations for the `ruff <https://pypi.org/project/ruff/>`_ code checker are contained in ``.ruff.toml`` file. 

The source code must be free of linting errors. You can check your code on linting errors locally by running the following command:

.. code-block:: bash

    hatch fmt --check

which should return:

.. code-block:: bash

    All checks passed!


.. _Compile Documentation:

Compile Documentation
~~~~~~~~~~~~~~~~~~~~~

The package documentation is provided under ``acoular/docs/source``. This directory contains the ``index.rst`` file, which is the root document embedding several other subdocuments (sub-pages).

You can check if the documentation can be compiled without errors by running

.. code-block:: bash

    hatch run docs:build

which should end with the following message:

.. code-block:: bash

    Build finished. The HTML pages are in build/html.


Testing
~~~~~~~

Acoular provides several tests to ensure the correct functionality of the package at different Python versions. The tests are located in the ``acoular/tests`` directory.

To run the tests locally, execute the following command:

.. code-block:: bash

    hatch run tests:test

Note that the tests are executed on different Python versions and take several minutes to complete. 