.. _Code Quality Standards:

Code Quality Standards
----------------------

Acoular requires a high standard of code quality, and we test if these requirements are met via Continuous Integration (CI) with `GitHub Actions <https://github.com/acoular/acoular/actions>`_. 

Currently, three main checks are performed on the code:

1.  `Linting and Formatting`_: The code must be correctly formatted and free of linting errors.
2. `Documentation Compilation`_: The documentation must be correctly compiled without errors.
3. `Testing`_: The tests must pass without errors.

We recommend running these checks before submitting a pull request. This can be done locally or on a GitHub runner. 
The latter is possible by using the `workflow_dispatch` event from the desired branch holding the changes. 
To do this, go to the `Actions` tab in the forked GitHub repository, select the `Tests` workflow and press the `Run workflow` button.


.. _Linting and Formatting:

Linting and Formatting
~~~~~~~~~~~~~~~~~~~~~~

Acoular primarily follows the coding style of `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.
To verify that the code meets the PEP8 standard, we use the `ruff <https://pypi.org/project/ruff/>`_ code linter and formatter. Configurations for the `ruff <https://pypi.org/project/ruff/>`_ code checker are contained in ``.ruff.toml`` file. 

The source code must be correctly formatted and free of linting errors. You can check your code locally by running the following command:

.. code-block:: bash

    hatch fmt --check

which should return:

.. code-block:: bash

    All checks passed!


.. _Documentation Compilation:

Documentation Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~

The package documentation is provided under ``acoular/docs/source``. This directory contains the ``index.rst`` file, which is the root document embedding several other subdocuments (sub-pages).

You can check if the documentation can be compiled without errors by running

.. code-block:: bash

    hatch run docs:build

which should end with the following message:

.. code-block:: bash

    Build finished. The HTML pages are in build/html.

.. _Testing:

Testing
~~~~~~~

Acoular provides several tests to ensure the correct functionality of the package at different Python versions. The tests are located in the ``acoular/tests`` directory.

To run the tests locally, execute the following command:

.. code-block:: bash

    hatch run tests:test

Note that the tests are executed on different Python versions and take several minutes to complete.

To check a specific Python version only, the command can be modified to:

.. code-block:: bash

    hatch -v run +py=3.11 tests:test

Here the tests will be executed for the Python version 3.11 only. The ``-v`` flag increases verbosity.