.. _Code Quality Standards:

Code Quality Standards
----------------------

Acoular requires a high standard of code quality, and we test if these requirements are met via Continuous Integration (CI) with `GitHub Actions <https://github.com/acoular/acoular/actions>`_. 

Currently, three main checks are performed on the code:

1. `Linting and Formatting`_: The code must be correctly formatted and free of linting errors.
2. `Documentation Build`_: The documentation must be buildable without errors.
3. `Testing`_: The unit and regression tests must pass without errors.

We recommend running these checks before submitting a pull request. This can be done locally or on a GitHub runner. 
The latter is possible by using the `workflow_dispatch` event from the desired branch holding the changes. 
To do this, go to the `Actions` tab in the forked GitHub repository, select the `Tests` workflow and press the `Run workflow` button.

.. _Linting and Formatting:

Linting and Formatting
~~~~~~~~~~~~~~~~~~~~~~

Acoular primarily follows the coding style of `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.
We use `ruff <https://pypi.org/project/ruff/>`_ for linting and formatting.

Before merging, the source code must be correctly formatted and free of linting errors. You can **check your code locally** by running the following commands:

.. include:: ../commands/ruff.rst

In case there are any reported errors, you can either fix them manually or try to **let ruff fix them for you** by running:

.. include:: ../commands/ruff-fix.rst

The ``ruff`` configuration can be found in the ``.ruff.toml`` file:

.. literalinclude:: /../../.ruff.toml

.. include:: build-docs.rst

.. _Testing:

Testing
~~~~~~~

Acoular provides several tests to ensure the correct functionality of the package at different Python versions. The tests are located in the ``acoular/tests`` directory.

To run the tests locally, execute the following command from within the ``acoular`` directory:

.. include:: ../commands/tests.rst
