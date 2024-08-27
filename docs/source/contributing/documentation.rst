Documentation
=============

The package documentation is provided under ``acoular/docs/source``. This directory contains the ``index.rst`` file which serves as the root document (landing page) embedding several other subdocuments (sub-pages) written in  `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ format. 
The full documentation is built using the `Sphinx <https://www.sphinx-doc.org/en/master/>`_ package. See :ref:`Documentation Compilation` for instructions on how to build the documentation.

.. note::

    Some of the subdirectories (sub-pages) are created automatically during the documentation build process. Therefore, it is recommended to avoid editing the generated files directly.
    This concerns the ``docs/source/api_ref/generated`` directory, which contains the API documentation generated from the source code, and the ``docs/source/auto_examples`` directory, which contains the auto-generated examples.


.. _User Documentation:

User Documentation
------------------

This is the easiest way to contribute to the documentation. You can simply edit the corresponding ``.rst`` files in the ``acoular/docs/source`` directory which are not generated automatically. Make sure you are using the `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ format. If you want to add a new document, it should be referenced in the ``index.rst`` or any related file. We are happy about every fixed spelling mistake or improved explanation. 

.. _API Documentation:

Documenting the API
-------------------

There are several situations in which you want to change the API documentation. For example, when you add a new class, method, or function, or when you change the signature of an existing method. In some cases, you might also want to improve the existing documentation.

The API documentation is generated from the documentation strings of the source code using the Sphinx `autosummary <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_ extension. The API documentation follows the `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
The generated API documentation can be found in the ``source/api_ref`` subdirectory of the documentation after compilation. 


Documenting Classes, Methods and Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A class documentation that is suitable as a how-to example is given by the :class:`acoular.sources.TimeSamples` class.

**Writing docstrings**:

Docstrings should first contain a short summary line, followed by an extended summary using `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ syntax. An extended summary is not always necessary, but it is recommended not only for complex implementations. Ideally, the extended summary provides a detailed explanation of the method's purpose, behavior, and usage.

.. literalinclude:: ../../../acoular/sources.py
    :pyobject: TimeSamples
    :lines: 1-6

We welcome the addition of code examples to the docstrings of classes (and functions). The :class:`acoular.sources.TimeSamples` class provides a short snippet with explanation in the docstring:

.. literalinclude:: ../../../acoular/sources.py
    :pyobject: TimeSamples
    :lines: 8-34

To ensure that these snippets stay up-to-date, they are tested automatically within Acoular's CI workflow using `pytest <https://docs.pytest.org/>`_ with the python standard `doctest <https://docs.python.org/3/library/doctest.html>`_ module. This requires that the **code examples are written in a way that they can be executed as standalone code snippets**. In very rare cases, it is not possible to write an executable standalone code snippet. In such cases, the code snippet can be marked with the `# doctest: +SKIP` directive to be excluded from the doctests.

One can use the following command with hatch to run the doctests locally:

.. code-block:: bash

    hatch run tests:doctest


Cross-references can be included in the `See Also` section, which lists related classes or functions:

.. literalinclude:: ../../../acoular/sources.py
    :pyobject: TimeSamples
    :lines: 35-40

**Documenting class attributes**:

Acoular makes use of the `Traits API <https://docs.enthought.com/traits/traits_user_manual/intro.html>`_, which allows for the definition of class attributes with type checking and default values. Most of the Acoular classes will directly define their attributes, and the corresponding __init__ method will not be explicitly defined. It is recommended to document public class attributes using a comment line above the attribute definition starting with `#:`. Sphinx will automatically detect these comments and include them in the API documentation. 

.. literalinclude:: ../../../acoular/sources.py
    :pyobject: TimeSamples
    :lines: 51-59

**Documenting public methods and functions**:

Similar to classes, public methods and functions need to have a docstring. In addition to the summary line, the arguments and return values should be documented in `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
See the :func:`acoular.sources.TimeSamples.result` method as an example:

.. literalinclude:: ../../../acoular/sources.py
    :pyobject: TimeSamples.result
    :lines: 1-20


**Autosummary**:

To ensure that a new class, or function is included in the generated API documentation, it needs to be added to the `autosummary` section at the top of the respective Python module file so that it can be recognized by Sphinx. The `autosummary` section is a block with a list of the classes, functions, and constants defined in the module. 

For the :mod:`acoular.sources` module, the `autosummary` section looks like this:

.. literalinclude:: ../../../acoular/sources.py
    :language: python
    :lines: 1-22



Documenting Modules
~~~~~~~~~~~~~~~~~~~

Modules should be documented with a module docstring at the beginning of the file. The docstring should contain a summary of the module's purpose. The docstring should be enclosed in triple quotes and should be placed at the beginning of the file, before any other code.


Writing Python examples
-----------------------

Acoular provides a set of :doc:`examples <../auto_examples/index>` that demonstrate how to use the package. These examples are located in the ``acoular/examples`` subdirectory. 

We are always looking for new examples that demonstrate the functionality of Acoular. If you have a good example that you would like to share, please consider contributing it to the Acoular package.

We also encourage you to enhance the existing examples by adding more detailed explanations, improving the code, or adding new features.

We use the `Sphinx-Gallery <https://sphinx-gallery.github.io/stable/index.html>`_ extension to automatically generate `.rst` files and downloadable `.ipynb` files from the Python examples, located in the `docs/source/auto_examples` directory. 


Adding new examples
~~~~~~~~~~~~~~~~~~~

To add a new example, create a new Python script in one of the subdirectories of the `examples` directory.
If no subdirectory is suitable, you can create a new one. 
The name of the script should start with `example_` followed by a short descriptive name.
If the latter is the case, make sure to add the new subdirectory in the `acoular/docs/source/conf.py` file in the `sphinx_gallery_conf` dictionary:

.. code-block:: python

    sphinx_gallery_conf = {
        'subsection_order' : ExplicitOrder([
            "../../examples/introductory_examples",
            "../../examples/wind_tunnel_examples",
            "../../examples/moving_sources_examples",
            "../../examples/io_and_signal_processing_examples",
            "../../examples/tools",
            "<ADD YOUR NEW SUBDIRECTORY HERE>"
        ]),
    }  

All examples are executed during the documentation build process to ensure that they are working correctly. Therefore, it is important to keep an eye on the execution time of your example. We will only accept examples that run in a reasonable time frame (ideally within a few seconds). 

Citing literature
-----------------

If you are adding a new feature or method that is based on scientific literature, please make sure to include the corresponding references as a bibtex entry in the `acoular/docs/source/literature/literature.bib` file. Details on the bibtex format can be found `here <https://ctan.org/pkg/bibtex>`_.

You can then cite the reference using the directive ``:cite:`<BIBTEX_KEY>``` directive in the documentation. Your citation will be automatically included in the bibliography section of the documentation. 
