.. _examples:

Examples
========

A gallery of showcase examples suitable for first-time and advanced users.

Plot Style
----------

For consistent plot appearance, use the Acoular matplotlib style by calling
``ac.set_plot_style()`` at the beginning of your scripts or notebooks:

.. code-block:: python

   import acoular as ac
   import matplotlib.pyplot as plt
   
   # Apply Acoular plot style
   ac.set_plot_style()
   
   # Now create your plots
   # ...

This style is automatically applied when building the documentation with
Sphinx Gallery and in Jupyter notebooks generated from the examples.
