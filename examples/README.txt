.. _examples:

Examples
========

A gallery of showcase examples suitable for first-time and advanced users.

Plot Style
----------

The Acoular matplotlib style is automatically applied when you import the
acoular package. This ensures consistent plot appearance across all examples,
scripts, and notebooks.

The style is applied when you import acoular:

.. code-block:: python

   import acoular as ac
   import matplotlib.pyplot as plt
   
   # Style is already applied automatically
   # Now create your plots
   # ...

If you want to explicitly use the style (e.g., after changing to a different
style), you can call:

.. code-block:: python

   plt.style.use('acoular.plots')

This style is used when building the documentation with Sphinx Gallery and
in Jupyter notebooks.
