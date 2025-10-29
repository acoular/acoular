.. _examples:

Examples
========

A gallery of showcase examples suitable for first-time and advanced users.

Plot Style
----------

For consistent plot appearance, all examples use the Acoular matplotlib style.
The style is applied at the beginning of each example:

.. code-block:: python

   from pathlib import Path
   import acoular as ac
   import matplotlib.pyplot as plt
   
   # Apply Acoular plot style
   plt.style.use(Path(ac.__file__).parent / 'plots.mplstyle')
   
   # Now create your plots
   # ...

You can also use this style in your own scripts by loading it from the
acoular package directory.
