.. _examples:

Examples
========

A gallery of showcase examples suitable for first-time and advanced users.

Plot Style
----------

For consistent plot appearance, all examples use the Acoular matplotlib style.
The style is applied at the beginning of each example:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Apply Acoular plot style
   plt.style.use('acoular.plots')
   
   # Now create your plots
   # ...

You can also use this style in your own scripts by calling
``plt.style.use('acoular.plots')`` after importing matplotlib.
