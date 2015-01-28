.. acoular documentation master file

Welcome to acoular's documentation!
===================================

Acoular is a framework for acoustic beamforming that is written in the Python programming language. It is aimed at applications in acoustic testing. Multichannel data recorded by a microphone array can be processed and analyzed in order generate mappings of sound source distributions. The maps (acoustic photographs) can then be used to  locate sources of interest and to characterize them using their spectra. 

A few highlights of the framework:

    * covers several beamforming algorithms 
    * different advanced deconvolution algorithms
    * both time-domain and frequency-domain operation included
    * 3D mapping possible
    * application for stationary and for moving targets
    * supports both scripting and graphical user interface
    * efficient: intelligent caching, parallel computing with OpenMP
    * easily extendible and well documented


Contents:

.. toctree::
    :hidden:

    Getting Started <get_started/index>
    Developer Guides <dev_guides/index>
    Literature <literature/index>
    FAQs <faqs/index>
    Examples <examples/index>
    API Reference <api_ref/index>


.. list-table::
    :widths: 20 80

    * - :doc:`get_started/index` 

      - The basics for using Acoular explained with a simple example.

    * - :doc:`examples/index` 

      - Example scripts covering different use cases.

    * - :doc:`api_ref/index` 

      - All modules, classes and methods featured in acoular are described in detail
        here. They can easily be browsed through an inheritance tree and cross links.

    * - :doc:`literature/index` 

      - In here some of the publications used for this program package are 
        listed. Further reading to fully understand how the algorithms work is recommended.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
