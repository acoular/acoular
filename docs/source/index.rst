.. beamfpy documentation master file

Welcome to beamfpy's documentation!
===================================

Beamfpy is a framework for acoustic beamforming that is written in the Python programming language. It is aimed at applications in acoustic testing. Multichannel data recorded by a microphone array can be processed and analyzed in order generate mappings of sound source distributions. The maps (acoustic photographs) can then be used to  locate sources of interest and to characterize them using their spectra. 

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
    Architecture Reference <arch_ref/index>
    FAQs <faqs/index>
    Examples <examples/index>
    API Reference <api_ref/index>


.. list-table::
    :class: borderless

    * - :doc:`get_started/index`

        The first stop for all those new to Enaml. This section includes
        an introduction to Enaml, installation instructions, and all the
        information needed to write your first Enaml application.

      - :doc:`dev_guides/index`

        The stuff that wasn't covered in :doc:`get_started/index`. This
        section provides in-depth documentation on a wide range of topics
        you are likely to encounter when developing production applications
        with Enaml. Look here for details on Enaml's scoping rules, aliases,
        templates, best practices, and more.


    * - :doc:`literature/index`

        In here some of the publications used for this program package are 
        listed. Further reading to fully understand how the algorithms work
        is recommended.

      - :doc:`examples/index`

        "Just show me the code!" This section provides an easy-to-browse
        alternative to running Enaml's examples from the command line. We've
        even included screenshots!

    * - :doc:`faqs/index`

        If you think you may not be the only one to have thought a thought,
        you are probably right. Look here to see if your if your question has
        already been asked, then take solace in the realization that you are
        not alone.

      - :doc:`api_ref/index`

        All modules, classes and methods featured in beamfpy are described in detail
        here. They can easily be browsed through an inheritance tree and cross links.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
