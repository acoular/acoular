==============
User Guide
==============

Basic Concepts
==============

Object-oriented Programming (OOP)
---------------------------------


Lazy Evaluation
---------------


Generator Pipelines
-------------------

.. Time Domain vs. Frequency Domain vs General usage (InOut)


Data Simulation
===============

.. describe how to simulate data with Acoular

Signals
-------
.. reference to DAGA2025 Herold et al. Notebook / Acoular Blog


Sources
-------


Impulse Responses
-----------------


Input / Output 
==============

Data Import
-----------
.. file formats used by TimeSamples / MaskedTimeSamples / ImportGrid / MicGeom / AIAA Benchmark...

Data Export
-----------

.. list export options available (WriteH5, WriteWAV, MicGeom.export_mpos, etc.)    

Data Acquisition
----------------

.. copy from DAGA2025 Eschenhagen et al. Notebook
.. Using Multiple Streams


Microphone Array Methods in the Time Domain
============================================

.. copy from DAGA2025 Czuchaj Notebook

Time Domain Filtering
---------------------

Beamforming
-----------

Deconvolution
-------------

Sector Integration
------------------


Microphone Array Methods in Frequency Domain
============================================

.. copy from DAGA2025 Kujawski et al. Notebook

Steering Vector
---------------

Beamforming
-----------

Deconvolution
-------------

Inverse Methods
---------------

Gridless Methods
----------------

Sector Integration
------------------



Sound Propagation and Environments
==================================


Flow Environments
-----------------

.. copy from DAGA2025 Lylloff notbook
.. show for time and frequency domain methods


.. Virtual Rotation
.. -------------------


Moving Sources
==============

.. convective amplification
.. reference to Rotating Point Source example in Sphinx Gallery
.. copy from DAGA2025 Czuchaj Notebook

Trajectory
----------

Data Simulation
---------------

Time Domain Methods with Moving Focus
-------------------------------------


Caching
=======

.. explain digest

Cached Properties
-----------------

Persistent File Cache
---------------------

.. * Caching of beamforming results
.. * Caching of generator results
.. * global caching options



Anti-Patterns
=============

.. # creating a new instance of a class in a loop (instead of creating it once and reusing it)
.. # changing entries of arrays that cannot be captured by the trait listeners
.. # overwriting file names during a session


Acoular Extensions
==================

.. SpectAcoular / TraitsUI
.. AcouPipe 
.. others?