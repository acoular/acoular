.. README.rst

Acoular
=======

acoular is a Python module for acoustic beamforming that is distributed under the new BSD license. 

It is aimed at applications in acoustic testing. Multichannel data recorded by a microphone array can be processed and analyzed in order generate mappings of sound source distributions. The maps (acoustic photographs) can then be used to  locate sources of interest and to characterize them using their spectra. 

Features
========

    * covers several beamforming algorithms 
    * different advanced deconvolution algorithms
    * both time-domain and frequency-domain operation included
    * 3D mapping possible
    * application for stationary and for moving targets
    * supports both scripting and graphical user interface
    * efficient: intelligent caching, parallel computing with OpenMP
    * easily extendible and well documented

Dependencies
============

acoular runs under Linux, Windows and possibly other OS (untested), a Python 2.7 installation is needed with the latest Numpy, Scipy, Traits, Chaco, scikit-learn, pytables packages available.