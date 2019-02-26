.. README.rst

Acoular
=======

Acoular is a Python module for acoustic beamforming that is distributed under the new BSD license. 

It is aimed at applications in acoustic testing. Multichannel data recorded by a microphone array can be processed and analyzed in order to generate mappings of sound source distributions. The maps (acoustic photographs) can then be used to locate sources of interest and to characterize them using their spectra. 

Features
========

    * covers several beamforming algorithms 
    * different advanced deconvolution algorithms
    * both time-domain and frequency-domain operation included
    * 3D mapping possible
    * application for stationary and for moving targets
    * supports both scripting and graphical user interface
    * efficient: intelligent caching, parallel computing with Numba
    * easily extendible and well documented

Dependencies
============

Acoular runs under Linux, Windows and MacOS, a Python 3.7, 3.6, 3.5, 3.4 or 2.7 installation is needed with the latest Numpy, Scipy, Traits, traitsui, scikit-learn, pytables, numba, pyqt packages available.
