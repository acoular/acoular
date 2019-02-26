What's new
============

19.02
------------
    
    * Adds support for Python 3.7
    * Introduces new :class:`~acoular.fbeamform.SteeringVector` class (see :doc:`../get_started/index` and :doc:`../examples/index`  for usage). With this, some of the Beamformer and PointSource traits are deprecated and should no longer be used. While the current version is intended to be fully compatible with older scripts, deprecation warnings will be raised if necessary.
    * Introduces optional use of reference distance for SPL evaluation (current default: reference position at (x,y,z)=(0,0,0) )
    * Introduces some basic Unit tests to evaluate the beamformer results 
    * Bugfix: CLEAN algorithm now uses correct PSFs
    * some minor bugfixes



18.01
------------

    * Added new DAMAS solving strategies (BeamformerDamasPlus)
    * Added Generalized Inverse Beamforming
    * Floating point precision of CSM, PSF and beamformer customizable (default: float64) -- affects cache file size
    * PowerSpectra class now includes EigSpectra functionality (EigSpectra still callable for backwards compatibility)
    * Inverse methods: unit of sound pressure for internal calculation customizable (default: nPa) for better numeric stability with sklearn solvers. Still returns all values in Pa.
    * Bugfix: BeamformerFunctional works now with steering vector formulation II (inverse) and III (true level) which produced incorrect results in the past. 
    * Bugfix: BeamformerFunctional can only be called when the diagonal of the CSM is included
    * Bugfix: Corrected calculation of PSF for steering vector formulation IV
    * Bugfix: Behaviour of normalizing PSF at assumed source location (psf=1) is removed
	




17.11
------------

    * Added support for Python 3.4,3.5 and 3.6 
    * Implementation of fast/parallelized code now with Numba (instead of C++ and SciPy.weave)
    * cross spectral matrix (CSM) orientation changed (was transposed in earlier versions). Please do not use the cache files from earlier versions in version 17.11!
    	



