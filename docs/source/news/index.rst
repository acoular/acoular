What's new
============
24.03
------------
    * Improve test coverage for :class:`~acoular.fbeamform.BeamformerCMF`
    * Changes to :class:`~acoular.fbeamform.BeamformerSODIX`:
        * correction of wrong cost-function 
        * speedups through the use of `numpy.einsum_path` together with `numpy.einsum`
        * changed start value `pgtol` for the optimization with `scipy.optimize.fmin_l_bfgs_b` solver
    * Bugfixes:
        * fixes unrecognized sector arguments in :class:`~acoular.tools.MetricEvaluator`
        * handles version-dependent default values for `normalize` attribute in sklearn solvers (relevant for :class:`~acoular.fbeamform.BeamformerCMF` )
        * fixes bug in :class:`~acoular.fbeamform.BeamformerOrth`: assigned strongest source at grid index 0 when instead of `eva_list` the trait `n` was given
        * fixes broken :class:`~acoular.tprocess.SpatialInterpolator` 
        * minor bugfix for single microphone transfer functions calculated with :class:`~acoular.fbeamform.SteeringVector` 
        * fixes broken `NNLS` method in :class:`~acoular.fbeamform.BeamformerCMF` (wrong keyword argument `normalize`)
    * Internal:
        * new GitHub workflow for CI of the documentation 
        * added Zenodo metadata file
        * changes to author name in `pyproject.toml`


23.11
------------
    * New class :class:`~acoular.tools.MetricEvaluator` to evaluate the performance of source mapping methods according to Herold and Sarradj (2017)
    * New class :class:`~acoular.sources.PointSourceConvolve` to blockwise convolve an arbitrary source signal with a spatial room impulse response
    * All filter classes derived from :class:`~acoular.tprocess.Filter` use SOS filters now
    * No more version restrictions for scikit-learn 
    * Speedups for numba jitted functions by enforcing C-contiguous arguments and the efficient use SIMD processor instructions
    * :class:`~acoular.fbeamform.BeamformerOrth` now reimplements orthogonal deconvolution to be even faster and has a slightly different interface
    * Simple benchmark suite to compare the performance of Acoular core routines on different computers
    * Some internal rework in grid and sector classes
    * Test coverage is improved
    * Bugfixes:
        * minor bugfix for convective amplification in :class:`~acoular.tbeamform.BeamformerCleantTraj` and :class:`~acoular.tbeamform.BeamformerCleantTrajSq`
        * bugfix in some attributes of :class:`~acoular.grids.ImportGrid`

23.6
------------
    * Supports Python 3.7, 3.8, 3.9, 3.10, 3.11 on Linux, MacOS, Windows
    * New build system using hatch, purge setuptools
    * New base class :class:`~acoular.fbeamform.BeamformerAdaptiveGrid` for gridless algorithms
    * New class :class:`~acoular.fbeamform.BeamformerGridlessOrth` for gridless orthogonal beamforming
    * New class :class:`~acoular.grids.RectSector3D`
    * Improved ray casting implementation for :class:`~acoular.environments.GeneralFlowEnvironment`
    * Improved handling of spectra calculation:
        * New base class :class:`~acoular.spectra.BaseSpectra`
        * New class :class:`~acoular.spectra.FFTSpectra` for time-frequency analysis
        * New class :class:`~acoular.spectra.PowerSpectraImport` for cross spectral matrix import
    * :class:`~acoular.microphones.MicGeom` now has an aperture trait
    * Tests are improved
    * Bugfixes:
        * broken numpy.int import 
        * one off bug in :class:`~acoular.grids.LineGrid` 


22.3
------------
    * New class :class:`~acoular.fbeamform.BeamformerSodix`
    * New SplitBregman and FISTA solvers in :class:`~acoular.fbeamform.BeamformerCMF`
    * IDW is now available for virtual rotation
    * different steering vector formulations are now available for time beamformers as well
    * Speedups:
        * time domain beamformers and CleanT deconvolution now share a common core codebase and all do blockwise processing
    * Bugfixes:
        * broken digest in :class:`~acoular.grids.RectGrid3D` repaired 
        * :class:`~acoular.tbeamform.BeamformerCleant` and derived classes now never miss samples


21.05
------------

    * Supports Python 3.6, 3.7, 3.8, 3.9 on Linux, MacOS, Windows
    * New class :class:`~acoular.signals.FiltWNoiseGenerator`
    * New classes :class:`~acoular.sources.SphericalHarmonicSource`, :class:`~acoular.sources.Linesource`, :class:`~acoular.sources.MovingPointSourceDipole`, :class:`~acoular.sources.MovingLineSource`
    * New class :class:`~acoular.tprocess.TimeConvolve`
    * Speedups: 
        * CSM works now in parallel and is faster
        * frequency domain beamformers are abaout 30% faster
        * time domain beamformers and CLEAN-T is now about 10 x faster
    * Unittests for all major features, much better coverage
    * Bugfixes:
        * integration sectors
        * new grid classes from 20.10 are now imported into module namespace

20.10
------------
	
	* Supports Python 3.6, 3.7, 3.8
	* New base classes for time signal processing: :class:`~acoular.tprocess.Filter`, :class:`~acoular.tprocess.FilterBank`
		* New filter classes: :class:`~acoular.tprocess.TimeExpAverage`, :class:`~acoular.tprocess.FiltFreqWeight`, :class:`~acoular.tprocess.OctaveFilterBank`
		* Demo script is now part of module (see :doc:`../get_started/index` for usage)
	* New class for processing sound card input: :class:`~acoular.sdinput.SoundDeviceSamplesGenerator`
	* New class for cumulative averaging: :class:`~acoular.tprocess.TimeCumAverage`
	* New grid classes :class:`~acoular.grids.ImportGrid`, :class:`~acoular.grids.LineGrid`, :class:`~acoular.grids.MergeGrid`
	* New versatile integration sector classes :class:`~acoular.grids.RectSector`, :class:`~acoular.grids.CircSector`, :class:`~acoular.grids.PolySector`, :class:`~acoular.grids.ConvexSector`, :class:`~acoular.grids.MultiSector`
	* New class for mixing multiple channels :class:`~acoular.tprocess.ChannelMixer`
	* New class for using arbitrary sources as signal: :class:`~acoular.signals.GenericSignalGenerator`
	* New time-domain beamforming classes for CLEAN-T method: :class:`~acoular.tbeamform.BeamformerCleant`, :class:`~acoular.tbeamform.BeamformerCleantSq`, :class:`~acoular.tbeamform.BeamformerCleantTraj`, :class:`~acoular.tbeamform.BeamformerCleantSqTraj`
	* Adds possibility to store/load metadata to/from HDF files with :class:`~acoular.tprocess.WriteH5`/:class:`~acoular.sources.TimeSamples` classes
	* New submodule :mod:`~acoular.tools` containing several helper functions, e.g. for data aggregation
	* :class:`~acoular.tprocess.WriteWAV`: user can optionally set name of output wav file
	* Bugfix: PowerSpectra.freq_range / .ind_low / .ind_high are now correctly updated depending on changed attributes
	* Bugfix: :class:`~acoular.signals.SineGenerator` amplitude is now set via :attr:`~acoular.signals.SineGenerator.amplitude` attribute
	* Some minor fixes


20.02
------------

    * Adds gradient-based solver for BeamformerCMF (L_BFGS_B)
    * Adds possibilty to take into account convective amplification of moving sources in BeamformerTimeTraj
    * TraitsUI support of the classes has to be activated via :attr:`~acoular.configuration.config`
    * Bugfix: use left-orientated coordinate system in rotating flow environment
    * Bugfix: correct wrong angle shift in spline interpolation
    * Removes several Python 2.7-specific implementations


19.11
------------
    * Adds new classes for handling rotating data, including detection of trigger signals and interpolation of sensor data for virtual array emulation (:class:`~acoular.tprocess.Trigger`, :class:`~acoular.tprocess.AngleTracker`, :class:`~acoular.tprocess.SpatialInterpolator`, :class:`~acoular.tprocess.SpatialInterpolatorRotation`, :class:`~acoular.tprocess.SpatialInterpolatorConstantRotation`)
    * Introduces new :class:`~acoular.tprocess.SampleSplitter` class, which allows distribution of data streams 
    * Adds new (global) caching options for more flexible cache file handling (e.g. never cache results, always cache, cache read-only). See :class:`~acoular.configuration.config` for information on how to use this.
    * User can choose whether to use h5py or pytables package for handling hdf files. See :class:`~acoular.configuration.config` for information on how to use this.
    * Change: BeamformerGIB behaviour (not calculating sources with eigenvalue of zero)
    * Bugfix: BeamformerTime interpolation
    * Bugfix: Integer division in PNoiseGenerator
    * Test suite and CI updates


19.08
------------
    
    * Supports Python 3.5, 3.6, 3.7
    * This will be the last version to officially support Python 2.7
    * Cache and data directories are now always created in current directory (Linux and Windows)
    * Bugfix: Steering vector backwards compatibility
    * Bugfix: Ambiguous caching (changes in an object's class name as well as changes in the data file content are now monitored for caching)
    * PowerSpectra: Frequency range to be evaluated can be set directly
    * Some code clean-up
    * Renamed examples



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

    * Added support for Python 3.4, 3.5 and 3.6 
    * Implementation of fast/parallelized code now with Numba (instead of C++ and SciPy.weave)
    * cross spectral matrix (CSM) orientation changed (was transposed in earlier versions). Please do not use the cache files from earlier versions in version 17.11!
    	



