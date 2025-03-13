What's new
============

Upcoming Release
------------------------

25.03
------------------------
    **New features:**
        * Introduces `rps` trait in :class:`~acoular.environments.FlowField`, that now correctly defines direction of rotation

    **Documentation**
        * add known issue section to install guide
        * adds new docstrings to submodules :mod:`acoular.spectra`, :mod:`acoular.environments`, :mod:`acoular.fprocess`, :mod:`acoular.process`, :mod:`acoular.signals`, :mod:`acoular.sources`, :mod:`acoular.trajectory`

    **Internal**
        * fix broken meta.yaml (adds conda build stage to CI)

25.01
------------------------

    **New features:**
        * consistently use `file` as an attribute to specify a filename (instead of `from_file` or `name`)
        * consistently use `n_iter` in beamformer algorithms that require setting a fixed or maximum number of iterations (e.g. `max_iter` deprecated)
        * adds new function :meth:`~acoular.tools.helper.c_air` to calculate the speed of sound from temperature and humidity
        * :class:`~acoular.calib.Calib` can now be used as a separate processing block
        * enable varying block buffer sizes for :class:`~acoular.process.SampleSplitter`
        * Replaces `HasPrivateTraits` with `HasStrictTraits` and `ABCHasStrictTraits` for better implementation of ABCs.
        * Allow Path objects to specify the caching and time data directory via `acoular.Config`
        * Adds new abstract base classes :class:`~acoular.signals.PeriodicSignalGenerator` and :class:`~acoular.signals.NoiseGenerator`
        * adds support for Python 3.13

    **Bugfixes**
        * pass missing `n_iter` as an argument to `LassoLarsCV` in :class:`acoular.fbeamform.BeamformerGIB` (otherwise, `LassoLarsCV` runs up to 500 iterations)
        * fix broken `pylops` solvers in :class:`~acoular.fbeamform.BeamformerCMF`
        * fixes negative values in source maps obtained with the solvers `LassoLars`, `LassoLarsCV` and `LassoLarsBIC`
        * fix use of `Trait()` factory and replace by `Enum`, `Instance`, `Map`, `Union`
        * fix use of deprecated `CLong`, `Long`, and `ListInt` traits
        * The previously broken example in the main `README` now runs out of the box.

    **Documentation**
        * adds guide on how to submit a pull request
        * adds `intersphinx` extension to cross-link documentation from other projects
        * updates examples to use a right hand co-ordinate system

    **Tests**
        * tests now consequently use `pytest` framework instead of `unittest`
        * separate tests in into `tests/regression` and `tests/unittests` directories
        * increases coverage to 76 %
        * introduces snapshot / regression testing for all :class:`~acoular.base.Generator`, :class:`~acoular.fbeamform.BeamformerBase`, :class:`~acoular.environments.FlowField`, and :class:`~acoular.environments.Environment` derived classes (results in new snapshot data of size 1.1MB (see `tests/regression/_regtest_output`)
        * use `pytest-cases` to separate parameters from tests (cases are located under `tests/cases`)
        * full coverage of `sdinput.py` module through the use of `pytest-mock` (mocks the behavior of an `InputStream` object, which cannot be used for CI due to missing audio interfaces)
        * linting and formatting for tests directory
        * refactor :class:`~acoular.h5cache.HDF5Cache` class due to a bug encountered with the new tests (`acoular` now can handle changing caching directories during a session. Previously, only the basename was observed which caused problems with changing cache directories)
        * tests now log the 10 slowest test runs
        * adds `profile` options to hatch test environment to profile test run via `hatch run tests:profile` and save a `graphviz` chart as SVG file
        * test AIAA benchmark classes with the benchmark data
        * test `aperture`, `center`, `export_mpos` functions in :class:`~acoular.microphones.MicGeom`
        * fix a typo in `AngleTracker` that lead to a property being incorrectly accessed
        * exclude jited functions from test coverage report
        * treat warnings as errors in tests
        * recursively tests for empty dependencies in `Property` traits that are depended on

    **Internal**
        * adds a pull request template
        * dynamically set the version in the `pyproject.toml` file (from `version.py`)
        * activates maximum line length enforcement of 120 and 100 for comments and docstrings
        * adds CI workflow for inspecting regression test outputs
        * adds action that automatically assigns a team member to newly opened pull requests
        * `depends_on` now only accepts a list of strings
        * removes deprecated traits ending with version 25.01
        * include doctests in coverage report
        * no longer add docs label if `news/index.rst` is updated

24.10
----------------

    **New features:**
        * Sounddevice inputs now allow for user-settable sample rates and precision types
        * Block-wise frequency domain processing
            * Introduces new `acoular.base` module with abstract base classes :class:`~acoular.base.Generator`, :class:`~acoular.base.SpectraGenerator`, :class:`~acoular.base.InOut`, :class:`~acoular.base.TimeOut`, :class:`~acoular.base.SpectraOut`
            * Introduces new module `acoular.fprocess` including new frequency domain processing classes :class:`~acoular.fprocess.RFFT`, :class:`~acoular.fprocess.IRFFT`, :class:`~acoular.fprocess.AutoPowerSpectra` and :class:`~acoular.fprocess.CrossPowerSpectra`
            * Introduces new module `acoular.process` including the following classes for general purpose (domain invariant) processing: :class:`~acoular.process.Cache`, :class:`~acoular.process.Average`, :class:`~acoular.process.SampleSplitter`
            * Rename :class:`~acoular.tprocess.MaskedTimeInOut` to :class:`~acoular.tprocess.MaskedTimeOut`
            * Deprecates: :class:`~acoular.base.TimeInOut`, :class:`~acoular.fprocess.FFTSpectra`, :class:`~acoular.process.TimeAverage`, :class:`~acoular.tprocess.MaskedTimeInOut`, :class:`~acoular.process.TimeCache`
            * adds unittests `test_process.py`, `test_fprocess.py`
            * adds documentation example `example_fft.py`
                * allow more platforms to build the docs files including Linux, MacOS, and Windows
            * move submodule `acoular.tools.aiaa` to `acoular.aiaa`

    **Bugfixes**
        * flush file in :class:`~acoular.tprocess.TimeCache` to prevent data loss / corruption
        * fixes use of already deprecated traits
        * fixes import order to avoid the deactivation of parallelization of numba
        * fixes numpy.matlib deprecation warning
        * better handling of `basename` attributes:
            * adds new utils functions `find_basename` and `get_file_basename` for consistent handling of `basename` attributes
            * deprecates `basename` attribute of :class:`~acoular.tprocess.WriteWAV` and :class:`~acoular.tprocess.MaskedTimeOut` for reasons of inconsistency

    **Documentation**
        * updates literature and adds `sphinxcontrib-bibtex` extension to support bibtex citation in the documentation.
        * cleans-up and structures the conf.py file and fixes missing acoular logo in documentation
        * fixes missing and wrong docstrings for `result` methods in :mod:`acoular.tbeamform`
        * improves the explanation of the installation of dependencies
        * add explanation on how to test specific Python version only
        * test documentation build on all latest platforms

    **Internal**
        * refactoring of classes in :mod:`acoular.tbeamform` (moves buffer logic away from Beamformer classes)
            * adds new :class:`~acoular.process.SamplesBuffer` class located in :mod:`~acoular.tools.utils`
        * replaces the markdown-link-check by lychee
            * faster CI (written in RUST)
            * allows more files to be checked, including the .rst files in the documentation
            * adds a cron job that runs daily
        * sets final version for several deprecated traits. (Will be removed in version 25.01)
        * use scipy.linalg consistently over numpy.linalg
        * drops support for Python 3.8 and 3.9
        * enable Numpy version > 2.0
        * run unittests on the latest MacOS

24.07
------------

    New features:
        * Implement a lazy result array for :class:`acoular.fbeamform.BeamformerBase` derived classes
        * Adds flow establishment length traits
        * Updates acoular demo with ASCII map and gets rid of writing file

    Bugfixes:
        * temporary fix to PyTables - NumPy 2.0 incompatibility problem
        * Fixes :class:`acoular.fbeamform.BeamformerGridlessOrth` `shgo` params and `eva_list` initial value bug
        * makes :class:`acoular.sdinput.SoundDeviceSamplesGenerator` always importable

    Internal:
        * use hatch for publishing the Acoular package
        * add issue templates
        * adds Conda CI
        * CI for TestPyPI and PyPI
        * remove `plot_example.py`
        * add autolabel rule for `fix` and `linting`
        * fix linting rules
        * add zenodo release to `CITAITON.cff`


24.05
------------
    * adds support for Python version 3.12 on Linux, MacOS, Windows
    * drops official support for Python version 3.7
    * provides new tools to import  data in AIAA array benchmark format

    * Bugfixes:
        * changes to UMA-16 microphone array arrangement

    * Internal:
        * formatting and linting with ruff
        * introduce hatch
        * measure test coverage
        * replace `zenodo.json` by `CITATION.cff`
        * Bugfixes CI
        * update LICENSE
        * adds code of conduct
        * allow workflow dispatch for testing on different branches using GitHub
        * improve documentation
        * refine package structure
            * move test directory outside of the source directory
            * remove outdated submodules `fileimport` and `nidaqimport`
            * introduce new submodule `acoular/tools`



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
        * New class :class:`~acoular.fprocess.FFTSpectra` for time-frequency analysis
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
    * Introduces new :class:`~acoular.process.SampleSplitter` class, which allows distribution of data streams
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
    * Introduces new :class:`~acoular.fbeamform.SteeringVector` class (see :doc:`../get_started/index` and `../examples/index` for usage). With this, some of the Beamformer and PointSource traits are deprecated and should no longer be used. While the current version is intended to be fully compatible with older scripts, deprecation warnings will be raised if necessary.
    * Introduces optional use of reference distance for SPL evaluation (current default: reference position at (x,y,z)=(0,0,0) )
    * Introduces some basic Unit tests to evaluate the beamformer results
    * Bugfix: CLEAN algorithm now uses correct PSFs
    * some minor Bugfixes



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
