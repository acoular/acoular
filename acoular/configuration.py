# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements global configuration of Acoular.

.. autosummary::
    :toctree: generated/

    Config
    config
"""

import importlib.util
import sys
from os import environ, mkdir, path
from warnings import warn

# When numpy is using OpenBLAS then it runs with OPENBLAS_NUM_THREADS which may lead to
# overcommittment when called from within numba jitted function that run on
# NUMBA_NUM_THREADS. If overcommitted, things get extremely! slow. Therefore we make an
# attempt to avoid this situation. The main problem is that OPENBLAS_NUM_THREADS is
# only respected once numpy starts. Later on, it cannot be changed.

# we check if numpy already loaded
if 'numpy' in sys.modules:
    # numpy is loaded
    # temporarily route stdout to string
    import io

    import numpy as np

    orig_stdout = sys.stdout
    temp_stdout = io.StringIO()
    sys.stdout = temp_stdout
    np.show_config()
    sys.stdout = orig_stdout
    # check if it uses OpenBLAS or another library
    if 'openblas' in temp_stdout.getvalue().lower():
        # it's OpenBLAS, set numba threads=1 to avoid overcommittment
        import numba

        numba.set_num_threads(1)
        warn(
            'We detected that Numpy is already loaded and uses OpenBLAS. Because '
            'this conflicts with Numba parallel execution, we disable parallel '
            'execution for now and processing might be slower. To speed up, '
            'either import Numpy after Acoular or set environment variable '
            'OPENBLAS_NUM_THREADS=1 before start of the program.',
            UserWarning,
            stacklevel=2,
        )
else:
    # numpy is not loaded
    environ['OPENBLAS_NUM_THREADS'] = '1'

# this loads numpy, so we have to defer loading until OpenBLAS check is done
from traits.api import Bool, Either, HasStrictTraits, Property, Str, Trait, cached_property


class Config(HasStrictTraits):
    """Implements the global configuration of the Acoular package.

    An instance of this class can be accessed for adjustment of the following
    properties.
    General caching behaviour can be controlled by :attr:`global_caching`.
    The package used to read and write .h5 files can be specified
    by :attr:`h5library`.

    Example:
    --------
        For using Acoular with h5py package and overwrite existing cache:

        >>>    import acoular
        >>>    acoular.config.h5library = "h5py"
        >>>    acoular.config.global_caching = "overwrite"

    """

    def __init__(self):
        HasStrictTraits.__init__(self)
        self._assert_h5library()

    #: Flag that globally defines caching behaviour of Acoular classes
    #: defaults to 'individual'.
    #:
    #: * 'individual': Acoular classes handle caching behavior individually.
    #: * 'all': Acoular classes cache everything and read from cache if possible.
    #: * 'none': Acoular classes do not cache results. Cachefiles are not created.
    #: * 'readonly': Acoular classes do not actively cache, but read from cache if existing.
    #: * 'overwrite': Acoular classes replace existing cachefile content with new data.
    global_caching = Property()

    _global_caching = Trait('individual', 'all', 'none', 'readonly', 'overwrite')

    #: Flag that globally defines package used to read and write .h5 files
    #: defaults to 'pytables'. It is also possible to set it to 'tables', which is an alias for 'pytables'.
    #: If 'pytables' can not be imported, 'h5py' is used.
    h5library = Property()

    _h5library = Either('pytables', 'tables', 'h5py', default='pytables')

    #: Defines the path to the directory containing Acoulars cache files.
    #: If the specified :attr:`cache_dir` directory does not exist,
    #: it will be created. :attr:`cache_dir` defaults to current session path.
    cache_dir = Property()

    _cache_dir = Str('')

    #: Defines the working directory containing data files. Used only by
    #: :class:`~acoular.tprocess.WriteH5` class.
    #: Defaults to current session path.
    td_dir = Property()

    _td_dir = Str(path.curdir)

    #: Boolean Flag that determines whether user has access to traitsui features.
    #: Defaults to False.
    use_traitsui = Property()

    _use_traitsui = Bool(False)

    #: Boolean Flag that determines whether tables is installed.
    have_tables = Property()

    #: Boolean Flag that determines whether h5py is installed.
    have_h5py = Property()

    #: Boolean Flag that determines whether matplotlib is installed.
    have_matplotlib = Property()

    #: Boolean Flag that determines whether pylops is installed.
    have_pylops = Property()

    #: Boolean Flag that determines whether sounddevice is installed.
    have_sounddevice = Property()

    def _get_global_caching(self):
        return self._global_caching

    def _set_global_caching(self, globalCachingValue):
        self._global_caching = globalCachingValue

    def _get_h5library(self):
        return self._h5library

    def _set_h5library(self, libraryName):
        self._h5library = libraryName

    def _get_use_traitsui(self):
        return self._use_traitsui

    def _set_use_traitsui(self, use_tui):
        if use_tui:
            from . import traitsviews
            # If user tries to use traitsuis and it's not installed, this will throw an error.
        self._use_traitsui = use_tui

    def _assert_h5library(self):
        if not self.have_tables and not self.have_h5py:
            msg = 'Packages H5py and PyTables are missing! At least one of them is required for Acoular to work.'
            raise ImportError(msg)
        if not self.have_tables:
            self.h5library = 'h5py'

    def _get_cache_dir(self):
        if self._cache_dir == '':
            cache_dir = path.join(path.curdir, 'cache')
            if not path.exists(cache_dir):
                mkdir(cache_dir)
            self._cache_dir = cache_dir
        return self._cache_dir

    def _set_cache_dir(self, cdir):
        if not path.exists(cdir):
            mkdir(cdir)
        self._cache_dir = cdir

    def _get_td_dir(self):
        return self._td_dir

    def _set_td_dir(self, tddir):
        self._td_dir = tddir

    def _have_module(self, module_name):
        spec = importlib.util.find_spec(module_name)
        return spec is not None

    @cached_property
    def _get_have_matplotlib(self):
        return self._have_module('matplotlib')

    @cached_property
    def _get_have_pylops(self):
        return self._have_module('pylops')

    @cached_property
    def _get_have_sounddevice(self):
        return self._have_module('sounddevice')

    @cached_property
    def _get_have_tables(self):
        return self._have_module('tables')

    @cached_property
    def _get_have_h5py(self):
        return self._have_module('h5py')


config = Config()
"""
This instance implements the global configuration of the Acoular package.

General caching behaviour can be controlled by the :attr:`global_caching` attribute:
* 'individual': Acoular classes handle caching behavior individually.
* 'all': Acoular classes cache everything and read from cache if possible.
* 'none': Acoular classes do not cache results. Cachefiles are not created.
* 'readonly': Acoular classes do not actively cache, but read from cache if existing.
* 'overwrite': Acoular classes replace existing cachefile content with new data.

The package used to read and write .h5 files can be specified
by :attr:`h5library`:
* 'PyTables': Use 'tables' (or 'pytables', depending on python distribution).
* 'H5py': Use 'h5py'.

Some Acoular classes support GUI elements for usage with tools from the TraitsUI package.
If desired, this package has to be installed manually, as it is not a prerequisite for
installing Acoular.
To enable the functionality, the flag attribute :attr:`use_traitsui` has to be set to True (default: False).
Note: this is independent from the GUI tools implemented in the spectAcoular package.


Example:
    For using Acoular with h5py package and overwrite existing cache:

    >>>    import acoular
    >>>    acoular.config.h5library = "h5py"
    >>>    acoular.config.global_caching = "overwrite"
"""
