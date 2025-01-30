# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements beamformers in the frequency domain.

.. autosummary::
    :toctree: generated/

    SteeringVector


    BeamformerBase
    BeamformerFunctional
    BeamformerCapon
    BeamformerEig
    BeamformerMusic
    BeamformerClean
    BeamformerDamas
    BeamformerDamasPlus
    BeamformerOrth
    BeamformerCleansc
    BeamformerCMF
    BeamformerSODIX
    BeamformerGIB
    BeamformerAdaptiveGrid
    BeamformerGridlessOrth

    PointSpreadFunction
    L_p
    integrate

"""

# imports from other packages

import warnings
from warnings import warn

# check for sklearn version to account for incompatible behavior
import sklearn
from numpy import (
    absolute,
    arange,
    argsort,
    array,
    atleast_2d,
    clip,
    delete,
    diag,
    dot,
    einsum,
    einsum_path,
    eye,
    fill_diagonal,
    full,
    hsplit,
    hstack,
    index_exp,
    inf,
    integer,
    invert,
    isscalar,
    log10,
    ndarray,
    newaxis,
    ones,
    pi,
    real,
    reshape,
    round,  # noqa: A004
    searchsorted,
    sign,
    size,
    sqrt,
    sum,  # noqa: A004
    tile,
    trace,
    tril,
    unique,
    vstack,
    zeros,
    zeros_like,
)
from packaging.version import parse
from scipy.linalg import eigh, eigvals, fractional_matrix_power, inv, norm
from scipy.optimize import fmin_l_bfgs_b, linprog, nnls, shgo
from sklearn.linear_model import LassoLars, LassoLarsCV, LassoLarsIC, LinearRegression, OrthogonalMatchingPursuitCV
from traits.api import (
    Any,
    Bool,
    CArray,
    Dict,
    Enum,
    Float,
    HasStrictTraits,
    Instance,
    Int,
    List,
    Property,
    Range,
    Tuple,
    cached_property,
    on_trait_change,
    property_depends_on,
)
from traits.trait_errors import TraitError

# acoular imports
from .configuration import config
from .deprecation import deprecated_alias
from .environments import Environment
from .fastFuncs import beamformerFreq, calcPointSpreadFunction, calcTransfer, damasSolverGaussSeidel
from .grids import Grid, Sector
from .h5cache import H5cache
from .h5files import H5CacheFileBase
from .internal import digest
from .microphones import MicGeom
from .spectra import PowerSpectra
from .tfastfuncs import _steer_I, _steer_II, _steer_III, _steer_IV

sklearn_ndict = {}
if parse(sklearn.__version__) < parse('1.4'):
    sklearn_ndict['normalize'] = False  # pragma: no cover

BEAMFORMER_BASE_DIGEST_DEPENDENCIES = ['freq_data.digest', 'r_diag', 'r_diag_norm', 'precision', 'steer.digest']


class SteeringVector(HasStrictTraits):
    """Basic class for implementing steering vectors with monopole source transfer models.

    Handles four different steering vector formulations. See :cite:`Sarradj2012` for details.
    """

    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    grid = Instance(Grid, desc='beamforming grid')

    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Instance(MicGeom, desc='microphone geometry')

    #: Type of steering vectors, see also :cite:`Sarradj2012`. Defaults to 'true level'.
    steer_type = Enum('true level', 'true location', 'classic', 'inverse', desc='type of steering vectors used')

    #: :class:`~acoular.environments.Environment` or derived object,
    #: which provides information about the sound propagation in the medium.
    #: Defaults to standard :class:`~acoular.environments.Environment` object.
    env = Instance(Environment(), Environment)

    # Sound travel distances from microphone array center to grid
    # points or reference position (readonly). Feature may change.
    r0 = Property(desc='array center to grid distances')

    # Sound travel distances from array microphones to grid
    # points (readonly). Feature may change.
    rm = Property(desc='all array mics to grid distances')

    # mirror trait for ref
    _ref = Any(array([0.0, 0.0, 0.0]), desc='reference position or distance')

    #: Reference position or distance at which to evaluate the sound pressure
    #: of a grid point.
    #: If set to a scalar, this is used as reference distance to the grid points.
    #: If set to a vector, this is interpreted as x,y,z coordinates of the reference position.
    #: Defaults to [0.,0.,0.].
    ref = Property(desc='reference position or distance')

    _steer_funcs_freq = Dict(
        {
            'classic': lambda x: x / absolute(x) / x.shape[-1],
            'inverse': lambda x: 1.0 / x.conj() / x.shape[-1],
            'true level': lambda x: x / einsum('ij,ij->i', x, x.conj())[:, newaxis],
            'true location': lambda x: x / sqrt(einsum('ij,ij->i', x, x.conj()) * x.shape[-1])[:, newaxis],
        },
        desc='dictionary of frequency domain steering vector functions',
    )

    _steer_funcs_time = Dict(
        {
            'classic': _steer_I,
            'inverse': _steer_II,
            'true level': _steer_III,
            'true location': _steer_IV,
        },
        desc='dictionary of time domain steering vector functions',
    )

    def _set_ref(self, ref):
        if isscalar(ref):
            try:
                self._ref = absolute(float(ref))
            except ValueError as ve:
                raise TraitError(args=self, name='ref', info='Float or CArray(3,)', value=ref) from ve
        elif len(ref) == 3:
            self._ref = array(ref, dtype=float)
        else:
            raise TraitError(args=self, name='ref', info='Float or CArray(3,)', value=ref)

    def _get_ref(self):
        return self._ref

    # internal identifier
    digest = Property(depends_on=['steer_type', 'env.digest', 'grid.digest', 'mics.digest', '_ref'])

    # internal identifier, use for inverse methods, excluding steering vector type
    inv_digest = Property(depends_on=['env.digest', 'grid.digest', 'mics.digest', '_ref'])

    @property_depends_on(['grid.digest', 'env.digest', '_ref'])
    def _get_r0(self):
        if isscalar(self.ref):
            if self.ref > 0:
                return full((self.grid.size,), self.ref)
            return self.env._r(self.grid.pos())
        return self.env._r(self.grid.pos, self.ref[:, newaxis])

    @property_depends_on(['grid.digest', 'mics.digest', 'env.digest'])
    def _get_rm(self):
        return atleast_2d(self.env._r(self.grid.pos, self.mics.pos))

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_inv_digest(self):
        return digest(self)

    def transfer(self, f, ind=None):
        """Calculates the transfer matrix for one frequency.

        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the transfer function of the gridpoints addressed by
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed

        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the transfer matrix for the given frequency

        """
        # if self.cached:
        #    warn('Caching of transfer function is not yet supported!', Warning)
        #    self.cached = False

        if ind is None:
            trans = calcTransfer(self.r0, self.rm, array(2 * pi * f / self.env.c))
        elif not isinstance(ind, ndarray):
            trans = calcTransfer(self.r0[ind], self.rm[ind, :][newaxis], array(2 * pi * f / self.env.c))  # [0, :]
        else:
            trans = calcTransfer(self.r0[ind], self.rm[ind, :], array(2 * pi * f / self.env.c))
        return trans

    def steer_vector(self, f, ind=None):
        """Calculates the steering vectors based on the transfer function.

        See also :cite:`Sarradj2012`.

        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the steering vectors of the gridpoints addressed by
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed

        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the steering vectors for the given frequency

        """
        func = self._steer_funcs_freq[self.steer_type]
        return func(self.transfer(f, ind))


class LazyBfResult:
    """Manages lazy per-frequency evaluation."""

    # Internal helper class which works together with BeamformerBase to provide
    # calculation on demand; provides an 'intelligent' [] operator. This is
    # implemented as an extra class instead of as a method of BeamformerBase to
    # properly control the BeamformerBase.result attribute. Might be migrated to
    # be a method of BeamformerBase in the future.

    def __init__(self, bf):
        self.bf = bf

    def __getitem__(self, key):
        # 'intelligent' [] operator checks if results are available and triggers calculation
        sl = index_exp[key][0]
        if isinstance(sl, (int, integer)):
            sl = slice(sl, sl + 1)
        # indices which are missing
        missingind = arange(*sl.indices(self.bf._numfreq))[self.bf._fr[sl] == 0]
        # calc if needed
        if missingind.size:
            self.bf._calc(missingind)
            if self.bf.h5f:
                self.bf.h5f.flush()

        return self.bf._ac.__getitem__(key)


class BeamformerBase(HasStrictTraits):
    """Beamforming using the basic delay-and-sum algorithm in the frequency domain."""

    # Instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes
    # that contains information about the steering vector. This is a private trait.
    # Do not set this directly, use `steer` trait instead.
    steer = Instance(SteeringVector, args=())

    #: :class:`~acoular.spectra.PowerSpectra` object that provides the
    #: cross spectral matrix and eigenvalues
    freq_data = Instance(PowerSpectra, desc='freq data object')

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, desc='removal of diagonal')

    #: If r_diag==True: if r_diag_norm==0.0, the standard
    #: normalization = num_mics/(num_mics-1) is used.
    #: If r_diag_norm !=0.0, the user input is used instead.
    #: If r_diag==False, the normalization is 1.0 either way.
    r_diag_norm = Float(
        0.0,
        desc='If diagonal of the csm is removed, some signal energy is lost.'
        'This is handled via this normalization factor.'
        'Internally, the default is: num_mics / (num_mics - 1).',
    )

    #: Floating point precision of property result. Corresponding to numpy dtypes. Default = 64 Bit.
    precision = Enum('float64', 'float32', desc='precision (32/64 Bit) of result, corresponding to numpy dtypes')

    #: Boolean flag, if 'True' (default), the result is cached in h5 files.
    cached = Bool(True, desc='cached flag')

    # hdf5 cache file
    h5f = Instance(H5CacheFileBase, transient=True)

    #: The beamforming result as squared sound pressure values
    #: at all grid point locations (readonly).
    #: Returns a (number of frequencies, number of gridpoints) array-like
    #: of floats. Values can only be accessed via the index operator [].
    result = Property(desc='beamforming result')

    # internal identifier
    digest = Property(depends_on=BEAMFORMER_BASE_DIGEST_DEPENDENCIES)

    # private traits
    _ac = Any(desc='beamforming result')
    _fr = Any(desc='flag for beamforming result at frequency index')
    _f = CArray(dtype='float64', desc='frequencies')
    _numfreq = Int(desc='number of frequencies')

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_filecache(self):
        """Function collects cached results from file depending on
        global/local caching behaviour. Returns (None, None) if no cachefile/data
        exist and global caching mode is 'readonly'.
        """
        #        print("get cachefile:", self.freq_data.basename)
        H5cache.get_cache_file(self, self.freq_data.basename)
        if not self.h5f:
            #            print("no cachefile:", self.freq_data.basename)
            return (None, None, None)  # only happens in case of global caching readonly

        nodename = self.__class__.__name__ + self.digest
        #        print("collect filecache for nodename:",nodename)
        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
            #            print("remove existing data for nodename",nodename)
            self.h5f.remove_data(nodename)  # remove old data before writing in overwrite mode

        if not self.h5f.is_cached(nodename):
            #            print("no data existent for nodename:", nodename)
            if config.global_caching == 'readonly':
                return (None, None, None)
            numfreq = self.freq_data.fftfreq().shape[0]
            group = self.h5f.create_new_group(nodename)
            self.h5f.create_compressible_array(
                'freqs',
                (numfreq,),
                'int8',  #'bool',
                group,
            )
            if isinstance(self, BeamformerAdaptiveGrid):
                self.h5f.create_compressible_array('gpos', (3, self.size), 'float64', group)
                self.h5f.create_compressible_array('result', (numfreq, self.size), self.precision, group)
            elif isinstance(self, BeamformerSODIX):
                self.h5f.create_compressible_array(
                    'result',
                    (numfreq, self.steer.grid.size * self.steer.mics.num_mics),
                    self.precision,
                    group,
                )
            else:
                self.h5f.create_compressible_array('result', (numfreq, self.steer.grid.size), self.precision, group)

        ac = self.h5f.get_data_by_reference('result', '/' + nodename)
        fr = self.h5f.get_data_by_reference('freqs', '/' + nodename)
        if isinstance(self, BeamformerAdaptiveGrid):
            gpos = self.h5f.get_data_by_reference('gpos', '/' + nodename)
        else:
            gpos = None
        return (ac, fr, gpos)

    def _assert_equal_channels(self):
        num_channels = self.freq_data.num_channels
        if num_channels != self.steer.mics.num_mics or num_channels == 0:
            msg = f'{num_channels:d} channels do not fit {self.steer.mics.num_mics:d} mics'
            raise ValueError(msg)

    @property_depends_on(['digest'])
    def _get_result(self):
        """Implements the :attr:`result` getter routine.
        The beamforming result is either loaded or calculated.
        """
        # store locally for performance
        self._f = self.freq_data.fftfreq()
        self._numfreq = self._f.shape[0]
        self._assert_equal_channels()
        ac, fr = (None, None)
        if not (  # if result caching is active
            config.global_caching == 'none' or (config.global_caching == 'individual' and not self.cached)
        ):
            (ac, fr, gpos) = self._get_filecache()  # can also be (None, None, None)
            if gpos:  # we have an adaptive grid
                self._gpos = gpos
        if ac and fr:  # cached data is available
            if config.global_caching == 'readonly':
                (ac, fr) = (ac[:], fr[:])  # so never write back to disk
        else:
            # no caching or not activated, init numpy arrays
            if isinstance(self, BeamformerAdaptiveGrid):
                self._gpos = zeros((3, self.size), dtype=self.precision)
                ac = zeros((self._numfreq, self.size), dtype=self.precision)
            elif isinstance(self, BeamformerSODIX):
                ac = zeros((self._numfreq, self.steer.grid.size * self.steer.mics.num_mics), dtype=self.precision)
            else:
                ac = zeros((self._numfreq, self.steer.grid.size), dtype=self.precision)
            fr = zeros(self._numfreq, dtype='int8')
        self._ac = ac
        self._fr = fr
        return LazyBfResult(self)

    def sig_loss_norm(self):
        """If the diagonal of the CSM is removed one has to handle the loss
        of signal energy --> Done via a normalization factor.
        """
        if not self.r_diag:  # Full CSM --> no normalization needed
            normfactor = 1.0
        elif self.r_diag_norm == 0.0:  # Removed diag: standard normalization factor
            nMics = float(self.freq_data.num_channels)
            normfactor = nMics / (nMics - 1)
        elif self.r_diag_norm != 0.0:  # Removed diag: user defined normalization factor
            normfactor = self.r_diag_norm
        return normfactor

    def _beamformer_params(self):
        """Manages the parameters for calling of the core beamformer functionality.
        This is a workaround to allow faster calculation and may change in the
        future.

        Returns
        -------
            - String containing the steering vector type
            - Function for frequency-dependent steering vector calculation

        """
        if type(self.steer) is SteeringVector:  # for simple steering vector, use faster method
            param_type = self.steer.steer_type

            def param_steer_func(f):
                return (self.steer.r0, self.steer.rm, 2 * pi * f / self.steer.env.c)
        else:
            param_type = 'custom'
            param_steer_func = self.steer.steer_vector
        return param_type, param_steer_func

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        normfactor = self.sig_loss_norm()
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            # print(f'compute{i}')
            csm = array(self.freq_data.csm[i], dtype='complex128')
            beamformerOutput = beamformerFreq(
                param_steer_type,
                self.r_diag,
                normfactor,
                steer_vector(f[i]),
                csm,
            )[0]
            if self.r_diag:  # set (unphysical) negative output values to 0
                indNegSign = sign(beamformerOutput) < 0
                beamformerOutput[indNegSign] = 0.0
            self._ac[i] = beamformerOutput
            self._fr[i] = 1

    def synthetic(self, f, num=0):
        """Evaluates the beamforming result for an arbitrary frequency band.

        Parameters
        ----------
        f: float
            Band center frequency.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================

        Returns
        -------
        array of floats
            The synthesized frequency band values of the beamforming result at
            each grid point .
            Note that the frequency resolution and therefore the bandwidth
            represented by a single frequency line depends on
            the :attr:`sampling frequency<acoular.base.SamplesGenerator.sample_freq>` and
            used :attr:`FFT block size<acoular.spectra.PowerSpectra.block_size>`.

        """
        res = self.result  # trigger calculation
        freq = self.freq_data.fftfreq()
        if len(freq) == 0:
            return None

        if num == 0:
            # single frequency line
            ind = searchsorted(freq, f)
            if ind >= len(freq):
                warn(
                    f'Queried frequency ({f:g} Hz) not in resolved frequency range. Returning zeros.',
                    Warning,
                    stacklevel=2,
                )
                h = zeros_like(res[0])
            else:
                if freq[ind] != f:
                    warn(
                        f'Queried frequency ({f:g} Hz) not in set of '
                        'discrete FFT sample frequencies. '
                        f'Using frequency {freq[ind]:g} Hz instead.',
                        Warning,
                        stacklevel=2,
                    )
                h = res[ind]
        else:
            # fractional octave band
            if isinstance(num, list):
                f1 = num[0]
                f2 = num[-1]
            else:
                f1 = f * 2.0 ** (-0.5 / num)
                f2 = f * 2.0 ** (+0.5 / num)
            ind1 = searchsorted(freq, f1)
            ind2 = searchsorted(freq, f2)
            if ind1 == ind2:
                warn(
                    f'Queried frequency band ({f1:g} to {f2:g} Hz) does not '
                    'include any discrete FFT sample frequencies. '
                    'Returning zeros.',
                    Warning,
                    stacklevel=2,
                )
                h = zeros_like(res[0])
            else:
                h = sum(res[ind1:ind2], 0)
        if isinstance(self, BeamformerAdaptiveGrid):
            return h
        if isinstance(self, BeamformerSODIX):
            return h.reshape((self.steer.grid.size, self.steer.mics.num_mics))
        return h.reshape(self.steer.grid.shape)

    def integrate(self, sector, frange=None, num=0):
        """Integrates result map over a given sector.

        Parameters
        ----------
        sector: array of floats or :class:`~acoular.grids.Sector`
            either an array, tuple or list with arguments for the 'indices'
            method of a :class:`~acoular.grids.Grid`-derived class
            (e.g. :meth:`RectGrid.indices<acoular.grids.RectGrid.indices>`
            or :meth:`RectGrid3D.indices<acoular.grids.RectGrid3D.indices>`).
            Possible sectors would be *array([xmin, ymin, xmax, ymax])*
            or *array([x, y, radius])* or an instance of a
            :class:`~acoular.grids.Sector`-derived class

        frange: tuple or None
            a tuple of (fmin,fmax) frequencies to include in the result if *num*==0,
            or band center frequency/frequencies for which to return the results
            if *num*>0; if None, then the frequency range is determined from
            the settings of the :attr:`PowerSpectra.ind_low` and
            :attr:`PowerSpectra.ind_high` of :attr:`freq_data`

        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line). Only considered if *frange* is not None.

            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================


        Returns
        -------
        res or (f, res): array of floats or tuple(array of floats, array of floats)
            If *frange*==None or *num*>0, the spectrum (all calculated frequency bands)
            for the integrated sector is returned as *res*. The dimension of this array is the
            number of frequencies given by :attr:`freq_data` and entries not computed are zero.
            If *frange*!=None and *num*==0, then (f, res) is returned where *f* are the (band)
            frequencies and the dimension of both arrays is determined from *frange*
        """
        if isinstance(sector, Sector):
            ind = self.steer.grid.subdomain(sector)
        elif hasattr(self.steer.grid, 'indices'):
            ind = self.steer.grid.indices(*sector)
        else:
            msg = (
                f'Grid of type {self.steer.grid.__class__.__name__} does not have an indices method! '
                f'Please use a sector derived instance of type :class:`~acoular.grids.Sector` '
                'instead of type numpy.array.'
            )
            raise NotImplementedError(
                msg,
            )
        gshape = self.steer.grid.shape
        if num == 0 or frange is None:
            if frange is None:
                ind_low = self.freq_data.ind_low
                ind_high = self.freq_data.ind_high
                if ind_low is None:
                    ind_low = 0
                if ind_low < 0:
                    ind_low += self._numfreq
                if ind_high is None:
                    ind_high = self._numfreq
                if ind_high < 0:
                    ind_high += self._numfreq
                irange = (ind_low, ind_high)
                num = 0
            elif len(frange) == 2:
                irange = (searchsorted(self._f, frange[0]), searchsorted(self._f, frange[1]))
            else:
                msg = 'Only a tuple of length 2 is allowed for frange if num==0'
                raise TypeError(
                    msg,
                )
            h = zeros(self._numfreq, dtype=float)
            sl = slice(*irange)
            r = self.result[sl]
            for i in range(*irange):
                # we do this per frequency because r might not have fancy indexing
                h[i] = r[i - sl.start].reshape(gshape)[ind].sum()
            if frange is None:
                return h
            return self._f[sl], h[sl]

        h = zeros(len(frange), dtype=float)
        for i, f in enumerate(frange):
            h[i] = self.synthetic(f, num).reshape(gshape)[ind].sum()
        return h


class BeamformerFunctional(BeamformerBase):
    """Functional beamforming algorithm.

    See :cite:`Dougherty2014` for details.
    """

    #: Functional exponent, defaults to 1 (= Classic Beamforming).
    gamma = Float(1, desc='functional exponent')

    #: Functional Beamforming is only well defined for full CSM
    r_diag = Enum(False, desc='False, as Functional Beamformer is only well defined for the full CSM')

    #: Normalization factor in case of CSM diagonal removal. Defaults to 1.0 since Functional
    #: Beamforming is only well defined for full CSM.
    r_diag_norm = Enum(
        1.0,
        desc='No normalization needed. Functional Beamforming is only well defined for full CSM.',
    )

    # internal identifier
    digest = Property(depends_on=BEAMFORMER_BASE_DIGEST_DEPENDENCIES + ['gamma'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        normfactor = self.sig_loss_norm()
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            if self.r_diag:  # pragma: no cover
                # This case is not used at the moment (see Trait r_diag)
                # It would need some testing as structural changes were not tested...
                # ==============================================================================
                #                     One cannot use spectral decomposition when diagonal of csm is
                #                     removed, as the resulting modified eigenvectors are not
                #                     orthogonal to each other anymore. Therefore potentiating
                #                     cannot be applied only to the eigenvalues. --> To avoid this
                #                     the root of the csm (removed diag) is calculated directly.
                #                     WATCH OUT: This doesn't really produce good results.
                # ==============================================================================
                csm = self.freq_data.csm[i]
                fill_diagonal(csm, 0)
                csmRoot = fractional_matrix_power(csm, 1.0 / self.gamma)
                beamformerOutput, steerNorm = beamformerFreq(
                    param_steer_type,
                    self.r_diag,
                    normfactor,
                    steer_vector(f[i]),
                    csmRoot,
                )
                beamformerOutput /= steerNorm  # take normalized steering vec

                # set (unphysical) negative output values to 0
                indNegSign = sign(beamformerOutput) < 0
                beamformerOutput[indNegSign] = 0.0
            else:
                eva = array(self.freq_data.eva[i], dtype='float64') ** (1.0 / self.gamma)
                eve = array(self.freq_data.eve[i], dtype='complex128')
                beamformerOutput, steerNorm = beamformerFreq(
                    param_steer_type,
                    self.r_diag,
                    1.0,
                    steer_vector(f[i]),
                    (eva, eve),
                )
                beamformerOutput /= steerNorm  # take normalized steering vec
            self._ac[i] = (
                (beamformerOutput**self.gamma) * steerNorm * normfactor
            )  # the normalization must be done outside the beamformer
            self._fr[i] = 1


class BeamformerCapon(BeamformerBase):
    """Beamforming using the Capon (Mininimum Variance) algorithm.

    See :cite:`Capon1969` for details.
    """

    # Boolean flag, if 'True', the main diagonal is removed before beamforming;
    # for Capon beamforming r_diag is set to 'False'.
    r_diag = Enum(False, desc='removal of diagonal')

    #: Normalization factor in case of CSM diagonal removal. Defaults to 1.0 since Beamformer Capon
    #: is only well defined for full CSM.
    r_diag_norm = Enum(
        1.0,
        desc='No normalization. BeamformerCapon is only well defined for full CSM.',
    )

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        nMics = self.freq_data.num_channels
        normfactor = self.sig_loss_norm() * nMics**2
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            csm = array(inv(array(self.freq_data.csm[i], dtype='complex128')), order='C')
            beamformerOutput = beamformerFreq(param_steer_type, self.r_diag, normfactor, steer_vector(f[i]), csm)[0]
            self._ac[i] = 1.0 / beamformerOutput
            self._fr[i] = 1


class BeamformerEig(BeamformerBase):
    """Beamforming using eigenvalue and eigenvector techniques.

    See :cite:`Sarradj2005` for details.
    """

    #: Number of component to calculate:
    #: 0 (smallest) ... :attr:`~acoular.base.SamplesGenerator.num_channels`-1;
    #: defaults to -1, i.e. num_channels-1
    n = Int(-1, desc='No. of eigenvalue')

    # Actual component to calculate, internal, readonly.
    na = Property(desc='No. of eigenvalue')

    # internal identifier
    digest = Property(depends_on=BEAMFORMER_BASE_DIGEST_DEPENDENCIES + ['n'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property_depends_on(['steer.mics', 'n'])
    def _get_na(self):
        na = self.n
        nm = self.steer.mics.num_mics
        if na < 0:
            na = max(nm + na, 0)
        return min(nm - 1, na)

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        na = int(self.na)  # eigenvalue taken into account
        normfactor = self.sig_loss_norm()
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            eva = array(self.freq_data.eva[i], dtype='float64')
            eve = array(self.freq_data.eve[i], dtype='complex128')
            beamformerOutput = beamformerFreq(
                param_steer_type,
                self.r_diag,
                normfactor,
                steer_vector(f[i]),
                (eva[na : na + 1], eve[:, na : na + 1]),
            )[0]
            if self.r_diag:  # set (unphysical) negative output values to 0
                indNegSign = sign(beamformerOutput) < 0
                beamformerOutput[indNegSign] = 0
            self._ac[i] = beamformerOutput
            self._fr[i] = 1


class BeamformerMusic(BeamformerEig):
    """Beamforming using the MUSIC algorithm.

    See :cite:`Schmidt1986` for details.
    """

    # Boolean flag, if 'True', the main diagonal is removed before beamforming;
    # for MUSIC beamforming r_diag is set to 'False'.
    r_diag = Enum(False, desc='removal of diagonal')

    #: Normalization factor in case of CSM diagonal removal. Defaults to 1.0 since BeamformerMusic
    #: is only well defined for full CSM.
    r_diag_norm = Enum(
        1.0,
        desc='No normalization. BeamformerMusic is only well defined for full CSM.',
    )

    # assumed number of sources, should be set to a value not too small
    # defaults to 1
    n = Int(1, desc='assumed number of sources')

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        nMics = self.freq_data.num_channels
        n = int(self.steer.mics.num_mics - self.na)
        normfactor = self.sig_loss_norm() * nMics**2
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            eva = array(self.freq_data.eva[i], dtype='float64')
            eve = array(self.freq_data.eve[i], dtype='complex128')
            beamformerOutput = beamformerFreq(
                param_steer_type,
                self.r_diag,
                normfactor,
                steer_vector(f[i]),
                (eva[:n], eve[:, :n]),
            )[0]
            self._ac[i] = 4e-10 * beamformerOutput.min() / beamformerOutput
            self._fr[i] = 1


class PointSpreadFunction(HasStrictTraits):
    """The point spread function.

    This class provides tools to calculate the PSF depending on the used
    microphone geometry, focus grid, flow environment, etc.
    The PSF is needed by several deconvolution algorithms to correct
    the aberrations when using simple delay-and-sum beamforming.
    """

    # Instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes
    # that contains information about the steering vector. This is a private trait.
    # Do not set this directly, use `steer` trait instead.
    steer = Instance(SteeringVector, args=())

    #: Indices of grid points to calculate the PSF for.
    grid_indices = CArray(
        dtype=int,
        value=array([]),
        desc='indices of grid points for psf',
    )  # value=array([]), value=self.steer.grid.pos(),

    #: Flag that defines how to calculate and store the point spread function
    #: defaults to 'single'.
    #:
    #: * 'full': Calculate the full PSF (for all grid points) in one go (should be used if the PSF
    #:           at all grid points is needed, as with :class:`DAMAS<BeamformerDamas>`)
    #: * 'single': Calculate the PSF for the grid points defined by :attr:`grid_indices`, one by one
    #:             (useful if not all PSFs are needed, as with :class:`CLEAN<BeamformerClean>`)
    #: * 'block': Calculate the PSF for the grid points defined by :attr:`grid_indices`, in one go
    #:            (useful if not all PSFs are needed, as with :class:`CLEAN<BeamformerClean>`)
    #: * 'readonly': Do not attempt to calculate the PSF since it should already be cached (useful
    #:               if multiple processes have to access the cache file)
    calcmode = Enum('single', 'block', 'full', 'readonly', desc='mode of calculation / storage')

    #: Floating point precision of property psf. Corresponding to numpy dtypes. Default = 64 Bit.
    precision = Enum('float64', 'float32', desc='precision (32/64 Bit) of result, corresponding to numpy dtypes')

    #: The actual point spread function.
    psf = Property(desc='point spread function')

    #: Frequency to evaluate the PSF for; defaults to 1.0.
    freq = Float(1.0, desc='frequency')

    # hdf5 cache file
    h5f = Instance(H5CacheFileBase, transient=True)

    # internal identifier
    digest = Property(depends_on=['steer.digest', 'precision'], cached=True)

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_filecache(self):
        """Function collects cached results from file depending on
        global/local caching behaviour. Returns (None, None) if no cachefile/data
        exist and global caching mode is 'readonly'.
        """
        filename = 'psf' + self.digest
        nodename = (f'Hz_{self.freq:.2f}').replace('.', '_')
        #        print("get cachefile:", filename)
        H5cache.get_cache_file(self, filename)
        if not self.h5f:  # only happens in case of global caching readonly
            #            print("no cachefile:", filename)
            return (None, None)  # only happens in case of global caching readonly

        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
            #            print("remove existing data for nodename",nodename)
            self.h5f.remove_data(nodename)  # remove old data before writing in overwrite mode

        if not self.h5f.is_cached(nodename):
            #            print("no data existent for nodename:", nodename)
            if config.global_caching == 'readonly':
                return (None, None)
            gs = self.steer.grid.size
            group = self.h5f.create_new_group(nodename)
            self.h5f.create_compressible_array('result', (gs, gs), self.precision, group)
            self.h5f.create_compressible_array(
                'gridpts',
                (gs,),
                'int8',  #'bool',
                group,
            )
        ac = self.h5f.get_data_by_reference('result', '/' + nodename)
        gp = self.h5f.get_data_by_reference('gridpts', '/' + nodename)
        return (ac, gp)

    def _get_psf(self):
        """Implements the :attr:`psf` getter routine.
        The point spread function is either loaded or calculated.
        """
        gs = self.steer.grid.size
        if not self.grid_indices.size:
            self.grid_indices = arange(gs)

        if config.global_caching != 'none':
            #            print("get filecache..")
            (ac, gp) = self._get_filecache()
            if ac and gp:
                #                print("cached data existent")
                if not gp[:][self.grid_indices].all():
                    #                    print("calculate missing results")
                    if self.calcmode == 'readonly':
                        msg = "Cannot calculate missing PSF (points) in 'readonly' mode."
                        raise ValueError(msg)
                    if config.global_caching == 'readonly':
                        (ac, gp) = (ac[:], gp[:])
                        self.calc_psf(ac, gp)
                        return ac[:, self.grid_indices]
                    self.calc_psf(ac, gp)
                    self.h5f.flush()
                    return ac[:, self.grid_indices]
                #                else:
                #                    print("cached results are complete! return.")
                return ac[:, self.grid_indices]
            #           print("no caching, calculate result")
            ac = zeros((gs, gs), dtype=self.precision)
            gp = zeros((gs,), dtype='int8')
            self.calc_psf(ac, gp)
        else:  # no caching activated
            #            print("no caching activated, calculate result")
            ac = zeros((gs, gs), dtype=self.precision)
            gp = zeros((gs,), dtype='int8')
            self.calc_psf(ac, gp)
        return ac[:, self.grid_indices]

    def calc_psf(self, ac, gp):
        """point-spread function calculation."""
        if self.calcmode != 'full':
            # calc_ind has the form [True, True, False, True], except
            # when it has only 1 entry (value True/1 would be ambiguous)
            calc_ind = [0] if self.grid_indices.size == 1 else invert(gp[:][self.grid_indices])

            # get indices which have the value True = not yet calculated
            g_ind_calc = self.grid_indices[calc_ind]

        if self.calcmode == 'single':  # calculate selected psfs one-by-one
            for ind in g_ind_calc:
                ac[:, ind] = self._psf_call([ind])[:, 0]
                gp[ind] = 1
        elif self.calcmode == 'full':  # calculate all psfs in one go
            gp[:] = 1
            ac[:] = self._psf_call(arange(self.steer.grid.size))
        else:  # 'block' # calculate selected psfs in one go
            hh = self._psf_call(g_ind_calc)
            for indh, ind in enumerate(g_ind_calc):
                gp[ind] = 1
                ac[:, ind] = hh[:, indh]
                indh += 1

    def _psf_call(self, ind):
        """Manages the calling of the core psf functionality.

        Parameters
        ----------
        ind : list of int
            Indices of gridpoints which are assumed to be sources. Normalization factor for the
            beamforming result (e.g. removal of diag is compensated with this.)

        Returns
        -------
        The psf [1, nGridPoints, len(ind)]
        """
        if type(self.steer) is SteeringVector:  # for simple steering vector, use faster method
            result = calcPointSpreadFunction(
                self.steer.steer_type,
                self.steer.r0,
                self.steer.rm,
                2 * pi * self.freq / self.steer.env.c,
                ind,
                self.precision,
            )
        else:
            # for arbitrary steering sectors, use general calculation. there is a version of this in
            # fastFuncs, may be used later after runtime testing and debugging
            product = dot(self.steer.steer_vector(self.freq).conj(), self.steer.transfer(self.freq, ind).T)
            result = (product * product.conj()).real
        return result


class BeamformerDamas(BeamformerBase):
    """DAMAS deconvolution algorithm.

    See :cite:`Brooks2006` for details.
    """

    #: (only for backward compatibility) :class:`BeamformerBase` object
    #: if set, provides :attr:`freq_data`, :attr:`steer`, :attr:`r_diag`
    #: if not set, these have to be set explicitly.
    beamformer = Property()

    # private storage of beamformer instance
    _beamformer = Instance(BeamformerBase)

    #: The floating-number-precision of the PSFs. Default is 64 bit.
    psf_precision = Enum('float64', 'float32', desc='precision of PSF')

    #: Number of iterations, defaults to 100.
    n_iter = Int(100, desc='number of iterations')

    #: Damping factor in modified gauss-seidel
    damp = Float(1.0, desc='damping factor in modified gauss-seidel-DAMAS-approach')

    #: Flag that defines how to calculate and store the point spread function,
    #: defaults to 'full'. See :attr:`PointSpreadFunction.calcmode` for details.
    calcmode = Enum('full', 'single', 'block', 'readonly', desc='mode of psf calculation / storage')

    # internal identifier
    digest = Property(
        depends_on=BEAMFORMER_BASE_DIGEST_DEPENDENCIES + ['n_iter', 'damp', 'psf_precision'],
    )

    def _get_beamformer(self):
        return self._beamformer

    def _set_beamformer(self, beamformer):
        msg = (
            f"Deprecated use of 'beamformer' trait in class {self.__class__.__name__}. "
            'Please set :attr:`freq_data`, :attr:`steer`, :attr:`r_diag` directly. '
            "Using the 'beamformer' trait will be removed in version 25.07."
        )
        warn(
            msg,
            DeprecationWarning,
            stacklevel=2,
        )
        self._beamformer = beamformer

    @cached_property
    def _get_digest(self):
        return digest(self)

    @on_trait_change('_beamformer.digest')
    def delegate_beamformer_traits(self):
        self.freq_data = self.beamformer.freq_data
        self.r_diag = self.beamformer.r_diag
        self.steer = self.beamformer.steer

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        normfactor = self.sig_loss_norm()
        p = PointSpreadFunction(steer=self.steer, calcmode=self.calcmode, precision=self.psf_precision)
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            csm = array(self.freq_data.csm[i], dtype='complex128')
            y = beamformerFreq(
                param_steer_type,
                self.r_diag,
                normfactor,
                steer_vector(f[i]),
                csm,
            )[0]
            if self.r_diag:  # set (unphysical) negative output values to 0
                indNegSign = sign(y) < 0
                y[indNegSign] = 0.0
            x = y.copy()
            p.freq = f[i]
            psf = p.psf[:]
            damasSolverGaussSeidel(psf, y, self.n_iter, self.damp, x)
            self._ac[i] = x
            self._fr[i] = 1


@deprecated_alias({'max_iter': 'n_iter'})
class BeamformerDamasPlus(BeamformerDamas):
    """DAMAS deconvolution :cite:`Brooks2006` for solving the system of equations, instead of the
    original Gauss-Seidel iterations, this class employs the NNLS or linear programming solvers from
    scipy.optimize or one of several optimization algorithms from the scikit-learn module. Needs
    a-priori delay-and-sum beamforming (:class:`BeamformerBase`).
    """

    #: Type of fit method to be used ('LassoLars',
    #: 'OMPCV', 'LP', or 'NNLS', defaults to 'NNLS').
    #: These methods are implemented in
    #: the `scikit-learn <http://scikit-learn.org/stable/user_guide.html>`_
    #: module or within scipy.optimize respectively.
    method = Enum('NNLS', 'LP', 'LassoLars', 'OMPCV', desc='method used for solving deconvolution problem')

    #: Weight factor for LassoLars method,
    #: defaults to 0.0.
    # (Values in the order of 10^⁻9 should produce good results.)
    alpha = Range(0.0, 1.0, 0.0, desc='Lasso weight factor')

    #: Maximum number of iterations,
    #: tradeoff between speed and precision;
    #: defaults to 500
    n_iter = Int(500, desc='maximum number of iterations')

    #: Unit multiplier for evaluating, e.g., nPa instead of Pa.
    #: Values are converted back before returning.
    #: Temporary conversion may be necessary to not reach machine epsilon
    #: within fitting method algorithms. Defaults to 1e9.
    unit_mult = Float(1e9, desc='unit multiplier')

    # internal identifier
    digest = Property(
        depends_on=BEAMFORMER_BASE_DIGEST_DEPENDENCIES + ['alpha', 'method', 'n_iter', 'unit_mult'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        p = PointSpreadFunction(steer=self.steer, calcmode=self.calcmode, precision=self.psf_precision)
        unit = self.unit_mult
        normfactor = self.sig_loss_norm()
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            csm = array(self.freq_data.csm[i], dtype='complex128')
            y = beamformerFreq(
                param_steer_type,
                self.r_diag,
                normfactor,
                steer_vector(f[i]),
                csm,
            )[0]
            if self.r_diag:  # set (unphysical) negative output values to 0
                indNegSign = sign(y) < 0
                y[indNegSign] = 0.0
            y *= unit
            p.freq = f[i]
            psf = p.psf[:]

            if self.method == 'NNLS':
                self._ac[i] = nnls(psf, y)[0] / unit
            elif self.method == 'LP':  # linear programming (Dougherty)
                if self.r_diag:
                    warn(
                        'Linear programming solver may fail when CSM main '
                        'diagonal is removed for delay-and-sum beamforming.',
                        Warning,
                        stacklevel=5,
                    )
                cT = -1 * psf.sum(1)  # turn the minimization into a maximization
                self._ac[i] = linprog(c=cT, A_ub=psf, b_ub=y).x / unit  # defaults to simplex method and non-negative x
            else:
                if self.method == 'LassoLars':
                    model = LassoLars(alpha=self.alpha * unit, max_iter=self.n_iter, positive=True)
                elif self.method == 'OMPCV':
                    model = OrthogonalMatchingPursuitCV()
                else:
                    msg = f'Method {self.method} not implemented.'
                    raise NotImplementedError(msg)
                model.normalize = False
                # from sklearn 1.2, normalize=True does not work the same way anymore and the
                # pipeline approach with StandardScaler does scale in a different way, thus we
                # monkeypatch the code and normalize ourselves to make results the same over
                # different sklearn versions
                norms = norm(psf, axis=0)
                # get rid of annoying sklearn warnings that appear
                # for sklearn<1.2 despite any settings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=FutureWarning)
                    # normalized psf
                    model.fit(psf / norms, y)
                # recover normalization in the coef's
                self._ac[i] = model.coef_[:] / norms / unit
            self._fr[i] = 1


class BeamformerOrth(BeamformerBase):
    """Orthogonal deconvolution algorithm.

    See :cite:`Sarradj2010` for details.
    New faster implementation without explicit (:class:`BeamformerEig`).
    """

    #: (only for backward compatibility) :class:`BeamformerEig` object
    #: if set, provides :attr:`freq_data`, :attr:`steer`, :attr:`r_diag`
    #: if not set, these have to be set explicitly.
    beamformer = Property()

    # private storage of beamformer instance
    _beamformer = Instance(BeamformerEig)

    #: List of components to consider, use this to directly set the eigenvalues
    #: used in the beamformer. Alternatively, set :attr:`n`.
    eva_list = CArray(dtype=int, value=array([-1]), desc='components')

    #: Number of components to consider, defaults to 1. If set,
    #: :attr:`eva_list` will contain
    #: the indices of the n largest eigenvalues. Setting :attr:`eva_list`
    #: afterwards will override this value.
    n = Int(1)

    # internal identifier
    digest = Property(
        depends_on=BEAMFORMER_BASE_DIGEST_DEPENDENCIES + ['eva_list'],
    )

    def _get_beamformer(self):
        return self._beamformer

    def _set_beamformer(self, beamformer):
        msg = (
            f"Deprecated use of 'beamformer' trait in class {self.__class__.__name__}. "
            'Please set :attr:`freq_data`, :attr:`steer`, :attr:`r_diag` directly. '
            "Using the 'beamformer' trait will be removed in version 25.07."
        )
        warn(
            msg,
            DeprecationWarning,
            stacklevel=2,
        )
        self._beamformer = beamformer

    @cached_property
    def _get_digest(self):
        return digest(self)

    @on_trait_change('_beamformer.digest')
    def delegate_beamformer_traits(self):
        self.freq_data = self.beamformer.freq_data
        self.r_diag = self.beamformer.r_diag
        self.steer = self.beamformer.steer

    @on_trait_change('n')
    def set_eva_list(self):
        """Sets the list of eigenvalues to consider."""
        self.eva_list = arange(-1, -1 - self.n, -1)

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        num_channels = self.freq_data.num_channels
        normfactor = self.sig_loss_norm()
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            eva = array(self.freq_data.eva[i], dtype='float64')
            eve = array(self.freq_data.eve[i], dtype='complex128')
            for n in self.eva_list:
                beamformerOutput = beamformerFreq(
                    param_steer_type,
                    self.r_diag,
                    normfactor,
                    steer_vector(f[i]),
                    (ones(1), eve[:, n].reshape((-1, 1))),
                )[0]
                self._ac[i, beamformerOutput.argmax()] += eva[n] / num_channels
            self._fr[i] = 1


@deprecated_alias({'n': 'n_iter'})
class BeamformerCleansc(BeamformerBase):
    """CLEAN-SC deconvolution algorithm.

    See :cite:`Sijtsma2007` for details.
    Classic delay-and-sum beamforming is already included.
    """

    #: no of CLEAN-SC iterations
    #: defaults to 0, i.e. automatic (max 2*num_channels)
    n_iter = Int(0, desc='no of iterations')

    #: iteration damping factor
    #: defaults to 0.6
    damp = Range(0.01, 1.0, 0.6, desc='damping factor')

    #: iteration stop criterion for automatic detection
    #: iteration stops if power[i]>power[i-stopn]
    #: defaults to 3
    stopn = Int(3, desc='stop criterion index')

    # internal identifier
    digest = Property(depends_on=BEAMFORMER_BASE_DIGEST_DEPENDENCIES + ['n_iter', 'damp', 'stopn'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        normfactor = self.sig_loss_norm()
        num_channels = self.freq_data.num_channels
        result = zeros((self.steer.grid.size), 'f')
        J = num_channels * 2 if not self.n_iter else self.n_iter
        powers = zeros(J, 'd')

        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            csm = array(self.freq_data.csm[i], dtype='complex128', copy=1)
            # h = self.steer._beamformerCall(f[i], self.r_diag, normfactor, (csm,))[0]
            h = beamformerFreq(param_steer_type, self.r_diag, normfactor, steer_vector(f[i]), csm)[0]
            # CLEANSC Iteration
            result *= 0.0
            for j in range(J):
                xi_max = h.argmax()  # index of maximum
                powers[j] = hmax = h[xi_max]  # maximum
                result[xi_max] += self.damp * hmax
                if j > self.stopn and hmax > powers[j - self.stopn]:
                    break
                wmax = self.steer.steer_vector(f[i], xi_max) * sqrt(normfactor)
                wmax = wmax[0].conj()  # as old code worked with conjugated csm..should be updated
                hh = wmax.copy()
                D1 = dot(csm.T - diag(diag(csm)), wmax) / hmax
                ww = wmax.conj() * wmax
                for _m in range(20):
                    H = hh.conj() * hh
                    hh = (D1 + H * wmax) / sqrt(1 + dot(ww, H))
                hh = hh[:, newaxis]
                csm1 = hmax * (hh * hh.conj().T)

                # h1 = self.steer._beamformerCall(f[i], self.r_diag, normfactor, \
                # (array((hmax, ))[newaxis, :], hh[newaxis, :].conjugate()))[0]
                h1 = beamformerFreq(
                    param_steer_type,
                    self.r_diag,
                    normfactor,
                    steer_vector(f[i]),
                    (array((hmax,)), hh.conj()),
                )[0]
                h -= self.damp * h1
                csm -= self.damp * csm1.T  # transpose(0,2,1)
            self._ac[i] = result
            self._fr[i] = 1


class BeamformerClean(BeamformerBase):
    """CLEAN deconvolution algorithm.

    See :cite:`Hoegbom1974` for details.
    """

    #: (only for backward compatibility) :class:`BeamformerBase` object
    #: if set, provides :attr:`freq_data`, :attr:`steer`, :attr:`r_diag`
    #: if not set, these have to be set explicitly.
    beamformer = Property()

    # private storage of beamformer instance
    _beamformer = Instance(BeamformerBase)

    #: The floating-number-precision of the PSFs. Default is 64 bit.
    psf_precision = Enum('float64', 'float32', desc='precision of PSF.')

    # iteration damping factor
    # defaults to 0.6
    damp = Range(0.01, 1.0, 0.6, desc='damping factor')

    # max number of iterations
    n_iter = Int(100, desc='maximum number of iterations')

    # how to calculate and store the psf
    calcmode = Enum('block', 'full', 'single', 'readonly', desc='mode of psf calculation / storage')

    # internal identifier
    digest = Property(
        depends_on=BEAMFORMER_BASE_DIGEST_DEPENDENCIES + ['n_iter', 'damp', 'psf_precision'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_beamformer(self):
        return self._beamformer

    def _set_beamformer(self, beamformer):
        msg = (
            f"Deprecated use of 'beamformer' trait in class {self.__class__.__name__}. "
            'Please set :attr:`freq_data`, :attr:`steer`, :attr:`r_diag` directly. '
            "Using the 'beamformer' trait will be removed in version 25.07."
        )
        warn(
            msg,
            DeprecationWarning,
            stacklevel=2,
        )
        self._beamformer = beamformer

    @on_trait_change('_beamformer.digest')
    def delegate_beamformer_traits(self):
        self.freq_data = self.beamformer.freq_data
        self.r_diag = self.beamformer.r_diag
        self.steer = self.beamformer.steer

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        gs = self.steer.grid.size
        normfactor = self.sig_loss_norm()

        if self.calcmode == 'full':
            warn(
                "calcmode = 'full', possibly slow CLEAN performance. Better use 'block' or 'single'.",
                Warning,
                stacklevel=2,
            )
        p = PointSpreadFunction(steer=self.steer, calcmode=self.calcmode, precision=self.psf_precision)
        param_steer_type, steer_vector = self._beamformer_params()
        for i in ind:
            p.freq = f[i]
            csm = array(self.freq_data.csm[i], dtype='complex128')
            dirty = beamformerFreq(
                param_steer_type,
                self.r_diag,
                normfactor,
                steer_vector(f[i]),
                csm,
            )[0]
            if self.r_diag:  # set (unphysical) negative output values to 0
                indNegSign = sign(dirty) < 0
                dirty[indNegSign] = 0.0

            clean = zeros(gs, dtype=dirty.dtype)
            i_iter = 0
            flag = True
            while flag:
                dirty_sum = abs(dirty).sum(0)
                next_max = dirty.argmax(0)
                p.grid_indices = array([next_max])
                psf = p.psf.reshape(gs)
                new_amp = self.damp * dirty[next_max]  # / psf[next_max]
                clean[next_max] += new_amp
                dirty -= psf * new_amp
                i_iter += 1
                flag = dirty_sum > abs(dirty).sum(0) and i_iter < self.n_iter and max(dirty) > 0

            self._ac[i] = clean
            self._fr[i] = 1


@deprecated_alias({'max_iter': 'n_iter'})
class BeamformerCMF(BeamformerBase):
    """Covariance Matrix Fitting algorithm.

    This is not really a beamformer, but an inverse method.
    See :cite:`Yardibi2008` for details.
    """

    #: Type of fit method to be used ('LassoLars', 'LassoLarsBIC',
    #: 'OMPCV' or 'NNLS', defaults to 'LassoLars').
    #: These methods are implemented in
    #: the `scikit-learn <http://scikit-learn.org/stable/user_guide.html>`_
    #: module.
    method = Enum(
        'LassoLars',
        'LassoLarsBIC',
        'OMPCV',
        'NNLS',
        'fmin_l_bfgs_b',
        'Split_Bregman',
        'FISTA',
        desc='fit method used',
    )

    #: Weight factor for LassoLars method,
    #: defaults to 0.0.
    #: (Use values in the order of 10^⁻9 for good results.)
    alpha = Range(0.0, 1.0, 0.0, desc='Lasso weight factor')

    #: Total or maximum number of iterations
    #: (depending on :attr:`method`),
    #: tradeoff between speed and precision;
    #: defaults to 500
    n_iter = Int(500, desc='maximum number of iterations')

    #: Unit multiplier for evaluating, e.g., nPa instead of Pa.
    #: Values are converted back before returning.
    #: Temporary conversion may be necessary to not reach machine epsilon
    #: within fitting method algorithms. Defaults to 1e9.
    unit_mult = Float(1e9, desc='unit multiplier')

    #: If True, shows the status of the PyLops solver. Only relevant in case of FISTA or
    #: Split_Bregman
    show = Bool(False, desc='show output of PyLops solvers')

    #: Energy normalization in case of diagonal removal not implemented for inverse methods.
    r_diag_norm = Enum(
        None,
        desc='Energy normalization in case of diagonal removal not implemented for inverse methods',
    )

    # internal identifier
    digest = Property(
        depends_on=[
            'freq_data.digest',
            'alpha',
            'method',
            'n_iter',
            'unit_mult',
            'r_diag',
            'precision',
            'steer.inv_digest',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @on_trait_change('method')
    def _validate(self):
        if self.method in ['FISTA', 'Split_Bregman'] and not config.have_pylops:
            msg = (
                'Cannot import Pylops package. No Pylops installed.'
                f'Solver for {self.method} in BeamformerCMF not available.'
            )
            raise ImportError(msg)

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f

        # function to repack complex matrices to deal with them in real number space
        def realify(matrix):
            return vstack([matrix.real, matrix.imag])

        # prepare calculation
        nc = self.freq_data.num_channels
        num_points = self.steer.grid.size
        unit = self.unit_mult

        for i in ind:
            csm = array(self.freq_data.csm[i], dtype='complex128', copy=1)

            h = self.steer.transfer(f[i]).T

            # reduced Kronecker product (only where solution matrix != 0)
            Bc = (h[:, :, newaxis] * h.conjugate().T[newaxis, :, :]).transpose(2, 0, 1)
            Ac = Bc.reshape(nc * nc, num_points)

            # get indices for upper triangular matrices (use tril b/c transposed)
            ind = reshape(tril(ones((nc, nc))), (nc * nc,)) > 0

            ind_im0 = (reshape(eye(nc), (nc * nc,)) == 0)[ind]
            if self.r_diag:
                # omit main diagonal for noise reduction
                ind_reim = hstack([ind_im0, ind_im0])
            else:
                # take all real parts -- also main diagonal
                ind_reim = hstack([ones(size(ind_im0)) > 0, ind_im0])
                ind_reim[0] = True  # why this ?

            A = realify(Ac[ind, :])[ind_reim, :]
            # use csm.T for column stacking reshape!
            R = realify(reshape(csm.T, (nc * nc, 1))[ind, :])[ind_reim, :] * unit
            # choose method
            if self.method == 'LassoLars':
                model = LassoLars(alpha=self.alpha * unit, max_iter=self.n_iter, positive=True, **sklearn_ndict)
            elif self.method == 'LassoLarsBIC':
                model = LassoLarsIC(criterion='bic', max_iter=self.n_iter, positive=True, **sklearn_ndict)
            elif self.method == 'OMPCV':
                model = OrthogonalMatchingPursuitCV(**sklearn_ndict)
            elif self.method == 'NNLS':
                model = LinearRegression(positive=True)

            if self.method == 'Split_Bregman' and config.have_pylops:
                from pylops import Identity, MatrixMult
                from pylops.optimization.sparsity import splitbregman

                Oop = MatrixMult(A)  # transfer operator
                Iop = self.alpha * Identity(num_points)  # regularisation
                self._ac[i], iterations, cost = splitbregman(
                    Op=Oop,
                    RegsL1=[Iop],
                    y=R[:, 0],
                    niter_outer=self.n_iter,
                    niter_inner=5,
                    RegsL2=None,
                    dataregsL2=None,
                    mu=1.0,
                    epsRL1s=[1],
                    tol=1e-10,
                    tau=1.0,
                    show=self.show,
                )
                self._ac[i] /= unit

            elif self.method == 'FISTA' and config.have_pylops:
                from pylops import MatrixMult
                from pylops.optimization.sparsity import fista

                Oop = MatrixMult(A)  # transfer operator
                self._ac[i], iterations, cost = fista(
                    Op=Oop,
                    y=R[:, 0],
                    niter=self.n_iter,
                    eps=self.alpha,
                    alpha=None,
                    tol=1e-10,
                    show=self.show,
                )
                self._ac[i] /= unit
            elif self.method == 'fmin_l_bfgs_b':
                # function to minimize
                def function(x):
                    # function
                    func = x.T @ A.T @ A @ x - 2 * R.T @ A @ x + R.T @ R
                    # derivitaive
                    der = 2 * A.T @ A @ x.T[:, newaxis] - 2 * A.T @ R
                    return func[0].T, der[:, 0]

                # initial guess
                x0 = ones([num_points])
                # boundaries - set to non negative
                boundaries = tile((0, +inf), (len(x0), 1))

                # optimize
                self._ac[i], yval, dicts = fmin_l_bfgs_b(
                    function,
                    x0,
                    fprime=None,
                    args=(),
                    approx_grad=0,
                    bounds=boundaries,
                    m=10,
                    factr=10000000.0,
                    pgtol=1e-05,
                    epsilon=1e-08,
                    iprint=-1,
                    maxfun=15000,
                    maxiter=self.n_iter,
                    disp=None,
                    callback=None,
                    maxls=20,
                )

                self._ac[i] /= unit
            else:
                # from sklearn 1.2, normalize=True does not work the same way anymore and the
                # pipeline approach with StandardScaler does scale in a different way, thus we
                # monkeypatch the code and normalize ourselves to make results the same over
                # different sklearn versions
                norms = norm(A, axis=0)
                # get rid of sklearn warnings that appear for sklearn<1.2 despite any settings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=FutureWarning)
                    # normalized A
                    model.fit(A / norms, R[:, 0])
                # recover normalization in the coef's
                self._ac[i] = model.coef_[:] / norms / unit
            self._fr[i] = 1


@deprecated_alias({'max_iter': 'n_iter'})
class BeamformerSODIX(BeamformerBase):
    """Source directivity modeling in the cross-spectral matrix (SODIX) algorithm.

    See :cite:`Funke2017` and :cite:`Oertwig2019` for details.
    """

    #: Type of fit method to be used ('fmin_l_bfgs_b').
    #: These methods are implemented in
    #: the scipy module.
    method = Enum('fmin_l_bfgs_b', desc='fit method used')

    #: Maximum number of iterations,
    #: tradeoff between speed and precision;
    #: defaults to 200
    n_iter = Int(200, desc='maximum number of iterations')

    #: Weight factor for regularization,
    #: defaults to 0.0.
    alpha = Range(0.0, 1.0, 0.0, desc='regularization factor')

    #: Unit multiplier for evaluating, e.g., nPa instead of Pa.
    #: Values are converted back before returning.
    #: Temporary conversion may be necessary to not reach machine epsilon
    #: within fitting method algorithms. Defaults to 1e9.
    unit_mult = Float(1e9, desc='unit multiplier')

    #: Energy normalization in case of diagonal removal not implemented for inverse methods.
    r_diag_norm = Enum(
        None,
        desc='Energy normalization in case of diagonal removal not implemented for inverse methods',
    )

    # internal identifier
    digest = Property(
        depends_on=[
            'freq_data.digest',
            'alpha',
            'method',
            'n_iter',
            'unit_mult',
            'r_diag',
            'precision',
            'steer.inv_digest',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        # prepare calculation
        f = self._f
        num_points = self.steer.grid.size
        # unit = self.unit_mult
        num_mics = self.steer.mics.num_mics
        # SODIX needs special treatment as the result from one frequency is used to
        # determine the initial guess for the next frequency in order to speed up
        # computation. Instead of just solving for only the frequencies in ind, we
        # start with index 1 (minimum frequency) and also check if the result is
        # already computed
        for i in range(1, ind.max() + 1):
            if not self._fr[i]:
                # measured csm
                csm = array(self.freq_data.csm[i], dtype='complex128', copy=1)
                # transfer function
                h = self.steer.transfer(f[i]).T

                if self.method == 'fmin_l_bfgs_b':
                    # function to minimize
                    def function(directions):
                        """Parameters
                        ----------
                        directions
                        [num_points*num_mics]

                        Returns
                        -------
                        func - Sodix function to optimize
                             [1]
                        derdrl - derivitaives in direction of D
                            [num_mics*num_points].

                        """
                        #### the sodix function ####
                        Djm = directions.reshape([num_points, num_mics])
                        p = h.T * Djm
                        csm_mod = dot(p.T, p.conj())
                        Q = csm - csm_mod
                        func = sum((absolute(Q)) ** 2)

                        # subscripts and operands for numpy einsum and einsum_path
                        subscripts = 'rl,rm,ml->rl'
                        operands = (h.T, h.T.conj() * Djm, Q)
                        es_path = einsum_path(subscripts, *operands, optimize='greedy')[0]

                        #### the sodix derivative ####
                        derdrl = einsum(subscripts, *operands, optimize=es_path)
                        derdrl = -4 * real(derdrl)
                        return func, derdrl.ravel()

                    ##### initial guess ####
                    if not self._fr[(i - 1)]:
                        D0 = ones([num_points, num_mics])
                    else:
                        D0 = sqrt(
                            self._ac[(i - 1)]
                            * real(trace(csm) / trace(array(self.freq_data.csm[i - 1], dtype='complex128', copy=1))),
                        )

                    # boundaries - set to non negative [2*(num_points*num_mics)]
                    boundaries = tile((0, +inf), (num_points * num_mics, 1))

                    # optimize with gradient solver
                    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

                    qi = ones([num_points, num_mics])
                    qi, yval, dicts = fmin_l_bfgs_b(
                        function,
                        D0,
                        fprime=None,
                        args=(),
                        approx_grad=0,
                        bounds=boundaries,
                        factr=100.0,
                        pgtol=1e-12,
                        epsilon=1e-08,
                        iprint=-1,
                        maxfun=1500000,
                        maxiter=self.n_iter,
                        disp=-1,
                        callback=None,
                        maxls=20,
                    )
                    # squared pressure
                    self._ac[i] = qi**2
                else:
                    pass
                self._fr[i] = 1


@deprecated_alias({'max_iter': 'n_iter'})
class BeamformerGIB(BeamformerEig):  # BeamformerEig #BeamformerBase
    """Beamforming GIB methods with different normalizations.

    See :cite:`Suzuki2011` for details.
    """

    #: Unit multiplier for evaluating, e.g., nPa instead of Pa.
    #: Values are converted back before returning.
    #: Temporary conversion may be necessary to not reach machine epsilon
    #: within fitting method algorithms. Defaults to 1e9.
    unit_mult = Float(1e9, desc='unit multiplier')

    #: Total or maximum number of iterations
    #: (depending on :attr:`method`),
    #: tradeoff between speed and precision;
    #: defaults to 10
    n_iter = Int(10, desc='maximum number of iterations')

    #: Type of fit method to be used ('Suzuki', 'LassoLars', 'LassoLarsCV', 'LassoLarsBIC',
    #: 'OMPCV' or 'NNLS', defaults to 'Suzuki').
    #: These methods are implemented in
    #: the `scikit-learn <http://scikit-learn.org/stable/user_guide.html>`_
    #: module.
    method = Enum(
        'Suzuki',
        'InverseIRLS',
        'LassoLars',
        'LassoLarsBIC',
        'LassoLarsCV',
        'OMPCV',
        'NNLS',
        desc='fit method used',
    )

    #: Weight factor for LassoLars method,
    #: defaults to 0.0.
    alpha = Range(0.0, 1.0, 0.0, desc='Lasso weight factor')
    # (use values in the order of 10^⁻9 for good results)

    #: Norm to consider for the regularization in InverseIRLS and Suzuki methods
    #: defaults to L-1 Norm
    pnorm = Float(1, desc='Norm for regularization')

    #: Beta - Fraction of sources maintained after each iteration
    #: defaults to 0.9
    beta = Float(0.9, desc='fraction of sources maintained')

    #: eps - Regularization parameter for Suzuki algorithm
    #: defaults to 0.05.
    eps_perc = Float(0.05, desc='regularization parameter')

    # This feature is not fully supported may be changed in the next release
    # First eigenvalue to consider. Defaults to 0.
    m = Int(0, desc='First eigenvalue to consider')

    #: Energy normalization in case of diagonal removal not implemented for inverse methods.
    r_diag_norm = Enum(
        None,
        desc='Energy normalization in case of diagonal removal not implemented for inverse methods',
    )

    # internal identifier
    digest = Property(
        depends_on=[
            'steer.inv_digest',
            'freq_data.digest',
            'precision',
            'alpha',
            'method',
            'n_iter',
            'unit_mult',
            'eps_perc',
            'pnorm',
            'beta',
            'n',
            'm',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property_depends_on('n')
    def _get_na(self):
        na = self.n
        nm = self.steer.mics.num_mics
        if na < 0:
            na = max(nm + na, 0)
        return min(nm - 1, na)

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        n = int(self.na)  # number of eigenvalues
        m = int(self.m)  # number of first eigenvalue
        num_channels = self.freq_data.num_channels  # number of channels
        num_points = self.steer.grid.size
        hh = zeros((1, num_points, num_channels), dtype='D')

        # Generate a cross spectral matrix, and perform the eigenvalue decomposition
        for i in ind:
            # for monopole and source strength Q needs to define density
            # calculate a transfer matrix A
            hh = self.steer.transfer(f[i])
            A = hh.T
            # eigenvalues and vectors
            csm = array(self.freq_data.csm[i], dtype='complex128', copy=1)
            eva, eve = eigh(csm)
            eva = eva[::-1]
            eve = eve[:, ::-1]
            # set small values zo 0, lowers numerical errors in simulated data
            eva[eva < max(eva) / 1e12] = 0
            # init sources
            qi = zeros([n + m, num_points], dtype='complex128')
            # Select the number of coherent modes to be processed referring to the eigenvalue
            # distribution.
            for s in list(range(m, n + m)):
                if eva[s] > 0:
                    # Generate the corresponding eigenmodes
                    emode = array(sqrt(eva[s]) * eve[:, s], dtype='complex128')
                    # choose method for computation
                    if self.method == 'Suzuki':
                        leftpoints = num_points
                        locpoints = arange(num_points)
                        weights = diag(ones(num_points))
                        epsilon = arange(self.n_iter)
                        for it in arange(self.n_iter):
                            if num_channels <= leftpoints:
                                AWA = dot(dot(A[:, locpoints], weights), A[:, locpoints].conj().T)
                                epsilon[it] = max(absolute(eigvals(AWA))) * self.eps_perc
                                qi[s, locpoints] = dot(
                                    dot(
                                        dot(weights, A[:, locpoints].conj().T),
                                        inv(AWA + eye(num_channels) * epsilon[it]),
                                    ),
                                    emode,
                                )
                            elif num_channels > leftpoints:
                                AA = dot(A[:, locpoints].conj().T, A[:, locpoints])
                                epsilon[it] = max(absolute(eigvals(AA))) * self.eps_perc
                                qi[s, locpoints] = dot(
                                    dot(inv(AA + inv(weights) * epsilon[it]), A[:, locpoints].conj().T),
                                    emode,
                                )
                            if self.beta < 1 and it > 1:
                                # Reorder from the greatest to smallest magnitude to define a
                                # reduced-point source distribution, and reform a reduced transfer
                                # matrix
                                leftpoints = int(round(num_points * self.beta ** (it + 1)))
                                idx = argsort(abs(qi[s, locpoints]))[::-1]
                                # print(it, leftpoints, locpoints, idx )
                                locpoints = delete(locpoints, [idx[leftpoints::]])
                                qix = zeros([n + m, leftpoints], dtype='complex128')
                                qix[s, :] = qi[s, locpoints]
                                # calc weights for next iteration
                                weights = diag(absolute(qix[s, :]) ** (2 - self.pnorm))
                            else:
                                weights = diag(absolute(qi[s, :]) ** (2 - self.pnorm))

                    elif self.method == 'InverseIRLS':
                        weights = eye(num_points)
                        locpoints = arange(num_points)
                        for _it in arange(self.n_iter):
                            if num_channels <= num_points:
                                wtwi = inv(dot(weights.T, weights))
                                aH = A.conj().T
                                qi[s, :] = dot(dot(wtwi, aH), dot(inv(dot(A, dot(wtwi, aH))), emode))
                                weights = diag(absolute(qi[s, :]) ** ((2 - self.pnorm) / 2))
                                weights = weights / sum(absolute(weights))
                            elif num_channels > num_points:
                                wtw = dot(weights.T, weights)
                                qi[s, :] = dot(dot(inv(dot(dot(A.conj.T, wtw), A)), dot(A.conj().T, wtw)), emode)
                                weights = diag(absolute(qi[s, :]) ** ((2 - self.pnorm) / 2))
                                weights = weights / sum(absolute(weights))
                    else:
                        locpoints = arange(num_points)
                        unit = self.unit_mult
                        AB = vstack([hstack([A.real, -A.imag]), hstack([A.imag, A.real])])
                        R = hstack([emode.real.T, emode.imag.T]) * unit
                        if self.method == 'LassoLars':
                            model = LassoLars(alpha=self.alpha * unit, max_iter=self.n_iter, positive=True)
                        elif self.method == 'LassoLarsBIC':
                            model = LassoLarsIC(criterion='bic', max_iter=self.n_iter, positive=True)
                        elif self.method == 'OMPCV':
                            model = OrthogonalMatchingPursuitCV()
                        elif self.method == 'LassoLarsCV':
                            model = LassoLarsCV(max_iter=self.n_iter, positive=True)
                        elif self.method == 'NNLS':
                            model = LinearRegression(positive=True)
                        model.normalize = False
                        # from sklearn 1.2, normalize=True does not work
                        # the same way anymore and the pipeline approach
                        # with StandardScaler does scale in a different
                        # way, thus we monkeypatch the code and normalize
                        # ourselves to make results the same over different
                        # sklearn versions
                        norms = norm(AB, axis=0)
                        # get rid of annoying sklearn warnings that appear
                        # for sklearn<1.2 despite any settings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', category=FutureWarning)
                            # normalized A
                            model.fit(AB / norms, R)
                        # recover normalization in the coef's
                        qi_real, qi_imag = hsplit(model.coef_[:] / norms / unit, 2)
                        # print(s,qi.size)
                        qi[s, locpoints] = qi_real + qi_imag * 1j
                else:
                    warn(
                        f'Eigenvalue {s:g} <= 0 for frequency index {i:g}. Will not be calculated!',
                        Warning,
                        stacklevel=2,
                    )
                # Generate source maps of all selected eigenmodes, and superpose source intensity
                # for each source type.
            temp = zeros(num_points)
            temp[locpoints] = sum(absolute(qi[:, locpoints]) ** 2, axis=0)
            self._ac[i] = temp
            self._fr[i] = 1


class BeamformerAdaptiveGrid(BeamformerBase, Grid):
    """Abstract base class for array methods without predefined grid."""

    # the grid positions live in a shadow trait
    _gpos = Any

    def _get_shape(self):
        return (self.size,)

    def _get_pos(self):
        return self._gpos

    def integrate(self, sector):
        """Integrates result map over a given sector.

        Parameters
        ----------
        sector: :class:`~acoular.grids.Sector` or derived
            Gives the sector over which to integrate

        Returns
        -------
        array of floats
            The spectrum (all calculated frequency bands) for the integrated sector.

        """
        if not isinstance(sector, Sector):
            msg = (
                f'Please use a sector derived instance of type :class:`~acoular.grids.Sector` '
                f'instead of type {type(sector)}.'
            )
            raise NotImplementedError(
                msg,
            )

        ind = self.subdomain(sector)
        r = self.result
        h = zeros(r.shape[0])
        for i in range(r.shape[0]):
            h[i] = r[i][ind].sum()
        return h


class BeamformerGridlessOrth(BeamformerAdaptiveGrid):
    """Orthogonal beamforming without predefined grid.

    See :cite:`Sarradj2022` for details.
    """

    #: List of components to consider, use this to directly set the eigenvalues
    #: used in the beamformer. Alternatively, set :attr:`n`.
    eva_list = CArray(dtype=int, value=array([-1]), desc='components')

    #: Number of components to consider, defaults to 1. If set,
    #: :attr:`eva_list` will contain
    #: the indices of the n largest eigenvalues. Setting :attr:`eva_list`
    #: afterwards will override this value.
    n = Int(1)

    #: Geometrical bounds of the search domain to consider.
    #: :attr:`bound` is a list that contains exactly three tuple of
    #: (min,max) for each of the coordinates x, y, z.
    #: Defaults to [(-1.,1.),(-1.,1.),(0.01,1.)]
    bounds = List(Tuple(Float, Float), minlen=3, maxlen=3, value=[(-1.0, 1.0), (-1.0, 1.0), (0.01, 1.0)])

    #: options dictionary for the SHGO solver, see
    #: `scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html>`_.
    #: Default is Sobol sampling Nelder-Mead local minimizer, 256 initial sampling points
    #: and 1 iteration
    shgo = Dict

    #: No normalization implemented. Defaults to 1.0.
    r_diag_norm = Enum(
        1.0,
        desc='If diagonal of the csm is removed, some signal energy is lost.'
        'This is handled via this normalization factor.'
        'For this class, normalization is not implemented. Defaults to 1.0.',
    )

    # internal identifier
    digest = Property(
        depends_on=['freq_data.digest', 'steer.digest', 'precision', 'r_diag', 'eva_list', 'bounds', 'shgo'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @on_trait_change('n')
    def set_eva_list(self):
        """Sets the list of eigenvalues to consider."""
        self.eva_list = arange(-1, -1 - self.n, -1)

    @property_depends_on('n')
    def _get_size(self):
        return self.n * self.freq_data.fftfreq().shape[0]

    def _calc(self, ind):
        """Calculates the result for the frequencies defined by :attr:`freq_data`.

        This is an internal helper function that is automatically called when
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.

        Parameters
        ----------
        ind : array of int
            This array contains all frequency indices for which (re)calculation is
            to be performed

        Returns
        -------
        This method only returns values through :attr:`_ac` and :attr:`_fr`

        """
        f = self._f
        normfactor = self.sig_loss_norm()
        num_channels = self.freq_data.num_channels
        # eigenvalue number list in standard form from largest to smallest
        eva_list = unique(self.eva_list % self.steer.mics.num_mics)[::-1]
        steer_type = self.steer.steer_type
        if steer_type == 'custom':
            msg = 'custom steer_type is not implemented'
            raise NotImplementedError(msg)
        mpos = self.steer.mics.pos
        env = self.steer.env
        shgo_opts = {
            'n': 256,
            'iters': 1,
            'sampling_method': 'sobol',
            'minimizer_kwargs': {'method': 'Nelder-Mead'},
        }
        shgo_opts.update(self.shgo)
        roi = []
        for x in self.bounds[0]:
            for y in self.bounds[1]:
                for z in self.bounds[2]:
                    roi.append((x, y, z))
        self.steer.env.roi = array(roi).T
        bmin = array(tuple(map(min, self.bounds)))
        bmax = array(tuple(map(max, self.bounds)))
        for i in ind:
            eva = array(self.freq_data.eva[i], dtype='float64')
            eve = array(self.freq_data.eve[i], dtype='complex128')
            k = 2 * pi * f[i] / env.c
            for j, n in enumerate(eva_list):
                # print(f[i],n)

                def func(xy):
                    # function to minimize globally
                    xy = clip(xy, bmin, bmax)
                    r0 = env._r(xy[:, newaxis])
                    rm = env._r(xy[:, newaxis], mpos)
                    return -beamformerFreq(
                        steer_type,
                        self.r_diag,
                        normfactor,
                        (r0, rm, k),
                        (ones(1), eve[:, n : n + 1]),
                    )[0][0]  # noqa: B023

                # simplical global homotopy optimizer
                oR = shgo(func, self.bounds, **shgo_opts)
                # index in grid
                i1 = i * self.n + j
                # store result for position
                self._gpos[:, i1] = oR['x']
                # store result for level
                self._ac[i, i1] = eva[n] / num_channels
                # print(oR['x'],eva[n]/num_channels,oR)
            self._fr[i] = 1


def L_p(x):  # noqa: N802
    r"""Calculates the sound pressure level from the squared sound pressure.

    :math:`L_p = 10 \lg ( x / 4\cdot 10^{-10})`

    Parameters
    ----------
    x: array of floats
        The squared sound pressure values

    Returns
    -------
    array of floats
        The corresponding sound pressure levels in dB.
        If `x<0`, -350.0 dB is returned.

    """
    # new version to prevent division by zero warning for float32 arguments
    return 10 * log10(clip(x / 4e-10, 1e-35, None))


#    return where(x>0, 10*log10(x/4e-10), -1000.)


def integrate(data, grid, sector):
    """Integrates a sound pressure map over a given sector.

    This function can be applied on beamforming results to
    quantitatively analyze the sound pressure in a given sector.
    If used with :meth:`Beamformer.result()<acoular.fbeamform.BeamformerBase.result>`,
    the output is identical to the result of the intrinsic
    :meth:`Beamformer.integrate<acoular.fbeamform.BeamformerBase.integrate>` method.
    It can, however, also be used with the
    :meth:`Beamformer.synthetic<acoular.fbeamform.BeamformerBase.synthetic>`
    output.

    Parameters
    ----------
    data: array of floats
        Contains the calculated squared sound pressure values in Pa**2.
        If data has the same number of entries than the number of grid points
        only one value is returned.
        In case of a 2-D array with the second dimension identical
        to the number of grid points an array containing as many entries as
        the first dimension is returned.
    grid: Grid object
        Object of a :class:`~acoular.grids.Grid`-derived class
        that provides the grid locations.
    sector: array of floats or :class:`~acoular.grids.Sector`-derived object
        Tuple with arguments for the `indices` method
        of a :class:`~acoular.grids.Grid`-derived class
        (e.g. :meth:`RectGrid.indices<acoular.grids.RectGrid.indices>`
        or :meth:`RectGrid3D.indices<acoular.grids.RectGrid3D.indices>`).
        Possible sectors would be `array([xmin, ymin, xmax, ymax])`
        or `array([x, y, radius])`.
        Alternatively, a :class:`~acoular.grids.Sector`-derived object
        can be used.

    Returns
    -------
    array of floats
        The spectrum (all calculated frequency bands) for the integrated sector.

    """
    if isinstance(sector, Sector):
        ind = grid.subdomain(sector)
    elif hasattr(grid, 'indices'):
        ind = grid.indices(*sector)
    else:
        msg = (
            f'Grid of type {grid.__class__.__name__} does not have an indices method! '
            f'Please use a sector derived instance of type :class:`~acoular.grids.Sector` '
            'instead of type numpy.array.'
        )
        raise NotImplementedError(
            msg,
        )

    gshape = grid.shape
    gsize = grid.size
    if size(data) == gsize:  # one value per grid point
        h = data.reshape(gshape)[ind].sum()
    elif data.ndim == 2 and data.shape[1] == gsize:
        h = zeros(data.shape[0])
        for i in range(data.shape[0]):
            h[i] = data[i].reshape(gshape)[ind].sum()
    return h
