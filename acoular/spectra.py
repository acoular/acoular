# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Estimation of power spectra and related tools.

.. autosummary::
    :toctree: generated/

    BaseSpectra
    PowerSpectra
    PowerSpectraImport
"""

from abc import abstractmethod
from warnings import warn

from numpy import (
    arange,
    array,
    bartlett,
    blackman,
    dot,
    empty,
    fill_diagonal,
    hamming,
    hanning,
    imag,
    linalg,
    ndarray,
    newaxis,
    ones,
    real,
    searchsorted,
    sum,  # noqa A004
    zeros,
)
from scipy import fft
from traits.api import (
    ABCHasStrictTraits,
    Bool,
    CArray,
    Delegate,
    Enum,
    Float,
    Instance,
    Int,
    Map,
    Property,
    Union,
    cached_property,
    property_depends_on,
)

# acoular imports
from .base import SamplesGenerator
from .configuration import config
from .deprecation import deprecated_alias
from .fastFuncs import calcCSM
from .h5cache import H5cache
from .h5files import H5CacheFileBase
from .internal import digest
from .tools.utils import find_basename


@deprecated_alias({'numchannels': 'num_channels'}, read_only=True)
class BaseSpectra(ABCHasStrictTraits):
    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of time data channels
    num_channels = Delegate('source')

    #: Window function for FFT, one of:
    #:   * 'Rectangular' (default)
    #:   * 'Hanning'
    #:   * 'Hamming'
    #:   * 'Bartlett'
    #:   * 'Blackman'
    window = Map(
        {'Rectangular': ones, 'Hanning': hanning, 'Hamming': hamming, 'Bartlett': bartlett, 'Blackman': blackman},
        default_value='Rectangular',
        desc='type of window for FFT',
    )

    #: Overlap factor for averaging: 'None'(default), '50%', '75%', '87.5%'.
    overlap = Map({'None': 1, '50%': 2, '75%': 4, '87.5%': 8}, default_value='None', desc='overlap of FFT blocks')

    #: FFT block size, one of: 128, 256, 512, 1024, 2048 ... 65536,
    #: defaults to 1024.
    block_size = Enum(
        1024,
        128,
        256,
        512,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        desc='number of samples per FFT block',
    )

    #: The floating-number-precision of the resulting spectra, corresponding to numpy dtypes.
    #: Default is 'complex128'.
    precision = Enum('complex128', 'complex64', desc='precision of the fft')

    # internal identifier
    digest = Property(depends_on=['precision', 'block_size', 'window', 'overlap'])

    @abstractmethod
    def _get_digest(self):
        """Return internal identifier."""

    def fftfreq(self):
        """Return the Discrete Fourier Transform sample frequencies.

        Returns
        -------
        f : ndarray
            Array of length *block_size/2+1* containing the sample frequencies.

        """
        if self.source is not None:
            return abs(fft.fftfreq(self.block_size, 1.0 / self.source.sample_freq)[: int(self.block_size / 2 + 1)])
        return None

    # generator that yields the time data blocks for every channel (with optional overlap)
    def _get_source_data(self):
        bs = self.block_size
        temp = empty((2 * bs, self.num_channels))
        pos = bs
        posinc = bs / self.overlap_
        for data_block in self.source.result(bs):
            ns = data_block.shape[0]
            temp[bs : bs + ns] = data_block  # fill from right
            while pos + bs <= bs + ns:
                yield temp[int(pos) : int(pos + bs)]
                pos += posinc
            else:
                temp[0:bs] = temp[bs:]  # copy to left
                pos -= bs


class PowerSpectra(BaseSpectra):
    """Provides the cross spectral matrix of multichannel time data
     and its eigen-decomposition.

    This class includes the efficient calculation of the full cross spectral
    matrix using the Welch method with windows and overlap (:cite:`Welch1967`). It also contains
    the CSM's eigenvalues and eigenvectors and additional properties.

    The result is computed only when needed, that is when the :attr:`csm`,
    :attr:`eva`, or :attr:`eve` attributes are actually read.
    Any change in the input data or parameters leads to a new calculation,
    again triggered when an attribute is read. The result may be
    cached on disk in HDF5 files and need not to be recomputed during
    subsequent program runs with identical input data and parameters. The
    input data is taken to be identical if the source has identical parameters
    and the same file name in case of that the data is read from a file.
    """

    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Shadow trait, should not be set directly, for internal use.
    _ind_low = Int(1, desc='index of lowest frequency line')

    # Shadow trait, should not be set directly, for internal use.
    _ind_high = Union(Int(-1), None, desc='index of highest frequency line')

    #: Index of lowest frequency line to compute, integer, defaults to 1,
    #: is used only by objects that fetch the csm, PowerSpectra computes every
    #: frequency line.
    ind_low = Property(_ind_low, desc='index of lowest frequency line')

    #: Index of highest frequency line to compute, integer,
    #: defaults to -1 (last possible line for default block_size).
    ind_high = Property(_ind_high, desc='index of lowest frequency line')

    # Stores the set lower frequency, for internal use, should not be set directly.
    _freqlc = Float(0)

    # Stores the set higher frequency, for internal use, should not be set directly.
    _freqhc = Union(Float(0), None)

    # Saves whether the user set indices or frequencies last, for internal use only,
    # not to be set directly, if True (default), indices are used for setting
    # the freq_range interval.
    _index_set_last = Bool(True)

    #: Flag, if true (default), the result is cached in h5 files and need not
    #: to be recomputed during subsequent program runs.
    cached = Bool(True, desc='cached flag')

    #: Number of FFT blocks to average, readonly
    #: (set from block_size and overlap).
    num_blocks = Property(desc='overall number of FFT blocks')

    #: 2-element array with the lowest and highest frequency. If set,
    #: will overwrite :attr:`_freqlc` and :attr:`_freqhc` according to
    #: the range.
    #: The freq_range interval will be the smallest discrete frequency
    #: inside the half-open interval [_freqlc, _freqhc[ and the smallest
    #: upper frequency outside of the interval.
    #: If user chooses the higher frequency larger than the max frequency,
    #: the max frequency will be the upper bound.
    freq_range = Property(desc='frequency range')

    #: Array with a sequence of indices for all frequencies
    #: between :attr:`ind_low` and :attr:`ind_high` within the result, readonly.
    indices = Property(desc='index range')

    #: Name of the cache file without extension, readonly.
    basename = Property(depends_on=['source.digest'], desc='basename for cache file')

    #: The cross spectral matrix,
    #: (number of frequencies, num_channels, num_channels) array of complex;
    #: readonly.
    csm = Property(desc='cross spectral matrix')

    #: Eigenvalues of the cross spectral matrix as an
    #: (number of frequencies) array of floats, readonly.
    eva = Property(desc='eigenvalues of cross spectral matrix')

    #: Eigenvectors of the cross spectral matrix as an
    #: (number of frequencies, num_channels, num_channels) array of floats,
    #: readonly.
    eve = Property(desc='eigenvectors of cross spectral matrix')

    # internal identifier
    digest = Property(
        depends_on=['source.digest', 'block_size', 'window', 'overlap', 'precision'],
    )

    # hdf5 cache file
    h5f = Instance(H5CacheFileBase, transient=True)

    @property_depends_on(['source.num_samples', 'block_size', 'overlap'])
    def _get_num_blocks(self):
        return self.overlap_ * self.source.num_samples / self.block_size - self.overlap_ + 1

    @property_depends_on(['source.sample_freq', 'block_size', 'ind_low', 'ind_high'])
    def _get_freq_range(self):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._ind_high is None:
                return array([fftfreq[self.ind_low], None])
            return fftfreq[[self.ind_low, self.ind_high]]
        return None

    def _set_freq_range(self, freq_range):  # by setting this the user sets _freqlc and _freqhc
        self._index_set_last = False
        self._freqlc = freq_range[0]
        self._freqhc = freq_range[1]

    @property_depends_on(['source.sample_freq', 'block_size', '_ind_low', '_freqlc'])
    def _get_ind_low(self):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._index_set_last:
                return min(self._ind_low, fftfreq.shape[0] - 1)
            return searchsorted(fftfreq[:-1], self._freqlc)
        return None

    @property_depends_on(['source.sample_freq', 'block_size', '_ind_high', '_freqhc'])
    def _get_ind_high(self):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._index_set_last:
                if self._ind_high is None:
                    return None
                return min(self._ind_high, fftfreq.shape[0] - 1)
            if self._freqhc is None:
                return None
            return searchsorted(fftfreq[:-1], self._freqhc)
        return None

    def _set_ind_high(self, ind_high):  # by setting this the user sets the lower index
        self._index_set_last = True
        self._ind_high = ind_high

    def _set_ind_low(self, ind_low):  # by setting this the user sets the higher index
        self._index_set_last = True
        self._ind_low = ind_low

    @property_depends_on(['block_size', 'ind_low', 'ind_high'])
    def _get_indices(self):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            try:
                indices = arange(fftfreq.shape[0], dtype=int)
                if self.ind_high is None:
                    return indices[self.ind_low :]
                return indices[self.ind_low : self.ind_high]
            except IndexError:
                return range(0)
        return None

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        return find_basename(self.source, alternative_basename=self.source.__class__.__name__ + self.source.digest)

    def calc_csm(self):
        """Csm calculation."""
        t = self.source
        wind = self.window_(self.block_size)
        weight = dot(wind, wind)
        wind = wind[newaxis, :].swapaxes(0, 1)
        numfreq = int(self.block_size / 2 + 1)
        csm_shape = (numfreq, t.num_channels, t.num_channels)
        csm_upper = zeros(csm_shape, dtype=self.precision)
        # get time data blockwise
        for data in self._get_source_data():
            ft = fft.rfft(data * wind, None, 0).astype(self.precision)
            calcCSM(csm_upper, ft)  # only upper triangular part of matrix is calculated (for speed reasons)
        # create the full csm matrix via transposing and complex conj.
        csm_lower = csm_upper.conj().transpose(0, 2, 1)
        [fill_diagonal(csm_lower[cntFreq, :, :], 0) for cntFreq in range(csm_lower.shape[0])]
        csm = csm_lower + csm_upper
        # onesided spectrum: multiplication by 2.0=sqrt(2)^2
        return csm * (2.0 / self.block_size / weight / self.num_blocks)

    def calc_ev(self):
        """Eigenvalues / eigenvectors calculation."""
        if self.precision == 'complex128':
            eva_dtype = 'float64'
        elif self.precision == 'complex64':
            eva_dtype = 'float32'
        #        csm = self.csm #trigger calculation
        csm_shape = self.csm.shape
        eva = empty(csm_shape[0:2], dtype=eva_dtype)
        eve = empty(csm_shape, dtype=self.precision)
        for i in range(csm_shape[0]):
            (eva[i], eve[i]) = linalg.eigh(self.csm[i])
        return (eva, eve)

    def calc_eva(self):
        """Calculates eigenvalues of csm."""
        return self.calc_ev()[0]

    def calc_eve(self):
        """Calculates eigenvectors of csm."""
        return self.calc_ev()[1]

    def _get_filecache(self, traitname):
        """Function handles result caching of csm, eigenvectors and eigenvalues
        calculation depending on global/local caching behaviour.
        """
        if traitname == 'csm':
            func = self.calc_csm
            numfreq = int(self.block_size / 2 + 1)
            shape = (numfreq, self.source.num_channels, self.source.num_channels)
            precision = self.precision
        elif traitname == 'eva':
            func = self.calc_eva
            shape = self.csm.shape[0:2]
            if self.precision == 'complex128':
                precision = 'float64'
            elif self.precision == 'complex64':
                precision = 'float32'
        elif traitname == 'eve':
            func = self.calc_eve
            shape = self.csm.shape
            precision = self.precision

        H5cache.get_cache_file(self, self.basename)
        if not self.h5f:  # in case of global caching readonly
            return func()

        nodename = traitname + '_' + self.digest
        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
            # print("remove existing node",nodename)
            self.h5f.remove_data(nodename)  # remove old data before writing in overwrite mode

        if not self.h5f.is_cached(nodename):
            if config.global_caching == 'readonly':
                return func()
            #            print("create array, data not cached for",nodename)
            self.h5f.create_compressible_array(nodename, shape, precision)

        ac = self.h5f.get_data_by_reference(nodename)
        if ac[:].sum() == 0:  # only initialized
            #            print("write {} to:".format(traitname),nodename)
            ac[:] = func()
            self.h5f.flush()
        return ac

    @property_depends_on(['digest'])
    def _get_csm(self):
        """Main work is done here:
        Cross spectral matrix is either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        if config.global_caching == 'none' or (config.global_caching == 'individual' and self.cached is False):
            return self.calc_csm()
        return self._get_filecache('csm')

    @property_depends_on(['digest'])
    def _get_eva(self):
        """Eigenvalues of cross spectral matrix are either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        if config.global_caching == 'none' or (config.global_caching == 'individual' and self.cached is False):
            return self.calc_eva()
        return self._get_filecache('eva')

    @property_depends_on(['digest'])
    def _get_eve(self):
        """Eigenvectors of cross spectral matrix are either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        if config.global_caching == 'none' or (config.global_caching == 'individual' and self.cached is False):
            return self.calc_eve()
        return self._get_filecache('eve')

    def synthetic_ev(self, freq, num=0):
        """Return synthesized frequency band values of the eigenvalues.

        Parameters
        ----------
        freq : float
            Band center frequency for which to return the results.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            3 (third-octave band).

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
        float
            Synthesized frequency band value of the eigenvalues (the sum of
            all values that are contained in the band).

        """
        f = self.fftfreq()
        if num == 0:
            # single frequency line
            return self.eva[searchsorted(f, freq)]
        f1 = searchsorted(f, freq * 2.0 ** (-0.5 / num))
        f2 = searchsorted(f, freq * 2.0 ** (0.5 / num))
        if f1 == f2:
            return self.eva[f1]
        return sum(self.eva[f1:f2], 0)


class PowerSpectraImport(PowerSpectra):
    """Provides a dummy class for using pre-calculated cross-spectral
    matrices.

    This class does not calculate the cross-spectral matrix. Instead,
    the user can inject one or multiple existing CSMs by setting the
    :attr:`csm` attribute. This can be useful when algorithms shall be
    evaluated with existing CSM matrices.
    The frequency or frequencies contained by the CSM must be set via the
    attr:`frequencies` attribute. The attr:`num_channels` attributes
    is determined on the basis of the CSM shape.
    In contrast to the PowerSpectra object, the attributes
    :attr:`sample_freq`, :attr:`source`, :attr:`block_size`, :attr:`window`,
    :attr:`overlap`, :attr:`cached`, and :attr:`num_blocks`
    have no functionality.
    """

    #: The cross spectral matrix,
    #: (number of frequencies, num_channels, num_channels) array of complex;
    csm = Property(desc='cross spectral matrix')

    #: frequencies included in the cross-spectral matrix in ascending order.
    #: Compound trait that accepts arguments of type list, array, and float
    frequencies = Union(CArray, Float, desc='frequencies included in the cross-spectral matrix')

    #: Number of time data channels
    num_channels = Property(depends_on=['digest'])

    source = Enum(None, desc='PowerSpectraImport cannot consume time data')

    # Sampling frequency of the signal, defaults to None
    sample_freq = Enum(None, desc='sampling frequency')

    block_size = Enum(None, desc='PowerSpectraImport does not operate on blocks of time data')

    window = Enum(None, desc='PowerSpectraImport does not perform windowing')

    overlap = Enum(None, desc='PowerSpectraImport does not consume time data')

    cached = Enum(False, desc='PowerSpectraImport has no caching capabilities')

    num_blocks = Enum(None, desc='PowerSpectraImport cannot determine the number of blocks')

    # Shadow trait, should not be set directly, for internal use.
    _ind_low = Int(0, desc='index of lowest frequency line')

    # Shadow trait, should not be set directly, for internal use.
    _ind_high = Union(None, Int, desc='index of highest frequency line')

    # internal identifier
    digest = Property(depends_on=['_csmsum'])

    #: Name of the cache file without extension, readonly.
    basename = Property(depends_on=['digest'], desc='basename for cache file')

    # csm shadow trait, only for internal use.
    _csm = CArray()

    # CSM checksum to trigger digest calculation, only for internal use.
    _csmsum = Float()

    def _get_basename(self):
        return 'csm_import_' + self.digest

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_num_channels(self):
        return self.csm.shape[1]

    def _get_csm(self):
        return self._csm

    def _set_csm(self, csm):
        if (len(csm.shape) != 3) or (csm.shape[1] != csm.shape[2]):
            msg = 'The cross spectral matrix must have the following shape: \
            (number of frequencies, num_channels, num_channels)!'
            raise ValueError(msg)
        self._csmsum = real(self._csm).sum() + (imag(self._csm) ** 2).sum()  # to trigger new digest creation
        self._csm = csm

    @property_depends_on(['digest'])
    def _get_eva(self):
        """Eigenvalues of cross spectral matrix are either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        return self.calc_eva()

    @property_depends_on(['digest'])
    def _get_eve(self):
        """Eigenvectors of cross spectral matrix are either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        return self.calc_eve()

    def fftfreq(self):
        """Return the Discrete Fourier Transform sample frequencies.

        Returns
        -------
        f : ndarray
            Array containing the frequencies.

        """
        if isinstance(self.frequencies, float):
            return array([self.frequencies])
        if isinstance(self.frequencies, ndarray):
            return self.frequencies
        if self.frequencies is None:
            warn('No frequencies defined for PowerSpectraImport object!', stacklevel=1)
        return self.frequencies
