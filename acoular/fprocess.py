# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements blockwise processing methods in the frequency domain.

.. autosummary::
    :toctree: generated/

    RFFT
    IRFFT
    AutoPowerSpectra
    CrossPowerSpectra
    FFTSpectra
"""

from warnings import warn

import numpy as np
from scipy import fft
from traits.api import Bool, CArray, Enum, Instance, Int, Property, Union, cached_property

# acoular imports
from .base import SamplesGenerator, SpectraGenerator, SpectraOut, TimeOut
from .deprecation import deprecated_alias
from .fastFuncs import calcCSM
from .internal import digest
from .process import SamplesBuffer
from .spectra import BaseSpectra


@deprecated_alias({'numfreqs': 'num_freqs', 'numsamples': 'num_samples'}, read_only=True)
class RFFT(BaseSpectra, SpectraOut):
    """Provides the one-sided Fast Fourier Transform (FFT) for real-valued multichannel time data.

    The FFT is calculated block-wise, i.e. the input data is divided into blocks of length
    :attr:`block_size` and the FFT is calculated for each block. Optionally, a window function
    can be applied to the data before the FFT calculation via the :attr:`window` attribute.
    """

    #: Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    #: Number of workers to use for the FFT calculation. If negative values are used,
    #: all available logical CPUs will be considered (``scipy.fft.rfft`` implementation wraps around
    #: from ``os.cpu_count()``).
    #: Default is `None` (handled by scipy)
    workers = Union(Int(), None, default_value=None, desc='number of workers to use')

    #: Scaling method, either 'amplitude', 'energy' or :code:`none`.
    #: Default is :code:`none`.
    #: 'energy': compensates for the energy loss due to truncation of the FFT result. The resulting
    #: one-sided spectrum is multiplied by 2.0, except for the DC and Nyquist frequency.
    #: 'amplitude': scales the one-sided spectrum so that the amplitude of discrete tones does not
    #: depend on the block size.
    scaling = Enum('none', 'energy', 'amplitude')

    #: block size of the FFT. Default is 1024.
    block_size = Property()

    #: Number of frequencies in the output.
    num_freqs = Property(depends_on=['_block_size'])

    #: Number of snapshots in the output.
    num_samples = Property(depends_on=['source.num_samples', '_block_size'])

    #: 1-D array of FFT sample frequencies.
    freqs = Property()

    # internal block size variable
    _block_size = Int(1024, desc='block size of the FFT')

    # internal identifier
    digest = Property(depends_on=['source.digest', 'scaling', 'precision', '_block_size', 'window', 'overlap'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_num_freqs(self):
        return int(self.block_size / 2 + 1)

    @cached_property
    def _get_num_samples(self):
        if self.source.num_samples >= 0:
            return int(np.floor(self.source.num_samples / self.block_size))
        return -1

    def _get_block_size(self):
        return self._block_size

    def _set_block_size(self, value):
        if value % 2 != 0:
            msg = 'Block size must be even'
            raise ValueError(msg)
        self._block_size = value

    def _scale(self, data, scaling_value):
        """Corrects the energy of the one-sided FFT data."""
        if self.scaling == 'amplitude' or self.scaling == 'energy':
            data[1:-1] *= 2.0
        data *= scaling_value
        return data

    def _get_freqs(self):
        """Return the Discrete Fourier Transform sample frequencies.

        Returns
        -------
        f : ndarray
            1-D Array of length *block_size/2+1* containing the sample frequencies.

        """
        if self.source is not None:
            return abs(fft.fftfreq(self.block_size, 1.0 / self.source.sample_freq)[: int(self.block_size / 2 + 1)])
        return np.array([])

    def result(self, num=1):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the number of multi-channel spectra (i.e. snapshots) per block
            returned by the generator.

        Returns
        -------
        Spectra block of shape (num, :attr:`num_channels` * :attr:`num_freqs`).
            The last block may be shorter than num.

        """
        wind = self.window_(self.block_size)
        if self.scaling == 'none' or self.scaling == 'energy':  # only compensate for the window
            svalue = 1 / np.sqrt(np.dot(wind, wind) / self.block_size)
        elif self.scaling == 'amplitude':  # compensates for the window and the energy loss
            svalue = 1 / wind.sum()
        wind = wind[:, np.newaxis]
        fftdata = np.zeros((num, self.num_channels * self.num_freqs), dtype=self.precision)
        j = 0
        for i, data in enumerate(self._get_source_data()):  # yields one block of time data
            j = i % num
            fftdata[j] = self._scale(
                fft.rfft(data * wind, n=self.block_size, axis=0, workers=self.workers).astype(self.precision),
                scaling_value=svalue,
            ).reshape(-1)
            if j == num - 1:
                yield fftdata
        if j < num - 1:  # yield remaining fft spectra
            yield fftdata[: j + 1]


@deprecated_alias({'numsamples': 'num_samples'}, read_only=True)
class IRFFT(TimeOut):
    """Calculates the inverse Fast Fourier Transform (IFFT) for one-sided multi-channel spectra."""

    #: Data source; :class:`~acoular.base.SpectraGenerator` or derived object.
    source = Instance(SpectraGenerator)

    #: Number of workers to use for the FFT calculation. If negative values are used,
    #: all available logical CPUs will be considered (``scipy.fft.rfft`` implementation wraps around
    #: from ``os.cpu_count()``).
    #: Default is `None` (handled by scipy)
    workers = Union(Int(), None, default_value=None, desc='number of workers to use')

    #: The floating-number-precision of the resulting time signals, corresponding to numpy dtypes.
    #: Default is 64 bit.
    precision = Enum('float64', 'float32', desc='precision of the time signal after the ifft')

    #: Number of time samples in the output.
    num_samples = Property(depends_on=['source.num_samples', 'source._block_size'])

    # internal time signal buffer to handle arbitrary output block sizes
    _buffer = CArray(desc='signal buffer')

    # internal identifier
    digest = Property(depends_on=['source.digest', 'scaling', 'precision', '_block_size', 'window', 'overlap'])

    def _get_num_samples(self):
        if self.source.num_samples >= 0:
            return int(self.source.num_samples * self.source.block_size)
        return -1

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _validate(self):
        if not self.source.block_size or self.source.block_size < 0:
            msg = (
                f'Source of class {self.__class__.__name__} has an unknown blocksize: {self.source.block_size}.'
                'This is likely due to incomplete spectral data from which the inverse FFT cannot be calculated.'
            )
            raise ValueError(msg)
        if (self.source.num_freqs - 1) * 2 != self.source.block_size:
            msg = (
                f'Block size must be 2*(num_freqs-1) but is {self.source.block_size}.'
                'This is likely due to incomplete spectral data from which the inverse FFT cannot be calculated.'
            )
            raise ValueError(msg)
        if self.source.block_size % 2 != 0:
            msg = f'Block size must be even but is {self.source.block_size}.'
            raise ValueError(msg)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). The last block may be shorter than num.

        Yields
        ------
        numpy.ndarray
            Yields blocks of shape (num, num_channels).
        """
        self._validate()
        bs = self.source.block_size
        if num != bs:
            buffer_length = (int(np.ceil(num / bs)) + 1) * bs
            buffer = SamplesBuffer(source=self, source_num=bs, length=buffer_length, dtype=self.precision)
            yield from buffer.result(num)
        else:
            for spectra in self.source.result(1):
                yield fft.irfft(
                    spectra.reshape(self.source.num_freqs, self.num_channels), n=num, axis=0, workers=self.workers
                )


class AutoPowerSpectra(SpectraOut):
    """Calculates the real-valued auto-power spectra."""

    #: Data source; :class:`~acoular.base.SpectraGenerator` or derived object.
    source = Instance(SpectraGenerator)

    #: Scaling method, either 'power' or 'psd' (Power Spectral Density).
    #: Only relevant if the source is a :class:`~acoular.fprocess.FreqInOut` object.
    scaling = Enum('power', 'psd')

    #: Determines if the spectra yielded by the source are single-sided spectra.
    single_sided = Bool(True, desc='single sided spectrum')

    #: The floating-number-precision of entries, corresponding to numpy dtypes. Default is 64 bit.
    precision = Enum('float64', 'float32', desc='floating-number-precision')

    # internal identifier
    digest = Property(depends_on=['source.digest', 'precision', 'scaling', 'single_sided'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_scaling_value(self):
        scale = 1 / self.block_size**2
        if self.single_sided:
            scale *= 2
        if self.scaling == 'psd':
            scale *= self.block_size * self.source.sample_freq
        return scale

    def result(self, num=1):
        """Python generator that yields the real-valued auto-power spectra.

        Parameters
        ----------
        num : integer
            This parameter defines the number of snapshots within each output data block.

        Yields
        ------
        numpy.ndarray
            Yields blocks of shape (num, num_channels * num_freqs).
            The last block may be shorter than num.

        """
        scale = self._get_scaling_value()
        for temp in self.source.result(num):
            yield ((temp * temp.conjugate()).real * scale).astype(self.precision)


@deprecated_alias({'numchannels': 'num_channels'}, read_only=True)
class CrossPowerSpectra(AutoPowerSpectra):
    """Calculates the complex-valued auto- and cross-power spectra.

    Receives the complex-valued spectra from the source and returns the cross-spectral matrix (CSM)
    in a flattened representation (i.e. the auto- and cross-power spectra are concatenated along the
    last axis). If :attr:`calc_mode` is 'full', the full CSM is calculated, if 'upper', only the
    upper triangle is calculated.
    """

    #: Data source; :class:`~acoular.base.SpectraGenerator` or derived object.
    source = Instance(SpectraGenerator)

    #: The floating-number-precision of entries of csm, eigenvalues and
    #: eigenvectors, corresponding to numpy dtypes. Default is 64 bit.
    precision = Enum('complex128', 'complex64', desc='precision of the fft')

    #: Calculation mode, either 'full' or 'upper'.
    #: 'full' calculates the full cross-spectral matrix, 'upper' calculates
    # only the upper triangle. Default is 'full'.
    calc_mode = Enum('full', 'upper', 'lower', desc='calculation mode')

    #: Number of channels in output. If :attr:`calc_mode` is 'full', then
    #: :attr:`num_channels` is :math:`n^2`, where :math:`n` is the number of
    #: channels in the input. If :attr:`calc_mode` is 'upper', then
    #: :attr:`num_channels` is :math:`n + n(n-1)/2`.
    num_channels = Property(depends_on=['source.num_channels'])

    # internal identifier
    digest = Property(depends_on=['source.digest', 'precision', 'scaling', 'single_sided', 'calc_mode'])

    @cached_property
    def _get_num_channels(self):
        n = self.source.num_channels
        return n**2 if self.calc_mode == 'full' else int(n + n * (n - 1) / 2)

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=1):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Yields
        ------
        numpy.ndarray
            Yields blocks of shape (num, num_channels * num_freqs).
        """
        nc_src = self.source.num_channels
        nc = self.num_channels
        nf = self.num_freqs
        scale = self._get_scaling_value()

        csm_flat = np.zeros((num, nc * nf), dtype=self.precision)
        csm_upper = np.zeros((nf, nc_src, nc_src), dtype=self.precision)
        for data in self.source.result(num):
            for i in range(data.shape[0]):
                calcCSM(csm_upper, data[i].astype(self.precision).reshape(nf, nc_src))
                if self.calc_mode == 'full':
                    csm_lower = csm_upper.conj().transpose(0, 2, 1)
                    [np.fill_diagonal(csm_lower[cntFreq, :, :], 0) for cntFreq in range(csm_lower.shape[0])]
                    csm_flat[i] = (csm_lower + csm_upper).reshape(-1)
                elif self.calc_mode == 'upper':
                    csm_flat[i] = csm_upper[:, :nc].reshape(-1)
                else:  # lower
                    csm_lower = csm_upper.conj().transpose(0, 2, 1)
                    csm_flat[i] = csm_lower[:, :nc].reshape(-1)
                csm_upper[...] = 0  # calcCSM adds cumulative
            yield csm_flat[: i + 1] * scale


class FFTSpectra(RFFT):
    """Provides the one-sided Fast Fourier Transform (FFT) for multichannel time data.

    Alias for :class:`~acoular.fprocess.RFFT`.

    .. deprecated:: 24.10
        Using :class:`~acoular.fprocess.FFTSpectra` is deprecated and will be removed in Acoular
        version 25.07. Use :class:`~acoular.fprocess.RFFT` instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(
            'Using FFTSpectra is deprecated and will be removed in Acoular version 25.07. Use class RFFT instead.',
            DeprecationWarning,
            stacklevel=2,
        )
