# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implements blockwise processing methods in the frequency domain.

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
    """
    Compute the one-sided Fast Fourier Transform (FFT) for real-valued multichannel time data.

    The FFT is performed block-wise, dividing the input data into blocks of length specified
    by the :attr:`block_size` attribute. A window function can be optionally
    applied to each block before the FFT calculation, controlled via the :attr:`window` attribute.

    This class provides flexibility in scaling the FFT results for different use cases, such as
    preserving amplitude or energy, by setting the :attr:`scaling` attribute.
    """

    #: Data source; an instance of :class:`~acoular.base.SamplesGenerator` or a derived object.
    #: This provides the input time-domain data for FFT processing.
    source = Instance(SamplesGenerator)

    #: The number of workers to use for FFT calculation.
    #: If set to a negative value, all available logical CPUs are used.
    #: Default is ``None``, which relies on the :func:`scipy.fft.rfft` implementation.
    workers = Union(Int(), None, default_value=None, desc='number of workers to use')

    #: Defines the scaling method for the FFT result. Options are:
    #:
    #:     - ``'none'``: No scaling is applied.
    #:     - ``'energy'``: Compensates for energy loss in the FFT result due to truncation,
    #:       doubling the values for frequencies other than DC and the Nyquist frequency.
    #:     - ``'amplitude'``: Scales the result so that the amplitude
    #:       of discrete tones is independent of the block size.
    scaling = Enum('none', 'energy', 'amplitude')

    #: The length of each block of time-domain data used for the FFT. Must be an even number.
    #: Default is ``1024``.
    block_size = Property()

    #: The number of frequency bins in the FFT result, calculated as ``block_size // 2 + 1``.
    num_freqs = Property(depends_on=['_block_size'])

    #: The total number of snapshots (blocks) available in the FFT result,
    #: determined by the size of the input data and the block size.
    num_samples = Property(depends_on=['source.num_samples', '_block_size'])

    #: A 1-D array containing the Discrete Fourier Transform
    #: sample frequencies corresponding to the FFT output.
    freqs = Property()

    # Internal representation of the block size for FFT processing.
    # Used for validation and property management.
    _block_size = Int(1024, desc='block size of the FFT')

    #: A unique identifier based on the process properties.
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
        # Corrects the energy of the one-sided FFT data.
        if self.scaling == 'amplitude' or self.scaling == 'energy':
            data[1:-1] *= 2.0
        data *= scaling_value
        return data

    def _get_freqs(self):
        # Return the Discrete Fourier Transform sample frequencies.

        # Returns
        # -------
        # f : ndarray
        #     1-D Array of length *block_size // 2 + 1* containing the sample frequencies.
        if self.source is not None:
            return abs(fft.fftfreq(self.block_size, 1.0 / self.source.sample_freq)[: int(self.block_size / 2 + 1)])
        return np.array([])

    def result(self, num=1):
        """
        Yield the FFT results block-wise as multi-channel spectra.

        This generator processes the input data block-by-block, applying the specified
        window function and FFT parameters. The output consists of the FFT spectra for
        each block, scaled according to the selected :attr:`scaling` method.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of multi-channel spectra (snapshots) per block to return. Default is ``1``.

        Yields
        ------
        :class:`numpy.ndarray`
            A block of FFT spectra with shape (num, :attr:`num_channels` ``*`` :attr:`num_freqs`).
            The final block may contain fewer than ``num`` spectra if the input data is insufficient
            to fill the last block.

        Notes
        -----
        - The generator compensates for energy or amplitude loss based on the :attr:`scaling`
          attribute.
        - If the input data source provides fewer samples than required for a complete block,
          the remaining spectra are padded or adjusted accordingly.
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
    """
    Perform the inverse Fast Fourier Transform (IFFT) for one-sided multi-channel spectra.

    This class converts spectral data from the frequency domain back into time-domain signals.
    The IFFT is calculated block-wise, where the block size is defined by the spectral data
    source. The output time-domain signals are scaled and processed according to the precision
    defined by the :attr:`precision` attribute.
    """

    #: Data source providing one-sided spectra, implemented as an instance of
    # :class:`~acoular.base.SpectraGenerator` or a derived object.
    source = Instance(SpectraGenerator)

    #: The number of workers (threads) to use for the IFFT calculation.
    #: A negative value utilizes all available logical CPUs.
    #: Default is ``None``, which relies on the :func:`scipy.fft.irfft` implementation.
    workers = Union(Int(), None, default_value=None, desc='number of workers to use')

    #: Determines the floating-point precision of the resulting time-domain signals.
    #: Options include ``'float64'`` and ``'float32'``.
    #: Default is ``'float64'``, ensuring high precision.
    precision = Enum('float64', 'float32', desc='precision of the time signal after the ifft')

    #: The total number of time-domain samples in the output.
    #: Computed as the product of the number of input samples and the block size.
    #: Returns ``-1`` if the number of input samples is negative.
    num_samples = Property(depends_on=['source.num_samples', 'source._block_size'])

    # Internal signal buffer used for handling arbitrary output block sizes. Optimizes
    # processing when the requested output block size does not match the source block size.
    _buffer = CArray(desc='signal buffer')

    #: A unique identifier based on the process properties.
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
        """
        Generate time-domain signal blocks from spectral data.

        This generator processes spectral data block-by-block, performing an inverse Fast
        Fourier Transform (IFFT) to convert the input spectra into time-domain signals.
        The output is yielded in blocks of the specified size.

        Parameters
        ----------
        num : :class:`int`
            The number of time samples per output block. If ``num`` differs from the
            source block size, an internal buffer is used to assemble the required output.

        Yields
        ------
        :class:`numpy.ndarray`
            Blocks of time-domain signals with shape (num, :attr:`num_channels`). The last block may
            contain fewer samples if the input data is insufficient to fill the requested size.

        Notes
        -----
        - The method ensures that the source block size and frequency data are compatible for IFFT.
        - If the requested block size does not match the source block size, a buffer is used
          to assemble the output, allowing arbitrary block sizes to be generated.
        - For performance optimization, the number of workers (threads) can be specified via
          the :attr:`workers` attribute.
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
    """
    Compute the real-valued auto-power spectra from multi-channel frequency-domain data.

    The auto-power spectra provide a measure of the power contained in each frequency bin
    for each channel. This class processes spectral data from the source block-by-block,
    applying scaling and precision adjustments as configured by the :attr:`scaling` and
    :attr:`precision` attributes.
    """

    #: The data source that provides frequency-domain spectra,
    #: implemented as an instance of :class:`~acoular.base.SpectraGenerator` or a derived object.
    source = Instance(SpectraGenerator)

    #: Specifies the scaling method for the auto-power spectra. Options are:
    #:
    #:     - ``'power'``: Outputs the raw power of the spectra.
    #:     - ``'psd'``: Outputs the Power Spectral Density (PSD),
    #:       normalized by the block size and sampling frequency.
    scaling = Enum('power', 'psd')

    #: A Boolean flag indicating whether the input spectra are single-sided. Default is ``True``.
    single_sided = Bool(True, desc='single sided spectrum')

    #: Specifies the floating-point precision of the computed auto-power spectra.
    #: Options are ``'float64'`` and ``'float32'``. Default is ``'float64'``.
    precision = Enum('float64', 'float32', desc='floating-number-precision')

    #: A unique identifier based on the computation properties.
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
        """
        Generate real-valued auto-power spectra blocks.

        This generator computes the auto-power spectra by taking the element-wise squared
        magnitude of the input spectra and applying the appropriate scaling. The results
        are yielded block-by-block with the specified number of snapshots.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of snapshots in each output block. Default is ``1``.

        Yields
        ------
        :class:`numpy.ndarray`
            (num, :attr:`num_channels` ``*`` :attr:`num_freqs`). The last block may contain fewer
            snapshots if the input data does not completely fill the requested block size.

        Notes
        -----
        - The auto-power spectra are computed as the squared magnitude of the spectra
          :math:`|S(f)|^2`, where :math:`S(f)` is the frequency-domain signal.
        - Scaling is applied based on the configuration of the :attr:`scaling` and
          :attr:`single_sided` attributes.
        - The floating-point precision of the output is determined by the
          :attr:`precision` attribute.
        """
        scale = self._get_scaling_value()
        for temp in self.source.result(num):
            yield ((temp * temp.conjugate()).real * scale).astype(self.precision)


@deprecated_alias({'numchannels': 'num_channels'}, read_only=True)
class CrossPowerSpectra(AutoPowerSpectra):
    """
    Compute the complex-valued auto- and cross-power spectra from frequency-domain data.

    This class generates the cross-spectral matrix (CSM) in a flattened representation, which
    includes the auto-power spectra (diagonal elements) and cross-power spectra (off-diagonal
    elements). Depending on the :attr:`calc_mode`, the class can compute:

    - The full CSM, which includes all elements.
    - Only the upper triangle of the CSM.
    - Only the lower triangle of the CSM.

    The results are computed block-by-block and scaled according to the specified configuration.
    """

    #: The data source providing the input spectra,
    #: implemented as an instance of :class:`~acoular.base.SpectraGenerator` or a derived object.
    source = Instance(SpectraGenerator)

    #: Specifies the floating-point precision of the computed cross-spectral matrix (CSM).
    #: Options are ``'complex128'`` and ``'complex64'``. Default is ``'complex128'``.
    precision = Enum('complex128', 'complex64', desc='precision of the fft')

    #: Defines the calculation mode for the cross-spectral matrix:
    #:
    #:     - ``'full'``: Computes the full CSM, including all auto- and cross-power spectra.
    #:     - ``'upper'``: Computes only the upper triangle of the CSM,
    #        excluding redundant lower-triangle elements.
    #:     - ``'lower'``: Computes only the lower triangle of the CSM,
    #:       excluding redundant upper-triangle elements.
    #:
    #: Default is ``'full'``.
    calc_mode = Enum('full', 'upper', 'lower', desc='calculation mode')

    #: The number of channels in the output data. The value depends on the number of input channels
    #: :math:`n` and the selected :attr:`calc_mode`:
    #:
    #:     - ``'full'``: :math:`n^2` (all elements in the CSM).
    #:     - ``'upper'``: :math:`n + n(n-1)/2` (diagonal + upper triangle elements).
    #:     - ``'lower'``: :math:`n + n(n-1)/2` (diagonal + lower triangle elements).
    num_channels = Property(depends_on=['source.num_channels'])

    #: A unique identifier based on the computation properties.
    digest = Property(depends_on=['source.digest', 'precision', 'scaling', 'single_sided', 'calc_mode'])

    @cached_property
    def _get_num_channels(self):
        n = self.source.num_channels
        return n**2 if self.calc_mode == 'full' else int(n + n * (n - 1) / 2)

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=1):
        """
        Generate blocks of complex-valued auto- and cross-power spectra.

        This generator computes the cross-spectral matrix (CSM) for input spectra block-by-block.
        Depending on the :attr:`calc_mode`, the resulting CSM is flattened in one of three ways:

        - ``'full'``: Includes all elements of the CSM.
        - ``'upper'``: Includes only the diagonal and upper triangle.
        - ``'lower'``: Includes only the diagonal and lower triangle.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of snapshots (blocks) in each output data block. Default is ``1``.

        Yields
        ------
        :class:`numpy.ndarray`
            Blocks of complex-valued auto- and cross-power spectra with shape
            ``(num, :attr:`num_channels` * :attr:`num_freqs`)``.
            The last block may contain fewer than ``num`` elements if the input data
            does not completely fill the requested block size.
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
    """
    Provide the one-sided Fast Fourier Transform (FFT) for multichannel time data.

    .. deprecated:: 24.10
        The :class:`~acoular.fprocess.FFTSpectra` class is deprecated and will be removed
        in Acoular version 25.07. Please use :class:`~acoular.fprocess.RFFT` instead.

    Alias for the :class:`~acoular.fprocess.RFFT` class, which computes the one-sided
    Fast Fourier Transform (FFT) for multichannel time data.

    Warnings
    --------
    This class remains temporarily available for backward compatibility but should not be used in
    new implementations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(
            'Using FFTSpectra is deprecated and will be removed in Acoular version 25.07. Use class RFFT instead.',
            DeprecationWarning,
            stacklevel=2,
        )
