# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements calibration of multichannel time signals.

.. autosummary::
    :toctree: generated/

    Calib
"""

# imports from other packages
import xml.dom.minidom

from numpy import array, newaxis
from traits.api import CArray, CInt, File, List, Property, cached_property, on_trait_change

import acoular as ac

from .base import InOut

# acoular imports
from .deprecation import deprecated_alias
from .internal import digest


@deprecated_alias({'from_file': 'file'})
class Calib(InOut):
    """Processing block for handling calibration data in `*.xml` or NumPy format.

    This class implements the application of calibration factors to the data obtained from its
    :attr:`source`. The calibrated data can be accessed (e.g. for use in a block chain) via the
    :meth:`result` generator. Depending on the source type, calibration can be performed in the
    time or frequency domain.


    Examples
    --------
    Consider calibrating a time signal by specifying the calibration factors in NumPy format.
    Assume that the following white noise signal is in Volt and the sensitivity of the virtual
    sensor is 1e-2 V/Pa. Then, the voltage signal can be converted to a calibrated sound
    pressure signal by multiplying it with a calibration factor of  100 Pa/V.

    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> signal = ac.WNoiseGenerator(sample_freq=51200,# doctest: +SKIP
    >>>                             num_samples=51200,# doctest: +SKIP
    >>>                             rms=0.01).signal()# doctest: +SKIP
    >>> ts = ac.TimeSamples(data=signal[:, np.newaxis], sample_freq=51200)  # doctest: +SKIP
    >>> calib = ac.Calib(source=ts)  # doctest: +SKIP
    >>> calib.data = np.array([100])  # doctest: +SKIP
    >>> print(next(calib.result(num=1)))  # doctest: +SKIP
    [[1.76405235]]

    The calibrated data can then be further processed, e.g. by calculating the FFT of the
    calibrated data.

    >>> fft = ac.RFFT(source=calib, block_size=16)  # doctest: +SKIP
    >>> print(next(fft.result(num=1)))  # doctest: +SKIP
    [[10.63879909+0.j          3.25957562-1.57652611j -2.27342854-3.39108312j
    0.07458428+0.49657939j  1.772696  +3.92233098j  3.19543248+0.17988554j
    0.3379041 -3.93342331j  0.93949242+2.5328611j   2.97352574+0.j        ]]

    One could also apply the calibration after the FFT calculation.

    >>> fft = ac.RFFT(source=ts, block_size=16)  # doctest: +SKIP
    >>> calib = ac.Calib(source=fft)  # doctest: +SKIP
    >>> calib.data = 100 * np.ones(ts.num_channels * fft.num_freqs)  # doctest: +SKIP
    >>> print(next(calib.result(num=1)))  # doctest: +SKIP
    [[10.63879909+0.j          3.25957562-1.57652611j -2.27342854-3.39108312j
    0.07458428+0.49657939j  1.772696  +3.92233098j  3.19543248+0.17988554j
    0.3379041 -3.93342331j  0.93949242+2.5328611j   2.97352574+0.j        ]]

    Deprecated and will be removed in Acoular 25.10:
    This class serves as interface to load calibration data for the used
    microphone array. The calibration factors are stored as [Pa/unit].
    """

    #: Name of the .xml file to be imported.
    file = File(filter=['*.xml'], exists=True, desc='name of the xml file to import')

    #: Number of microphones in the calibration data,
    #: is set automatically when read from file or when data is set.
    num_mics = CInt(0, desc='number of microphones in the geometry')

    #: Array of calibration factors,
    #: is set automatically when read from file.
    #: Can be set manually by specifying a NumPy array with shape (num_channels, ) if
    #: :attr:`source` yields time domain signals. For frequency domain signals, the expected
    #: shape is (num_channels * num_freqs).
    data = CArray(desc='calibration data')

    #: Channels that are to be treated as invalid.
    invalid_channels = List(int, desc='list of invalid channels')

    #: Channel mask to serve as an index for all valid channels, is set automatically.
    channels = Property(depends_on=['invalid_channels', 'num_mics'], desc='channel mask')

    # Internal identifier
    digest = Property(depends_on=['source.digest', 'data'])

    @on_trait_change('data')
    def set_num_mics(self):
        self.num_mics = self.data.shape[0]

    @cached_property
    def _get_channels(self):
        if len(self.invalid_channels) == 0:
            return slice(0, None, None)
        allr = [i for i in range(self.num_mics) if i not in self.invalid_channels]
        return array(allr)

    @cached_property
    def _get_digest(self):
        return digest(self)

    @on_trait_change('file')
    def import_data(self):
        """Loads the calibration data from `*.xml` file ."""
        doc = xml.dom.minidom.parse(self.file)
        names = []
        data = []
        for element in doc.getElementsByTagName('pos'):
            names.append(element.getAttribute('Name'))
            data.append(float(element.getAttribute('factor')))
        self.data = array(data, 'd')
        self.num_mics = self.data.shape[0]

    def __validate_data(self):
        """Validates the calibration data."""
        if self.data is None:
            msg = 'No calibration data available.'
            raise ValueError(msg)
        if self.source is None:
            msg = 'No source data available.'
            raise ValueError(msg)
        tobj = self.source
        while isinstance(tobj, ac.InOut):
            tobj = tobj.source
        if isinstance(tobj, ac.SamplesGenerator) and (self.data[self.channels].shape[0] != tobj.num_channels):
            msg = f'calibration data shape {self.data[self.channels].shape[0]} does not match \
                source data shape {tobj.num_channels}'
            raise ValueError(msg)
        if isinstance(tobj, ac.SpectraGenerator) and (
            self.data[self.channels].shape[0] != tobj.num_channels * tobj.num_freqs
        ):
            msg = f'calibration data shape {self.data[self.channels].shape[0]} does not match \
                source data shape {tobj.num_channels * tobj.num_freqs}'
            raise ValueError(msg)

    def result(self, num):
        """Python generator that processes the source data and yields the time-signal block-wise.

        This method needs to be implemented by the derived classes.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block)

        Yields
        ------
        numpy.ndarray
            Two-dimensional output data block of shape (num, sourcechannels)
            where sourcechannels is num_channels if the source data is in the time domain
            or sourcechannels is num_channels*num_freqs if the source data is in the frequency
            domain.
        """
        self.__validate_data()
        for block in self.source.result(num):
            yield block * self.data[self.channels][newaxis]
