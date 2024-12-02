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
from traits.api import CArray, CLong, File, ListInt, Property, cached_property, on_trait_change

from .base import InOut

# acoular imports
from .deprecation import deprecated_alias
from .internal import digest


@deprecated_alias({'from_file': 'file'})
class Calib(InOut):
    """Container for calibration data in `*.xml` format or Numpy format (manually set).

    This class implements the application of calibration factors to the data obtained from its
    :attr:`source`. The calibrated data can be accessed (e.g. for use in a block chain) via the
    :meth:`result` generator. Depending on the source type, calibration can be performed in the
    time or frequency domain.


    Examples
    --------
        For example, to calibrate a time signal, the calibration data can be set manually and the
        Consider calibrating a time signal by specifying the calibration factors in NumPy format.
        Assume that the following white noise signal is in Volt and the sensitivity of the virtual
        sensor is 1e-2 V/Pa. Then, the voltage signal can be converted to a calibrated sound
        pressure signal by multiplying it with a calibration factor of  100 Pa/V.

        >>> import acoular as ac
        >>> import numpy as np
        >>>
        >>> signal = ac.WNoiseGenerator(sample_freq=51200, numsamples=51200, rms=2.0).signal()
        >>> ts = ac.TimeSamples(data=signal[:, np.newaxis], sample_freq=51200)
        >>> calib = ac.Calib(source=ts)
        >>> calib.data = np.array([100])
        >>> print(next(calib.result(num=1)))
        [[352.81046919]]

        The calibrated data can then be further processed, e.g. by calculating the FFT of the
        calibrated data.

        >>> fft = ac.RFFT(source=calib, block_size=16)
        >>> print(next(fft.result(num=1)))
        [[2127.75981743  +0.j          651.91512352-315.3052216j
          -454.68570762-678.21662333j   14.91685605 +99.31587754j
           354.53920079+784.46619566j  639.0864956  +35.97710777j
            67.58081917-786.68466133j  187.89848304+506.57222063j
           594.70514857  +0.j        ]]

        One could also apply the calibration after the FFT calculation.

        >>> fft = ac.RFFT(source=ts, block_size=16)
        >>> calib = ac.Calib(source=fft)
        >>> calib.data = 100 * np.ones(ts.num_channels * fft.num_freqs)
        >>> print(next(calib.result(num=1)))
        [[2127.75981743  +0.j          651.91512352-315.3052216j
          -454.68570762-678.21662333j   14.91685605 +99.31587754j
           354.53920079+784.46619566j  639.0864956  +35.97710777j
            67.58081917-786.68466133j  187.89848304+506.57222063j
           594.70514857  +0.j        ]]

    Depracted and will be removed in Acoular 25.10:
    This class serves as interface to load calibration data for the used
    microphone array. The calibration factors are stored as [Pa/unit].
    """

    #: Name of the .xml file to be imported.
    file = File(filter=['*.xml'], exists=True, desc='name of the xml file to import')

    #: Number of microphones in the calibration data,
    #: is set automatically when read from file or when data is set.
    num_mics = CLong(0, desc='number of microphones in the geometry')

    #: Array of calibration factors,
    #: is set automatically when read from file.
    #: Can be set manually by specifying a NumPy array with shape (num_channels, ) if
    #: :attr:`source` yields time domain signals. For frequency domain signals, the expected
    #: shape is (num_channels * num_freqs).
    data = CArray(desc='calibration data')

    #: Channels that are to be treated as invalid.
    invalid_channels = ListInt(desc='list of invalid channels')

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
        for block in self.source.result(num):
            if self.data[self.channels].shape[0] == block.shape[1]:
                yield block * self.data[self.channels][newaxis]
            else:
                msg = f'calibration data shape does not match source data shape: \
{self.data[self.channels].shape[0]}, {block.shape[1]}'
                raise ValueError(msg)
