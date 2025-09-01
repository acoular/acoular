# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implement blockwise processing in the time domain.

.. autosummary::
    :toctree: generated/

    MaskedTimeOut
    Trigger
    AngleTracker
    ChannelMixer
    SpatialInterpolator
    SpatialInterpolatorRotation
    SpatialInterpolatorConstantRotation
    Mixer
    TimePower
    TimeCumAverage
    TimeReverse
    Filter
    FilterBank
    FiltFiltOctave
    FiltOctave
    TimeExpAverage
    FiltFreqWeight
    OctaveFilterBank
    WriteWAV
    WriteH5
    TimeConvolve
"""

# imports from other packages
import wave
from abc import abstractmethod
from datetime import datetime, timezone
from os import path
from warnings import warn

import numba as nb
import numpy as np
import scipy.linalg as spla
from scipy.fft import irfft, rfft
from scipy.interpolate import CloughTocher2DInterpolator, CubicSpline, LinearNDInterpolator, Rbf, splev, splrep
from scipy.signal import bilinear, butter, sosfilt, sosfiltfilt, tf2sos
from scipy.spatial import Delaunay
from traits.api import (
    Bool,
    CArray,
    CInt,
    Constant,
    Delegate,
    Dict,
    Either,
    Enum,
    File,
    Float,
    Instance,
    Int,
    List,
    Map,
    Property,
    Str,
    Union,
    cached_property,
    observe,
)

# acoular imports
from .base import SamplesGenerator, TimeOut
from .configuration import config
from .deprecation import deprecated_alias
from .environments import cartToCyl, cylToCart
from .h5files import _get_h5file_class
from .internal import digest, ldigest
from .microphones import MicGeom
from .process import Cache
from .tools.utils import find_basename


@deprecated_alias(
    {'numchannels_total': 'num_channels_total', 'numsamples_total': 'num_samples_total'}, removal_version='25.10'
)
class MaskedTimeOut(TimeOut):
    """
    A signal processing block that allows for the selection of specific channels and time samples.

    The :class:`MaskedTimeOut` class is designed to filter data from a given
    :class:`~acoular.sources.SamplesGenerator` (or a derived object) by defining valid time samples
    and excluding specific channels. It acts as an intermediary between the data source and
    subsequent processing steps, ensuring that only the selected portion of the data is passed
    along.

    This class is useful for selecting specific portions of data for analysis. The processed data is
    accessed through the generator method :meth:`result`, which returns data in block-wise fashion
    for efficient streaming.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    #: This object provides the raw time-domain signals that will be filtered based on the
    #: :attr:`start`, :attr:`stop`, and :attr:`invalid_channels` attributes.
    source = Instance(SamplesGenerator)

    #: The index of the first valid sample. Default is ``0``.
    start = CInt(0, desc='start of valid samples')

    #: The index of the last valid sample (exclusive).
    #: If set to :obj:`None`, the selection continues until the end of the available data.
    stop = Union(None, CInt, desc='stop of valid samples')

    #: List of channel indices to be excluded from processing.
    invalid_channels = List(int, desc='list of invalid channels')

    #: A mask or index array representing valid channels. (automatically updated)
    channels = Property(depends_on=['invalid_channels', 'source.num_channels'], desc='channel mask')

    #: Total number of input channels, including invalid channels, as given by
    #: :attr:`~acoular.base.TimeOut.source`. (read-only).
    num_channels_total = Delegate('source', 'num_channels')

    #: Total number of input channels, including invalid channels. (read-only).
    num_samples_total = Delegate('source', 'num_samples')

    #: Number of valid input channels after excluding :attr:`invalid_channels`. (read-only)
    num_channels = Property(
        depends_on=['invalid_channels', 'source.num_channels'], desc='number of valid input channels'
    )

    #: Number of valid time-domain samples, based on :attr:`start` and :attr:`stop` indices.
    #: (read-only)
    num_samples = Property(
        depends_on=['start', 'stop', 'source.num_samples'], desc='number of valid samples per channel'
    )

    #: The name of the cache file (without extension). It serves as an internal reference for data
    #: caching and tracking processed files. (automatically generated)
    basename = Property(depends_on=['source.digest'], desc='basename for cache file')

    #: A unique identifier for the object, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'start', 'stop', 'invalid_channels'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        warn(
            (
                f'The basename attribute of a {self.__class__.__name__} object is deprecated'
                ' and will be removed in Acoular 26.01!'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return find_basename(self.source, alternative_basename=self.source.__class__.__name__ + self.source.digest)

    @cached_property
    def _get_channels(self):
        if len(self.invalid_channels) == 0:
            return slice(0, None, None)
        allr = [i for i in range(self.num_channels_total) if i not in self.invalid_channels]
        return np.array(allr)

    @cached_property
    def _get_num_channels(self):
        if len(self.invalid_channels) == 0:
            return self.num_channels_total
        return len(self.channels)

    @cached_property
    def _get_num_samples(self):
        sli = slice(self.start, self.stop).indices(self.num_samples_total)
        return sli[1] - sli[0]

    def result(self, num):
        """
        Generate blocks of processed data, selecting only valid samples and channels.

        This method fetches data from the :attr:`source` object, applies the defined :attr:`start`
        and :attr:`stop` constraints on time samples, and filters out :attr:`invalid_channels`. The
        data is then yielded in block-wise fashion to facilitate efficient streaming.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array of shape (``num``, :attr:`MaskedTimeOut.num_channels`), contatining blocks of
            a filtered time-domain signal. The last block may contain fewer samples if the total
            number of samples is not a multiple of ``num``. `MaskedTimeOut.num_channels` is not
            inherited directly and may be smaller than the :attr:`source`'s number of channels.

        Raises
        ------
        :obj:`OSError`
            If no valid samples are available within the defined :attr:`start` and :attr:`stop`
            range. This can occur if :attr:`start` is greater than or equal to :attr:`stop` or if
            the :attr:`source` is not containing any valid samples in the given range.
        """
        sli = slice(self.start, self.stop).indices(self.num_samples_total)
        start = sli[0]
        stop = sli[1]
        if start >= stop:
            msg = 'no samples available'
            raise OSError(msg)

        if start != 0 or stop != self.num_samples_total:
            offset = -start % num
            if offset == 0:
                offset = num
            buf = np.empty((num + offset, self.num_channels), dtype=float)
            bsize = 0
            i = 0
            fblock = True
            for block in self.source.result(num):
                bs = block.shape[0]
                i += bs
                if fblock and i >= start:  # first block in the chosen interval
                    if i >= stop:  # special case that start and stop are in one block
                        yield block[bs - (i - start) : bs - (i - stop), self.channels]
                        break
                    bsize += i - start
                    buf[: (i - start), :] = block[bs - (i - start) :, self.channels]
                    fblock = False
                elif i >= stop:  # last block
                    buf[bsize : bsize + bs - (i - stop), :] = block[: bs - (i - stop), self.channels]
                    bsize += bs - (i - stop)
                    if bsize > num:
                        yield buf[:num]
                        buf[: bsize - num, :] = buf[num:bsize, :]
                        bsize -= num
                    yield buf[:bsize, :]
                    break
                elif i >= start:
                    buf[bsize : bsize + bs, :] = block[:, self.channels]
                    bsize += bs
                if bsize >= num:
                    yield buf[:num]
                    buf[: bsize - num, :] = buf[num:bsize, :]
                    bsize -= num

        else:  # if no start/stop given, don't do the resorting thing
            for block in self.source.result(num):
                yield block[:, self.channels]


class ChannelMixer(TimeOut):
    """
    A signal processing block that mixes multiple input channels into a single output channel.

    The :class:`ChannelMixer` class takes a multi-channel signal from a
    :class:`~acoular.sources.SamplesGenerator` (or a derived object) and applies an optional set of
    amplitude weights to each channel. The resulting weighted sum is then output as a single-channel
    signal.

    This class is particularly useful for cases where a combined signal representation is needed,
    such as beamforming, array signal processing, or for reducing the dimensionality of
    multi-channel time signal data.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    #: It provides the multi-channel time-domain signals that will be mixed.
    source = Instance(SamplesGenerator)

    #: An array of amplitude weight factors applied to each input channel before summation.
    #: If not explicitly set, all channels are weighted equally (delault is ``1``).
    #: The shape of :attr:`weights` must match the :attr:`number of input channels<num_channels>`.
    #: If an incompatible shape is provided, a :obj:`ValueError` will be raised.
    weights = CArray(desc='channel weights')

    #: The number of output channels, which is always ``1`` for this class since it produces a
    #: single mixed output. (read-only)
    num_channels = Constant(1)

    #: A unique identifier for the object, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'weights'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num):
        """
        Generate the mixed output signal in blocks.

        This method retrieves data from the :attr:`source` object, applies the specified amplitude
        :attr:`weights` to each channel, and sums them to produce a single-channel output. The data
        is processed and yielded in block-wise fashion for efficient memory handling.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array of shape ``(num, 1)`` containing blocks a of single-channel mixed signal.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.

        Raises
        ------
        :obj:`ValueError`
            If the :attr:`weights` array is provided but its shape does not match the expected shape
            (:attr:`num_channels`,) or (``1``,), a :obj:`ValueError` is raised indicating that the
            weights cannot be broadcasted properly.
        """
        if self.weights.size:
            if self.weights.shape in {(self.source.num_channels,), (1,)}:
                weights = self.weights
            else:
                msg = f'Weight factors can not be broadcasted: {self.weights.shape}, {(self.source.num_channels,)}'
                raise ValueError(msg)
        else:
            weights = 1

        for block in self.source.result(num):
            yield np.sum(weights * block, 1, keepdims=True)


class Trigger(TimeOut):  # pragma: no cover
    """
    A signal processing class for detecting and analyzing trigger signals in time-series data.

    The :class:`Trigger` class identifies trigger events in a single-channel signal provided by a
    :class:`~acoular.base.SamplesGenerator` source. The detection process involves:

    1. Identifying peaks that exceed a specified positive or negative threshold.
    2. Estimating the approximate duration of one revolution based on the largest
       sample distance between consecutive peaks.
    3. Dividing the estimated revolution duration into segments called "hunks,"
       allowing only one peak per hunk.
    4. Selecting the most appropriate peak per hunk based on a chosen criterion
       (e.g., first occurrence or extremum value).
    5. Validating the consistency of the detected peaks by ensuring the revolutions
       have a stable duration with minimal variation.

    This class is typically used for rotational speed analysis, where trigger events
    correspond to periodic markers in a signal (e.g., TDC signals in engine diagnostics).
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    #: The signal must be single-channel.
    source = Instance(SamplesGenerator)

    #: The threshold value for detecting trigger peaks. The meaning of this threshold depends
    #: on the trigger type (:attr;`trigger_type`). The sign is relevant:
    #:
    #: - A positive threshold detects peaks above this value.
    #: - A negative threshold detects peaks below this value.
    #:
    #: If :obj:`None`, an estimated threshold is used, calculated as 75% of the extreme deviation
    #: from the mean signal value. Default is :obj:`None`.
    #:
    #: E.g: If the mean value is :math:`0` and there are positive extrema at :math:`400` and
    #: negative extrema at :math:`-800`. Then the estimated threshold would be
    #: :math:`0.75 \cdot (-800) = -600`.
    threshold = Union(None, Float)

    #: The maximum allowable variation in duration between two trigger instances. If any revolution
    #: exceeds this variation threshold, a warning is issued. Default is ``0.02``.
    max_variation_of_duration = Float(0.02)

    #: Defines the length of "hunks" as a fraction of the estimated duration between two trigger
    #: instances. If multiple peaks occur within a hunk, only one is retained based on
    #: :attr:`multiple_peaks_in_hunk`. Default is ``0.1``.
    hunk_length = Float(0.1)

    #: Specifies the type of trigger detection:
    #:
    #: - ``'dirac'``: A single impulse is considered a trigger. The sign of :attr:`threshold`
    #:   determines whether positive or negative peaks are detected.
    #: - ``'rect'``: A repeating rectangular waveform is assumed. Only every second edge is
    #:   considered a trigger. The sign of :attr:`threshold` determines whether rising (``+``) or
    #:   falling (``-``) edges are used.
    #:
    #: Default is ``'dirac'``.
    trigger_type = Enum('dirac', 'rect')

    #: Defines the criterion for selecting a peak when multiple occur within a hunk (see
    #: :attr:`hunk_length`):
    #:
    #: - ``'extremum'``: Selects the most extreme peak.
    #: - ``'first'``: Selects the first peak encountered.
    #:
    #: Default is ``'extremum'``.
    multiple_peaks_in_hunk = Enum('extremum', 'first')

    #: A tuple containing:
    #:
    #: - A :class:`numpy.ndarray` of sample indices corresponding to detected trigger events.
    #: - The maximum number of samples between consecutive trigger peaks.
    #: - The minimum number of samples between consecutive trigger peaks.
    trigger_data = Property(
        depends_on=[
            'source.digest',
            'threshold',
            'max_variation_of_duration',
            'hunk_length',
            'trigger_type',
            'multiple_peaks_in_hunk',
        ],
    )

    #: A unique identifier for the trigger, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'source.digest',
            'threshold',
            'max_variation_of_duration',
            'hunk_length',
            'trigger_type',
            'multiple_peaks_in_hunk',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_trigger_data(self):
        self._check_trigger_existence()
        triggerFunc = {'dirac': self._trigger_dirac, 'rect': self._trigger_rect}[self.trigger_type]
        num = 2048  # number samples for result-method of source
        threshold = self._threshold(num)

        # get all samples which surpasse the threshold
        peakLoc = np.array([], dtype='int')  # all indices which surpasse the threshold
        trigger_data = np.array([])
        x0 = []
        dSamples = 0
        for triggerSignal in self.source.result(num):
            localTrigger = np.flatnonzero(triggerFunc(x0, triggerSignal, threshold))
            if len(localTrigger) != 0:
                peakLoc = np.append(peakLoc, localTrigger + dSamples)
                trigger_data = np.append(trigger_data, triggerSignal[localTrigger])
            dSamples += num
            x0 = triggerSignal[-1]
        if len(peakLoc) <= 1:
            msg = 'Not enough trigger info. Check *threshold* sign and value!'
            raise Exception(msg)

        peakDist = peakLoc[1:] - peakLoc[:-1]
        maxPeakDist = max(peakDist)  # approximate distance between the revolutions

        # if there are hunks which contain multiple peaks -> check for each hunk,
        # which peak is the correct one -> delete the other one.
        # if there are no multiple peaks in any hunk left -> leave the while
        # loop and continue with program
        multiplePeaksWithinHunk = np.flatnonzero(peakDist < self.hunk_length * maxPeakDist)
        while len(multiplePeaksWithinHunk) > 0:
            peakLocHelp = multiplePeaksWithinHunk[0]
            indHelp = [peakLocHelp, peakLocHelp + 1]
            if self.multiple_peaks_in_hunk == 'extremum':
                values = trigger_data[indHelp]
                deleteInd = indHelp[np.argmin(abs(values))]
            elif self.multiple_peaks_in_hunk == 'first':
                deleteInd = indHelp[1]
            peakLoc = np.delete(peakLoc, deleteInd)
            trigger_data = np.delete(trigger_data, deleteInd)
            peakDist = peakLoc[1:] - peakLoc[:-1]
            multiplePeaksWithinHunk = np.flatnonzero(peakDist < self.hunk_length * maxPeakDist)

        # check whether distances between peaks are evenly distributed
        meanDist = np.mean(peakDist)
        diffDist = abs(peakDist - meanDist)
        faultyInd = np.flatnonzero(diffDist > self.max_variation_of_duration * meanDist)
        if faultyInd.size != 0:
            warn(
                f'In Trigger-Identification: The distances between the peaks (and therefore the lengths of the \
                revolutions) vary too much (check samples {peakLoc[faultyInd] + self.source.start}).',
                Warning,
                stacklevel=2,
            )
        return peakLoc, max(peakDist), min(peakDist)

    def _trigger_dirac(self, x0, x, threshold):  # noqa: ARG002
        # x0 not needed here, but needed in _trigger_rect
        return self._trigger_value_comp(x, threshold)

    def _trigger_rect(self, x0, x, threshold):
        # x0 stores the last value of the the last generator cycle
        xNew = np.append(x0, x)
        # indPeakHunk = abs(xNew[1:] - xNew[:-1]) > abs(threshold)
        # with above line, every edge would be located
        return self._trigger_value_comp(xNew[1:] - xNew[:-1], threshold)

    def _trigger_value_comp(self, trigger_data, threshold):
        return trigger_data > threshold if threshold > 0.0 else trigger_data < threshold

    def _threshold(self, num):
        if self.threshold is None:  # take a guessed threshold
            # get max and min values of whole trigger signal
            maxVal = -np.inf
            minVal = np.inf
            meanVal = 0
            cntMean = 0
            for trigger_data in self.source.result(num):
                maxVal = max(maxVal, trigger_data.max())
                minVal = min(minVal, trigger_data.min())
                meanVal += trigger_data.mean()
                cntMean += 1
            meanVal /= cntMean

            # get 75% of maximum absolute value of trigger signal
            maxTriggerHelp = [minVal, maxVal] - meanVal
            argInd = np.argmax(abs(maxTriggerHelp))
            thresh = maxTriggerHelp[argInd] * 0.75  # 0.75 for 75% of max trigger signal
            warn(f'No threshold was passed. An estimated threshold of {thresh} is assumed.', Warning, stacklevel=2)
        else:  # take user defined  threshold
            thresh = self.threshold
        return thresh

    def _check_trigger_existence(self):
        nChannels = self.source.num_channels
        if nChannels != 1:
            msg = f'Trigger signal must consist of ONE channel, instead {nChannels} channels are given!'
            raise Exception(msg)
        return 0

    def result(self, num):
        """
        Generate signal data from the source without modification.

        This method acts as a pass-through, providing data blocks directly from the :attr:`source`
        generator. It is included for interface consistency but does not apply trigger-based
        transformations to the data.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array containing ``num`` samples from the source signal.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.

        Warnings
        --------
        This method is not implemented for trigger-based transformations.
        A warning is issued, indicating that data is passed unprocessed.
        """
        msg = 'result method not implemented yet! Data from source will be passed without transformation.'
        warn(msg, Warning, stacklevel=2)
        yield from self.source.result(num)


class AngleTracker(MaskedTimeOut):
    """
    Compute the rotational angle and RPM per sample from a trigger signal in the time domain.

    This class retrieves samples from the specified :attr:`trigger` signal and interpolates angular
    position and rotational speed. The results are stored in the properties :attr:`angle` and
    :attr:`rpm`.

    The algorithm assumes a periodic trigger signal marking rotational events (e.g., a tachometer
    pulse or an encoder signal) and interpolates the angle and RPM using cubic splines. It is
    capable of handling different rotational directions and numbers of triggers per revolution.
    """

    #: Trigger data source, expected to be an instance of :class:`Trigger`.
    trigger = Instance(Trigger)

    #: A unique identifier for the tracker, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'source.digest',
            'trigger.digest',
            'trigger_per_revo',
            'rot_direction',
            'interp_points',
            'start_angle',
        ],
    )

    #: Number of trigger signals per revolution. This allows tracking scenarios where multiple
    #: trigger pulses occur per rotation. Default is ``1``, meaning a single trigger per revolution.
    trigger_per_revo = Int(1, desc='trigger signals per revolution')

    #: Rotation direction flag:
    #:
    #: - ``1``: counter-clockwise rotation.
    #: - ``-1``: clockwise rotation.
    #:
    #: Default is ``-1``.
    rot_direction = Int(-1, desc='mathematical direction of rotation')

    #: Number of points used for spline interpolation. Default is ``4``.
    interp_points = Int(4, desc='Points of interpolation used for spline')

    #: Initial rotation angle (in radians) corresponding to the first trigger event. This allows
    #: defining a custom starting reference angle. Default is ``0``.
    start_angle = Float(0, desc='rotation angle for trigger position')

    #: Revolutions per minute (RPM) computed for each sample.
    #: It is based on the trigger data. (read-only)
    rpm = Property(depends_on=['digest'], desc='revolutions per minute for each sample')

    #: Average revolutions per minute over the entire dataset.
    #: It is computed based on the trigger intervals. (read-only)
    average_rpm = Property(depends_on=['digest'], desc='average revolutions per minute')

    #: Computed rotation angle (in radians) for each sample.
    #: It is interpolated from the trigger data. (read-only)
    angle = Property(depends_on=['digest'], desc='rotation angle for each sample')

    # Internal flag to determine whether rpm and angle calculation has been processed,
    # prevents recalculation
    _calc_flag = Bool(False)

    # Revolutions per minute, internal use
    _rpm = CArray()

    # Rotation angle in radians, internal use
    _angle = CArray()

    @cached_property
    def _get_digest(self):
        return digest(self)

    # helperfunction for trigger index detection
    def _find_nearest_idx(self, peakarray, value):
        peakarray = np.asarray(peakarray)
        return (abs(peakarray - value)).argmin()

    def _to_rpm_and_angle(self):
        # Internal helper function.
        # Calculates angles in radians for one or more instants in time.

        # Current version supports only trigger and sources with the same samplefreq.
        # This behaviour may change in future releases.

        # init
        ind = 0
        # trigger data
        peakloc, maxdist, mindist = self.trigger.trigger_data
        TriggerPerRevo = self.trigger_per_revo
        rotDirection = self.rot_direction
        num = self.source.num_samples
        samplerate = self.source.sample_freq
        self._rpm = np.zeros(num)
        self._angle = np.zeros(num)
        # number of spline points
        InterpPoints = self.interp_points

        # loop over all timesamples
        while ind < num:
            # when starting spline forward
            if ind < peakloc[InterpPoints]:
                peakdist = (
                    peakloc[self._find_nearest_idx(peakarray=peakloc, value=ind) + 1]
                    - peakloc[self._find_nearest_idx(peakarray=peakloc, value=ind)]
                )
                splineData = np.stack(
                    (range(InterpPoints), peakloc[ind // peakdist : ind // peakdist + InterpPoints]),
                    axis=0,
                )
            # spline backwards
            else:
                peakdist = (
                    peakloc[self._find_nearest_idx(peakarray=peakloc, value=ind)]
                    - peakloc[self._find_nearest_idx(peakarray=peakloc, value=ind) - 1]
                )
                splineData = np.stack(
                    (range(InterpPoints), peakloc[ind // peakdist - InterpPoints : ind // peakdist]),
                    axis=0,
                )
            # calc angles and rpm
            Spline = splrep(splineData[:, :][1], splineData[:, :][0], k=3)
            self._rpm[ind] = splev(ind, Spline, der=1, ext=0) * 60 * samplerate
            self._angle[ind] = (
                splev(ind, Spline, der=0, ext=0) * 2 * np.pi * rotDirection / TriggerPerRevo + self.start_angle
            ) % (2 * np.pi)
            # next sample
            ind += 1
        # calculation complete
        self._calc_flag = True

    # reset calc flag if something has changed
    @observe('digest')
    def _reset_calc_flag(self, event):  # noqa ARG002
        self._calc_flag = False

    # calc rpm from trigger data
    @cached_property
    def _get_rpm(self):
        if not self._calc_flag:
            self._to_rpm_and_angle()
        return self._rpm

    # calc of angle from trigger data
    @cached_property
    def _get_angle(self):
        if not self._calc_flag:
            self._to_rpm_and_angle()
        return self._angle

    # calc average rpm from trigger data
    @cached_property
    def _get_average_rpm(self):
        # trigger indices data
        peakloc = self.trigger.trigger_data[0]
        # calculation of average rpm in 1/min
        return (len(peakloc) - 1) / (peakloc[-1] - peakloc[0]) / self.trigger_per_revo * self.source.sample_freq * 60


class SpatialInterpolator(TimeOut):  # pragma: no cover
    """
    Base class for spatial interpolation of microphone data.

    This class retrieves samples from a specified source and performs spatial interpolation to
    generate output at virtual microphone positions. The interpolation is executed using various
    methods such as linear, spline, radial basis function (RBF), and inverse distance weighting
    (IDW).

    See Also
    --------
    :class:`SpatialInterpolatorRotation` : Spatial interpolation class for rotating sound sources.
    :class:`SpatialInterpolatorConstantRotation` :
        Performs spatial linear interpolation for sources undergoing constant rotation.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    #: It provides the time-domain pressure samples from microphones.
    source = Instance(SamplesGenerator)

    #: The physical microphone geometry. An instance of :class:`~acoular.microphones.MicGeom` that
    #: defines the positions of the real microphones used for measurement.
    mics = Instance(MicGeom(), desc='microphone geometry')

    #: The virtual microphone geometry. This property defines the positions
    #: of virtual microphones where interpolated pressure values are computed.
    #: Default is the physical microphone geometry (:attr:`mics`).
    mics_virtual = Property(desc='microphone geometry')

    _mics_virtual = Instance(MicGeom, desc='internal microphone geometry;internal usage, read only')

    def _get_mics_virtual(self):
        if not self._mics_virtual and self.mics:
            self._mics_virtual = self.mics
        return self._mics_virtual

    def _set_mics_virtual(self, mics_virtual):
        self._mics_virtual = mics_virtual

    #: Interpolation method used for spatial data estimation.
    #:
    #: Options:
    #:
    #: - ``'linear'``: Uses NumPy linear interpolation.
    #: - ``'spline'``: Uses SciPy's CubicSpline interpolator
    #: - ``'rbf-multiquadric'``: Radial basis function (RBF) interpolation with a multiquadric
    #:   kernel.
    #: - ``'rbf-cubic'``: RBF interpolation with a cubic kernel.
    #: - ``'IDW'``: Inverse distance weighting interpolation.
    #: - ``'custom'``: Allows user-defined interpolation methods.
    #: - ``'sinc'``: Uses sinc-based interpolation for signal reconstruction.
    method = Enum(
        'linear',
        'spline',
        'rbf-multiquadric',
        'rbf-cubic',
        'IDW',
        'custom',
        'sinc',
        desc='method for interpolation used',
    )

    #: Defines the spatial dimensionality of the microphone array.
    #:
    #: Possible values:
    #:
    #: - ``'1D'``: Linear microphone arrays.
    #: - ``'2D'``: Planar microphone arrays.
    #: - ``'ring'``: Circular arrays where rotation needs to be considered.
    #: - ``'3D'``: Three-dimensional microphone distributions.
    #: - ``'custom'``: User-defined microphone arrangements.
    array_dimension = Enum('1D', '2D', 'ring', '3D', 'custom', desc='spatial dimensionality of the array geometry')

    #: Sampling frequency of the output signal, inherited from the :attr:`source`. This defines the
    #: rate at which microphone pressure samples are acquired and processed.
    sample_freq = Delegate('source', 'sample_freq')

    #: Number of channels in the output data. This corresponds to the number of virtual microphone
    #: positions where interpolated pressure values are computed. The value is ´determined based on
    #: the :attr:`mics_virtual` geometry.
    num_channels = Property()

    #: Number of time-domain samples in the output signal, inherited from the :attr:`source`.
    num_samples = Delegate('source', 'num_samples')

    #: Whether to interpolate a virtual microphone at the origin. If set to ``True``, an additional
    #: virtual microphone position at the coordinate origin :math:`(0,0,0)` will be interpolated.
    interp_at_zero = Bool(False)

    #: Transformation matrix for coordinate system alignment.
    #:
    #: This 3x3 orthogonal matrix is used to align the microphone coordinates such that rotations
    #: occur around the z-axis. If the original coordinates do not conform to the expected alignment
    #: (where the x-axis transitions into the y-axis upon rotation), applying this matrix modifies
    #: the coordinates accordingly. The transformation is defined as
    #:
    #: .. math::
    #:     \begin{bmatrix}x'\\y'\\z'\end{bmatrix} = Q \cdot \begin{bmatrix}x\\y\\z\end{bmatrix}
    #:
    #: where :math:`Q` is the transformation matrix and :math:`(x', y', z')` are the modified
    #: coordinates. If no transformation is needed, :math:`Q` defaults to the identity matrix.
    Q = CArray(dtype=np.float64, shape=(3, 3), value=np.identity(3))

    #: Number of neighboring microphones used in IDW interpolation. This parameter determines how
    #: many physical microphones contribute to the weighted sum in inverse distance weighting (IDW)
    #: interpolation.
    num_IDW = Int(3, desc='number of neighboring microphones, DEFAULT=3')  # noqa: N815

    #: Weighting exponent for IDW interpolation. This parameter controls the influence of distance
    #: in inverse distance weighting (IDW). A higher value gives more weight to closer microphones.
    p_weight = Float(
        2,
        desc='used in interpolation for virtual microphone, weighting power exponent for IDW',
    )

    # Stores the output of :meth:`_virtNewCoord_func`; Read-Only
    _virtNewCoord_func = Property(  # noqa: N815
        depends_on=['mics.digest', 'mics_virtual.digest', 'method', 'array_dimension', 'interp_at_zero'],
    )

    #: Unique identifier for the current configuration of the interpolator. (read-only)
    digest = Property(
        depends_on=[
            'mics.digest',
            'mics_virtual.digest',
            'source.digest',
            'method',
            'array_dimension',
            'Q',
            'interp_at_zero',
        ],
    )

    def _get_num_channels(self):
        return self.mics_virtual.num_mics

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_virtNewCoord(self):  # noqa N802
        return self._virtNewCoord_func(self.mics.mpos, self.mics_virtual.mpos, self.method, self.array_dimension)

    def sinc_mic(self, r):
        """
        Compute a modified sinc function for use in Radial Basis Function (RBF) approximation.

        This function is used as a kernel in sinc-based interpolation methods, where the sinc
        function serves as a basis function for reconstructing signals based on spatially
        distributed microphone data. The function is scaled according to the number of virtual
        microphone positions, ensuring accurate signal approximation.

        Parameters
        ----------
        r : :obj:`float` or :obj:`list` of :obj:`floats<float>`
            The radial distance(s) at which to evaluate the sinc function, typically representing
            the spatial separation between real and virtual microphone positions.

        Returns
        -------
        :class:`numpy.ndarray`
            Evaluated sinc function values at the given radial distances.
        """
        return np.sinc((r * self.mics_virtual.mpos.shape[1]) / (np.pi))

    def _virtNewCoord_func(self, mpos, mpos_virt, method, array_dimension):  # noqa N802
        # Core functionality for getting the interpolation.
        #
        # Parameters
        # ----------
        # mpos : float[3, nPhysicalMics]
        #     The mic positions of the physical (really existing) mics
        # mpos_virt : float[3, nVirtualMics]
        #     The mic positions of the virtual mics
        # method : string
        #     The Interpolation method to use
        # array_dimension : string
        #     The Array Dimensions in cylinder coordinates
        #
        # Returns
        # -------
        # mesh : List[]
        #     The items of these lists depend on the reduced interpolation dimension of each
        #     subarray.
        #     If the Array is 1D the list items are:
        #         1. item : float64[nMicsInSpecificSubarray]
        #             Ordered positions of the real mics on the new 1d axis,
        #             to be used as inputs for numpys interp.
        #         2. item : int64[nMicsInArray]
        #             Indices identifying how the measured pressures must be evaluated, s.t. the
        #             entries of the previous item (see last line) correspond to their initial
        #             pressure values.
        #     If the Array is 2D or 3d the list items are:
        #         1. item : Delaunay mesh object
        #             Delaunay mesh (see scipy.spatial.Delaunay) for the specific Array
        #         2. item : int64[nMicsInArray]
        #             same as 1d case, BUT with the difference, that here the rotational periodicity
        #             is handled, when constructing the mesh. Therefore, the mesh could have more
        #             vertices than the actual Array mics.
        #
        # virtNewCoord : float64[3, nVirtualMics]
        #     Projection of each virtual mic onto its new coordinates. The columns of virtNewCoord
        #     correspond to [phi, rho, z].
        #
        # newCoord : float64[3, nMics]
        #     Projection of each mic onto its new coordinates. The columns of newCoordinates
        #     correspond to [phi, rho, z].

        # init positions of virtual mics in cyl coordinates
        nVirtMics = mpos_virt.shape[1]
        virtNewCoord = np.zeros((3, nVirtMics))
        virtNewCoord.fill(np.nan)
        # init real positions in cyl coordinates
        nMics = mpos.shape[1]
        newCoord = np.zeros((3, nMics))
        newCoord.fill(np.nan)
        # empty mesh object
        mesh = []

        if self.array_dimension == '1D' or self.array_dimension == 'ring':
            # get projections onto new coordinate, for real mics
            projectionOnNewAxis = cartToCyl(mpos, self.Q)[0]
            indReorderHelp = np.argsort(projectionOnNewAxis)
            mesh.append([projectionOnNewAxis[indReorderHelp], indReorderHelp])

            # new coordinates of real mics
            indReorderHelp = np.argsort(cartToCyl(mpos, self.Q)[0])
            newCoord = (cartToCyl(mpos, self.Q).T)[indReorderHelp].T

            # and for virtual mics
            virtNewCoord = cartToCyl(mpos_virt)

        elif self.array_dimension == '2D':  # 2d case0
            # get virtual mic projections on new coord system
            virtNewCoord = cartToCyl(mpos_virt, self.Q)

            # new coordinates of real mics
            indReorderHelp = np.argsort(cartToCyl(mpos, self.Q)[0])
            newCoord = cartToCyl(mpos, self.Q)

            # scipy delauney triangulation
            # Delaunay
            tri = Delaunay(newCoord.T[:, :2], incremental=True)  #

            if self.interp_at_zero:
                # add a point at zero
                tri.add_points(np.array([[0], [0]]).T)

            # extend mesh with closest boundary points of repeating mesh
            pointsOriginal = np.arange(tri.points.shape[0])
            hull = tri.convex_hull
            hullPoints = np.unique(hull)

            addRight = tri.points[hullPoints]
            addRight[:, 0] += 2 * np.pi
            addLeft = tri.points[hullPoints]
            addLeft[:, 0] -= 2 * np.pi

            indOrigPoints = np.concatenate((pointsOriginal, pointsOriginal[hullPoints], pointsOriginal[hullPoints]))
            # add all hull vertices to original mesh and check which of those
            # are actual neighbors of the original array. Cancel out all others.
            tri.add_points(np.concatenate([addLeft, addRight]))
            indices, indptr = tri.vertex_neighbor_vertices
            hullNeighbor = np.empty((0), dtype='int32')
            for currHull in hullPoints:
                neighborOfHull = indptr[indices[currHull] : indices[currHull + 1]]
                hullNeighbor = np.append(hullNeighbor, neighborOfHull)
            hullNeighborUnique = np.unique(hullNeighbor)
            pointsNew = np.unique(np.append(pointsOriginal, hullNeighborUnique))
            tri = Delaunay(tri.points[pointsNew])  # re-meshing
            mesh.append([tri, indOrigPoints[pointsNew]])

        elif self.array_dimension == '3D':  # 3d case
            # get virtual mic projections on new coord system
            virtNewCoord = cartToCyl(mpos_virt, self.Q)
            # get real mic projections on new coord system
            indReorderHelp = np.argsort(cartToCyl(mpos, self.Q)[0])
            newCoord = cartToCyl(mpos, self.Q)
            # Delaunay
            tri = Delaunay(newCoord.T, incremental=True)  # , incremental=True,qhull_options =  "Qc QJ Q12"

            if self.interp_at_zero:
                # add a point at zero
                tri.add_points(np.array([[0], [0], [0]]).T)

            # extend mesh with closest boundary points of repeating mesh
            pointsOriginal = np.arange(tri.points.shape[0])
            hull = tri.convex_hull
            hullPoints = np.unique(hull)

            addRight = tri.points[hullPoints]
            addRight[:, 0] += 2 * np.pi
            addLeft = tri.points[hullPoints]
            addLeft[:, 0] -= 2 * np.pi

            indOrigPoints = np.concatenate((pointsOriginal, pointsOriginal[hullPoints], pointsOriginal[hullPoints]))
            # add all hull vertices to original mesh and check which of those
            # are actual neighbors of the original array. Cancel out all others.
            tri.add_points(np.concatenate([addLeft, addRight]))
            indices, indptr = tri.vertex_neighbor_vertices
            hullNeighbor = np.empty((0), dtype='int32')
            for currHull in hullPoints:
                neighborOfHull = indptr[indices[currHull] : indices[currHull + 1]]
                hullNeighbor = np.append(hullNeighbor, neighborOfHull)
            hullNeighborUnique = np.unique(hullNeighbor)
            pointsNew = np.unique(np.append(pointsOriginal, hullNeighborUnique))
            tri = Delaunay(tri.points[pointsNew])  # re-meshing
            mesh.append([tri, indOrigPoints[pointsNew]])

        return mesh, virtNewCoord, newCoord

    def _result_core_func(self, p, phi_delay=None, period=None, Q=Q, interp_at_zero=False):  # noqa: N803, ARG002 (see #226)
        # Performs the actual Interpolation.
        #
        # Parameters
        # ----------
        # p : float[num, nMicsReal]
        #     The pressure field of the yielded sample at real mics.
        # phi_delay : empty list (default) or float[num]
        #     If passed (rotational case), this list contains the angular delay
        #     of each sample in rad.
        # period : None (default) or float
        #     If periodicity can be assumed (rotational case)
        #     this parameter contains the periodicity length
        #
        # Returns
        # -------
        # pInterp : float[num, nMicsVirtual]
        #     The interpolated time data at the virtual mics
        if phi_delay is None:
            phi_delay = []
        # number of time samples
        nTime = p.shape[0]
        # number of virtual mixcs
        nVirtMics = self.mics_virtual.mpos.shape[1]
        # mesh and projection onto polar Coordinates
        meshList, virtNewCoord, newCoord = self._get_virtNewCoord()
        # pressure interpolation init
        pInterp = np.zeros((nTime, nVirtMics))
        # Coordinates in cartesian CO - for IDW interpolation
        newCoordCart = cylToCart(newCoord)

        if self.interp_at_zero:
            # interpolate point at 0 in Kartesian CO
            interpolater = LinearNDInterpolator(
                cylToCart(newCoord[:, np.argsort(newCoord[0])])[:2, :].T,
                p[:, (np.argsort(newCoord[0]))].T,
                fill_value=0,
            )
            pZero = interpolater((0, 0))
            # add the interpolated pressure at origin to pressure channels
            p = np.concatenate((p, pZero[:, np.newaxis]), axis=1)

        # helpfunction reordered for reordered pressure values
        pHelp = p[:, meshList[0][1]]

        # Interpolation for 1D Arrays
        if self.array_dimension == '1D' or self.array_dimension == 'ring':
            # for rotation add phi_delay
            if not np.array_equal(phi_delay, []):
                xInterpHelp = np.tile(virtNewCoord[0, :], (nTime, 1)) + np.tile(phi_delay, (virtNewCoord.shape[1], 1)).T
                xInterp = ((xInterpHelp + np.pi) % (2 * np.pi)) - np.pi  #  shifting phi into feasible area [-pi, pi]
            # if no rotation given
            else:
                xInterp = np.tile(virtNewCoord[0, :], (nTime, 1))
            # get ordered microphone positions in radiant
            x = newCoord[0]
            for cntTime in range(nTime):
                if self.method == 'linear':
                    # numpy 1-d interpolation
                    pInterp[cntTime] = np.interp(
                        xInterp[cntTime, :],
                        x,
                        pHelp[cntTime, :],
                        period=period,
                        left=np.nan,
                        right=np.nan,
                    )

                elif self.method == 'spline':
                    # scipy cubic spline interpolation
                    SplineInterp = CubicSpline(
                        np.append(x, (2 * np.pi) + x[0]),
                        np.append(pHelp[cntTime, :], pHelp[cntTime, :][0]),
                        axis=0,
                        bc_type='periodic',
                        extrapolate=None,
                    )
                    pInterp[cntTime] = SplineInterp(xInterp[cntTime, :])

                elif self.method == 'sinc':
                    # compute using 3-D Rbfs for sinc
                    rbfi = Rbf(
                        x,
                        newCoord[1],
                        newCoord[2],
                        pHelp[cntTime, :],
                        function=self.sinc_mic,
                    )  # radial basis function interpolator instance

                    pInterp[cntTime] = rbfi(xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2])

                elif self.method == 'rbf-cubic':
                    # compute using 3-D Rbfs with multiquadratics
                    rbfi = Rbf(
                        x,
                        newCoord[1],
                        newCoord[2],
                        pHelp[cntTime, :],
                        function='cubic',
                    )  # radial basis function interpolator instance

                    pInterp[cntTime] = rbfi(xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2])

        # Interpolation for arbitrary 2D Arrays
        elif self.array_dimension == '2D':
            # check rotation
            if not np.array_equal(phi_delay, []):
                xInterpHelp = np.tile(virtNewCoord[0, :], (nTime, 1)) + np.tile(phi_delay, (virtNewCoord.shape[1], 1)).T
                xInterp = ((xInterpHelp + np.pi) % (2 * np.pi)) - np.pi  # shifting phi into feasible area [-pi, pi]
            else:
                xInterp = np.tile(virtNewCoord[0, :], (nTime, 1))

            mesh = meshList[0][0]
            for cntTime in range(nTime):
                # points for interpolation
                newPoint = np.concatenate(
                    (xInterp[cntTime, :][:, np.newaxis], virtNewCoord[1, :][:, np.newaxis]), axis=1
                )
                # scipy 1D interpolation
                if self.method == 'linear':
                    interpolater = LinearNDInterpolator(mesh, pHelp[cntTime, :], fill_value=0)
                    pInterp[cntTime] = interpolater(newPoint)

                elif self.method == 'spline':
                    # scipy CloughTocher interpolation
                    f = CloughTocher2DInterpolator(mesh, pHelp[cntTime, :], fill_value=0)
                    pInterp[cntTime] = f(newPoint)

                elif self.method == 'sinc':
                    # compute using 3-D Rbfs for sinc
                    rbfi = Rbf(
                        newCoord[0],
                        newCoord[1],
                        newCoord[2],
                        pHelp[cntTime, : len(newCoord[0])],
                        function=self.sinc_mic,
                    )  # radial basis function interpolator instance

                    pInterp[cntTime] = rbfi(xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2])

                elif self.method == 'rbf-cubic':
                    # compute using 3-D Rbfs
                    rbfi = Rbf(
                        newCoord[0],
                        newCoord[1],
                        newCoord[2],
                        pHelp[cntTime, : len(newCoord[0])],
                        function='cubic',
                    )  # radial basis function interpolator instance

                    virtshiftcoord = np.array([xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2]])
                    pInterp[cntTime] = rbfi(virtshiftcoord[0], virtshiftcoord[1], virtshiftcoord[2])

                elif self.method == 'rbf-multiquadric':
                    # compute using 3-D Rbfs
                    rbfi = Rbf(
                        newCoord[0],
                        newCoord[1],
                        newCoord[2],
                        pHelp[cntTime, : len(newCoord[0])],
                        function='multiquadric',
                    )  # radial basis function interpolator instance

                    virtshiftcoord = np.array([xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2]])
                    pInterp[cntTime] = rbfi(virtshiftcoord[0], virtshiftcoord[1], virtshiftcoord[2])
                # using inverse distance weighting
                elif self.method == 'IDW':
                    newPoint2_M = newPoint.T
                    newPoint3_M = np.append(newPoint2_M, np.zeros([1, self.num_channels]), axis=0)
                    newPointCart = cylToCart(newPoint3_M)
                    for ind in np.arange(len(newPoint[:, 0])):
                        newPoint_Rep = np.tile(newPointCart[:, ind], (len(newPoint[:, 0]), 1)).T
                        subtract = newPoint_Rep - newCoordCart
                        normDistance = spla.norm(subtract, axis=0)
                        index_norm = np.argsort(normDistance)[: self.num_IDW]
                        pHelpNew = pHelp[cntTime, index_norm]
                        normNew = normDistance[index_norm]
                        if normNew[0] < 1e-3:
                            pInterp[cntTime, ind] = pHelpNew[0]
                        else:
                            wholeD = np.sum(1 / normNew**self.p_weight)
                            weight = (1 / normNew**self.p_weight) / wholeD
                            pInterp[cntTime, ind] = np.sum(pHelpNew * weight)

        # Interpolation for arbitrary 3D Arrays
        elif self.array_dimension == '3D':
            # check rotation
            if not np.array_equal(phi_delay, []):
                xInterpHelp = np.tile(virtNewCoord[0, :], (nTime, 1)) + np.tile(phi_delay, (virtNewCoord.shape[1], 1)).T
                xInterp = ((xInterpHelp + np.pi) % (2 * np.pi)) - np.pi  # shifting phi into feasible area [-pi, pi]
            else:
                xInterp = np.tile(virtNewCoord[0, :], (nTime, 1))

            mesh = meshList[0][0]
            for cntTime in range(nTime):
                # points for interpolation
                newPoint = np.concatenate((xInterp[cntTime, :][:, np.newaxis], virtNewCoord[1:, :].T), axis=1)

                if self.method == 'linear':
                    interpolater = LinearNDInterpolator(mesh, pHelp[cntTime, :], fill_value=0)
                    pInterp[cntTime] = interpolater(newPoint)

                elif self.method == 'sinc':
                    # compute using 3-D Rbfs for sinc
                    rbfi = Rbf(
                        newCoord[0],
                        newCoord[1],
                        newCoord[2],
                        pHelp[cntTime, : len(newCoord[0])],
                        function=self.sinc_mic,
                    )  # radial basis function interpolator instance

                    pInterp[cntTime] = rbfi(xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2])

                elif self.method == 'rbf-cubic':
                    # compute using 3-D Rbfs
                    rbfi = Rbf(
                        newCoord[0],
                        newCoord[1],
                        newCoord[2],
                        pHelp[cntTime, : len(newCoord[0])],
                        function='cubic',
                    )  # radial basis function interpolator instance

                    pInterp[cntTime] = rbfi(xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2])

                elif self.method == 'rbf-multiquadric':
                    # compute using 3-D Rbfs
                    rbfi = Rbf(
                        newCoord[0],
                        newCoord[1],
                        newCoord[2],
                        pHelp[cntTime, : len(newCoord[0])],
                        function='multiquadric',
                    )  # radial basis function interpolator instance

                    pInterp[cntTime] = rbfi(xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2])

        # return interpolated pressure values
        return pInterp

    def result(self, num):
        """
        Generate interpolated microphone data over time.

        This method retrieves pressure samples from the physical microphones and applies spatial
        interpolation to estimate the pressure at virtual microphone locations.
        The interpolation method is determined by :attr:`method`.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array of shape (``num``, `n`), where `n` is the number of virtual microphones,
            containing interpolated pressure values for the virtual microphones at each time step.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.
        """
        msg = 'result method not implemented yet! Data from source will be passed without transformation.'
        warn(msg, Warning, stacklevel=2)
        yield from self.source.result(num)


class SpatialInterpolatorRotation(SpatialInterpolator):  # pragma: no cover
    """
    Spatial interpolation class for rotating sound sources.

    This class extends :attr:`SpatialInterpolator` to handle sources that undergo rotational
    movement. It retrieves samples from the :attr:`source` attribute and angle data from the
    :attr:`AngleTracker` instance (:attr:`angle_source`). Using these inputs, it computes
    interpolated outputs through the :meth:`result` generator method.

    See Also
    --------
    :class:`SpatialInterpolator`: Base class for spatial interpolation of microphone data.
    """

    #: Provides real-time tracking of the source's rotation angles,
    #: instance of :attr:`AngleTracker`.
    angle_source = Instance(AngleTracker)

    #: Unique identifier for the current configuration of the interpolator. (read-only)
    digest = Property(
        depends_on=[
            'source.digest',
            'angle_source.digest',
            'mics.digest',
            'mics_virtual.digest',
            'method',
            'array_dimension',
            'Q',
            'interp_at_zero',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=128):
        """
        Generate interpolated output samples in block-wise fashion.

        This method acts as a generator, yielding time-domain time signal samples that have been
        spatially interpolated based on rotational movement.

        Parameters
        ----------
        num : :obj:`int`, optional
            Number of samples per block. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            Interpolated time signal samples in blocks of shape
            (``num``, :attr:`~SpatialInterpolator.num_channels`), where
            :attr:`~SpatialInterpolator.num_channels` is inherited from the
            :class:`SpatialInterpolator` base class.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.
        """
        # period for rotation
        period = 2 * np.pi
        # get angle
        angle = self.angle_source.angle()
        # counter to track angle position in time for each block
        count = 0
        for timeData in self.source.result(num):
            phi_delay = angle[count : count + num]
            interpVal = self._result_core_func(timeData, phi_delay, period, self.Q, interp_at_zero=False)
            yield interpVal
            count += num


class SpatialInterpolatorConstantRotation(SpatialInterpolator):  # pragma: no cover
    """
    Performs spatial linear interpolation for sources undergoing constant rotation.

    This class interpolates signals from a rotating sound source based on a constant rotational
    speed. It retrieves samples from the :attr:`source` and applies interpolation before
    generating output through the :meth:`result` generator.

    See Also
    --------
    :class:`SpatialInterpolator` : Base class for spatial interpolation of microphone data.
    :class:`SpatialInterpolatorRotation` : Spatial interpolation class for rotating sound sources.
    """

    #: Rotational speed of the source in revolutions per second (rps). A positive value indicates
    #: counterclockwise rotation around the positive z-axis, meaning motion from the x-axis toward
    #: the y-axis.
    rotational_speed = Float(0.0)

    #: Unique identifier for the current configuration of the interpolator. (read-only)
    digest = Property(
        depends_on=[
            'source.digest',
            'mics.digest',
            'mics_virtual.digest',
            'method',
            'array_dimension',
            'Q',
            'interp_at_zero',
            'rotational_speed',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=1):
        """
        Generate interpolated time signal data in blocks of size ``num``.

        This generator method continuously processes incoming time signal data while applying
        rotational interpolation. The phase delay is computed based on the rotational speed and
        applied to the signal.

        Parameters
        ----------
        num : :obj:`int`, optional
            Number of samples per block.
            Default is ``1``.

        Yields
        ------
        :class:`numpy.ndarray`
            An array containing the interpolated time signal samples in blocks of shape
            (``num``, :attr:`~SpatialInterpolator.num_channels`), where
            :attr:`~SpatialInterpolator.num_channels` is inherited from the
            :class:`SpatialInterpolator` base class.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.
        """
        omega = 2 * np.pi * self.rotational_speed
        period = 2 * np.pi
        phiOffset = 0.0
        for timeData in self.source.result(num):
            nTime = timeData.shape[0]
            phi_delay = phiOffset + np.linspace(0, nTime / self.sample_freq * omega, nTime, endpoint=False)
            interpVal = self._result_core_func(timeData, phi_delay, period, self.Q, interp_at_zero=False)
            phiOffset = phi_delay[-1] + omega / self.sample_freq
            yield interpVal


class Mixer(TimeOut):
    """
    Mix signals from multiple sources into a single output.

    This class takes a :attr:`primary time signal source<source>` and a list of
    :attr:`additional sources<sources>` with the same sampling rates and channel counts across all
    :attr:`primary time signal source<source>`, and outputs a mixed signal.
    The mixing process is performed block-wise using a generator.

    If one of the :attr:`additional sources<sources>` holds a shorter signal than the other
    sources the :meth:`result` method will stop yielding mixed time signal at that point.
    """

    #: The primary time signal source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: A list of additional time signal sources to be mixed with the primary source, each must be an
    #: instance of :class:`~acoular.base.SamplesGenerator`.
    sources = List(Instance(SamplesGenerator, ()))

    #: The sampling frequency of the primary time signal, delegated from :attr:`source`.
    sample_freq = Delegate('source')

    #: The number of channels in the output, delegated from :attr:`source`.
    num_channels = Delegate('source')

    #: The number of samples in the output, delegated from :attr:`source`.
    num_samples = Delegate('source')

    #: Internal identifier that tracks changes in the :attr:`sources` list.
    sdigest = Str()

    @observe('sources.items.digest')
    def _set_sourcesdigest(self, event):  # noqa ARG002
        self.sdigest = ldigest(self.sources)

    #: A unique identifier for the Mixer instance, based on the :attr:`primary source<source>` and
    #: the :attr:`list of additional sources<sources>`.
    digest = Property(depends_on=['source.digest', 'sdigest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def validate_sources(self):
        """
        Validate whether the additional sources are compatible with the primary source.

        This method checks if all sources have the same sampling frequency and the same number of
        channels. If a mismatch is detected, a :obj:`ValueError` is raised.

        Raises
        ------
        :obj:`ValueError`
            If any source in :attr:`sources` has a different sampling frequency or
            number of channels than :attr:`source`.
        """
        if self.source:
            for s in self.sources:
                if self.sample_freq != s.sample_freq:
                    msg = f'Sample frequency of {s} does not fit'
                    raise ValueError(msg)
                if self.num_channels != s.num_channels:
                    msg = f'Channel count of {s} does not fit'
                    raise ValueError(msg)

    def result(self, num):
        """
        Generate mixed time signal data in blocks of ``num`` samples.

        This generator method retrieves time signal data from all sources and sums them together
        to produce a combined output. The data from each source is processed in blocks of the
        same size, ensuring synchronized mixing.

        .. note::

            Yielding stops when one of the additionally provied signals ends; i.e. if one of the
            additional sources holds a signal of shorter length than that of the
            :attr:`primary source<source>` that (shorter) signal forms the lower bound of the length
            of the mixed time signal yielded.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array containing the mixed time samples in blocks of shape
            (``num``, :attr:`~acoular.base.TimeOut.num_channels`), where
            :attr:`~acoular.base.TimeOut.num_channels` is inhereted from the
            :class:`~acoular.base.TimeOut` base class.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.
        """
        # check whether all sources fit together
        self.validate_sources()

        gens = [i.result(num) for i in self.sources]
        for temp in self.source.result(num):
            sh = temp.shape[0]
            for g in gens:
                try:
                    temp1 = next(g)
                except StopIteration:
                    return
                if temp.shape[0] > temp1.shape[0]:
                    temp = temp[: temp1.shape[0]]
                temp += temp1[: temp.shape[0]]
            yield temp
            if sh > temp.shape[0]:
                break


class TimePower(TimeOut):
    """
    Calculate the time-dependent power of a signal by squaring its samples.

    This class computes the power of the input signal by squaring the value of each sample. It
    processes the signal in blocks, making it suitable for large datasets or real-time signal
    processing. The power is calculated on a per-block basis, and each block of the output is
    yielded as a NumPy array.

    Attributes
    ----------
    source : SamplesGenerator
        The input data source, which provides the time signal or signal samples
        to be processed. It must be an instance of :class:`~acoular.base.SamplesGenerator`
        or any derived class that provides a `result()` method.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    def result(self, num):
        """
        Generate the time-dependent power of the input signal in blocks.

        This method iterates through the signal samples provided by the :attr:`source` and
        calculates the power by squaring each sample. The output is yielded block-wise to
        facilitate processing large signals in chunks.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array containing the squared samples from the :attr:`source`. Each block will have
            the shape (``num``, :attr:`~acoular.base.TimeOut.num_channels`), where
            :attr:`~acoular.base.TimeOut.num_channels` is inhereted from the
            :class:`~acoular.base.TimeOut` base class.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.
        """
        for temp in self.source.result(num):
            yield temp * temp


class TimeCumAverage(TimeOut):
    """
    Calculates the cumulative average of the signal.

    This class computes the cumulative average of the input signal over time, which is useful for
    metrics like the Equivalent Continuous Sound Level (Leq). It processes the signal in blocks,
    maintaining a running average of the samples. The result is yielded in blocks, allowing for
    memory-efficient processing of large datasets.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    def result(self, num):
        """
        Generate the cumulative average of the input signal in blocks.

        This method iterates through the signal samples provided by the :attr:`source`, and for each
        block, it computes the cumulative average of the samples up to that point. The result is
        yielded in blocks, with each block containing the cumulative average of the signal up to
        that sample.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array containing the cumulative average of the samples. Each block will have the
            shape (``num``, :attr:`~acoular.base.TimeOut.num_channels`), where
            :attr:`~acoular.base.TimeOut.num_channels` is inhereted from the :attr:`source`.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.

        Notes
        -----
        The cumulative average is updated iteratively by considering the previously accumulated sum
        and the current block of samples. For each new sample, the cumulative average is
        recalculated by summing the previous cumulative value and the new samples, then dividing by
        the total number of samples up to that point.
        """
        count = (np.arange(num) + 1)[:, np.newaxis]
        for i, temp in enumerate(self.source.result(num)):
            ns, nc = temp.shape
            if not i:
                accu = np.zeros((1, nc))
            temp = (accu * (count[0] - 1) + np.cumsum(temp, axis=0)) / count[:ns]
            accu = temp[-1]
            count += ns
            yield temp


class TimeReverse(TimeOut):
    """
    Calculates the time-reversed signal of a source.

    This class takes the input signal from a source and computes the time-reversed version of the
    signal. It processes the signal in blocks, yielding the time-reversed signal block by block.
    This can be useful for various signal processing tasks, such as creating echoes or reversing
    the playback of time signal signals.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    def result(self, num):
        """
        Generate the time-reversed version of the input signal block-wise.

        This method processes the signal provided by the :attr:`source` in blocks, and for each
        block, it produces the time-reversed version of the signal. The result is yielded in blocks,
        with each block containing the time-reversed version of the signal for that segment.
        The signal is reversed in time by flipping the order of samples within each block.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array containing the time-reversed version of the signal for the current block.
            Each block will have the shape (``num``, :attr:`acoular.base.TimeOut.num_channels`),
            where :attr:`~acoular.base.TimeOut.num_channels` is inherited from the :attr:`source`.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.

        Notes
        -----
        The time-reversal is achieved by reversing the order of samples in each block of the signal.
        The :meth:`result` method first collects all the blocks from the source, then processes them
        in reverse order, yielding the time-reversed signal in blocks. The first block yielded
        corresponds to the last block of the source signal, and so on, until the entire signal has
        been processed in reverse.
        """
        result_list = []
        result_list.extend(self.source.result(num))
        temp = np.empty_like(result_list[0])
        h = result_list.pop()
        nsh = h.shape[0]
        temp[:nsh] = h[::-1]
        for h in result_list[::-1]:
            temp[nsh:] = h[: nsh - 1 : -1]
            yield temp
            temp[:nsh] = h[nsh - 1 :: -1]
        yield temp[:nsh]


class Filter(TimeOut):
    """
    Abstract base class for IIR filters using SciPy's :func:`~scipy.signal.lfilter`.

    This class implements a digital Infinite Impulse Response (IIR) filter that applies filtering to
    a given signal in a block-wise manner. The filter coefficients can be dynamically changed during
    processing.

    See Also
    --------
    :func:`scipy.signal.lfilter` :
        Filter data along one-dimension with an IIR or FIR (finite impulse response) filter.
    :func:`scipy.signal.sosfilt` :
        Filter data along one dimension using cascaded second-order sections.
    :class:`FiltOctave` :
        Octave or third-octave bandpass filter (causal, with non-zero phase delay).
    :class:`FiltFiltOctave` : Octave or third-octave bandpass filter with zero-phase distortion.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: Second-order sections representation of the filter coefficients.
    #: This property is dynamically updated and can change during signal processing.
    sos = Property()

    def _get_sos(self):
        return tf2sos([1], [1])

    def result(self, num):
        """
        Apply the IIR filter to the input signal and yields filtered data block-wise.

        This method processes the signal provided by :attr:`source`, applying the defined filter
        coefficients (:attr:`sos`) using the :func:`scipy.signal.sosfilt` function. The filtering
        is performed in a streaming fashion, yielding blocks of filtered signal data.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array containing the bandpass-filtered signal for the current block. Each block has
            the shape (``num``, :attr:`~acoular.base.TimeOut.num_channels`), where
            :attr:`~acoular.base.TimeOut.num_channels` is inherited from the :attr:`source`.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.
        """
        sos = self.sos
        zi = np.zeros((sos.shape[0], 2, self.source.num_channels))
        for block in self.source.result(num):
            sos = self.sos  # this line is useful in case of changes
            # to self.sos during generator lifetime
            block, zi = sosfilt(sos, block, axis=0, zi=zi)
            yield block


class FiltOctave(Filter):
    """
    Octave or third-octave bandpass filter (causal, with non-zero phase delay).

    This class implements a bandpass filter that conforms to octave or third-octave frequency band
    standards. The filter is designed using a second-order section (SOS) Infinite Impulse Response
    (IIR) approach.

    The filtering process introduces a non-zero phase delay due to its causal nature. The center
    frequency and the octave fraction determine the frequency band characteristics.

    See Also
    --------
    :class:`Filter` : The base class implementing a general IIR filter.
    :class:`FiltFiltOctave` : Octave or third-octave bandpass filter with zero-phase distortion.
    """

    #: The center frequency of the octave or third-octave band. Default is ``1000``.
    band = Float(1000.0, desc='band center frequency')

    #: Defines whether the filter is an octave-band or third-octave-band filter.
    #:
    #: - ``'Octave'``: Full octave band filter.
    #: - ``'Third octave'``: Third-octave band filter.
    #:
    #: Default is ``'Octave'``.
    fraction = Map({'Octave': 1, 'Third octave': 3}, default_value='Octave', desc='fraction of octave')

    #: The order of the IIR filter, which affects the steepness of the filter's roll-off.
    #: Default is ``3``.
    order = Int(3, desc='IIR filter order')

    #: Second-order sections representation of the filter coefficients. This property depends on
    #: :attr:`band`, :attr:`fraction`, :attr:`order`, and the source's digest.
    sos = Property(depends_on=['band', 'fraction', 'source.digest', 'order'])

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'band', 'fraction', 'order'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        # Compute the second-order section coefficients for the bandpass filter.

        # The filter design follows ANSI S1.11-1987 standards and adjusts
        # filter edge frequencies to maintain correct power bandwidth.

        # The filter is implemented using a Butterworth design, with
        # appropriate frequency scaling to match the desired octave band.

        # Returns
        # -------
        # :class:`numpy.ndarray`
        #     SOS (second-order section) coefficients for the filter.

        # Raises
        # ------
        # :obj:`ValueError`
        #     If the center frequency (:attr:`band`) is too high relative to
        #     the sampling frequency.

        # filter design
        fs = self.sample_freq
        # adjust filter edge frequencies for correct power bandwidth (see ANSI 1.11 1987
        # and Kalb,J.T.: "A thirty channel real time audio analyzer and its applications",
        # PhD Thesis: Georgia Inst. of Techn., 1975
        beta = np.pi / (2 * self.order)
        alpha = pow(2.0, 1.0 / (2.0 * self.fraction_))
        beta = 2 * beta / np.sin(beta) / (alpha - 1 / alpha)
        alpha = (1 + np.sqrt(1 + beta * beta)) / beta
        fr = 2 * self.band / fs
        if fr > 1 / np.sqrt(2):
            msg = f'band frequency too high:{self.band:f},{fs:f}'
            raise ValueError(msg)
        om1 = fr / alpha
        om2 = fr * alpha
        return butter(self.order, [om1, om2], 'bandpass', output='sos')


class FiltFiltOctave(FiltOctave):
    """
    Octave or third-octave bandpass filter with zero-phase distortion.

    This filter applies an IIR bandpass filter in both forward and reverse directions, effectively
    eliminating phase distortion. It provides zero-phase filtering but requires significantly more
    memory compared to causal filtering.

    See Also
    --------
    :class:`Filter` : The base class implementing a general IIR filter.
    :class:`FiltOctave` : The standard octave or third-octave filter with causal filtering.

    Notes
    -----
    - Due to the double-pass filtering, additional bandwidth correction is applied to maintain
      accurate frequency response.
    - This approach requires storing the entire signal in memory before processing, making it
      unsuitable for real-time applications with large datasets.
    """

    #: The half-order of the IIR filter, applied twice (once forward and once backward). This
    #: results in a final filter order twice as large as the specified value. Default is ``2``.
    order = Int(2, desc='IIR filter half order')

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'band', 'fraction', 'order'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        # Compute the second-order section (SOS) coefficients for the filter.
        #
        # The filter design follows ANSI S1.11-1987 standards and incorporates additional bandwidth
        # correction to compensate for the double-pass filtering effect.
        #
        # Returns
        # -------
        # :class:`numpy.ndarray`
        #     SOS (second-order section) coefficients for the filter.
        #
        # Raises
        # ------
        # :obj:`ValueError`
        #     If the center frequency (:attr:`band`) is too high relative to the
        #     sampling frequency.

        # filter design
        fs = self.sample_freq
        # adjust filter edge frequencies for correct power bandwidth (see FiltOctave)
        beta = np.pi / (2 * self.order)
        alpha = pow(2.0, 1.0 / (2.0 * self.fraction_))
        beta = 2 * beta / np.sin(beta) / (alpha - 1 / alpha)
        alpha = (1 + np.sqrt(1 + beta * beta)) / beta
        # additional bandwidth correction for double-pass
        alpha = alpha * {6: 1.01, 5: 1.012, 4: 1.016, 3: 1.022, 2: 1.036, 1: 1.083}.get(self.order, 1.0) ** (
            3 / self.fraction_
        )
        fr = 2 * self.band / fs
        if fr > 1 / np.sqrt(2):
            msg = f'band frequency too high:{self.band:f},{fs:f}'
            raise ValueError(msg)
        om1 = fr / alpha
        om2 = fr * alpha
        return butter(self.order, [om1, om2], 'bandpass', output='sos')

    def result(self, num):
        """
        Apply the filter to the input signal and yields filtered data block-wise.

        The input signal is first stored in memory, then filtered in both forward and reverse
        directions to achieve zero-phase distortion. The processed signal is yielded in blocks.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            An array containing the filtered signal for the current block. Each block has shape
            (``num``, :attr:`~acoular.base.TimeOut.num_channels`), where
            :attr:`~acoular.base.TimeOut.num_channels` is inherited from the :attr:`source`.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.

        Notes
        -----
        - This method requires the entire signal to be stored in memory, making it unsuitable for
          streaming or real-time applications.
        - Filtering is performed separately for each channel to optimize memory usage.
        """
        sos = self.sos
        data = np.empty((self.source.num_samples, self.source.num_channels))
        j = 0
        for block in self.source.result(num):
            ns, nc = block.shape
            data[j : j + ns] = block
            j += ns
        # filter one channel at a time to save memory
        for j in range(self.source.num_channels):
            data[:, j] = sosfiltfilt(sos, data[:, j])
        j = 0
        ns = data.shape[0]
        while j < ns:
            yield data[j : j + num]
            j += num


class TimeExpAverage(Filter):
    """
    Compute an exponentially weighted moving average of the input signal.

    This filter implements exponential averaging as defined in IEC 61672-1, which is commonly used
    for sound level measurements. The time weighting determines how quickly past values decay in
    significance.

    See Also
    --------
    :class:`Filter` : Base class for implementing IIR filters.

    Notes
    -----
    The `Impulse` (``'I'``) weighting is not part of IEC 61672-1 but is included for additional
    flexibility.
    """

    #: Time weighting constant, determining the exponential decay rate.
    #:
    #: - ``'F'`` (Fast) → 0.125
    #: - ``'S'`` (Slow) → 1.0
    #: - ``'I'`` (Impulse) → 0.035 (non-standard)
    #:
    #: Default is ``'F'``.
    weight = Map({'F': 0.125, 'S': 1.0, 'I': 0.035}, default_value='F', desc='time weighting')

    #: Filter coefficients in second-order section (SOS) format.
    sos = Property(depends_on=['weight', 'source.digest'])

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'weight'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        # Compute the second-order section (SOS) coefficients for the exponential filter.
        #
        # The filter follows the form of a first-order IIR filter:
        #
        # .. math::
        #     y[n] = \\alpha x[n] + (1 - \\alpha) y[n-1]
        #
        # where :math:`\\alpha` is determined by the selected time weighting.
        #
        # Returns
        # -------
        # :class:`numpy.ndarray`
        #     SOS (second-order section) coefficients representing the filter.
        #
        # Notes
        # -----
        # The coefficient :math:`\\alpha` is calculated as:
        #
        # .. math::
        #     \\alpha = 1 - e^{-1 / (\\tau f_s)}
        #
        # where:
        #
        # - :math:`\\tau` is the selected time constant (:attr:`weight`).
        # - :math:`f_s` is the sampling frequency of the source.
        #
        # This implementation ensures that the filter adapts dynamically
        # based on the source's sampling frequency.
        alpha = 1 - np.exp(-1 / self.weight_ / self.sample_freq)
        a = [1, alpha - 1]
        b = [alpha]
        return tf2sos(b, a)


class FiltFreqWeight(Filter):
    """
    Apply frequency weighting according to IEC 61672-1.

    This filter implements frequency weighting curves commonly used in sound level meters for noise
    measurement. It provides A-weighting, C-weighting, and Z-weighting options.

    See Also
    --------
    :class:`Filter` : Base class for implementing IIR filters.

    Notes
    -----
    - The filter is designed following IEC 61672-1:2002, the standard for sound level meters.
    - The weighting curves are implemented using bilinear transformation of analog filter
      coefficients to the discrete domain.
    """

    #: Defines the frequency weighting curve:
    #:
    #: - ``'A'``: Mimics human hearing sensitivity at low sound levels.
    #: - ``'C'``: Used for high-level sound measurements with less attenuation at low frequencies.
    #: - ``'Z'``: A flat response with no frequency weighting.
    #:
    #: Default is ``'A'``.
    weight = Enum('A', 'C', 'Z', desc='frequency weighting')

    #: Second-order sections (SOS) representation of the filter coefficients. This property is
    #: dynamically computed based on :attr:`weight` and the :attr:`Filter.source`'s digest.
    sos = Property(depends_on=['weight', 'source.digest'])

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'weight'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        # Compute the second-order section (SOS) coefficients for the frequency weighting filter.
        #
        # The filter design is based on analog weighting functions defined in IEC 61672-1,
        # transformed into the discrete-time domain using the bilinear transformation.
        #
        # Returns
        # -------
        # :class:`numpy.ndarray`
        #     SOS (second-order section) coefficients representing the filter.
        #
        # Notes
        # -----
        # The analog weighting functions are defined as:
        #
        # - **A-weighting**:
        #
        #   .. math::
        #       H(s) = \\frac{(2 \\pi f_4)^2 (s + 2 \\pi f_3) (s + 2 \\pi f_2)}
        #       {(s + 2 \\pi f_4) (s + 2 \\pi f_1) (s^2 + 4 \\pi f_1 s + (2 \\pi f_1)^2)}
        #
        #   where the parameters are:
        #
        #   - :math:`f_1 = 20.598997` Hz
        #   - :math:`f_2 = 107.65265` Hz
        #   - :math:`f_3 = 737.86223` Hz
        #   - :math:`f_4 = 12194.217` Hz
        #
        # - **C-weighting** follows a similar approach but without the low-frequency roll-off.
        #
        # - **Z-weighting** is implemented as a flat response (no filtering).
        #
        # The bilinear transformation is used to convert these analog functions into
        # the digital domain, preserving the frequency response characteristics.
        #
        # Raises
        # ------
        # :obj:`ValueError`
        #     If an invalid weight type is provided.

        # s domain coefficients
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        a = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
        if self.weight == 'A':
            a = np.polymul(np.polymul(a, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
            b = [(2 * np.pi * f4) ** 2 * 10 ** (1.9997 / 20), 0, 0, 0, 0]
            b, a = bilinear(b, a, self.sample_freq)
        elif self.weight == 'C':
            b = [(2 * np.pi * f4) ** 2 * 10 ** (0.0619 / 20), 0, 0]
            b, a = bilinear(b, a, self.sample_freq)
            b = np.append(b, np.zeros(2))  # make 6th order
            a = np.append(a, np.zeros(2))
        else:
            b = np.zeros(7)
            b[0] = 1.0
            a = b  # 6th order flat response
        return tf2sos(b, a)


@deprecated_alias({'numbands': 'num_bands'}, read_only=True, removal_version='25.10')
class FilterBank(TimeOut):
    """
    Abstract base class for IIR filter banks based on :mod:`scipy.signal.lfilter`.

    Implements a bank of parallel filters. This class should not be instantiated by itself.

    Inherits from :class:`acoular.base.TimeOut`, and defines the structure for working with filter
    banks for processing multi-channel time series data, such as time signal signals.

    See Also
    --------
    :class:`acoular.base.TimeOut` :
        ABC for signal processing blocks that interact with data from a source.
    :class:`acoular.base.SamplesGenerator` :
        Interface for any generating multi-channel time domain signal processing block.
    :mod:`scipy.signal` :
        SciPy module for signal processing.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: The list containing second order section (SOS) coefficients for the filters in the filter
    #: bank.
    sos = Property()

    #: A list of labels describing the different frequency bands of the filter bank.
    bands = Property()

    #: The total number of bands in the filter bank.
    num_bands = Property()

    #: The total number of output channels resulting from the filter bank operation.
    num_channels = Property()

    @abstractmethod
    def _get_sos(self):
        """Return a list of second order section coefficients."""

    @abstractmethod
    def _get_bands(self):
        """Return a list of labels for the bands."""

    @abstractmethod
    def _get_num_bands(self):
        """Return the number of bands."""

    def _get_num_channels(self):
        return self.num_bands * self.source.num_channels

    def result(self, num):
        """
        Yield the bandpass filtered output of the source in blocks of samples.

        This method uses the second order section coefficients (:attr:`sos`) to filter the input
        samples provided by the source in blocks. The result is returned as a generator.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :obj:`numpy.ndarray`
            An array of shape (``num``, :attr:`num_channels`), delivering the filtered
            samples for each band.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.

        Notes
        -----
        The returned samples are bandpass filtered according to the coefficients in
        :attr:`sos`. Each block corresponds to the filtered samples for each frequency band.
        """
        numbands = self.num_bands
        snumch = self.source.num_channels
        sos = self.sos
        zi = [np.zeros((sos[0].shape[0], 2, snumch)) for _ in range(numbands)]
        res = np.zeros((num, self.num_channels), dtype='float')
        for block in self.source.result(num):
            for i in range(numbands):
                res[:, i * snumch : (i + 1) * snumch], zi[i] = sosfilt(sos[i], block, axis=0, zi=zi[i])
            yield res


class OctaveFilterBank(FilterBank):
    """
    Octave or third-octave filter bank.

    Inherits from :class:`FilterBank` and implements an octave or third-octave filter bank.
    This class is used for filtering multi-channel time series data, such as time signal signals,
    using bandpass filters with center frequencies at octave or third-octave intervals.

    See Also
    --------
    :class:`FilterBank` :
        The base class for implementing IIR filter banks.
    :class:`acoular.base.SamplesGenerator` :
        Interface for generating multi-channel time domain signal processing blocks.
    :mod:`scipy.signal` :
        SciPy module for signal processing.
    """

    #: The lowest band center frequency index. Default is ``21``.
    #: This index refers to the position in the scale of octave or third-octave bands.
    lband = Int(21, desc='lowest band center frequency index')

    #: The highest band center frequency index + 1. Default is ``40``.
    #: This is the position in the scale of octave or third-octave bands.
    hband = Int(40, desc='highest band center frequency index + 1')

    #: The fraction of an octave, either ``'Octave'`` or ``'Third octave'``.
    #: Default is ``'Octave'``.
    #: Determines the width of the frequency bands. 'Octave' refers to full octaves,
    #: and ``'Third octave'`` refers to third-octave bands.
    fraction = Map({'Octave': 1, 'Third octave': 3}, default_value='Octave', desc='fraction of octave')

    #: The list of filter coefficients for all filters in the filter bank.
    #: The coefficients are computed based on the :attr:`lband`, :attr:`hband`,
    #: and :attr:`fraction` attributes.
    ba = Property(depends_on=['lband', 'hband', 'fraction', 'source.digest'])

    #: The list of labels describing the frequency bands in the filter bank.
    bands = Property(depends_on=['lband', 'hband', 'fraction'])

    #: The total number of bands in the filter bank.
    num_bands = Property(depends_on=['lband', 'hband', 'fraction'])

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'lband', 'hband', 'fraction', 'order'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_bands(self):
        return [10 ** (i / 10) for i in range(self.lband, self.hband, 4 - self.fraction_)]

    @cached_property
    def _get_num_bands(self):
        return len(self.bands)

    @cached_property
    def _get_sos(self):
        # Generate and return the second-order section (SOS) coefficients for each filter.
        #
        # For each frequency band in the filter bank, the SOS coefficients are calculated using
        # the :class:`FiltOctave` object with the appropriate `fraction` setting. The coefficients
        # are then returned as a list.
        #
        # Returns
        # -------
        # :obj:`list` of :obj:`numpy.ndarray`
        #     A list of SOS coefficients for each filter in the filter bank.
        of = FiltOctave(source=self.source, fraction=self.fraction)
        sos = []
        for i in range(self.lband, self.hband, 4 - self.fraction_):
            of.band = 10 ** (i / 10)
            sos_ = of.sos
            sos.append(sos_)
        return sos


@deprecated_alias({'name': 'file'}, removal_version='25.10')
class WriteWAV(TimeOut):
    """
    Saves time signal from one or more channels as mono, stereo, or multi-channel ``.wav`` file.

    Inherits from :class:`~acoular.base.TimeOut` and allows for exporting time-series data from one
    or more channels to a WAV file. Supports saving mono, stereo, or multi-channel signals to disk
    with automatic or user-defined file naming.

    See Also
    --------
    :class:`acoular.base.TimeOut` :
        ABC for signal processing blocks that interact with data from a source.
    :class:`acoular.base.SamplesGenerator` :
        Interface for generating multi-channel time domain signal processing blocks.
    :mod:`wave` :
        Python module for handling WAV files.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: The name of the file to be saved. If none is given, the name will be automatically
    #: generated from the source.
    file = File(filter=['*.wav'], desc='name of wave file')

    #: The name of the cache file (without extension). It serves as an internal reference for data
    #: caching and tracking processed files. (automatically generated)
    basename = Property(depends_on=['digest'])

    #: The list of channels to save. Can only contain one or two channels.
    channels = List(int, desc='channels to save')

    # Bit depth of the output file.
    encoding = Enum('uint8', 'int16', 'int32', desc='bit depth of the output file')

    # Maximum value to scale the output to. If `None`, the maximum value of the data is used.
    max_val = Either(None, Float, desc='Maximum value to scale the output to.')

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'channels'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        warn(
            (
                f'The basename attribute of a {self.__class__.__name__} object is deprecated'
                ' and will be removed in Acoular 26.01!'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return find_basename(self.source)

    def _type_info(self):
        dtype = np.dtype(self.encoding)
        info = np.iinfo(dtype)
        return dtype, info.min, info.max, int(info.bits / 8)

    def _encode(self, data):
        """Encodes the data according to self.encoding."""
        dtype, dmin, dmax, _ = self._type_info()
        if dtype == np.dtype('uint8'):
            data = (data + 1) / 2 * dmax
        else:
            data *= -dmin
        data = np.round(data)
        if data.min() < dmin or data.max() > dmax:
            warn(
                f'Clipping occurred in WAV export. Data type {dtype} cannot represent all values in data. \
            Consider raising max_val.',
                stacklevel=1,
            )
        return data.clip(dmin, dmax).astype(dtype).tobytes()

    def result(self, num):
        """
        Generate and save time signal data as a WAV file in blocks.

        This generator method retrieves time signal data from the :attr:`source` and writes it to a
        WAV file in blocks of size ``num``. The data is scaled and encoded according to the selected
        bit depth and channel configuration. If no file name is specified, a name is generated
        automatically. The method yields each block of data after it is written to the file,
        allowing for streaming or real-time processing.

        Parameters
        ----------
        num : :class:`int`
            Number of samples per block to write and yield.

        Yields
        ------
        :class:`numpy.ndarray`
            The block of time signal data that was written to the WAV file, with shape
            (``num``, number of channels).

        Raises
        ------
        :class:`ValueError`
            If no channels are specified for output.
        :class:`Warning`
            If more than two channels are specified, or if the sample frequency is not an integer.
            Also warns if clipping occurs due to data range limitations.

        See Also
        --------
        :meth:`save` : Save the entire source output to a WAV file in one call.
        """
        nc = len(self.channels)
        if nc == 0:
            msg = 'No channels given for output.'
            raise ValueError(msg)
        elif nc > 2:
            warn(f'More than two channels given for output, exported file will have {nc:d} channels', stacklevel=1)
        if self.sample_freq.is_integer():
            fs = self.sample_freq
        else:
            fs = int(round(self.sample_freq))
            msg = f'Sample frequency {self.sample_freq} is not a whole number. Proceeding with sampling frequency {fs}.'
            warn(msg, Warning, stacklevel=1)
        dtype, _, dmax, sw = self._type_info()
        if self.file == '':
            name = self.basename
            for nr in self.channels:
                name += f'{nr:d}'
            name += '.wav'
        else:
            name = self.file

        with wave.open(name, 'w') as wf:
            wf.setnchannels(nc)
            wf.setsampwidth(sw)
            wf.setframerate(fs)
            ind = np.array(self.channels)
            if self.max_val is None:
                # compute maximum and remember result to avoid calling source twice
                if not isinstance(self.source, Cache):
                    self.source = Cache(source=self.source)

                # distinguish cases to use full dynamic range of dtype
                if dtype == np.dtype('uint8'):
                    mx = 0
                    for data in self.source.result(num):
                        mx = max(np.abs(data).max(), mx)
                elif dtype in (np.dtype('int16'), np.dtype('int32')):
                    # for signed integers, we need special treatment because of asymmetry
                    negmax, posmax = 0, 0
                    for data in self.source.result(num):
                        negmax, posmax = max(abs(data.min()), negmax), max(data.max(), posmax)
                    mx = negmax if negmax > posmax else posmax + 1 / dmax  # correction for asymmetry
            else:
                mx = self.max_val

            # write scaled data to file
            for data in self.source.result(num):
                frames = self._encode(data[:, ind] / mx)
                wf.writeframes(frames)
                yield data

    def save(self):
        """
        Save the entire source output to a WAV file.

        This method writes all available time signal data from the :attr:`source` to the specified
        WAV file in blocks. It calls the :meth:`result` method internally and discards the yielded
        data. The file is written according to the current :attr:`channels`, :attr:`encoding`, and
        scaling settings. If no file name is specified, a name is generated automatically.

        See Also
        --------
        :meth:`result` : Generator for writing and yielding data block-wise.
        """
        for _ in self.result(1024):
            pass


@deprecated_alias(
    {'name': 'file', 'numsamples_write': 'num_samples_write', 'writeflag': 'write_flag'}, removal_version='25.10'
)
class WriteH5(TimeOut):
    """
    Saves time signal data as a ``.h5`` (HDF5) file.

    Inherits from :class:`~acoular.base.TimeOut` and provides functionality for saving multi-channel
    time-domain signal data to an HDF5 file. The file can be written in blocks and supports
    metadata storage, precision control, and dynamic file generation based on timestamps.

    See Also
    --------
    :class:`acoular.base.TimeOut` :
        ABC for signal processing blocks interacting with data from a source.
    :class:`acoular.base.SamplesGenerator` :
        Interface for generating multi-channel time-domain signal processing blocks.
    h5py :
        Python library for reading and writing HDF5 files.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: The name of the file to be saved. If none is given, the name is automatically
    #: generated based on the current timestamp.
    file = File(filter=['*.h5'], desc='name of data file')

    #: The number of samples to write to file per call to `result` method.
    #: Default is ``-1``, meaning all available data from the source will be written.
    num_samples_write = Int(-1)

    #: A flag that can be set to stop file writing. Default is ``True``.
    write_flag = Bool(True)

    #: A unique identifier for the object, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest'])

    #: Precision of the entries in the HDF5 file, represented as numpy data types.
    #: Default is ``'float32'``.
    precision = Enum('float32', 'float64', desc='precision of H5 File')

    #: Metadata to be stored in the HDF5 file.
    metadata = Dict(desc='metadata to be stored in .h5 file')

    @cached_property
    def _get_digest(self):
        return digest(self)

    def create_filename(self):
        """
        Generate a filename for the HDF5 file if needed.

        Generate a filename for the HDF5 file based on the current timestamp if no filename is
        provided. If a filename is provided, it is used as the file name.
        """
        if self.file == '':
            name = datetime.now(tz=timezone.utc).isoformat('_').replace(':', '-').replace('.', '_')
            self.file = path.join(config.td_dir, name + '.h5')

    def get_initialized_file(self):
        """
        Initialize the HDF5 file and prepare the necessary datasets and metadata.

        This method creates the file (if it doesn't exist), sets up the main data array,
        and appends metadata to the file.

        Returns
        -------
        :class:`h5py.File`
            The initialized HDF5 file object ready for data insertion.
        """
        file = _get_h5file_class()
        self.create_filename()
        f5h = file(self.file, mode='w')
        f5h.create_extendable_array('time_data', (0, self.num_channels), self.precision)
        ac = f5h.get_data_by_reference('time_data')
        f5h.set_node_attribute(ac, 'sample_freq', self.sample_freq)
        self.add_metadata(f5h)
        return f5h

    def save(self):
        """
        Save the source output to a HDF5 file.

        This method writes the processed time-domain signal data from the source to the
        specified HDF5 file. Data is written in blocks and appended to the extendable
        ``'time_data'`` array.

        Notes
        -----
        - If no file is specified, a file name is automatically generated.
        - Metadata defined in the :attr:`metadata` attribute is stored in the file.
        """
        f5h = self.get_initialized_file()
        ac = f5h.get_data_by_reference('time_data')
        for data in self.source.result(4096):
            f5h.append_data(ac, data)
        f5h.close()

    def add_metadata(self, f5h):
        """
        Add metadata to the HDF5 file.

        Metadata is stored in a separate 'metadata' group within the HDF5 file. The metadata
        is stored as arrays with each key-value pair corresponding to a separate array.

        Parameters
        ----------
        f5h : :obj:`h5py.File`
            The HDF5 file object to which metadata will be added.
        """
        nitems = len(self.metadata.items())
        if nitems > 0:
            f5h.create_new_group('metadata', '/')
            for key, value in self.metadata.items():
                if isinstance(value, str):
                    value = np.array(value, dtype='S')
                f5h.create_array('/metadata', key, value)

    def result(self, num):
        """
        Python generator that saves source output to an HDF5 file.

        This method processes data from the source in blocks and writes the data to the HDF5 file.
        It yields the processed blocks while the data is being written.

        Parameters
        ----------
        num : :obj:`int`
            Number of samples per block.

        Yields
        ------
        :obj:`numpy.ndarray`
            A numpy array of shape (``num``, :attr:`~acoular.base.SamplesGenerator.num_channels`),
            where :attr:`~acoular.base.SamplesGenerator.num_channels` is inhereted from the
            :attr:`source`, delivering the processed time-domain signal data.
            The last block may contain fewer samples if the total number of samples is not
            a multiple of ``num``.

        Notes
        -----
        - If :attr:`num_samples_write` is set to a value other than ``-1``, only that number of
          samples will be written to the file.
        - The data is echoed as it is yielded, after being written to the file.
        """
        self.write_flag = True
        f5h = self.get_initialized_file()
        ac = f5h.get_data_by_reference('time_data')
        scount = 0
        stotal = self.num_samples_write
        source_gen = self.source.result(num)
        while self.write_flag:
            sleft = stotal - scount
            if stotal != -1 and sleft > 0:
                anz = min(num, sleft)
            elif stotal == -1:
                anz = num
            else:
                break
            try:
                data = next(source_gen)
            except StopIteration:
                break
            f5h.append_data(ac, data[:anz])
            f5h.flush()
            yield data
            scount += anz
        f5h.close()


class TimeConvolve(TimeOut):
    """
    Perform frequency domain convolution with the uniformly partitioned overlap-save (UPOLS) method.

    This class convolves a source signal with a kernel in the frequency domain. It uses the UPOLS
    method, which efficiently computes convolutions by processing signal blocks and kernel blocks
    separately in the frequency domain. For detailed theoretical background,
    refer to :cite:`Wefers2015`.

    Inherits from :class:`~acoular.base.TimeOut`, which allows the class to process signals
    generated by a source object. The kernel used for convolution can be one-dimensional or
    two-dimensional, and it can be applied across one or more channels of the source signal.

    See Also
    --------
    :class:`acoular.base.TimeOut` :
        The parent class for signal processing blocks.
    :class:`acoular.base.SamplesGenerator` :
        The interface for generating multi-channel time-domain signals.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: Convolution kernel in the time domain.
    #: The second dimension of the kernel array has to be either ``1`` or match
    #: the :attr:`source`'s :attr:`~acoular.base.SamplesGenerator.num_channels` attribute.
    #: If only a single kernel is supplied, it is applied to all channels.
    kernel = CArray(dtype=float, desc='Convolution kernel.')

    # Internal block size for partitioning signals into smaller segments during processing.
    _block_size = Int(desc='Block size')

    # Blocks of the convolution kernel in the frequency domain.
    # Computed using Fast Fourier Transform (FFT).
    _kernel_blocks = Property(
        depends_on=['kernel', '_block_size'],
        desc='Frequency domain Kernel blocks',
    )

    #: A unique identifier for the object, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'kernel'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _validate_kernel(self):
        # Validate the dimensions of the convolution kernel.
        #
        # Reshapes the kernel to match the required dimensions for broadcasting. Checks if the
        # kernel is either one-dimensional or two-dimensional, and ensures that the second dimension
        # matches the number of channels in the source signal.
        #
        # Raises
        # ------
        # ValueError
        #     If the kernel's shape is invalid or incompatible with the source signal.
        if self.kernel.ndim == 1:
            self.kernel = self.kernel.reshape([-1, 1])
            return
        # check dimensionality
        if self.kernel.ndim > 2:
            msg = 'Only one or two dimensional kernels accepted.'
            raise ValueError(msg)
        # check if number of kernels matches num_channels
        if self.kernel.shape[1] not in (1, self.source.num_channels):
            msg = 'Number of kernels must be either `num_channels` or one.'
            raise ValueError(msg)

    # compute the rfft of the kernel blockwise
    @cached_property
    def _get__kernel_blocks(self):
        # Compute the frequency-domain blocks of the kernel using the FFT.
        #
        # This method splits the kernel into blocks and applies the Fast Fourier Transform (FFT)
        # to each block. The result is used in the convolution process for efficient computation.
        #
        # Returns
        # -------
        # :class:`numpy.ndarray`
        #     A 3D array of complex values representing the frequency-domain blocks of the kernel.
        [L, N] = self.kernel.shape
        num = self._block_size
        P = int(np.ceil(L / num))
        trim = num * (P - 1)
        blocks = np.zeros([P, num + 1, N], dtype='complex128')

        if P > 1:
            for i, block in enumerate(np.split(self.kernel[:trim], P - 1, axis=0)):
                blocks[i] = rfft(np.concatenate([block, np.zeros([num, N])], axis=0), axis=0)

        blocks[-1] = rfft(
            np.concatenate([self.kernel[trim:], np.zeros([2 * num - L + trim, N])], axis=0),
            axis=0,
        )
        return blocks

    def result(self, num=128):
        """
        Convolve the source signal with the kernel and yield the result in blocks.

        The method generates the convolution of the source signal with the kernel by processing the
        signal in small blocks, performing the convolution in the frequency domain, and yielding the
        results block by block.

        Parameters
        ----------
        num : :obj:`int`, optional
            Number of samples per block.
            Default is ``128``.

        Yields
        ------
        :obj:`numpy.ndarray`
            A array of shape (``num``, :attr:`~acoular.base.SamplesGenerator.num_channels`),
            where :attr:`~acoular.base.SamplesGenerator.num_channels` is inhereted from the
            :attr:`source`, representing the convolution result in blocks.

        Notes
        -----
        - The kernel is first validated and reshaped if necessary.
        - The convolution is computed efficiently using the FFT in the frequency domain.
        """
        self._validate_kernel()
        # initialize variables
        self._block_size = num
        L = self.kernel.shape[0]
        N = self.source.num_channels
        M = self.source.num_samples
        numblocks_kernel = int(np.ceil(L / num))  # number of kernel blocks
        Q = int(np.ceil(M / num))  # number of signal blocks
        R = int(np.ceil((L + M - 1) / num))  # number of output blocks
        last_size = (L + M - 1) % num  # size of final block

        idx = 0
        fdl = np.zeros([numblocks_kernel, num + 1, N], dtype='complex128')
        buff = np.zeros([2 * num, N])  # time-domain input buffer
        spec_sum = np.zeros([num + 1, N], dtype='complex128')

        signal_blocks = self.source.result(num)
        temp = next(signal_blocks)
        buff[num : num + temp.shape[0]] = temp  # append new time-data

        # for very short signals, we are already done
        if R == 1:
            _append_to_fdl(fdl, idx, numblocks_kernel, rfft(buff, axis=0))
            spec_sum = _spectral_sum(spec_sum, fdl, self._kernel_blocks)
            # truncate s.t. total length is L+M-1 (like numpy convolve w/ mode="full")
            yield irfft(spec_sum, axis=0)[num : last_size + num]
            return

        # stream processing of source signal
        for temp in signal_blocks:
            _append_to_fdl(fdl, idx, numblocks_kernel, rfft(buff, axis=0))
            spec_sum = _spectral_sum(spec_sum, fdl, self._kernel_blocks)
            yield irfft(spec_sum, axis=0)[num:]
            buff = np.concatenate(
                [buff[num:], np.zeros([num, N])],
                axis=0,
            )  # shift input buffer to the left
            buff[num : num + temp.shape[0]] = temp  # append new time-data

        for _ in range(R - Q):
            _append_to_fdl(fdl, idx, numblocks_kernel, rfft(buff, axis=0))
            spec_sum = _spectral_sum(spec_sum, fdl, self._kernel_blocks)
            yield irfft(spec_sum, axis=0)[num:]
            buff = np.concatenate(
                [buff[num:], np.zeros([num, N])],
                axis=0,
            )  # shift input buffer to the left

        _append_to_fdl(fdl, idx, numblocks_kernel, rfft(buff, axis=0))
        spec_sum = _spectral_sum(spec_sum, fdl, self._kernel_blocks)
        # truncate s.t. total length is L+M-1 (like numpy convolve w/ mode="full")
        yield irfft(spec_sum, axis=0)[num : last_size + num]


@nb.jit(nopython=True, cache=True)
def _append_to_fdl(fdl, idx, numblocks_kernel, buff):  # pragma: no cover
    fdl[idx] = buff
    idx = int(idx + 1 % numblocks_kernel)


@nb.jit(nopython=True, cache=True)
def _spectral_sum(out, fdl, kb):  # pragma: no cover
    P, B, N = kb.shape
    for n in range(N):
        for b in range(B):
            out[b, n] = 0
            for i in range(P):
                out[b, n] += fdl[i, b, n] * kb[i, b, n]

    return out
