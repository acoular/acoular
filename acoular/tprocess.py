# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implement blockwise processing in the time domain.

.. inheritance-diagram::
                acoular.tprocess
    :top-classes:
                acoular.base.TimeOut
    :parts: 1

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
    NotchFilter
    AdaptiveNotchFilter
    ZeroPhaseNotchFilter
    CascadeNotchFilter
    OctaveFilterBank
    WriteWAV
    WriteH5
    TimeConvolve
"""

# imports from other packages
import contextlib
import wave
from abc import abstractmethod
from collections import deque
from datetime import UTC, datetime
from os import path
from warnings import warn as _warn

# acoular imports
from .base import SamplesGenerator, TimeOut
from .configuration import config
from .environments import cartToCyl, cylToCart
from .h5files import _get_h5file_class
from .internal import digest, ldigest
from .microphones import MicGeom
from .process import Cache
from .tfastfuncs import (
    iir_harmonic_cascade_lms_kernel,
    iir_lms_kernel,
    iir_time_varying_kernel,
)

import numba as nb
import numpy as np
import scipy.linalg as spla
from scipy.fft import irfft, rfft, rfftfreq
from scipy.interpolate import CloughTocher2DInterpolator, CubicSpline, LinearNDInterpolator, Rbf, splev, splrep
from scipy.signal import bilinear, butter, lfilter, lfilter_zi, sosfilt, sosfiltfilt, tf2sos
from scipy.spatial import Delaunay
from traits.api import (
    Any,
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


class MaskedTimeOut(TimeOut):
    """
    A signal processing block that allows for the selection of specific channels and time samples.

    The :class:`MaskedTimeOut` class is designed to filter data from a given
    :class:`~acoular.base.SamplesGenerator` (or a derived object) by defining valid time samples
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
    start = CInt(0)

    #: The index of the last valid sample (exclusive).
    #: If set to :obj:`None`, the selection continues until the end of the available data.
    stop = Union(None, CInt)

    #: List of channel indices to be excluded from processing.
    invalid_channels = List(int)

    #: A mask or index array representing valid channels. (automatically updated)
    channels = Property(depends_on=['invalid_channels', 'source.num_channels'])

    #: Total number of input channels, including invalid channels, as given by
    #: :attr:`~acoular.base.TimeOut.source`. (read-only).
    num_channels_total = Delegate('source', 'num_channels')

    #: Total number of input channels, including invalid channels. (read-only).
    num_samples_total = Delegate('source', 'num_samples')

    #: Number of valid input channels after excluding :attr:`invalid_channels`. (read-only)
    num_channels = Property(depends_on=['invalid_channels', 'source.num_channels'])

    #: Number of valid time-domain samples, based on :attr:`start` and :attr:`stop` indices.
    #: (read-only)
    num_samples = Property(depends_on=['start', 'stop', 'source.num_samples'])

    #: The name of the cache file (without extension). It serves as an internal reference for data
    #: caching and tracking processed files. (automatically generated)
    basename = Property(depends_on=['source.digest'])

    #: A unique identifier for the object, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'start', 'stop', 'invalid_channels'])

    @cached_property
    def _get_digest(self):
        return digest(self)

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
    :class:`~acoular.base.SamplesGenerator` (or a derived object) and applies an optional set of
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
    weights = CArray()

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

    .. deprecated::
        :class:`Trigger` is deprecated and will be removed in version 27.01.

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

    def __init__(self, *args, **kwargs):
        _warn(
            'Trigger is deprecated and will be removed in version 27.01.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_trigger_data(self):
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
            raise ValueError(msg)

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
            _warn(
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
            _warn(f'No threshold was passed. An estimated threshold of {thresh} is assumed.', Warning, stacklevel=2)
        else:  # take user defined  threshold
            thresh = self.threshold
        return thresh

    def _check_trigger_existence(self):
        nChannels = self.source.num_channels
        if nChannels != 1:
            msg = f'Trigger signal must consist of ONE channel, instead {nChannels} channels are given!'
            raise ValueError(msg)
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
        _warn(msg, Warning, stacklevel=2)
        yield from self.source.result(num)


class AngleTracker(MaskedTimeOut):
    """
    Compute the rotational angle and RPM per sample from a trigger signal in the time domain.

    .. deprecated::
        :class:`AngleTracker` is deprecated and will be removed in version 27.01.

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
    trigger_per_revo = Int(1)

    #: Rotation direction flag:
    #:
    #: - ``1``: counter-clockwise rotation.
    #: - ``-1``: clockwise rotation.
    #:
    #: Default is ``-1``.
    rot_direction = Int(-1)

    #: Number of points used for spline interpolation. Default is ``4``.
    interp_points = Int(4)

    #: Initial rotation angle (in radians) corresponding to the first trigger event. This allows
    #: defining a custom starting reference angle. Default is ``0``.
    start_angle = Float(0)

    #: Revolutions per minute (RPM) computed for each sample.
    #: It is based on the trigger data. (read-only)
    rpm = Property(depends_on=['digest'])

    #: Average revolutions per minute over the entire dataset.
    #: It is computed based on the trigger intervals. (read-only)
    average_rpm = Property(depends_on=['digest'])

    #: Computed rotation angle (in radians) for each sample.
    #: It is interpolated from the trigger data. (read-only)
    angle = Property(depends_on=['digest'])

    # Internal flag to determine whether rpm and angle calculation has been processed,
    # prevents recalculation
    _calc_flag = Bool(False)

    # Revolutions per minute, internal use
    _rpm = CArray()

    # Rotation angle in radians, internal use
    _angle = CArray()

    def __init__(self, *args, **kwargs):
        _warn(
            'AngleTracker is deprecated and will be removed in version 27.01.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

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
    def _reset_calc_flag(self, event):  # noqa: ARG002
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

    .. deprecated::
        :class:`SpatialInterpolator` is deprecated and will be removed in version 27.01.

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
    mics = Instance(MicGeom())

    #: The virtual microphone geometry. This property defines the positions
    #: of virtual microphones where interpolated pressure values are computed.
    #: Default is the physical microphone geometry (:attr:`mics`).
    mics_virtual = Property()

    #: internal microphone geometry;internal usage, read only
    _mics_virtual = Instance(MicGeom)

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
    array_dimension = Enum('1D', '2D', 'ring', '3D', 'custom')

    #: Sampling frequency of the output signal, inherited from the :attr:`source`. This defines the
    #: rate at which microphone pressure samples are acquired and processed.
    sample_freq = Delegate('source', 'sample_freq')

    #: Number of channels in the output data. This corresponds to the number of virtual microphone
    #: positions where interpolated pressure values are computed. The value is determined based on
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
    num_IDW = Int(3)  # noqa: N815

    #: Weighting exponent for IDW interpolation. This parameter controls the influence of distance
    #: in inverse distance weighting (IDW). A higher value gives more weight to closer microphones.
    p_weight = Float(2)

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

    def __init__(self, *args, **kwargs):
        _warn(
            f'{self.__class__.__name__} is deprecated and will be removed in version 27.01.',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def _get_num_channels(self):
        return self.mics_virtual.num_mics

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_virtNewCoord(self):  # noqa: N802
        return self._virtNewCoord_func(self.mics.pos, self.mics_virtual.pos, self.method, self.array_dimension)

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

    def _virtNewCoord_func(self, mpos, mpos_virt, _method, _array_dimension):  # noqa: N802
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

        if self.array_dimension in {'1D', 'ring'}:
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
            tri = Delaunay(newCoord.T[:, :2], incremental=True)

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
        nVirtMics = self.mics_virtual.pos.shape[1]
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
        if self.array_dimension in {'1D', 'ring'}:
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
        _warn(msg, Warning, stacklevel=2)
        yield from self.source.result(num)


class SpatialInterpolatorRotation(SpatialInterpolator):  # pragma: no cover
    """
    Spatial interpolation class for rotating sound sources.

    .. deprecated::
        :class:`SpatialInterpolatorRotation` is deprecated and will be removed in version 27.01.

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
        angle = self.angle_source.angle
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

    .. deprecated::
        :class:`SpatialInterpolatorConstantRotation` is deprecated and will be removed in version
        27.01.

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
    def _set_sourcesdigest(self, event):  # noqa: ARG002
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
            Each block will have the shape (``num``, :attr:`~acoular.base.TimeOut.num_channels`),
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
    Abstract base class for IIR filters using SciPy's :func:`~scipy.signal.sosfilt`.

    This class implements a digital Infinite Impulse Response (IIR) filter that applies filtering to
    a given signal in a block-wise manner. The filter coefficients can be dynamically changed during
    processing.

    See Also
    --------
    :class:`FiltOctave` :
        Octave or third-octave bandpass filter (causal, with non-zero phase delay).
    :class:`FiltFiltOctave` :
        Octave or third-octave bandpass filter with zero-phase distortion.
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
    band = Float(1000.0)

    #: Defines whether the filter is an octave-band or third-octave-band filter.
    #:
    #: - ``'Octave'``: Full octave band filter.
    #: - ``'Third octave'``: Third-octave band filter.
    #:
    #: Default is ``'Octave'``.
    fraction = Map({'Octave': 1, 'Third octave': 3}, default_value='Octave')

    #: The order of the IIR filter, which affects the steepness of the filter's roll-off.
    #: Default is ``3``.
    order = Int(3)

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
    order = Int(2)

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
    weight = Map({'F': 0.125, 'S': 1.0, 'I': 0.035}, default_value='F')

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
    weight = Enum('A', 'C', 'Z')

    #: Second-order sections (SOS) representation of the filter coefficients. This property is
    #: dynamically computed based on :attr:`weight` and the
    #: :attr:`~acoular.tprocess.Filter.source`'s digest.
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


class FilterBank(TimeOut):
    """
    Abstract base class for IIR filter banks based on SOS coefficients.

    Implements a bank of parallel filters. This class should not be instantiated by itself.

    Inherits from :class:`~acoular.base.TimeOut`, and defines the structure for working with filter
    banks for processing multi-channel time series data, such as time signal signals.

    See Also
    --------
    :class:`~acoular.base.TimeOut` :
        ABC for signal processing blocks that interact with data from a source.
    :class:`~acoular.base.SamplesGenerator` :
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
            len_block = block.shape[0]
            for i in range(numbands):
                res[:len_block, i * snumch : (i + 1) * snumch], zi[i] = sosfilt(sos[i], block, axis=0, zi=zi[i])
            yield res[:len_block]


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
    :class:`~acoular.base.SamplesGenerator` :
        Interface for generating multi-channel time domain signal processing blocks.
    :mod:`scipy.signal` :
        SciPy module for signal processing.
    """

    #: The lowest band center frequency index. Default is ``21``.
    #: This index refers to the position in the scale of octave or third-octave bands.
    lband = Int(21)

    #: The highest band center frequency index + 1. Default is ``40``.
    #: This is the position in the scale of octave or third-octave bands.
    hband = Int(40)

    #: The fraction of an octave, either ``'Octave'`` or ``'Third octave'``.
    #: Default is ``'Octave'``.
    #: Determines the width of the frequency bands. 'Octave' refers to full octaves,
    #: and ``'Third octave'`` refers to third-octave bands.
    fraction = Map({'Octave': 1, 'Third octave': 3}, default_value='Octave')

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


class WriteWAV(TimeOut):
    """
    Saves time signal from one or more channels as mono, stereo, or multi-channel ``.wav`` file.

    Inherits from :class:`~acoular.base.TimeOut` and allows for exporting time-series data from one
    or more channels to a WAV file. Supports saving mono, stereo, or multi-channel signals to disk
    with automatic or user-defined file naming.

    See Also
    --------
    :class:`~acoular.base.TimeOut` :
        ABC for signal processing blocks that interact with data from a source.
    :class:`~acoular.base.SamplesGenerator` :
        Interface for generating multi-channel time domain signal processing blocks.
    :mod:`wave` :
        Python module for handling WAV files.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: The name of the file to be saved. If none is given, the name will be automatically
    #: generated from the source.
    file = File(filter=['*.wav'])

    #: The name of the cache file (without extension). It serves as an internal reference for data
    #: caching and tracking processed files. (automatically generated)
    basename = Property(depends_on=['digest'])

    #: The list of channels to save. Can only contain one or two channels.
    channels = List(int)

    # Bit depth of the output file.
    #: bit depth of the output file
    encoding = Enum('uint8', 'int16', 'int32')

    # Maximum value to scale the output to. If `None`, the maximum value of the data is used.
    #: Maximum value to scale the output to.
    max_val = Either(None, Float)

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'channels'])

    @cached_property
    def _get_digest(self):
        return digest(self)

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
            _warn(
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
            _warn(f'More than two channels given for output, exported file will have {nc:d} channels', stacklevel=1)
        if self.sample_freq.is_integer():
            fs = self.sample_freq
        else:
            fs = round(self.sample_freq)
            msg = f'Sample frequency {self.sample_freq} is not a whole number. Proceeding with sampling frequency {fs}.'
            _warn(msg, Warning, stacklevel=1)
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


class WriteH5(TimeOut):
    """
    Saves time signal data as a ``.h5`` (HDF5) file.

    Inherits from :class:`~acoular.base.TimeOut` and provides functionality for saving multi-channel
    time-domain signal data to an HDF5 file. The file can be written in blocks and supports
    metadata storage, precision control, and dynamic file generation based on timestamps.

    See Also
    --------
    :class:`~acoular.base.TimeOut` :
        ABC for signal processing blocks interacting with data from a source.
    :class:`~acoular.base.SamplesGenerator` :
        Interface for generating multi-channel time-domain signal processing blocks.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: The name of the file to be saved. If none is given, the name is automatically
    #: generated based on the current timestamp.
    file = File(filter=['*.h5'])

    #: The number of samples to write to file per call to `result` method.
    #: Default is ``-1``, meaning all available data from the source will be written.
    num_samples_write = Int(-1)

    #: A flag that can be set to stop file writing. Default is ``True``.
    write_flag = Bool(True)

    #: A unique identifier for the object, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest'])

    #: Precision of the entries in the HDF5 file, represented as numpy data types.
    #: Default is ``'float32'``.
    precision = Enum('float32', 'float64')

    #: Metadata to be stored in the HDF5 file.
    metadata = Dict()

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
            name = datetime.now(tz=UTC).isoformat('_').replace(':', '-').replace('.', '_')
            self.file = path.join(config.td_dir, name + '.h5')

    def get_initialized_file(self):
        """
        Initialize the HDF5 file and prepare the necessary datasets and metadata.

        This method creates the file (if it doesn't exist), sets up the main data array,
        and appends metadata to the file.

        Returns
        -------
        `h5py.File`
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
        f5h : `h5py.File`
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
    :class:`~acoular.base.TimeOut` :
        The parent class for signal processing blocks.
    :class:`~acoular.base.SamplesGenerator` :
        The interface for generating multi-channel time-domain signals.
    """

    #: The input data source. It must be an instance of a
    #: :class:`~acoular.base.SamplesGenerator`-derived class.
    source = Instance(SamplesGenerator)

    #: Convolution kernel in the time domain.
    #: The second dimension of the kernel array has to be either ``1`` or match
    #: the :attr:`source`'s :attr:`~acoular.base.Generator.num_channels` attribute.
    #: If only a single kernel is supplied, it is applied to all channels.
    kernel = CArray(dtype=float)

    #: Controls whether to extend the output to include the full convolution result.
    #:
    #: - If ``False`` (default): Output length is :math:`\\max(L, M)`, where :math:`L` is the
    #:   kernel length and :math:`M` is the signal length. This mode keeps the output length
    #:   equal to the longest input (different from NumPy's ``mode='same'``, since it does not
    #:   pad the output).
    #: - If ``True``: Output length is :math:`L + M - 1`, returning the full convolution at
    #:   each overlap point (similar to NumPy's ``mode='full'``).
    #:
    #: Default is ``False``.
    extend_signal = Bool(False)

    # Internal block size for partitioning signals into smaller segments during processing.
    #: Block size
    _block_size = Int()

    # Blocks of the convolution kernel in the frequency domain.
    # Computed using Fast Fourier Transform (FFT).
    _kernel_blocks = Property(
        depends_on=['kernel', '_block_size'],
    )

    #: A unique identifier for the object, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'kernel', 'extend_signal'])

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
        r"""
        Convolve the source signal with the kernel and yield the result in blocks.

        The method generates the convolution of the source signal (length :math:`M`) with the kernel
        (length :math:`L`) by processing the signal in small blocks, performing the convolution in
        the frequency domain, and yielding the results block by block.

        Parameters
        ----------
        num : :obj:`int`, optional
            Number of samples per block.
            Default is ``128``.

        Yields
        ------
        :obj:`numpy.ndarray`
            An array of shape (``num``, :attr:`~acoular.base.Generator.num_channels`),
            where :attr:`~acoular.base.Generator.num_channels` is inherited from the
            :attr:`source`, representing the convolution result in blocks.

        Notes
        -----
        - The kernel is first validated and reshaped if necessary.
        - The convolution is computed efficiently using the FFT in the frequency domain.
        - The output length is determined by the :attr:`extend_signal` property.
        """
        self._validate_kernel()
        # initialize variables
        self._block_size = num
        L = self.kernel.shape[0]
        N = self.source.num_channels
        M = self.source.num_samples

        output_size = max(L, M) if not self.extend_signal else L + M - 1

        numblocks_kernel = int(np.ceil(L / num))  # number of kernel blocks
        Q = int(np.ceil(M / num))  # number of signal blocks
        R = int(np.ceil(output_size / num))  # number of output blocks
        last_size = output_size % num  # size of final output block

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
            final_len = last_size if last_size != 0 else num
            yield irfft(spec_sum, axis=0)[num : final_len + num]
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
        final_len = last_size if last_size != 0 else num
        yield irfft(spec_sum, axis=0)[num : final_len + num]


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


def find_spectral_peaks(
    signal_1d,
    sample_freq,
    n_peaks=None,
    freq_hint=None,
    fft_tolerance=0.2,
    threshold_db=20.0,
):
    """FFT-based spectral peak detection with parabolic interpolation.

    Computes the power spectrum of a 1-D signal, identifies local maxima
    above a noise-relative threshold, and refines each peak location via
    three-point parabolic interpolation for sub-bin accuracy.

    Parameters
    ----------
    signal_1d : numpy.ndarray
        Input signal, shape ``(num_samples,)``.
    sample_freq : float
        Sampling frequency in Hz.
    n_peaks : int or None
        Maximum number of peaks to return.  ``None`` returns all.
    freq_hint : float or None
        If given, only peaks within ``freq_hint * (1 +- fft_tolerance)``
        are considered.
    fft_tolerance : float
        Fractional tolerance for the *freq_hint* search window
        (default 0.2 = +-20 %).
    threshold_db : float
        Minimum dB above the median power-spectrum level for a peak to
        be considered (default 20 dB).

    Returns
    -------
    list of tuple
        ``(freq_hz, power_db)`` pairs sorted by descending power.

    References
    ----------
    .. [1] Jacobsen, E. and Kootsookos, P. (2007). Fast, Accurate Frequency
           Estimators [DSP Tips & Tricks]. IEEE Signal Processing Magazine,
           24(3), 123-125. https://doi.org/10.1109/MSP.2007.361611
    .. [2] Smith, J.O. (2011). Spectral Audio Signal Processing. W3K Publishing.
           Parabolic peak interpolation:
           https://ccrma.stanford.edu/~jos/sasp/Peak_Detection_Steps_3.html
    """
    fft_values = rfft(signal_1d)
    fft_freqs = rfftfreq(len(signal_1d), 1.0 / sample_freq)
    power_db = 10.0 * np.log10(np.abs(fft_values) ** 2 + 1e-12)
    threshold = np.median(power_db) + threshold_db

    peaks = []
    for i in range(1, len(power_db) - 1):
        if power_db[i] > power_db[i - 1] and power_db[i] > power_db[i + 1] and power_db[i] > threshold:
            y0, y1, y2 = power_db[i - 1], power_db[i], power_db[i + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-10:
                # Three-point parabolic interpolation: d = 0.5*(y0 - y2)/(y0 - 2*y1 + y2)
                # Achieves sub-bin frequency accuracy from three dB-scale FFT samples.
                # (Jacobsen & Kootsookos 2007; Smith SASP appendix C.2)
                delta = 0.5 * (y0 - y2) / denom
                refined = fft_freqs[i] + delta * (fft_freqs[1] - fft_freqs[0])
                peaks.append((refined, y1))

    if not peaks:
        return []

    # Filter by frequency hint if provided
    if freq_hint is not None and abs(freq_hint) > 1e-6:
        lo = freq_hint * (1.0 - fft_tolerance)
        hi = freq_hint * (1.0 + fft_tolerance)
        nearby = [(f, p) for f, p in peaks if lo <= f <= hi]
        if nearby:
            peaks = nearby

    peaks.sort(key=lambda x: x[1], reverse=True)
    if n_peaks is not None:
        peaks = peaks[:n_peaks]
    return peaks


class NotchFilter(Filter):
    r"""Second-order IIR notch filter with configurable center frequency.

    Implements the transfer function:

    .. math::

        H(z) = \frac{1 - 2\cos(\theta)z^{-1} + z^{-2}}
               {1 - 2r\cos(\theta)z^{-1} + r^2 z^{-2}}

    where :math:`\theta = 2\pi f_{\text{notch}}/f_s` is the normalized notch
    frequency and *r* is the pole radius controlling notch width.

    The filter places zeros on the unit circle at the notch frequency
    and poles inside the unit circle for stability. This achieves sharp
    tonal suppression while preserving other frequency components.

    The notch biquad is exposed as a single second-order section via the
    :attr:`~acoular.tprocess.Filter.sos` property, so streaming and
    per-channel state handling are inherited from
    :class:`~acoular.tprocess.Filter` (:func:`scipy.signal.sosfilt`).

    The filter bank this class belongs to follows :cite:`Harvey2019`; for the
    notch (antiresonance) biquad itself see :cite:`Smith2007`.

    See Also
    --------
    :class:`AdaptiveNotchFilter` : Notch filter with time-varying frequency.
    :class:`ZeroPhaseNotchFilter` : Phase-preserving notch filter.
    :class:`CascadeNotchFilter` : Bank of notches for harmonic series.

    Examples
    --------
    >>> import acoular as ac
    >>> from acoular import NotchFilter
    >>> source = ac.TimeSamples(name='measurement.h5')  # doctest: +SKIP
    >>> filt = NotchFilter(f_notch=440.0, pole_radius=0.95, source=source)  # doctest: +SKIP
    """

    #: Center frequency of the notch in Hz. Must be less than the Nyquist
    #: frequency (sample_freq / 2).
    f_notch = Float(desc='center frequency of the notch in Hz')

    #: Pole radius controlling notch width (0 < r < 1, default 0.99).
    #: Closer to 1 gives a narrower notch; closer to 0 gives a wider notch.
    pole_radius = Float(0.99, desc='pole radius (0 < r < 1)')

    #: Second-order sections representation of the notch biquad, a single
    #: ``(1, 6)`` section derived from :attr:`f_notch` and :attr:`pole_radius`.
    sos = Property(depends_on=['source.digest', 'f_notch', 'pole_radius'])

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(depends_on=['source.digest', 'f_notch', 'pole_radius'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        # Single second-order section [b0, b1, b2, a0, a1, a2] for the notch
        # biquad; consumed by the inherited Filter.result() via scipy.sosfilt.
        b, a = self._compute_coefficients()
        return np.hstack([b, a]).reshape(1, 6)

    def _compute_coefficients_for_freq(self, f_notch):
        """Compute IIR coefficients for an arbitrary notch frequency.

        Parameters
        ----------
        f_notch : float
            Notch center frequency in Hz.

        Returns
        -------
        b : numpy.ndarray
            Numerator coefficients [1, -2*cos(theta), 1].
        a : numpy.ndarray
            Denominator coefficients [1, -2*r*cos(theta), r**2].
        """
        # Zeros at e^{+-j theta} (on unit circle) -> complete cancellation at f_notch.
        # Poles at r*e^{+-j theta} (inside unit circle) -> BIBO stability.
        # Narrow notch for r -> 1; wider notch for r -> 0.
        # H(z) = (1 - 2cos(theta)z^-1 + z^-2) / (1 - 2r*cos(theta)z^-1 + r^2 z^-2)
        # See: https://ccrma.stanford.edu/~jos/filters/Peaking_Equalizers.html
        theta = 2 * np.pi * f_notch / self.sample_freq
        r = self.pole_radius
        b = np.array([1.0, -2.0 * np.cos(theta), 1.0])
        a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
        return b, a

    def _compute_coefficients(self):
        """Compute IIR filter coefficients from the current :attr:`f_notch`.

        Returns
        -------
        b : numpy.ndarray
            Numerator coefficients.
        a : numpy.ndarray
            Denominator coefficients.
        """
        return self._compute_coefficients_for_freq(self.f_notch)

    @property
    def coefficients(self):
        """Filter coefficients for frequency response analysis.

        Returns
        -------
        tuple of numpy.ndarray
            ``(b, a)`` where *b* is the numerator and *a* the denominator.
        """
        b, a = self._compute_coefficients()
        return b.copy(), a.copy()


class AdaptiveNotchFilter(NotchFilter):
    """Adaptive notch filter with time-varying frequency tracking.

    Extends :class:`NotchFilter` with dynamic frequency updating based on
    either external frequency trajectories (external mode) or autonomous LMS
    adaptation (auto mode). Supports multi-channel input.

    With no *freq_source* and ``mode=None``, behaves identically to
    :class:`NotchFilter`.

    Notes
    -----
    External mode updates filter coefficients dynamically based on
    *freq_source* blocks while preserving filter state across frequency
    changes to maintain signal continuity.

    Filter coefficients are recomputed per sample when the frequency changes,
    which is expected behaviour for time-varying filters. State preservation
    ensures no discontinuities at frequency transitions.

    In auto mode the frequency is tracked with a normalised LMS update of the
    recursive gradient after :cite:`Nehorai1985`; drift is monitored and reset
    by a global FFT search following :cite:`TanJiang2009`. The overall approach
    follows :cite:`Harvey2019`.

    See Also
    --------
    :class:`NotchFilter` : Static notch filter, the base class.
    :class:`ZeroPhaseNotchFilter` : Phase-preserving variant.
    """

    #: Streaming frequency source for external-mode frequency tracking.
    #: Must be an object with a ``result(num)`` method yielding
    #: ``(num_samples,)`` frequency arrays. Works for both offline
    #: (wrap a pre-computed array) and real-time scenarios.
    freq_source = Instance(
        SamplesGenerator,
        desc='Streaming source with result(num) yielding (num_samples,) frequency arrays '
        'for external-mode frequency tracking.',
    )

    #: Adaptation mode. ``None`` infers from *freq_source* (external if
    #: provided, static otherwise). ``'external'`` uses direct frequency
    #: control. ``'auto'`` uses autonomous LMS adaptation.
    mode = Enum(None, 'external', 'auto', desc="adaptation mode: None, 'external', or 'auto'")

    #: LMS step size for auto mode. A single float or a list of step sizes.
    mu = Union(Float, List, desc='LMS step size for auto mode')

    #: Moving-average window for frequency smoothing in auto mode.
    smooth_window = Int(256, desc='moving-average window for frequency smoothing')

    #: Leak factor for the recursive gradient (0 = instantaneous,
    #: 1 = full recursive). Recommended 0.95 for single-tone tracking with
    #: ``mu`` ~ 0.06 and ``pole_radius`` ~ 0.95.
    gradient_leak = Float(0.0, desc='leak factor for recursive LMS gradient')

    # Internal per-channel processing state (transient, not part of digest).
    # Allocated lazily in _ensure_states and reset at the start of result().
    _zi = Any()
    _beta_state = Any()
    _current_freq_per_ch = Any()
    _freq_history = Any()
    _freq_history_sum = Any()
    _f0_per_ch = Any()
    _learned_frequencies = Any(np.zeros(0))
    _initialized = Bool(False)
    _current_position = Int(0)

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'source.digest',
            'f_notch',
            'pole_radius',
            'freq_source.digest',
            'mode',
            'mu',
            'smooth_window',
            'gradient_leak',
        ]
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property
    def learned_frequencies(self):
        """Per-sample frequency estimate of the most recently processed block.

        Only meaningful in auto mode, where it holds the frequency the LMS
        update tracked for every sample of the last block yielded by
        :meth:`result`, shape ``(num,)``. It is taken from the first channel;
        in other modes it stays empty.

        Note that this covers the last block only -- collect it while
        iterating over :meth:`result` to obtain the full trajectory.
        """
        return self._learned_frequencies

    def _ensure_states(self, num_channels):
        """Allocate per-channel state arrays if not yet sized correctly."""
        zi = self._zi
        if zi is None or zi.shape[0] != num_channels:
            self._zi = np.zeros((num_channels, 2))
            self._beta_state = np.zeros((num_channels, 5))
            self._current_freq_per_ch = np.full(num_channels, self.f_notch)
            self._freq_history = np.tile(
                np.full(self.smooth_window, self.f_notch),
                (num_channels, 1),
            )
            self._freq_history_sum = np.full(
                num_channels,
                self.f_notch * self.smooth_window,
            )
            self._f0_per_ch = np.full(num_channels, self.f_notch)

    def _get_effective_mode(self):
        """Infer mode from *freq_source* if not set explicitly.

        Returns
        -------
        str or None
            ``'external'``, ``'auto'``, or ``None`` (static mode).
        """
        if self.mode is not None:
            return self.mode
        if self.freq_source is not None:
            return 'external'
        return None

    def _resolve_step_size(self):
        """Resolve LMS step size from the :attr:`mu` trait."""
        if isinstance(self.mu, list):
            return self.mu[0] if self.mu else 0.001
        if self.mu is not None:
            return float(self.mu)
        return 0.001

    def _filter_block(self, data, freq_trajectory=None):
        """Process one multi-channel block.

        Called by :class:`CascadeNotchFilter` to apply this filter to all
        *K* channels.

        Parameters
        ----------
        data : numpy.ndarray
            Input block, shape ``(num_samples, num_channels)``.
        freq_trajectory : numpy.ndarray or None
            Per-sample frequency, shape ``(num_samples,)``.
            If given, time-varying filtering is applied.
            If ``None`` and ``mode == 'auto'``, LMS adaptation is used.
            Otherwise, static filtering with the current :attr:`f_notch`.

        Returns
        -------
        numpy.ndarray
            Filtered block, same shape as *data*.
        """
        num_channels = data.shape[1]
        self._ensure_states(num_channels)

        if freq_trajectory is not None:
            cos_theta = np.cos(
                2.0 * np.pi * np.ascontiguousarray(freq_trajectory, dtype=np.float64) / self.sample_freq,
            )
            data_c = np.ascontiguousarray(data, dtype=np.float64)
            return iir_time_varying_kernel(
                data_c,
                cos_theta,
                self.pole_radius,
                self._zi,
            )

        if self.mode == 'auto':
            step_size = self._resolve_step_size()
            data_c = np.ascontiguousarray(data, dtype=np.float64)
            output, learned, _ = iir_lms_kernel(
                data_c,
                self.pole_radius,
                self.sample_freq,
                step_size,
                self.smooth_window,
                self._zi,
                self._current_freq_per_ch,
                self._beta_state,
                self._freq_history,
                self._freq_history_sum,
                gradient_leak=self.gradient_leak,
            )
            self._learned_frequencies = learned
            return output

        # Static mode - per-channel lfilter
        b, a = self._compute_coefficients()
        output = np.zeros_like(data)
        for ch in range(num_channels):
            output[:, ch], self._zi[ch] = lfilter(
                b,
                a,
                data[:, ch],
                zi=self._zi[ch],
            )
        return output

    def result(self, num):
        """Apply adaptive filter to input signal blocks.

        Dispatches based on the effective mode (inferred or explicit).

        Parameters
        ----------
        num : int
            Number of samples per output block.

        Yields
        ------
        numpy.ndarray
            Filtered signal blocks with shape ``(num, num_channels)``.
            The final block may contain fewer than *num* samples.

        Raises
        ------
        ValueError
            If mode is ``'external'`` but no *freq_source* is provided.
        """
        # Reset transient per-channel state at the start of each pass.
        self._zi = None
        self._beta_state = None
        self._current_freq_per_ch = None
        self._freq_history = None
        self._freq_history_sum = None
        self._f0_per_ch = None
        self._learned_frequencies = np.zeros(0)
        self._initialized = False
        self._current_position = 0

        effective_mode = self._get_effective_mode()

        if effective_mode == 'external' and self.freq_source is None:
            msg = (
                "mode is 'external' but no freq_source provided. "
                "Set freq_source or use mode='auto' / None."
            )
            raise ValueError(msg)

        if effective_mode is None:
            yield from super().result(num)
        elif effective_mode == 'external':
            yield from self._result_external(num)
        elif effective_mode == 'auto':
            yield from self._result_auto(num)

    def _result_external(self, num):
        """Process signal in external RPM mode.

        Pulls frequency blocks from *freq_source* in lockstep with signal
        blocks and applies time-varying filtering with state preservation.
        """
        freq_iter = self.freq_source.result(num)

        for source_block in self.source.result(num):
            num_samples = source_block.shape[0]
            num_channels = source_block.shape[1]
            self._ensure_states(num_channels)

            try:
                block_freq_traj = np.asarray(next(freq_iter)).ravel()
            except StopIteration:
                msg = 'freq_source exhausted before source finished yielding blocks.'
                raise ValueError(msg) from None
            if len(block_freq_traj) != num_samples:
                msg = (
                    f'freq_source block size {len(block_freq_traj)} does not match '
                    f'source block size {num_samples}. Both must yield blocks of the same size.'
                )
                raise ValueError(msg)

            output = self._filter_block(source_block, block_freq_traj)
            self._current_position = self._current_position + num_samples
            yield output

    def _apply_time_varying_filter(self, data, freq_trajectory, ch):
        """Apply time-varying IIR filtering to one channel.

        Thin wrapper around the vectorised kernel used by
        :class:`ZeroPhaseNotchFilter` for single-channel passes.

        Parameters
        ----------
        data : numpy.ndarray
            Input data, shape ``(num_samples,)``.
        freq_trajectory : numpy.ndarray
            Frequency at each sample, shape ``(num_samples,)``.
        ch : int
            Channel index.

        Returns
        -------
        numpy.ndarray
            Filtered data, shape ``(num_samples,)``.
        """
        data_2d = np.ascontiguousarray(data[:, np.newaxis], dtype=np.float64)
        cos_theta = np.cos(
            2.0 * np.pi * np.ascontiguousarray(freq_trajectory, dtype=np.float64) / self.sample_freq,
        )
        zi_slice = self._zi[ch : ch + 1].copy()
        output = iir_time_varying_kernel(
            data_2d,
            cos_theta,
            self.pole_radius,
            zi_slice,
        )
        self._zi[ch] = zi_slice[0]
        return output[:, 0]

    def _result_auto(self, num):
        """Process signal in autonomous LMS mode.

        Performs FFT-based initialisation on the first block, then
        continuously adapts filter frequencies using the referenceless LMS
        algorithm.
        """
        for source_block in self.source.result(num):
            num_samples = source_block.shape[0]
            num_channels = source_block.shape[1]
            self._ensure_states(num_channels)

            if not self._initialized:
                self._initialize_from_fft(source_block)
                self._initialized = True
                init_f = self.f_notch
                self._current_freq_per_ch[:] = init_f
                self._freq_history[:] = init_f
                self._freq_history_sum[:] = init_f * self.smooth_window

            output = self._filter_block(source_block)
            self._monitor_and_reset(source_block)
            self._current_position = self._current_position + num_samples
            yield output

    def _initialize_from_fft(self, initial_block, fft_tolerance=0.2, *, global_search=False):
        """Initialise filter frequency using FFT peak detection near *f_notch*.

        Uses parabolic interpolation to refine peak frequency estimates
        beyond FFT bin resolution for sub-Hz accuracy.

        Parameters
        ----------
        initial_block : numpy.ndarray
            Initial signal block with shape ``(num_samples, num_channels)``.
        fft_tolerance : float
            Fractional tolerance for peak search around *f_notch*
            (default 0.2 = +-20 %).
        global_search : bool
            If ``True``, pick the globally strongest peak instead of
            restricting to the region around *f_notch*.
        """
        hint = None if global_search else self.f_notch
        peaks = find_spectral_peaks(
            initial_block[:, 0],
            self.sample_freq,
            n_peaks=1,
            freq_hint=hint,
            fft_tolerance=fft_tolerance,
        )
        if not peaks:
            return

        init_freq = peaks[0][0]
        self.f_notch = init_freq
        if self._f0_per_ch is not None and self._f0_per_ch.shape[0] > 0:
            self._f0_per_ch[:] = init_freq

    def _monitor_and_reset(self, current_block):
        """Check global-minimum drift and reset if needed.

        Implements the monitoring strategy of Tan & Jiang (2009, section II).
        When the smoothed frequency drifts beyond
        ``0.25 * (1 - r) * fs / pi`` from the last initialisation frequency,
        a global FFT search and reset is triggered.

        Parameters
        ----------
        current_block : numpy.ndarray
            Raw input block, shape ``(num_samples, num_channels)``.
        """
        if self._current_freq_per_ch is None or self._current_freq_per_ch.shape[0] == 0:
            return
        # Maximum allowable frequency drift before a global-search reset.
        # Derived from the notch bandwidth: BW ~= (1-r)*fs/pi  (-3 dB points),
        # so delta_f_max = 0.25*BW ensures the LMS gradient is still well-defined
        # at the current estimate. (Tan & Jiang 2009, section II; DOI: 10.1109/MSP.2009.934189)
        delta_f_max = 0.25 * (1.0 - self.pole_radius) * self.sample_freq / np.pi
        if abs(self._current_freq_per_ch[0] - self._f0_per_ch[0]) > delta_f_max:
            prev_f0 = self._f0_per_ch[0]
            self._initialize_from_fft(current_block, global_search=True)
            if abs(self._f0_per_ch[0] - prev_f0) > 1e-6:
                new_f = self.f_notch
                self._current_freq_per_ch[:] = new_f
                self._freq_history[:] = new_f
                self._freq_history_sum[:] = new_f * self.smooth_window
                self._beta_state[:] = 0.0


class ZeroPhaseNotchFilter(AdaptiveNotchFilter):
    """Zero-phase notch filter using forward-backward (filtfilt) algorithm.

    Inherits the full trait set and per-channel state machinery from
    :class:`AdaptiveNotchFilter` (which itself inherits from
    :class:`NotchFilter`).  The only responsibility of this class is to
    replace the causal :meth:`_filter_block` with a forward-backward
    implementation and to expose :meth:`_process_signal` / :meth:`result`
    helpers that collect the full signal first (required by any non-causal
    algorithm).

    Notes
    -----
    * Forward-backward filtering is a batch (non-causal) operation.  Both
      :meth:`_process_signal` and :meth:`result` therefore collect the entire
      source signal before processing.
    * The static path pads the signal with odd-extension (matching
      :func:`scipy.signal.filtfilt`) so that boundary transients are
      suppressed.
    * The external and auto paths initialise the forward and backward
      filter states from :func:`scipy.signal.lfilter_zi` scaled by the
      first / last sample of the (unpadded) signal - no explicit padding is
      needed because the trajectory already defines the filter at every
      sample.

    The boundary states follow :cite:`Gustafsson1996`, the method also used by
    :func:`scipy.signal.filtfilt`; see :cite:`Smith2007` for forward-backward
    filtering in general and :cite:`Harvey2019` for the application to
    propeller noise.

    See Also
    --------
    :class:`AdaptiveNotchFilter` : The causal base class.
    :class:`FiltFiltOctave` : Zero-phase octave band filter.
    """

    #: Core block size used when iterating in overlapping blocks (static /
    #: external modes inside :meth:`result`).
    block_size = Int(4096, desc='core block size for block-based iteration')

    #: Multiplier applied to the single-filter settling time to compute
    #: the overlap margin on each side of a core block.
    overlap_factor = Float(2.0, desc='overlap margin multiplier for settling time')

    #: Number of future blocks buffered for the backward pass in streaming
    #: mode.  Higher values improve boundary accuracy at the cost of latency
    #: and memory.
    num_lookahead_blocks = Int(1, desc='number of future blocks buffered for backward pass')

    #: When ``True``, collect the entire signal before processing
    #: (full-signal batch processing).  When ``False`` (default), use
    #: deque-based streaming with ``num_lookahead_blocks`` of latency.
    batch_mode = Bool(False, desc='use full-signal batch processing instead of streaming')

    # Internal state for the zero-phase passes (transient, not in digest).
    # Reset at the start of result(); the streaming buffers hold the
    # forward-pass output awaiting the backward pass.
    _learned_trajectory = Any(np.array([]))
    _stream_fwd_zi = Any()
    _stream_prev_fwd = Any()
    _stream_prev_traj = Any()
    _stream_prev_learned = Any()

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'source.digest',
            'f_notch',
            'pole_radius',
            'freq_source.digest',
            'mode',
            'mu',
            'smooth_window',
            'gradient_leak',
            'block_size',
            'overlap_factor',
            'num_lookahead_blocks',
            'batch_mode',
        ]
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property
    def overlap_samples(self):
        """Overlap margin for block-based iteration.

        ``overlap = overlap_factor * settling``, where
        ``settling = 7 / |ln(pole_radius)|``.
        For ``pole_radius = 0.95`` and ``overlap_factor = 2.0`` this gives
        ~ 274 samples.
        """
        tau = -1.0 / np.log(self.pole_radius)
        settling = int(7 * tau)
        return int(self.overlap_factor * settling)

    def _compute_padlen(self):
        """Return the padding length matching :func:`scipy.signal.filtfilt`."""
        b, a = self._compute_coefficients()
        return 3 * max(len(b), len(a))

    def _pad_signal(self, x):
        """Odd-extension padding at both ends of a 1-D signal.

        Reflects the signal anti-symmetrically about each boundary:
        ``pad[k] = 2*x[0] - x[k]`` (front) and ``2*x[-1] - x[-k]`` (back).
        This forces continuity of the signal at the endpoints, suppressing
        start-up and end transients in the forward-backward pass.
        Matches the padding used by :func:`scipy.signal.filtfilt`.
        (Gustafsson 1996, section III; DOI: 10.1109/78.492552)
        """
        padlen = self._compute_padlen()
        front_pad = 2 * x[0] - x[padlen:0:-1]
        back_pad = 2 * x[-1] - x[-2 : -padlen - 2 : -1]
        return np.concatenate([front_pad, x, back_pad])

    def _unpad_signal(self, x, original_length):
        """Strip the odd-extension padding."""
        padlen = self._compute_padlen()
        return x[padlen : padlen + original_length]

    def _forward_pass(self, x, b, a, zi=None):
        """Forward lfilter with fixed coefficients.

        If *zi* is ``None`` (first block) the initial condition is derived
        from :func:`scipy.signal.lfilter_zi` scaled by ``x[0]``.
        """
        if zi is None:
            zi = lfilter_zi(b, a) * x[0]
        return lfilter(b, a, x, zi=zi)

    def _backward_pass(self, x, b, a, zi=None):
        """Time-reverse -> lfilter -> time-reverse (fixed coefficients)."""
        x_reversed = x[::-1]
        if zi is None:
            zi = lfilter_zi(b, a) * x_reversed[0]
        y_reversed, zf = lfilter(b, a, x_reversed, zi=zi)
        return y_reversed[::-1], zf

    def _process_channel(self, x, b, a):
        """Full padded forward-backward pass for one channel (static mode)."""
        original_length = len(x)
        x_padded = self._pad_signal(x)

        zi = lfilter_zi(b, a)
        y_forward, _ = self._forward_pass(x_padded, b, a, zi * x_padded[0])

        zi_backward = lfilter_zi(b, a)
        y_backward, _ = self._backward_pass(y_forward, b, a, zi_backward * y_forward[-1])

        return self._unpad_signal(y_backward, original_length)

    def _forward_pass_adaptive(self, x, freq_trajectory, zi=None):
        """Sample-by-sample forward pass with time-varying coefficients.

        Returns ``(output, zi_final)`` for stateful block chaining.
        """
        x_2d = np.ascontiguousarray(x[:, np.newaxis], dtype=np.float64)
        cos_theta = np.cos(
            2.0 * np.pi * np.ascontiguousarray(freq_trajectory, dtype=np.float64) / self.sample_freq,
        )

        if zi is None:
            b, a = self._compute_coefficients_for_freq(freq_trajectory[0])
            zi = lfilter_zi(b, a) * x[0]

        zi_2d = np.ascontiguousarray(zi[np.newaxis, :], dtype=np.float64)
        output_2d = iir_time_varying_kernel(
            x_2d,
            cos_theta,
            self.pole_radius,
            zi_2d,
        )

        return output_2d[:, 0], zi_2d[0]

    def _backward_pass_adaptive(self, x, freq_trajectory, zi=None):
        """Time-reverse -> sample-by-sample filter -> time-reverse.

        *freq_trajectory* must already be reversed by the caller.
        Returns ``(output, zi_final)``.
        """
        x_reversed = x[::-1]
        x_2d = np.ascontiguousarray(x_reversed[:, np.newaxis], dtype=np.float64)
        cos_theta = np.cos(
            2.0 * np.pi * np.ascontiguousarray(freq_trajectory, dtype=np.float64) / self.sample_freq,
        )

        if zi is None:
            b, a = self._compute_coefficients_for_freq(freq_trajectory[0])
            zi = lfilter_zi(b, a) * x_reversed[0]

        zi_2d = np.ascontiguousarray(zi[np.newaxis, :], dtype=np.float64)
        output_2d = iir_time_varying_kernel(
            x_2d,
            cos_theta,
            self.pole_radius,
            zi_2d,
        )

        return output_2d[::-1, 0], zi_2d[0]

    def _forward_pass_lms(self, x, ch, zi=None):
        """LMS-adapting forward pass for one channel.

        Returns ``(output, learned_trajectory, zi_final)``.
        """
        step_size = self._resolve_step_size()
        x_2d = np.ascontiguousarray(x[:, np.newaxis], dtype=np.float64)

        zi_slice = self._zi[ch : ch + 1].copy()
        freq_slice = self._current_freq_per_ch[ch : ch + 1].copy()
        beta_slice = self._beta_state[ch : ch + 1].copy()
        hist_slice = self._freq_history[ch : ch + 1].copy()
        sum_slice = self._freq_history_sum[ch : ch + 1].copy()

        if zi is not None:
            zi_slice[0] = zi
        else:
            b, a = self._compute_coefficients_for_freq(freq_slice[0])
            zi_init = lfilter_zi(b, a) * x[0]
            zi_slice[0] = zi_init

        output_2d, learned, _ = iir_lms_kernel(
            x_2d,
            self.pole_radius,
            self.sample_freq,
            step_size,
            self.smooth_window,
            zi_slice,
            freq_slice,
            beta_slice,
            hist_slice,
            sum_slice,
            gradient_leak=self.gradient_leak,
        )

        # Write back per-channel state
        self._zi[ch] = zi_slice[0]
        self._current_freq_per_ch[ch] = freq_slice[0]
        self._beta_state[ch] = beta_slice[0]
        self._freq_history[ch] = hist_slice[0]
        self._freq_history_sum[ch] = sum_slice[0]

        return output_2d[:, 0], learned, zi_slice[0]

    def _process_channel_external(self, x, traj):
        """External-mode zero-phase for one channel."""
        y_forward, _ = self._forward_pass_adaptive(x, traj)
        y_backward, _ = self._backward_pass_adaptive(y_forward, traj[::-1])
        return y_backward

    def _process_channel_auto(self, x, ch):
        """Auto-mode zero-phase for one channel."""
        y_forward, learned, _ = self._forward_pass_lms(x, ch)
        self._learned_trajectory = learned
        y_backward, _ = self._backward_pass_adaptive(y_forward, learned[::-1])
        return y_backward

    def _filter_block(self, data, freq_trajectory=None):
        """Zero-phase forward-backward filtering of one multi-channel block.

        This is the method called by :class:`CascadeNotchFilter` to push
        data through a single stage of the serial chain.

        Parameters
        ----------
        data : numpy.ndarray
            Input block, shape ``(num_samples, num_channels)``.
        freq_trajectory : numpy.ndarray or None
            Per-sample frequency for external mode, shape ``(num_samples,)``.

        Returns
        -------
        numpy.ndarray
            Zero-phase filtered block, same shape as *data*.
        """
        num_channels = data.shape[1]
        self._ensure_states(num_channels)

        if self.mode == 'auto' and not self._initialized:
            self._initialize_from_fft(data)
            self._initialized = True
            # Sync per-channel state with FFT-detected frequency
            init_f = self.f_notch
            self._current_freq_per_ch[:] = init_f
            self._freq_history[:] = init_f
            self._freq_history_sum[:] = init_f * self.smooth_window

        b, a = self._compute_coefficients()
        output = np.zeros_like(data)
        for ch in range(num_channels):
            if freq_trajectory is not None:
                output[:, ch] = self._process_channel_external(
                    data[:, ch],
                    freq_trajectory,
                )
            elif self.mode == 'auto':
                output[:, ch] = self._process_channel_auto(data[:, ch], ch)
            else:
                output[:, ch] = self._process_channel(data[:, ch], b, a)

        if self.mode == 'auto':
            self._learned_frequencies = self._learned_trajectory

        return output

    def _resolve_trajectory(self):
        """Resolve the effective frequency trajectory.

        Priority:
        1. *freq_source* (streaming) - exhausted into an array.
        2. ``None`` - static or auto mode handles internally.
        """
        if self.freq_source is not None:
            blocks = list(self.freq_source.result(self.num_samples))
            return np.concatenate([np.asarray(b).ravel() for b in blocks])

        return None

    def _process_signal(self):
        """Process the entire signal with zero-phase filtering.

        Collects all blocks from *source*, resolves the frequency
        trajectory (if any), and delegates to :meth:`_filter_block`.

        Returns
        -------
        numpy.ndarray
            Filtered signal, shape ``(num_samples, num_channels)``.
        """
        blocks = list(self.source.result(self.num_samples))
        signal = np.vstack(blocks)

        traj = self._resolve_trajectory()
        if traj is not None:
            traj = traj[: signal.shape[0]]

        return self._filter_block(signal, freq_trajectory=traj)

    def _reset_streaming(self):
        """Initialise (or reset) the internal state for streaming mode.

        Must be called before the first :meth:`_filter_block_streaming`
        call.

        The backward pass needs future samples to settle.  Each call to
        :meth:`_filter_block_streaming` uses the *current* block as a
        one-block settling pad and returns the zero-phase output for the
        *previous* block.  Output is therefore delayed by exactly one
        block.  After the last block, call :meth:`_flush_streaming` to
        retrieve the final block.
        """
        self._stream_fwd_zi = None
        self._stream_prev_fwd = None
        self._stream_prev_traj = None
        self._stream_prev_learned = None

    def _filter_block_streaming(self, data, freq_trajectory=None):
        """Streaming zero-phase forward-backward filter for one block.

        Parameters
        ----------
        data : numpy.ndarray
            New input block, shape ``(num_samples, num_channels)``.
        freq_trajectory : numpy.ndarray or None
            Per-sample frequency for this block (external mode only).

        Returns
        -------
        numpy.ndarray or None
            Zero-phase filtered output for the **previous** block (same
            shape as *data*), or ``None`` on the very first call when the
            block is still being buffered.
        """
        num_samples, num_channels = data.shape
        self._ensure_states(num_channels)

        if self.mode == 'auto' and not self._initialized:
            self._initialize_from_fft(data)
            self._initialized = True

        if self._stream_fwd_zi is None:
            self._stream_fwd_zi = [None] * num_channels

        # Forward pass (streaming, state carried)
        fwd_out = np.zeros_like(data)
        cur_learned = None

        for ch in range(num_channels):
            if self.mode == 'auto':
                fwd_out[:, ch], learned_ch, self._stream_fwd_zi[ch] = self._forward_pass_lms(
                    data[:, ch], ch, self._stream_fwd_zi[ch]
                )
                if cur_learned is None:
                    cur_learned = np.zeros((num_samples, num_channels))
                cur_learned[:, ch] = learned_ch
            elif freq_trajectory is not None:
                fwd_out[:, ch], self._stream_fwd_zi[ch] = self._forward_pass_adaptive(
                    data[:, ch],
                    freq_trajectory,
                    self._stream_fwd_zi[ch],
                )
            else:
                b, a = self._compute_coefficients()
                fwd_out[:, ch], self._stream_fwd_zi[ch] = self._forward_pass(
                    data[:, ch], b, a, self._stream_fwd_zi[ch]
                )

        # First block: buffer only, no output yet
        if self._stream_prev_fwd is None:
            self._stream_prev_fwd = fwd_out
            self._stream_prev_traj = freq_trajectory
            self._stream_prev_learned = cur_learned
            return None

        # Backward pass: zero-phase the *previous* block
        prev_num_samples = self._stream_prev_fwd.shape[0]
        b, a = self._compute_coefficients()
        output = np.zeros_like(self._stream_prev_fwd)

        for ch in range(num_channels):
            context = np.concatenate(
                [
                    self._stream_prev_fwd[:, ch],
                    fwd_out[:, ch],
                ]
            )

            if self.mode == 'auto':
                traj_context = np.concatenate(
                    [
                        self._stream_prev_learned[:, ch],
                        cur_learned[:, ch],
                    ]
                )
                bwd, _ = self._backward_pass_adaptive(context, traj_context[::-1])
            elif self._stream_prev_traj is not None and freq_trajectory is not None:
                traj_context = np.concatenate(
                    [
                        self._stream_prev_traj,
                        freq_trajectory,
                    ]
                )
                bwd, _ = self._backward_pass_adaptive(context, traj_context[::-1])
            elif self._stream_prev_traj is not None:
                hold = np.full(num_samples, self._stream_prev_traj[-1])
                traj_context = np.concatenate([self._stream_prev_traj, hold])
                bwd, _ = self._backward_pass_adaptive(context, traj_context[::-1])
            else:
                bwd, _ = self._backward_pass(context, b, a)

            output[:, ch] = bwd[:prev_num_samples]

        self._stream_prev_fwd = fwd_out
        self._stream_prev_traj = freq_trajectory
        self._stream_prev_learned = cur_learned

        return output

    def _flush_streaming(self):
        """Retrieve the final buffered block after no more input arrives.

        Returns
        -------
        numpy.ndarray or None
            Zero-phase output for the last block, or ``None`` if the
            streaming buffer is already empty.
        """
        if self._stream_prev_fwd is None:
            return None

        num_channels = self._stream_prev_fwd.shape[1]
        b, a = self._compute_coefficients()
        output = np.zeros_like(self._stream_prev_fwd)

        for ch in range(num_channels):
            x = self._stream_prev_fwd[:, ch]

            if self.mode == 'auto' and self._stream_prev_learned is not None:
                bwd, _ = self._backward_pass_adaptive(
                    x,
                    self._stream_prev_learned[:, ch][::-1],
                )
            elif self._stream_prev_traj is not None:
                bwd, _ = self._backward_pass_adaptive(
                    x,
                    self._stream_prev_traj[::-1],
                )
            else:
                bwd, _ = self._backward_pass(x, b, a)

            output[:, ch] = bwd

        self._stream_prev_fwd = None
        self._stream_prev_traj = None
        self._stream_prev_learned = None

        return output

    def _filter_signal_two_pass(self, signal, traj):
        """Forward + backward sweep with state propagation across blocks.

        Produces results identical to single-shot filtfilt while keeping
        the per-block working set small.  For static mode the signal is
        odd-extension padded *once* at the signal boundaries (not per block)
        and unpadded after the backward sweep.

        Parameters
        ----------
        signal : numpy.ndarray
            Input signal, shape ``(total_samples, num_channels)``.
        traj : numpy.ndarray or None
            Per-sample frequency trajectory, shape ``(total_samples,)``.

        Returns
        -------
        numpy.ndarray
            Filtered signal, same shape as *signal*.
        """
        total_samples, num_channels = signal.shape
        block = self.block_size
        is_static = traj is None
        b, a = self._compute_coefficients()

        if is_static:
            padlen = self._compute_padlen()
            signal_work = np.column_stack([self._pad_signal(signal[:, ch]) for ch in range(num_channels)])
        else:
            padlen = 0
            signal_work = signal

        work_len = signal_work.shape[0]

        # Forward sweep (left -> right)
        forward_out = np.empty_like(signal_work)
        fwd_zi = [None] * num_channels

        for start in range(0, work_len, block):
            end = min(start + block, work_len)
            for ch in range(num_channels):
                x = signal_work[start:end, ch]
                if is_static:
                    forward_out[start:end, ch], fwd_zi[ch] = self._forward_pass(x, b, a, fwd_zi[ch])
                else:
                    forward_out[start:end, ch], fwd_zi[ch] = self._forward_pass_adaptive(
                        x, traj[start:end], fwd_zi[ch]
                    )

        # Backward sweep (right -> left)
        output_work = np.empty_like(signal_work)
        bwd_zi = [None] * num_channels

        for start in reversed(range(0, work_len, block)):
            end = min(start + block, work_len)
            for ch in range(num_channels):
                x = forward_out[start:end, ch]
                if is_static:
                    output_work[start:end, ch], bwd_zi[ch] = self._backward_pass(x, b, a, bwd_zi[ch])
                else:
                    output_work[start:end, ch], bwd_zi[ch] = self._backward_pass_adaptive(
                        x,
                        traj[start:end][::-1],
                        bwd_zi[ch],
                    )

        if is_static:
            return output_work[padlen : padlen + total_samples]
        return output_work

    def result(self, num):
        """Yield zero-phase filtered signal in chunks of *num* samples.

        By default a streaming approach is used: source blocks are
        forward-filtered and buffered; once ``num_lookahead_blocks``
        blocks of future context are available the backward pass is
        executed on the oldest block and the result yielded.

        Set ``batch_mode=True`` to collect the full signal before
        processing (original behaviour, required for very short signals
        or when exact :func:`scipy.signal.filtfilt` equivalence is
        needed).

        Parameters
        ----------
        num : int
            Requested samples per yielded block.

        Yields
        ------
        numpy.ndarray
            Filtered blocks, shape ``(n, num_channels)`` with ``n <= num``.
        """
        # Reset transient state at the start of each pass.
        self._zi = None
        self._beta_state = None
        self._current_freq_per_ch = None
        self._freq_history = None
        self._freq_history_sum = None
        self._f0_per_ch = None
        self._learned_frequencies = np.zeros(0)
        self._learned_trajectory = np.array([])
        self._initialized = False
        self._current_position = 0

        if self.batch_mode:
            yield from self._result_batch(num)
            return

        # Settling-time warning
        settling = int(7 / abs(np.log(self.pole_radius)))
        buffer_length = self.num_lookahead_blocks * num
        if settling > buffer_length:
            _warn(
                f'Settling time ({settling} samples) exceeds lookahead '
                f'buffer ({buffer_length} samples). Consider increasing '
                f'num_lookahead_blocks or using batch_mode=True.',
                stacklevel=2,
            )

        yield from self._result_streaming(num)

    def _result_batch(self, num):
        """Batch path: collect entire signal, process, yield chunks."""
        blocks = list(self.source.result(self.num_samples))
        signal = np.vstack(blocks)
        total_samples = signal.shape[0]

        traj = self._resolve_trajectory()
        if traj is not None:
            traj = traj[:total_samples]

        # Auto mode or short signal: single-shot
        if self.mode == 'auto' or total_samples <= self.block_size:
            output = self._filter_block(signal, freq_trajectory=traj)
            for i in range(0, output.shape[0], num):
                yield output[i : i + num]
            return

        # Static / external: stateful two-pass sweep
        output = self._filter_signal_two_pass(signal, traj)
        for i in range(0, total_samples, num):
            yield output[i : i + num]

    def _result_streaming(self, num):
        """Streaming path: deque-based forward-backward with lookahead.

        Forward-filters each source block (keeping IIR state across
        blocks) and appends the result to a deque.  Once the deque holds
        more than ``num_lookahead_blocks`` entries, the oldest block is
        backward-filtered using all newer entries as settling context and
        yielded.  Remaining blocks are flushed at the end.
        """
        L = self.num_lookahead_blocks
        fwd_buffer = deque()
        traj_buffer = deque()
        learned_buffer = deque()

        fwd_zi = None
        b, a = self._compute_coefficients()

        # Per-block trajectory iteration
        freq_iter = None
        if self.freq_source is not None:
            freq_iter = self.freq_source.result(num)

        for source_block in self.source.result(num):
            n_samp, n_ch = source_block.shape
            self._ensure_states(n_ch)

            if fwd_zi is None:
                fwd_zi = [None] * n_ch

            # Trajectory slice for this block
            traj_slice = None
            if freq_iter is not None:
                with contextlib.suppress(StopIteration):
                    traj_slice = np.asarray(next(freq_iter)).ravel()

            # Auto-mode FFT init on first block
            if self.mode == 'auto' and not self._initialized:
                self._initialize_from_fft(source_block)
                self._initialized = True
                init_f = self.f_notch
                self._current_freq_per_ch[:] = init_f
                self._freq_history[:] = init_f
                self._freq_history_sum[:] = init_f * self.smooth_window

            # Forward pass (stateful across blocks)
            fwd_out = np.zeros_like(source_block)
            cur_learned = None

            for ch in range(n_ch):
                if self.mode == 'auto':
                    fwd_out[:, ch], learned_ch, fwd_zi[ch] = self._forward_pass_lms(
                        source_block[:, ch], ch, fwd_zi[ch],
                    )
                    if cur_learned is None:
                        cur_learned = np.zeros((n_samp, n_ch))
                    cur_learned[:, ch] = learned_ch
                elif traj_slice is not None:
                    fwd_out[:, ch], fwd_zi[ch] = self._forward_pass_adaptive(
                        source_block[:, ch], traj_slice, fwd_zi[ch],
                    )
                else:
                    fwd_out[:, ch], fwd_zi[ch] = self._forward_pass(
                        source_block[:, ch], b, a, fwd_zi[ch],
                    )

            fwd_buffer.append(fwd_out)
            traj_buffer.append(traj_slice)
            learned_buffer.append(cur_learned)

            # Yield oldest block once buffer exceeds lookahead
            if len(fwd_buffer) > L:
                yield self._backward_pass_buffered(
                    fwd_buffer, traj_buffer, learned_buffer, b, a,
                )

        # Flush remaining buffered blocks
        while fwd_buffer:
            yield self._backward_pass_buffered(
                fwd_buffer, traj_buffer, learned_buffer, b, a,
            )

    def _backward_pass_buffered(self, fwd_buffer, traj_buffer, learned_buffer, b, a):
        """Pop oldest block from buffer and backward-pass with context.

        Uses all remaining blocks in the buffer as settling context for
        the backward IIR pass, then returns the output trimmed to the
        oldest block's length.
        """
        oldest_fwd = fwd_buffer.popleft()
        oldest_traj = traj_buffer.popleft()
        oldest_learned = learned_buffer.popleft()

        n_oldest = oldest_fwd.shape[0]
        n_ch = oldest_fwd.shape[1]

        # Context: oldest + all remaining (lookahead) blocks
        context = np.vstack([oldest_fwd] + list(fwd_buffer)) if fwd_buffer else oldest_fwd

        output = np.zeros((n_oldest, n_ch), dtype=oldest_fwd.dtype)

        for ch in range(n_ch):
            if self.mode == 'auto' and oldest_learned is not None:
                parts = [oldest_learned[:, ch]]
                for lb in learned_buffer:
                    if lb is not None:
                        parts.append(lb[:, ch])
                learned_ctx = np.concatenate(parts)
                bwd, _ = self._backward_pass_adaptive(
                    context[:, ch], learned_ctx[::-1],
                )
            elif oldest_traj is not None:
                parts = [oldest_traj]
                for tb in traj_buffer:
                    if tb is not None:
                        parts.append(tb)
                traj_ctx = np.concatenate(parts)
                bwd, _ = self._backward_pass_adaptive(
                    context[:, ch], traj_ctx[::-1],
                )
            else:
                bwd, _ = self._backward_pass(context[:, ch], b, a)

            output[:, ch] = bwd[:n_oldest]

        return output


class CascadeNotchFilter(TimeOut):
    """S x M serial cascade of notch filters across K channels.

    Manages a bank of filters to suppress multiple harmonics from multiple
    sources simultaneously. Supports both static (fixed frequencies) and
    adaptive (time-varying) modes.

    The filter bank applies S x M notch filters to each of K channels,
    where:

    - *S* = number of independent tonal sources
    - *M* = number of harmonics per source
    - *K* = number of input channels

    Internally holds a list of S x M :class:`AdaptiveNotchFilter` instances.
    The :meth:`result` method passes each block through the chain in
    sequence: each filter's output feeds the next.  Per-channel filter
    states are maintained inside each child filter.

    Notes
    -----
    Serial cascade provides sharper notch response but accumulates phase
    delay (for single-pass filtering).  When using zero-phase filtering
    (``zero_phase=True``), phase considerations are moot.

    The filter bank is initialised lazily on the first call to
    :meth:`result`.

    The cascade and its joint optimisation follow :cite:`Harvey2019`; the
    per-stage LMS update is based on :cite:`Nehorai1985` with the drift
    monitoring of :cite:`TanJiang2009`.

    See Also
    --------
    :class:`AdaptiveNotchFilter` : The single-notch building block.
    :class:`ZeroPhaseNotchFilter` : Used as child when ``zero_phase=True``.
    """

    #: Number of independent tonal sources (*S*).
    num_sources = Int(1, desc='number of independent tonal sources (S)')

    #: Number of harmonics per source (*M*).
    harmonics_per_source = Int(1, desc='number of harmonics per source (M)')

    #: Initial / static center frequencies with shape ``(S, M)`` in Hz.
    #: For source *s* and harmonic *m*: ``frequencies[s, m]``.
    frequencies = CArray(dtype=np.float64, desc='center frequencies, shape (S, M) in Hz')

    #: Streaming frequency source for external-mode frequency tracking.
    #: Must have a ``result(num)`` method yielding
    #: ``(num_samples, S*M)`` frequency blocks.
    freq_source = Instance(
        SamplesGenerator,
        desc='Streaming source with result(num) yielding (num_samples, S*M) frequency blocks '
        'for external-mode frequency tracking.',
    )

    #: Pole radius controlling notch width (0 < r < 1, default 0.95).
    #:
    #: Accepts either a scalar (applied to every cascade stage) or an
    #: array specifying per-stage values.  Accepted array shapes are
    #: ``(harmonics_per_source,)`` - same schedule for every source - or
    #: ``(num_sources, harmonics_per_source)`` for full per-stage control.
    #: Per-stage values are only honoured in external and static modes;
    #: auto (LMS) mode falls back to the array mean.
    pole_radius = Union(
        Float(0.95),
        CArray(dtype=np.float64),
        desc='pole radius per cascade stage - scalar, (M,), or (S, M)',
    )

    #: Adaptation mode. ``None`` infers from *freq_source*.
    mode = Enum(None, 'external', 'auto', desc="adaptation mode: None, 'external', or 'auto'")

    #: LMS step size (float) or per-source schedule (list).
    mu = Union(Float, List, desc='LMS step size for auto mode')

    #: Leak factor for recursive LMS gradient
    #: (0 = instantaneous, 1 = full recursive).
    gradient_leak = Float(0.0, desc='leak factor for recursive LMS gradient')

    #: Moving-average window for frequency smoothing in auto mode.
    smooth_window = Int(256, desc='moving-average window for frequency smoothing')

    #: When ``True``, children are :class:`ZeroPhaseNotchFilter` instances.
    zero_phase = Bool(False, desc='use zero-phase forward-backward filtering')

    #: Core block size for zero-phase block iteration.
    block_size = Int(4096, desc='core block size for zero-phase iteration')

    #: Number of lookahead blocks used as settling context in
    #: streaming zero-phase mode.  Higher values improve backward-pass
    #: settling at the cost of increased latency (``num_lookahead_blocks``
    #: blocks).  A value of 1 is sufficient when ``block_size`` exceeds the
    #: IIR settling time (about ``7 / abs(log(pole_radius))`` samples).
    num_lookahead_blocks = Int(
        1, desc='number of lookahead blocks for streaming zero-phase settling context',
    )

    #: Use thesis-aligned harmonic cascade LMS
    #: (shared theta per source, joint optimisation).
    joint_lms = Bool(False, desc='use joint harmonic cascade LMS optimisation')

    # Internal state (transient, not part of digest). The child filter
    # bank and the joint-LMS working arrays are (re)built in result().
    _filters = List()
    _filters_initialized = Bool(False)
    _current_position = Int(0)
    _jl_current_freq = Any()
    _jl_zi = Any()
    _jl_source_state = Any()
    _jl_harm_grad_state = Any()
    _jl_grad_prop_zi = Any()
    _jl_freq_history = Any()
    _jl_freq_history_sum = Any()
    _jl_step_sizes = Any()
    _jl_f0 = Any()
    _jl_learned = Any()
    _jl_initialized = Bool(False)
    _stream_fwd_zi = Any()
    _stream_jl_initialized = Bool(False)
    _stream_fwd_buf = Any()
    _stream_buf_full = Bool(False)

    #: A unique identifier for the filter, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'source.digest',
            'num_sources',
            'harmonics_per_source',
            'frequencies',
            'freq_source.digest',
            'pole_radius',
            'mode',
            'mu',
            'smooth_window',
            'gradient_leak',
            'zero_phase',
            'block_size',
            'num_lookahead_blocks',
            'joint_lms',
        ]
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property
    def learned_frequencies(self):
        """Per-sample learned frequencies from the last block.

        When :attr:`joint_lms` is ``True`` and joint LMS was used, returns a
        list of length *S* where each element is the fundamental frequency
        trajectory for that source, shape ``(N,)``.

        Otherwise returns one array per filter in the cascade (length S x M).
        """
        if self.joint_lms and self._jl_learned is not None:
            return [self._jl_learned[s, :] for s in range(self.num_sources)]
        return [f._learned_frequencies for f in self._filters]

    def _pole_radii(self):
        """Return the per-stage pole radii as an ``(S, M)`` ``float64`` array.

        Broadcasts a scalar :attr:`pole_radius` to all stages; accepts
        ``(M,)`` (same schedule per source) or ``(S, M)`` arrays.
        """
        S = self.num_sources
        M = self.harmonics_per_source
        pr = self.pole_radius
        if np.isscalar(pr) or (isinstance(pr, np.ndarray) and pr.shape == ()):
            return np.full((S, M), float(pr), dtype=np.float64)
        arr = np.asarray(pr, dtype=np.float64)
        if arr.ndim == 1 and arr.shape[0] == M:
            return np.broadcast_to(arr[np.newaxis, :], (S, M)).copy()
        if arr.ndim == 2 and arr.shape == (S, M):
            return arr.astype(np.float64, copy=True)
        msg = f'pole_radius array shape {arr.shape} does not match ({M},) or ({S}, {M})'
        raise ValueError(msg)

    def _scalar_pole_radius(self):
        """Return a representative scalar for kernels that don't support per-stage r.

        Used by the joint-LMS kernels and by ``_monitor_and_reset_joint``.
        """
        pr = self.pole_radius
        if np.isscalar(pr) or (isinstance(pr, np.ndarray) and pr.shape == ()):
            return float(pr)
        return float(np.asarray(pr, dtype=np.float64).mean())

    def _initialize_joint_lms_state(self, num_channels):
        """Allocate state arrays for the joint harmonic cascade LMS kernel."""
        S = self.num_sources
        M = self.harmonics_per_source
        K = num_channels
        W = self.smooth_window

        fundamentals = np.zeros(S, dtype=np.float64)
        freqs = self.frequencies
        if freqs.size > 0:
            for s in range(S):
                fundamentals[s] = freqs[s, 0]
        else:
            fundamentals[:] = 100.0

        self._jl_current_freq = fundamentals.copy()
        self._jl_zi = np.zeros((S, M, K, 2), dtype=np.float64)
        self._jl_source_state = np.zeros((S, 1), dtype=np.float64)
        self._jl_harm_grad_state = np.zeros((S, M, K, 4), dtype=np.float64)
        self._jl_grad_prop_zi = np.zeros((S, S, M, K, 2), dtype=np.float64)

        self._jl_freq_history = np.zeros((S, W), dtype=np.float64)
        self._jl_freq_history_sum = np.zeros(S, dtype=np.float64)
        for s in range(S):
            self._jl_freq_history[s, :] = fundamentals[s]
            self._jl_freq_history_sum[s] = fundamentals[s] * W

        step_sizes = np.zeros(S, dtype=np.float64)
        if isinstance(self.mu, list):
            for s in range(S):
                step_sizes[s] = self.mu[s] if s < len(self.mu) else self.mu[0]
        elif self.mu is not None:
            step_sizes[:] = float(self.mu)

        self._jl_step_sizes = step_sizes
        self._jl_f0 = fundamentals.copy()
        self._jl_initialized = True

    def _get_effective_mode(self):
        """Infer mode for cascade operation."""
        if self.mode is not None:
            return self.mode
        if self.freq_source is not None:
            return 'external'
        return None

    def _find_spectral_peaks(self, block, n_peaks=4):
        """FFT-based spectral peak detection with parabolic interpolation.

        Parameters
        ----------
        block : numpy.ndarray
            Input signal, shape ``(num_samples, num_channels)``.
            Only channel 0 is used.
        n_peaks : int
            Maximum number of peaks to return.

        Returns
        -------
        list of tuple
            ``(freq_hz, power_db)`` pairs sorted by descending power.
        """
        return find_spectral_peaks(
            block[:, 0], self.sample_freq, n_peaks=n_peaks,
        )

    def _monitor_and_reset_joint(self, current_block):
        """Global-minimum monitoring and reset for joint LMS.

        Implements the strategy of Tan & Jiang (2009, section 2.2).  For each
        source whose smoothed fundamental frequency has drifted beyond
        ``delta_f_max``, performs an FFT peak search and resets that source's
        frequency tracking state to the closest detected peak.

        Parameters
        ----------
        current_block : numpy.ndarray
            Raw input block, shape ``(num_samples, num_channels)``.
        """
        if self._jl_f0 is None or self._jl_current_freq is None:
            return
        S = self.num_sources
        W = self.smooth_window
        # delta_f_max = 0.25*(1-r)*fs/pi  (quarter of the notch -3 dB bandwidth).
        # Beyond this drift the LMS gradient becomes unreliable and a global
        # FFT search is triggered. (Tan & Jiang 2009, section II; DOI: 10.1109/MSP.2009.934189)
        delta_f_max = 0.25 * (1.0 - self._scalar_pole_radius()) * self.sample_freq / np.pi

        needs_reset = [s for s in range(S) if abs(self._jl_current_freq[s] - self._jl_f0[s]) > delta_f_max]
        if not needs_reset:
            return

        peaks = self._find_spectral_peaks(
            current_block,
            n_peaks=max(S, len(needs_reset)) * 2,
        )

        for s in needs_reset:
            if not peaks:
                break
            f_ref = self._jl_f0[s]
            closest_idx = min(range(len(peaks)), key=lambda i: abs(peaks[i][0] - f_ref))
            new_freq = peaks[closest_idx][0]
            peaks.pop(closest_idx)

            self._jl_current_freq[s] = new_freq
            self._jl_freq_history[s, :] = new_freq
            self._jl_freq_history_sum[s] = new_freq * W
            self._jl_harm_grad_state[s, :, :, :] = 0.0
            self._jl_f0[s] = new_freq

    def _initialize_filters(self):
        """Create S x M filter instances."""
        if self._filters_initialized:
            return

        effective_mode = self._get_effective_mode()
        filter_cls = ZeroPhaseNotchFilter if self.zero_phase else AdaptiveNotchFilter
        freqs = self.frequencies
        radii = self._pole_radii()  # (S, M)

        for s in range(self.num_sources):
            for m in range(self.harmonics_per_source):
                filter_idx = s * self.harmonics_per_source + m
                r_sm = float(radii[s, m])

                init_freq = freqs[s, m] if freqs.size > 0 else 100.0 * (m + 1)

                if effective_mode == 'auto':
                    if isinstance(self.mu, list):
                        step = self.mu[filter_idx] if filter_idx < len(self.mu) else self.mu[0]
                    elif self.mu is not None:
                        step = self.mu
                    else:
                        step = 0.001
                    f = filter_cls(
                        f_notch=init_freq,
                        pole_radius=r_sm,
                        mode='auto',
                        mu=step,
                        gradient_leak=self.gradient_leak,
                        smooth_window=self.smooth_window,
                        source=self.source,
                    )
                elif effective_mode == 'external':
                    f = filter_cls(
                        f_notch=init_freq,
                        pole_radius=r_sm,
                        mode='external',
                        source=self.source,
                    )
                else:
                    f = filter_cls(
                        f_notch=init_freq,
                        pole_radius=r_sm,
                        mode=None,
                        source=self.source,
                    )

                self._filters.append(f)

        self._filters_initialized = True

    def _result_zero_phase(self, num):
        """Zero-phase result path: full-signal batch processing.

        Collects the entire signal, applies the forward cascade, then
        the backward cascade.  No streaming latency - the full signal
        must be available before processing begins.

        Parameters
        ----------
        num : int
            Samples per yielded output block.

        Yields
        ------
        numpy.ndarray
            Filtered blocks, shape ``(n, num_channels)`` with ``n <= num``.
        """
        effective_mode = self._get_effective_mode()

        signal = np.vstack(list(self.source.result(self.num_samples)))
        total_samples = signal.shape[0]

        full_freq = None
        if effective_mode == 'external':
            freq_blocks = list(self.freq_source.result(self.num_samples))
            full_freq = np.vstack([np.asarray(b) for b in freq_blocks])
            if full_freq.ndim == 1:
                full_freq = full_freq[:, np.newaxis]

        # Auto mode
        if effective_mode == 'auto':
            if self.joint_lms:
                num_channels = signal.shape[1]
                self._initialize_joint_lms_state(num_channels)
                S = self.num_sources
                M = self.harmonics_per_source

                r_scalar = self._scalar_pole_radius()
                data_c = np.ascontiguousarray(signal, dtype=np.float64)
                output_forward, learned = iir_harmonic_cascade_lms_kernel(
                    data_c,
                    S,
                    M,
                    r_scalar,
                    self.sample_freq,
                    self._jl_step_sizes,
                    self.smooth_window,
                    self._jl_zi,
                    self._jl_current_freq,
                    self._jl_source_state,
                    self._jl_freq_history,
                    self._jl_freq_history_sum,
                    self._jl_harm_grad_state,
                    self._jl_grad_prop_zi,
                    gradient_leak=self.gradient_leak,
                )
                self._jl_learned = learned

                # Backward pass: reverse the signal, apply S*M time-varying
                # notch filters with the reversed learned frequency trajectories,
                # then reverse again - the two phase shifts cancel, giving
                # zero net phase distortion (forward-backward / filtfilt principle).
                # See: https://ccrma.stanford.edu/~jos/filters/
                radii = self._pole_radii()  # (S, M) - honoured per harmonic in backward pass
                data = output_forward[::-1].copy()
                for s in range(S):
                    freq_s_rev = learned[s, ::-1]
                    for m in range(1, M + 1):
                        cos_theta = np.cos(
                            2.0 * np.pi * m * freq_s_rev / self.sample_freq,
                        )
                        zi_back = np.zeros((num_channels, 2), dtype=np.float64)
                        data = iir_time_varying_kernel(
                            data,
                            cos_theta,
                            float(radii[s, m - 1]),
                            zi_back,
                        )
                output_zp = data[::-1]

                for i in range(0, total_samples, num):
                    yield output_zp[i : i + num]
                return

            # Fallback: serial greedy cascade
            data = signal.copy()
            for f in self._filters:
                data = f._filter_block(data, freq_trajectory=None)

            for i in range(0, total_samples, num):
                yield data[i : i + num]
            return

        # Static / external: sequential two-pass per filter
        data = signal.copy()
        for filter_idx, f in enumerate(self._filters):
            freq_traj = full_freq[:, filter_idx] if full_freq is not None else None
            data = f._filter_signal_two_pass(data, freq_traj)

        for i in range(0, total_samples, num):
            yield data[i : i + num]

    def result(self, num):
        """Apply serial cascade filter to input signal blocks.

        When :attr:`zero_phase` is ``True`` the entire signal is collected
        first and processed via the zero-phase path.  Otherwise blocks
        stream through the causal filter chain.

        Parameters
        ----------
        num : int
            Number of samples per output block.

        Yields
        ------
        numpy.ndarray
            Filtered signal blocks with shape ``(num, num_channels)``.

        Raises
        ------
        ValueError
            If mode is ``'external'`` but no *freq_source* is provided, or
            if *freq_source* yields blocks with wrong shape or runs out
            early.
        """
        effective_mode = self._get_effective_mode()

        if effective_mode == 'external' and self.freq_source is None:
            msg = (
                "mode is 'external' but no freq_source provided. "
                "Set freq_source or use mode='auto' / None."
            )
            raise ValueError(msg)

        # Reset transient state at the start of each pass.
        self._filters = []
        self._filters_initialized = False
        self._current_position = 0
        self._jl_zi = None
        self._jl_current_freq = None
        self._jl_source_state = None
        self._jl_freq_history = None
        self._jl_freq_history_sum = None
        self._jl_harm_grad_state = None
        self._jl_grad_prop_zi = None
        self._jl_learned = None
        self._jl_step_sizes = None
        self._jl_f0 = None
        self._jl_initialized = False

        self._initialize_filters()

        if self.zero_phase:
            yield from self._result_zero_phase(num)
            return

        # Causal streaming - joint LMS
        if effective_mode == 'auto' and self.joint_lms:
            first_block = True
            for source_block in self.source.result(num):
                num_samples = source_block.shape[0]
                num_channels = source_block.shape[1]
                if first_block:
                    self._initialize_joint_lms_state(num_channels)
                    first_block = False

                S = self.num_sources
                M = self.harmonics_per_source
                data_c = np.ascontiguousarray(source_block, dtype=np.float64)
                block_output, block_learned = iir_harmonic_cascade_lms_kernel(
                    data_c,
                    S,
                    M,
                    self._scalar_pole_radius(),
                    self.sample_freq,
                    self._jl_step_sizes,
                    self.smooth_window,
                    self._jl_zi,
                    self._jl_current_freq,
                    self._jl_source_state,
                    self._jl_freq_history,
                    self._jl_freq_history_sum,
                    self._jl_harm_grad_state,
                    self._jl_grad_prop_zi,
                    gradient_leak=self.gradient_leak,
                )
                self._jl_learned = block_learned
                self._monitor_and_reset_joint(source_block)
                self._current_position = self._current_position + num_samples
                yield block_output
            return

        # Causal streaming - serial cascade
        num_filters = self.num_sources * self.harmonics_per_source
        freq_iter = self.freq_source.result(num) if effective_mode == 'external' else None

        for source_block in self.source.result(num):
            num_samples = source_block.shape[0]

            freq_block = None
            if freq_iter is not None:
                try:
                    freq_block = np.asarray(next(freq_iter))
                except StopIteration:
                    msg = 'freq_source exhausted before source finished yielding blocks.'
                    raise ValueError(msg) from None
                if freq_block.ndim == 1:
                    freq_block = freq_block[:, np.newaxis]
                if freq_block.shape != (num_samples, num_filters):
                    msg = (
                        f'freq_source block shape {freq_block.shape} does not match '
                        f'expected ({num_samples}, {num_filters}).'
                    )
                    raise ValueError(msg)

            data = source_block.copy()
            for filter_idx, f in enumerate(self._filters):
                freq_traj = freq_block[:, filter_idx] if freq_block is not None else None
                data = f._filter_block(data, freq_trajectory=freq_traj)

            self._current_position = self._current_position + num_samples
            yield data

    def _reset_streaming(self):
        """Initialise streaming state for unified zero-phase cascade.

        Must be called before the first :meth:`_filter_block_streaming`
        call.  The entire S x M cascade shares a configurable lookahead
        buffer (:attr:`num_lookahead_blocks`), so total pipeline latency
        is ``num_lookahead_blocks`` blocks (independent of S and M).

        The forward cascade is processed as a unit for each incoming
        block.  The backward cascade uses the lookahead blocks' forward
        outputs as settling context - mirroring the batch-mode
        implementation in :meth:`_result_zero_phase`.
        """
        self._initialize_filters()
        if not self.zero_phase:
            msg = '_reset_streaming() requires zero_phase=True'
            raise RuntimeError(msg)

        L = self.num_lookahead_blocks
        # Forward-pass IIR state for each of S*M filters, shape (K, 2) each
        self._stream_fwd_zi = None  # list of (K, 2) arrays, one per filter
        # Joint LMS forward state (initialised on first block if needed)
        self._stream_jl_initialized = False
        # Ring buffer of forward cascade outputs + trajectories (length L+1)
        # Once full, the oldest entry is backward-processed using the
        # remaining L entries as settling context.
        self._stream_fwd_buf = deque(maxlen=L + 1)  # (fwd_output, traj)
        self._stream_buf_full = False

    def _forward_cascade(self, data, freq_block=None):
        """Run the full forward cascade on one block.

        Returns ``(forward_output, learned_or_freq_trajectories)``.
        The forward IIR state is carried across calls via
        ``_stream_fwd_zi`` / joint-LMS state arrays.
        """
        effective_mode = self._get_effective_mode()
        num_channels = data.shape[1]
        S = self.num_sources
        M = self.harmonics_per_source

        # --- Joint LMS auto mode ---
        if effective_mode == 'auto' and self.joint_lms:
            if not self._stream_jl_initialized:
                self._initialize_joint_lms_state(num_channels)
                self._stream_jl_initialized = True

            data_c = np.ascontiguousarray(data, dtype=np.float64)
            fwd_out, learned = iir_harmonic_cascade_lms_kernel(
                data_c,
                S,
                M,
                self._scalar_pole_radius(),
                self.sample_freq,
                self._jl_step_sizes,
                self.smooth_window,
                self._jl_zi,
                self._jl_current_freq,
                self._jl_source_state,
                self._jl_freq_history,
                self._jl_freq_history_sum,
                self._jl_harm_grad_state,
                self._jl_grad_prop_zi,
                gradient_leak=self.gradient_leak,
            )
            self._jl_learned = learned
            self._monitor_and_reset_joint(data)
            return fwd_out, learned  # learned shape (S, N)

        # --- Serial forward cascade (static / external / non-joint auto) ---
        if self._stream_fwd_zi is None:
            self._stream_fwd_zi = [None] * len(self._filters)

        current = data.copy()
        num_filters = len(self._filters)
        trajectories = np.zeros((num_filters, data.shape[0]), dtype=np.float64)
        radii_flat = self._pole_radii().reshape(-1)  # (S*M,) row-major s,m

        for filter_idx, f in enumerate(self._filters):
            freq_traj = freq_block[:, filter_idx] if effective_mode == 'external' and freq_block is not None else None

            # Forward pass per channel using child filter helpers
            fwd_out = np.zeros_like(current)
            zi = self._stream_fwd_zi[filter_idx]
            if zi is None:
                zi = np.zeros((num_channels, 2), dtype=np.float64)
            r_stage = float(radii_flat[filter_idx])

            if freq_traj is not None:
                # Time-varying forward pass
                cos_theta = np.cos(
                    2.0 * np.pi * np.ascontiguousarray(freq_traj, dtype=np.float64) / self.sample_freq,
                )
                zi_c = np.ascontiguousarray(zi, dtype=np.float64)
                fwd_out = iir_time_varying_kernel(
                    np.ascontiguousarray(current, dtype=np.float64),
                    cos_theta,
                    r_stage,
                    zi_c,
                )
                self._stream_fwd_zi[filter_idx] = zi_c
                trajectories[filter_idx, :] = freq_traj
            else:
                # Static coefficients
                b, a = f._compute_coefficients()
                for ch in range(num_channels):
                    zi_ch = lfilter_zi(b, a) * current[0, ch] if np.all(zi[ch] == 0.0) else zi[ch]
                    fwd_out[:, ch], zi[ch] = lfilter(b, a, current[:, ch], zi=zi_ch)
                self._stream_fwd_zi[filter_idx] = zi

            current = fwd_out

        return current, trajectories  # trajectories shape (S*M, N)

    def _backward_cascade(self, target_fwd, target_traj, context_entries):
        """Backward cascade with lookahead settling context.

        Concatenates ``[target_fwd, *context_fwds]``, reverses, applies
        all S x M notch filters, reverses back, and returns only the
        ``target_fwd``-sized portion.

        Parameters
        ----------
        target_fwd : numpy.ndarray
            Forward cascade output of the block to be zero-phase filtered.
        target_traj : numpy.ndarray
            Learned/external frequency trajectories for the target block.
        context_entries : list of (fwd, traj) tuples
            Lookahead blocks used as settling context (in chronological
            order, i.e. immediately following the target block).
        """
        num_channels = target_fwd.shape[1]
        target_len = target_fwd.shape[0]
        effective_mode = self._get_effective_mode()
        S = self.num_sources
        M = self.harmonics_per_source

        # Concatenate target + all context blocks, then reverse
        fwd_parts = [target_fwd] + [e[0] for e in context_entries]
        context = np.concatenate(fwd_parts, axis=0)
        data = context[::-1].copy()

        radii = self._pole_radii()  # (S, M)

        if effective_mode == 'auto' and self.joint_lms:
            # trajectories are (S, N) each
            for s in range(S):
                traj_parts = [target_traj[s]] + [e[1][s] for e in context_entries]
                traj_s = np.concatenate(traj_parts)
                freq_s_rev = traj_s[::-1]
                for m in range(1, M + 1):
                    cos_theta = np.cos(
                        2.0 * np.pi * m * freq_s_rev / self.sample_freq,
                    )
                    zi_back = np.zeros((num_channels, 2), dtype=np.float64)
                    data = iir_time_varying_kernel(
                        data, cos_theta, float(radii[s, m - 1]), zi_back,
                    )
        elif effective_mode == 'external':
            # trajectories are (S*M, N) each.  Cascade stages are ordered
            # (source-major, harmonic-minor) - same ordering as _pole_radii().
            num_filters = S * M
            radii_flat = radii.reshape(-1)
            for filter_idx in range(num_filters):
                traj_parts = [target_traj[filter_idx]] + [e[1][filter_idx] for e in context_entries]
                traj = np.concatenate(traj_parts)
                freq_rev = traj[::-1]
                cos_theta = np.cos(
                    2.0 * np.pi * freq_rev / self.sample_freq,
                )
                zi_back = np.zeros((num_channels, 2), dtype=np.float64)
                data = iir_time_varying_kernel(
                    data, cos_theta, float(radii_flat[filter_idx]), zi_back,
                )
        else:
            # Static mode: apply each filter's fixed coefficients
            for f in self._filters:
                b, a = f._compute_coefficients()
                for ch in range(num_channels):
                    zi_ch = lfilter_zi(b, a) * data[0, ch]
                    data[:, ch], _ = lfilter(b, a, data[:, ch], zi=zi_ch)

        output = data[::-1]
        return output[:target_len]

    def _filter_block_streaming(self, data, freq_block=None):
        """Push one block through the unified streaming cascade.

        The entire S x M forward cascade is applied first.  Once
        ``num_lookahead_blocks`` future blocks have been buffered, the
        oldest block is backward-processed using the buffered blocks as
        settling context.

        Pipeline latency is ``num_lookahead_blocks`` blocks (independent
        of S and M).

        Parameters
        ----------
        data : numpy.ndarray
            Input block, shape ``(num_samples, num_channels)``.
        freq_block : numpy.ndarray or None
            Per-sample frequency block for external mode, shape
            ``(num_samples, S*M)``.

        Returns
        -------
        numpy.ndarray or None
            Zero-phase filtered output for a previous block, or
            ``None`` while the lookahead buffer is still filling.
        """
        L = self.num_lookahead_blocks
        buf = self._stream_fwd_buf  # deque(maxlen=L+1)

        # Forward cascade through all S*M filters
        cur_fwd, cur_traj = self._forward_cascade(data, freq_block)
        buf.append((cur_fwd, cur_traj))

        # Buffer still filling (need L+1 entries: 1 target + L context)
        if len(buf) < L + 1:
            return None

        # Oldest entry is the target; remaining L entries are context
        target_fwd, target_traj = buf[0]
        context_entries = [buf[i] for i in range(1, len(buf))]

        output = self._backward_cascade(target_fwd, target_traj, context_entries)

        # Remove oldest (processed) entry
        buf.popleft()

        return output

    def _flush_streaming(self):
        """Drain the lookahead buffer after the last input block.

        Each remaining entry is backward-processed with whatever
        context is still available (shrinking as the buffer drains).

        Returns
        -------
        list of numpy.ndarray
            Up to ``num_lookahead_blocks`` remaining output blocks.
        """
        buf = self._stream_fwd_buf
        if not buf:
            return []

        outputs = []
        while buf:
            target_fwd, target_traj = buf[0]
            context_entries = [buf[i] for i in range(1, len(buf))]
            output = self._backward_cascade(target_fwd, target_traj, context_entries)
            outputs.append(output)
            buf.popleft()

        return outputs
