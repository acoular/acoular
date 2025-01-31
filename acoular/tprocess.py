# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements blockwise processing in the time domain.

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
    MaskedTimeInOut
"""

# imports from other packages
import wave
from abc import abstractmethod
from datetime import datetime, timezone
from os import path
from warnings import warn

import numba as nb
from numpy import (
    append,
    arange,
    argmax,
    argmin,
    argsort,
    array,
    array_equal,
    asarray,
    ceil,
    concatenate,
    cumsum,
    delete,
    empty,
    empty_like,
    exp,
    flatnonzero,
    float64,
    identity,
    inf,
    int16,
    interp,
    linspace,
    mean,
    nan,
    newaxis,
    pi,
    polymul,
    sin,
    sinc,
    split,
    sqrt,
    stack,
    sum,  # noqa: A004
    tile,
    unique,
    zeros,
)
from scipy.fft import irfft, rfft
from scipy.interpolate import CloughTocher2DInterpolator, CubicSpline, LinearNDInterpolator, Rbf, splev, splrep
from scipy.linalg import norm
from scipy.signal import bilinear, butter, sosfilt, sosfiltfilt, tf2sos
from scipy.spatial import Delaunay
from traits.api import (
    Bool,
    CArray,
    CInt,
    Constant,
    Delegate,
    Dict,
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
    on_trait_change,
)

# acoular imports
from .base import SamplesGenerator, TimeOut
from .configuration import config
from .deprecation import deprecated_alias
from .environments import cartToCyl, cylToCart
from .h5files import _get_h5file_class
from .internal import digest, ldigest
from .microphones import MicGeom
from .tools.utils import find_basename


@deprecated_alias({'numchannels_total': 'num_channels_total', 'numsamples_total': 'num_samples_total'})
class MaskedTimeOut(TimeOut):
    """Signal processing block for channel and sample selection.

    This class serves as intermediary to define (in)valid
    channels and samples for any
    :class:`~acoular.sources.SamplesGenerator` (or derived) object.
    It gets samples from :attr:`~acoular.base.TimeOut.source`
    and generates output via the generator :meth:`result`.
    """

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Index of the first sample to be considered valid.
    start = CInt(0, desc='start of valid samples')

    # Index of the last sample to be considered valid.
    stop = Union(None, CInt, desc='stop of valid samples')

    # Channels that are to be treated as invalid.
    invalid_channels = List(int, desc='list of invalid channels')

    # Channel mask to serve as an index for all valid channels, is set automatically.
    channels = Property(depends_on=['invalid_channels', 'source.num_channels'], desc='channel mask')

    # Number of channels in input, as given by :attr:`~acoular.base.TimeOut.source`.
    num_channels_total = Delegate('source', 'num_channels')

    # Number of samples in input, as given by :attr:`~acoular.base.TimeOut.source`.
    num_samples_total = Delegate('source', 'num_samples')

    # Number of valid channels, is set automatically.
    num_channels = Property(
        depends_on=['invalid_channels', 'source.num_channels'], desc='number of valid input channels'
    )

    # Number of valid time samples, is set automatically.
    num_samples = Property(
        depends_on=['start', 'stop', 'source.num_samples'], desc='number of valid samples per channel'
    )

    # Name of the cache file without extension, readonly.
    basename = Property(depends_on=['source.digest'], desc='basename for cache file')

    # internal identifier
    digest = Property(depends_on=['source.digest', 'start', 'stop', 'invalid_channels'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        warn(
            (
                f'The basename attribute of a {self.__class__.__name__} object is deprecated'
                ' and will be removed in a future release!'
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
        return array(allr)

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
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, :attr:`num_channels`).
            The last block may be shorter than num.

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
            buf = empty((num + offset, self.num_channels), dtype=float)
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
    """Class for directly mixing the channels of a multi-channel source.
    Outputs a single channel.
    """

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Amplitude weight(s) for the channels as array. If not set, all channels are equally weighted.
    weights = CArray(desc='channel weights')

    # Number of channels is always one here.
    num_channels = Constant(1)

    # internal identifier
    digest = Property(depends_on=['source.digest', 'weights'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, 1).
            The last block may be shorter than num.

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
            yield sum(weights * block, 1, keepdims=True)


class Trigger(TimeOut):  # pragma: no cover
    """Class for identifying trigger signals.
    Gets samples from :attr:`source` and stores the trigger samples in :meth:`trigger_data`.

    The algorithm searches for peaks which are above/below a signed threshold.
    A estimate for approximative length of one revolution is found via the greatest
    number of samples between the adjacent peaks.
    The algorithm then defines hunks as percentages of the estimated length of one revolution.
    If there are multiple peaks within one hunk, the algorithm just takes one of them
    into account (e.g. the first peak, the peak with extremum value, ...).
    In the end, the algorithm checks if the found peak locations result in rpm that don't
    vary too much.
    """

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Threshold of trigger. Has different meanings for different
    # :attr:`~acoular.tprocess.Trigger.trigger_type`. The sign is relevant.
    # If a sample of the signal is above/below the positive/negative threshold,
    # it is assumed to be a peak.
    # Default is None, in which case a first estimate is used: The threshold
    # is assumed to be 75% of the max/min difference between all extremums and the
    # mean value of the trigger signal. E.g: the mean value is 0 and there are positive
    # extremums at 400 and negative extremums at -800. Then the estimated threshold would be
    # 0.75 * -800 = -600.
    threshold = Float(None)

    # Maximum allowable variation of length of each revolution duration. Default is
    # 2%. A warning is thrown, if any revolution length surpasses this value:
    # abs(durationEachRev - meanDuration) > 0.02 * meanDuration
    max_variation_of_duration = Float(0.02)

    # Defines the length of hunks via lenHunk = hunk_length * maxOncePerRevDuration.
    # If there are multiple peaks within lenHunk, then the algorithm will
    # cancel all but one out (see :attr:`~acoular.tprocess.Trigger.multiple_peaks_in_hunk`).
    # Default is to 0.1.
    hunk_length = Float(0.1)

    # Type of trigger.
    #
    # 'dirac': a single pulse is assumed (sign of :attr:`~acoular.tprocess.Trigger.trigger_type` is
    # important). Sample will trigger if its value is above/below the pos/neg threshold.
    #
    # 'rect' : repeating rectangular functions. Only every second edge is assumed to be a trigger.
    # The sign of :attr:`~acoular.tprocess.Trigger.trigger_type` gives information on which edge
    # should be used (+ for rising edge, - for falling edge). Sample will trigger if the difference
    # between its value and its predecessors value is above/below the pos/neg threshold.
    #
    # Default is 'dirac'.
    trigger_type = Enum('dirac', 'rect')

    # Identifier which peak to consider, if there are multiple peaks in one hunk : (see
    # :attr:`~acoular.tprocess.Trigger.hunk_length`). Default is to 'extremum', : in which case the
    # extremal peak (maximum if threshold > 0, minimum if threshold < 0) is considered.
    multiple_peaks_in_hunk = Enum('extremum', 'first')

    # Tuple consisting of 3 entries:
    #
    # 1.: -Vector with the sample indices of the 1/Rev trigger samples
    #
    # 2.: -maximum of number of samples between adjacent trigger samples
    #
    # 3.: -minimum of number of samples between adjacent trigger samples
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

    # internal identifier
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
        peakLoc = array([], dtype='int')  # all indices which surpasse the threshold
        trigger_data = array([])
        x0 = []
        dSamples = 0
        for triggerSignal in self.source.result(num):
            localTrigger = flatnonzero(triggerFunc(x0, triggerSignal, threshold))
            if len(localTrigger) != 0:
                peakLoc = append(peakLoc, localTrigger + dSamples)
                trigger_data = append(trigger_data, triggerSignal[localTrigger])
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
        multiplePeaksWithinHunk = flatnonzero(peakDist < self.hunk_length * maxPeakDist)
        while len(multiplePeaksWithinHunk) > 0:
            peakLocHelp = multiplePeaksWithinHunk[0]
            indHelp = [peakLocHelp, peakLocHelp + 1]
            if self.multiple_peaks_in_hunk == 'extremum':
                values = trigger_data[indHelp]
                deleteInd = indHelp[argmin(abs(values))]
            elif self.multiple_peaks_in_hunk == 'first':
                deleteInd = indHelp[1]
            peakLoc = delete(peakLoc, deleteInd)
            trigger_data = delete(trigger_data, deleteInd)
            peakDist = peakLoc[1:] - peakLoc[:-1]
            multiplePeaksWithinHunk = flatnonzero(peakDist < self.hunk_length * maxPeakDist)

        # check whether distances between peaks are evenly distributed
        meanDist = mean(peakDist)
        diffDist = abs(peakDist - meanDist)
        faultyInd = flatnonzero(diffDist > self.max_variation_of_duration * meanDist)
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
        xNew = append(x0, x)
        # indPeakHunk = abs(xNew[1:] - xNew[:-1]) > abs(threshold)
        # with above line, every edge would be located
        return self._trigger_value_comp(xNew[1:] - xNew[:-1], threshold)

    def _trigger_value_comp(self, trigger_data, threshold):
        return trigger_data > threshold if threshold > 0.0 else trigger_data < threshold

    def _threshold(self, num):
        if self.threshold is None:  # take a guessed threshold
            # get max and min values of whole trigger signal
            maxVal = -inf
            minVal = inf
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
            argInd = argmax(abs(maxTriggerHelp))
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
        msg = 'result method not implemented yet! Data from source will be passed without transformation.'
        warn(msg, Warning, stacklevel=2)
        yield from self.source.result(num)


class AngleTracker(MaskedTimeOut):
    """Calculates rotation angle and rpm per sample from a trigger signal
    using spline interpolation in the time domain.

    Gets samples from :attr:`trigger` and stores the angle and rpm samples in :meth:`angle` and
    :meth:`rpm`.
    """

    # Trigger data from :class:`acoular.tprocess.Trigger`.
    trigger = Instance(Trigger)

    # internal identifier
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

    # Trigger signals per revolution,
    # defaults to 1.
    trigger_per_revo = Int(1, desc='trigger signals per revolution')

    # Flag to set counter-clockwise (1) or clockwise (-1) rotation,
    # defaults to -1.
    rot_direction = Int(-1, desc='mathematical direction of rotation')

    # Points of interpolation used for spline,
    # defaults to 4.
    interp_points = Int(4, desc='Points of interpolation used for spline')

    # rotation angle in radians for first trigger position
    start_angle = Float(0, desc='rotation angle for trigger position')

    # revolutions per minute for each sample, read-only
    rpm = Property(depends_on=['digest'], desc='revolutions per minute for each sample')

    # average revolutions per minute, read-only
    average_rpm = Property(depends_on=['digest'], desc='average revolutions per minute')

    # rotation angle in radians for each sample, read-only
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
        peakarray = asarray(peakarray)
        return (abs(peakarray - value)).argmin()

    def _to_rpm_and_angle(self):
        """Internal helper function
        Calculates angles in radians for one or more instants in time.

        Current version supports only trigger and sources with the same samplefreq.
        This behaviour may change in future releases
        """
        # init
        ind = 0
        # trigger data
        peakloc, maxdist, mindist = self.trigger.trigger_data
        TriggerPerRevo = self.trigger_per_revo
        rotDirection = self.rot_direction
        num = self.source.num_samples
        samplerate = self.source.sample_freq
        self._rpm = zeros(num)
        self._angle = zeros(num)
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
                splineData = stack(
                    (range(InterpPoints), peakloc[ind // peakdist : ind // peakdist + InterpPoints]),
                    axis=0,
                )
            # spline backwards
            else:
                peakdist = (
                    peakloc[self._find_nearest_idx(peakarray=peakloc, value=ind)]
                    - peakloc[self._find_nearest_idx(peakarray=peakloc, value=ind) - 1]
                )
                splineData = stack(
                    (range(InterpPoints), peakloc[ind // peakdist - InterpPoints : ind // peakdist]),
                    axis=0,
                )
            # calc angles and rpm
            Spline = splrep(splineData[:, :][1], splineData[:, :][0], k=3)
            self._rpm[ind] = splev(ind, Spline, der=1, ext=0) * 60 * samplerate
            self._angle[ind] = (
                splev(ind, Spline, der=0, ext=0) * 2 * pi * rotDirection / TriggerPerRevo + self.start_angle
            ) % (2 * pi)
            # next sample
            ind += 1
        # calculation complete
        self._calc_flag = True

    # reset calc flag if something has changed
    @on_trait_change('digest')
    def _reset_calc_flag(self):
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
        """Returns average revolutions per minute (rpm) over the source samples.

        Returns
        -------
        rpm : float
            rpm in 1/min.

        """
        # trigger indices data
        peakloc = self.trigger.trigger_data[0]
        # calculation of average rpm in 1/min
        return (len(peakloc) - 1) / (peakloc[-1] - peakloc[0]) / self.trigger_per_revo * self.source.sample_freq * 60


class SpatialInterpolator(TimeOut):  # pragma: no cover
    """Base class for spatial interpolation of microphone data.
    Gets samples from :attr:`source` and generates output via the
    generator :meth:`result`.
    """

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # :class:`~acoular.microphones.MicGeom` object that provides the real microphone locations.
    mics = Instance(MicGeom(), desc='microphone geometry')

    # :class:`~acoular.microphones.MicGeom` object that provides the virtual microphone locations.
    mics_virtual = Property(desc='microphone geometry')

    _mics_virtual = Instance(MicGeom, desc='internal microphone geometry;internal usage, read only')

    def _get_mics_virtual(self):
        if not self._mics_virtual and self.mics:
            self._mics_virtual = self.mics
        return self._mics_virtual

    def _set_mics_virtual(self, mics_virtual):
        self._mics_virtual = mics_virtual

    # Interpolation method in spatial domain, defaults to linear
    # linear uses numpy linear interpolation
    # spline uses scipy CloughTocher algorithm
    # rbf is scipy radial basis function with multiquadric, cubic and sinc functions
    # idw refers to the inverse distance weighting algorithm
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

    # spatial dimensionality of the array geometry
    array_dimension = Enum('1D', '2D', 'ring', '3D', 'custom', desc='spatial dimensionality of the array geometry')

    # Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source', 'sample_freq')

    # Number of channels in output.
    num_channels = Property()

    # Number of samples in output, as given by :attr:`source`.
    num_samples = Delegate('source', 'num_samples')

    # Interpolate a point at the origin of the Array geometry
    interp_at_zero = Bool(False)

    # The rotation must be around the z-axis, which means from x to y axis.
    # If the coordinates are not build like that, than this 3x3 orthogonal
    # transformation matrix Q can be used to modify the coordinates.
    # It is assumed that with the modified coordinates the rotation is around the z-axis.
    # The transformation is done via [x,y,z]_mod = Q * [x,y,z]. (default is Identity).
    Q = CArray(dtype=float64, shape=(3, 3), value=identity(3))

    num_IDW = Int(3, desc='number of neighboring microphones, DEFAULT=3')  # noqa: N815

    p_weight = Float(
        2,
        desc='used in interpolation for virtual microphone, weighting power exponent for IDW',
    )

    # Stores the output of :meth:`_virtNewCoord_func`; Read-Only
    _virtNewCoord_func = Property(  # noqa: N815
        depends_on=['mics.digest', 'mics_virtual.digest', 'method', 'array_dimension', 'interp_at_zero'],
    )

    # internal identifier
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
        """Modified Sinc function for Radial Basis function approximation."""
        return sinc((r * self.mics_virtual.mpos.shape[1]) / (pi))

    def _virtNewCoord_func(self, mpos, mpos_virt, method, array_dimension):  # noqa N802
        """Core functionality for getting the interpolation.

        Parameters
        ----------
        mpos : float[3, nPhysicalMics]
            The mic positions of the physical (really existing) mics
        mpos_virt : float[3, nVirtualMics]
            The mic positions of the virtual mics
        method : string
            The Interpolation method to use
        array_dimension : string
            The Array Dimensions in cylinder coordinates

        Returns
        -------
        mesh : List[]
            The items of these lists depend on the reduced interpolation dimension of each subarray.
            If the Array is 1D the list items are:
                1. item : float64[nMicsInSpecificSubarray]
                    Ordered positions of the real mics on the new 1d axis,
                    to be used as inputs for numpys interp.
                2. item : int64[nMicsInArray]
                    Indices identifying how the measured pressures must be evaluated, s.t. the
                    entries of the previous item (see last line) correspond to their initial
                    pressure values.
            If the Array is 2D or 3d the list items are:
                1. item : Delaunay mesh object
                    Delaunay mesh (see scipy.spatial.Delaunay) for the specific Array
                2. item : int64[nMicsInArray]
                    same as 1d case, BUT with the difference, that here the rotational periodicity
                    is handled, when constructing the mesh. Therefore, the mesh could have more
                    vertices than the actual Array mics.

        virtNewCoord : float64[3, nVirtualMics]
            Projection of each virtual mic onto its new coordinates. The columns of virtNewCoord
            correspond to [phi, rho, z].

        newCoord : float64[3, nMics]
            Projection of each mic onto its new coordinates. The columns of newCoordinates
            correspond to [phi, rho, z].
        """
        # init positions of virtual mics in cyl coordinates
        nVirtMics = mpos_virt.shape[1]
        virtNewCoord = zeros((3, nVirtMics))
        virtNewCoord.fill(nan)
        # init real positions in cyl coordinates
        nMics = mpos.shape[1]
        newCoord = zeros((3, nMics))
        newCoord.fill(nan)
        # empty mesh object
        mesh = []

        if self.array_dimension == '1D' or self.array_dimension == 'ring':
            # get projections onto new coordinate, for real mics
            projectionOnNewAxis = cartToCyl(mpos, self.Q)[0]
            indReorderHelp = argsort(projectionOnNewAxis)
            mesh.append([projectionOnNewAxis[indReorderHelp], indReorderHelp])

            # new coordinates of real mics
            indReorderHelp = argsort(cartToCyl(mpos, self.Q)[0])
            newCoord = (cartToCyl(mpos, self.Q).T)[indReorderHelp].T

            # and for virtual mics
            virtNewCoord = cartToCyl(mpos_virt)

        elif self.array_dimension == '2D':  # 2d case0
            # get virtual mic projections on new coord system
            virtNewCoord = cartToCyl(mpos_virt, self.Q)

            # new coordinates of real mics
            indReorderHelp = argsort(cartToCyl(mpos, self.Q)[0])
            newCoord = cartToCyl(mpos, self.Q)

            # scipy delauney triangulation
            # Delaunay
            tri = Delaunay(newCoord.T[:, :2], incremental=True)  #

            if self.interp_at_zero:
                # add a point at zero
                tri.add_points(array([[0], [0]]).T)

            # extend mesh with closest boundary points of repeating mesh
            pointsOriginal = arange(tri.points.shape[0])
            hull = tri.convex_hull
            hullPoints = unique(hull)

            addRight = tri.points[hullPoints]
            addRight[:, 0] += 2 * pi
            addLeft = tri.points[hullPoints]
            addLeft[:, 0] -= 2 * pi

            indOrigPoints = concatenate((pointsOriginal, pointsOriginal[hullPoints], pointsOriginal[hullPoints]))
            # add all hull vertices to original mesh and check which of those
            # are actual neighbors of the original array. Cancel out all others.
            tri.add_points(concatenate([addLeft, addRight]))
            indices, indptr = tri.vertex_neighbor_vertices
            hullNeighbor = empty((0), dtype='int32')
            for currHull in hullPoints:
                neighborOfHull = indptr[indices[currHull] : indices[currHull + 1]]
                hullNeighbor = append(hullNeighbor, neighborOfHull)
            hullNeighborUnique = unique(hullNeighbor)
            pointsNew = unique(append(pointsOriginal, hullNeighborUnique))
            tri = Delaunay(tri.points[pointsNew])  # re-meshing
            mesh.append([tri, indOrigPoints[pointsNew]])

        elif self.array_dimension == '3D':  # 3d case
            # get virtual mic projections on new coord system
            virtNewCoord = cartToCyl(mpos_virt, self.Q)
            # get real mic projections on new coord system
            indReorderHelp = argsort(cartToCyl(mpos, self.Q)[0])
            newCoord = cartToCyl(mpos, self.Q)
            # Delaunay
            tri = Delaunay(newCoord.T, incremental=True)  # , incremental=True,qhull_options =  "Qc QJ Q12"

            if self.interp_at_zero:
                # add a point at zero
                tri.add_points(array([[0], [0], [0]]).T)

            # extend mesh with closest boundary points of repeating mesh
            pointsOriginal = arange(tri.points.shape[0])
            hull = tri.convex_hull
            hullPoints = unique(hull)

            addRight = tri.points[hullPoints]
            addRight[:, 0] += 2 * pi
            addLeft = tri.points[hullPoints]
            addLeft[:, 0] -= 2 * pi

            indOrigPoints = concatenate((pointsOriginal, pointsOriginal[hullPoints], pointsOriginal[hullPoints]))
            # add all hull vertices to original mesh and check which of those
            # are actual neighbors of the original array. Cancel out all others.
            tri.add_points(concatenate([addLeft, addRight]))
            indices, indptr = tri.vertex_neighbor_vertices
            hullNeighbor = empty((0), dtype='int32')
            for currHull in hullPoints:
                neighborOfHull = indptr[indices[currHull] : indices[currHull + 1]]
                hullNeighbor = append(hullNeighbor, neighborOfHull)
            hullNeighborUnique = unique(hullNeighbor)
            pointsNew = unique(append(pointsOriginal, hullNeighborUnique))
            tri = Delaunay(tri.points[pointsNew])  # re-meshing
            mesh.append([tri, indOrigPoints[pointsNew]])

        return mesh, virtNewCoord, newCoord

    def _result_core_func(self, p, phi_delay=None, period=None, Q=Q, interp_at_zero=False):  # noqa: N803, ARG002 (see #226)
        """Performs the actual Interpolation.

        Parameters
        ----------
        p : float[num, nMicsReal]
            The pressure field of the yielded sample at real mics.
        phi_delay : empty list (default) or float[num]
            If passed (rotational case), this list contains the angular delay
            of each sample in rad.
        period : None (default) or float
            If periodicity can be assumed (rotational case)
            this parameter contains the periodicity length

        Returns
        -------
        pInterp : float[num, nMicsVirtual]
            The interpolated time data at the virtual mics

        """
        if phi_delay is None:
            phi_delay = []
        # number of time samples
        nTime = p.shape[0]
        # number of virtual mixcs
        nVirtMics = self.mics_virtual.mpos.shape[1]
        # mesh and projection onto polar Coordinates
        meshList, virtNewCoord, newCoord = self._get_virtNewCoord()
        # pressure interpolation init
        pInterp = zeros((nTime, nVirtMics))
        # Coordinates in cartesian CO - for IDW interpolation
        newCoordCart = cylToCart(newCoord)

        if self.interp_at_zero:
            # interpolate point at 0 in Kartesian CO
            interpolater = LinearNDInterpolator(
                cylToCart(newCoord[:, argsort(newCoord[0])])[:2, :].T,
                p[:, (argsort(newCoord[0]))].T,
                fill_value=0,
            )
            pZero = interpolater((0, 0))
            # add the interpolated pressure at origin to pressure channels
            p = concatenate((p, pZero[:, newaxis]), axis=1)

        # helpfunction reordered for reordered pressure values
        pHelp = p[:, meshList[0][1]]

        # Interpolation for 1D Arrays
        if self.array_dimension == '1D' or self.array_dimension == 'ring':
            # for rotation add phi_delay
            if not array_equal(phi_delay, []):
                xInterpHelp = tile(virtNewCoord[0, :], (nTime, 1)) + tile(phi_delay, (virtNewCoord.shape[1], 1)).T
                xInterp = ((xInterpHelp + pi) % (2 * pi)) - pi  #  shifting phi cootrdinate into feasible area [-pi, pi]
            # if no rotation given
            else:
                xInterp = tile(virtNewCoord[0, :], (nTime, 1))
            # get ordered microphone positions in radiant
            x = newCoord[0]
            for cntTime in range(nTime):
                if self.method == 'linear':
                    # numpy 1-d interpolation
                    pInterp[cntTime] = interp(
                        xInterp[cntTime, :],
                        x,
                        pHelp[cntTime, :],
                        period=period,
                        left=nan,
                        right=nan,
                    )

                elif self.method == 'spline':
                    # scipy cubic spline interpolation
                    SplineInterp = CubicSpline(
                        append(x, (2 * pi) + x[0]),
                        append(pHelp[cntTime, :], pHelp[cntTime, :][0]),
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
            if not array_equal(phi_delay, []):
                xInterpHelp = tile(virtNewCoord[0, :], (nTime, 1)) + tile(phi_delay, (virtNewCoord.shape[1], 1)).T
                xInterp = ((xInterpHelp + pi) % (2 * pi)) - pi  # shifting phi cootrdinate into feasible area [-pi, pi]
            else:
                xInterp = tile(virtNewCoord[0, :], (nTime, 1))

            mesh = meshList[0][0]
            for cntTime in range(nTime):
                # points for interpolation
                newPoint = concatenate((xInterp[cntTime, :][:, newaxis], virtNewCoord[1, :][:, newaxis]), axis=1)
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

                    virtshiftcoord = array([xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2]])
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

                    virtshiftcoord = array([xInterp[cntTime, :], virtNewCoord[1], virtNewCoord[2]])
                    pInterp[cntTime] = rbfi(virtshiftcoord[0], virtshiftcoord[1], virtshiftcoord[2])
                # using inverse distance weighting
                elif self.method == 'IDW':
                    newPoint2_M = newPoint.T
                    newPoint3_M = append(newPoint2_M, zeros([1, self.num_channels]), axis=0)
                    newPointCart = cylToCart(newPoint3_M)
                    for ind in arange(len(newPoint[:, 0])):
                        newPoint_Rep = tile(newPointCart[:, ind], (len(newPoint[:, 0]), 1)).T
                        subtract = newPoint_Rep - newCoordCart
                        normDistance = norm(subtract, axis=0)
                        index_norm = argsort(normDistance)[: self.num_IDW]
                        pHelpNew = pHelp[cntTime, index_norm]
                        normNew = normDistance[index_norm]
                        if normNew[0] < 1e-3:
                            pInterp[cntTime, ind] = pHelpNew[0]
                        else:
                            wholeD = sum(1 / normNew**self.p_weight)
                            weight = (1 / normNew**self.p_weight) / wholeD
                            pInterp[cntTime, ind] = sum(pHelpNew * weight)

        # Interpolation for arbitrary 3D Arrays
        elif self.array_dimension == '3D':
            # check rotation
            if not array_equal(phi_delay, []):
                xInterpHelp = tile(virtNewCoord[0, :], (nTime, 1)) + tile(phi_delay, (virtNewCoord.shape[1], 1)).T
                xInterp = ((xInterpHelp + pi) % (2 * pi)) - pi  # shifting phi cootrdinate into feasible area [-pi, pi]
            else:
                xInterp = tile(virtNewCoord[0, :], (nTime, 1))

            mesh = meshList[0][0]
            for cntTime in range(nTime):
                # points for interpolation
                newPoint = concatenate((xInterp[cntTime, :][:, newaxis], virtNewCoord[1:, :].T), axis=1)

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
        msg = 'result method not implemented yet! Data from source will be passed without transformation.'
        warn(msg, Warning, stacklevel=2)
        yield from self.source.result(num)


class SpatialInterpolatorRotation(SpatialInterpolator):  # pragma: no cover
    """Spatial  Interpolation for rotating sources. Gets samples from :attr:`source`
    and angles from  :attr:`AngleTracker`.Generates output via the generator :meth:`result`.

    """

    # Angle data from AngleTracker class
    angle_source = Instance(AngleTracker)

    # Internal identifier
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
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, :attr:`num_channels`).
            The last block may be shorter than num.

        """
        # period for rotation
        period = 2 * pi
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
    """Spatial linear Interpolation for constantly rotating sources.
    Gets samples from :attr:`source` and generates output via the
    generator :meth:`result`.
    """

    # Rotational speed in rps. Positive, if rotation is around positive z-axis sense,
    # which means from x to y axis.
    rotational_speed = Float(0.0)

    # internal identifier
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
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, :attr:`num_channels`).
            The last block may be shorter than num.

        """
        omega = 2 * pi * self.rotational_speed
        period = 2 * pi
        phiOffset = 0.0
        for timeData in self.source.result(num):
            nTime = timeData.shape[0]
            phi_delay = phiOffset + linspace(0, nTime / self.sample_freq * omega, nTime, endpoint=False)
            interpVal = self._result_core_func(timeData, phi_delay, period, self.Q, interp_at_zero=False)
            phiOffset = phi_delay[-1] + omega / self.sample_freq
            yield interpVal


class Mixer(TimeOut):
    """Mixes the signals from several sources."""

    # Data source; :class:`~acoular.base.SamplesGenerator` object.
    source = Instance(SamplesGenerator)

    # List of additional :class:`~acoular.base.SamplesGenerator` objects
    # to be mixed.
    sources = List(Instance(SamplesGenerator, ()))

    # Sampling frequency of the signal as given by :attr:`source`.
    sample_freq = Delegate('source')

    # Number of channels in output as given by :attr:`source`.
    num_channels = Delegate('source')

    # Number of samples in output as given by :attr:`source`.
    num_samples = Delegate('source')

    # internal identifier
    sdigest = Str()

    @observe('sources.items.digest')
    def _set_sourcesdigest(self, event):  # noqa ARG002
        self.sdigest = ldigest(self.sources)

    # internal identifier
    digest = Property(depends_on=['source.digest', 'sdigest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def validate_sources(self):
        """Validates if sources fit together."""
        if self.source:
            for s in self.sources:
                if self.sample_freq != s.sample_freq:
                    msg = f'Sample frequency of {s} does not fit'
                    raise ValueError(msg)
                if self.num_channels != s.num_channels:
                    msg = f'Channel count of {s} does not fit'
                    raise ValueError(msg)

    def result(self, num):
        """Python generator that yields the output block-wise.
        The output from the source and those in the list
        sources are being added.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Calculates time-depended power of the signal."""

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Squared output of source.
            Yields samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

        """
        for temp in self.source.result(num):
            yield temp * temp


class TimeCumAverage(TimeOut):
    """Calculates cumulative average of the signal, useful for Leq."""

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Cumulative average of the output of source.
            Yields samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

        """
        count = (arange(num) + 1)[:, newaxis]
        for i, temp in enumerate(self.source.result(num)):
            ns, nc = temp.shape
            if not i:
                accu = zeros((1, nc))
            temp = (accu * (count[0] - 1) + cumsum(temp, axis=0)) / count[:ns]
            accu = temp[-1]
            count += ns
            yield temp


class TimeReverse(TimeOut):
    """Calculates the time-reversed signal of a source."""

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Yields samples in blocks of shape (num, num_channels).
            Time-reversed output of source.
            The last block may be shorter than num.

        """
        result_list = []
        result_list.extend(self.source.result(num))
        temp = empty_like(result_list[0])
        h = result_list.pop()
        nsh = h.shape[0]
        temp[:nsh] = h[::-1]
        for h in result_list[::-1]:
            temp[nsh:] = h[: nsh - 1 : -1]
            yield temp
            temp[:nsh] = h[nsh - 1 :: -1]
        yield temp[:nsh]


class Filter(TimeOut):
    """Abstract base class for IIR filters based on scipy lfilter
    implements a filter with coefficients that may be changed
    during processing.

    Should not be instantiated by itself.
    """

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Filter coefficients
    sos = Property()

    def _get_sos(self):
        return tf2sos([1], [1])

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            Delivers the bandpass filtered output of source.
            The last block may be shorter than num.

        """
        sos = self.sos
        zi = zeros((sos.shape[0], 2, self.source.num_channels))
        for block in self.source.result(num):
            sos = self.sos  # this line is useful in case of changes
            # to self.sos during generator lifetime
            block, zi = sosfilt(sos, block, axis=0, zi=zi)
            yield block


class FiltOctave(Filter):
    """Octave or third-octave filter (causal, non-zero phase delay)."""

    # Band center frequency; defaults to 1000.
    band = Float(1000.0, desc='band center frequency')

    # Octave fraction: 'Octave' or 'Third octave'; defaults to 'Octave'.
    fraction = Map({'Octave': 1, 'Third octave': 3}, default_value='Octave', desc='fraction of octave')

    # Filter order
    order = Int(3, desc='IIR filter order')

    sos = Property(depends_on=['band', 'fraction', 'source.digest', 'order'])

    # internal identifier
    digest = Property(depends_on=['source.digest', 'band', 'fraction', 'order'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        # filter design
        fs = self.sample_freq
        # adjust filter edge frequencies for correct power bandwidth (see ANSI 1.11 1987
        # and Kalb,J.T.: "A thirty channel real time audio analyzer and its applications",
        # PhD Thesis: Georgia Inst. of Techn., 1975
        beta = pi / (2 * self.order)
        alpha = pow(2.0, 1.0 / (2.0 * self.fraction_))
        beta = 2 * beta / sin(beta) / (alpha - 1 / alpha)
        alpha = (1 + sqrt(1 + beta * beta)) / beta
        fr = 2 * self.band / fs
        if fr > 1 / sqrt(2):
            msg = f'band frequency too high:{self.band:f},{fs:f}'
            raise ValueError(msg)
        om1 = fr / alpha
        om2 = fr * alpha
        return butter(self.order, [om1, om2], 'bandpass', output='sos')


class FiltFiltOctave(FiltOctave):
    """Octave or third-octave filter with zero phase delay.

    This filter can be applied on time signals.
    It requires large amounts of memory!
    """

    # Filter order (applied for forward filter and backward filter)
    order = Int(2, desc='IIR filter half order')

    # internal identifier
    digest = Property(depends_on=['source.digest', 'band', 'fraction', 'order'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        # filter design
        fs = self.sample_freq
        # adjust filter edge frequencies for correct power bandwidth (see FiltOctave)
        beta = pi / (2 * self.order)
        alpha = pow(2.0, 1.0 / (2.0 * self.fraction_))
        beta = 2 * beta / sin(beta) / (alpha - 1 / alpha)
        alpha = (1 + sqrt(1 + beta * beta)) / beta
        # additional bandwidth correction for double-pass
        alpha = alpha * {6: 1.01, 5: 1.012, 4: 1.016, 3: 1.022, 2: 1.036, 1: 1.083}.get(self.order, 1.0) ** (
            3 / self.fraction_
        )
        fr = 2 * self.band / fs
        if fr > 1 / sqrt(2):
            msg = f'band frequency too high:{self.band:f},{fs:f}'
            raise ValueError(msg)
        om1 = fr / alpha
        om2 = fr * alpha
        return butter(self.order, [om1, om2], 'bandpass', output='sos')

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            Delivers the zero-phase bandpass filtered output of source.
            The last block may be shorter than num.

        """
        sos = self.sos
        data = empty((self.source.num_samples, self.source.num_channels))
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
    """Computes exponential averaging according to IEC 61672-1
    time constant: F -> 125 ms, S -> 1 s
    I (non-standard) -> 35 ms.
    """

    # time weighting
    weight = Map({'F': 0.125, 'S': 1.0, 'I': 0.035}, default_value='F', desc='time weighting')

    sos = Property(depends_on=['weight', 'source.digest'])

    # internal identifier
    digest = Property(depends_on=['source.digest', 'weight'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        alpha = 1 - exp(-1 / self.weight_ / self.sample_freq)
        a = [1, alpha - 1]
        b = [alpha]
        return tf2sos(b, a)


class FiltFreqWeight(Filter):
    """Frequency weighting filter according to IEC 61672."""

    # weighting characteristics
    weight = Enum('A', 'C', 'Z', desc='frequency weighting')

    sos = Property(depends_on=['weight', 'source.digest'])

    # internal identifier
    digest = Property(depends_on=['source.digest', 'weight'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sos(self):
        # s domain coefficients
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        a = polymul([1, 4 * pi * f4, (2 * pi * f4) ** 2], [1, 4 * pi * f1, (2 * pi * f1) ** 2])
        if self.weight == 'A':
            a = polymul(polymul(a, [1, 2 * pi * f3]), [1, 2 * pi * f2])
            b = [(2 * pi * f4) ** 2 * 10 ** (1.9997 / 20), 0, 0, 0, 0]
            b, a = bilinear(b, a, self.sample_freq)
        elif self.weight == 'C':
            b = [(2 * pi * f4) ** 2 * 10 ** (0.0619 / 20), 0, 0]
            b, a = bilinear(b, a, self.sample_freq)
            b = append(b, zeros(2))  # make 6th order
            a = append(a, zeros(2))
        else:
            b = zeros(7)
            b[0] = 1.0
            a = b  # 6th order flat response
        return tf2sos(b, a)


@deprecated_alias({'numbands': 'num_bands'}, read_only=True)
class FilterBank(TimeOut):
    """Abstract base class for IIR filter banks based on scipy lfilter
    implements a bank of parallel filters.

    Should not be instantiated by itself.
    """

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # List of filter coefficients for all filters
    sos = Property()

    # List of labels for bands
    bands = Property()

    # Number of bands
    num_bands = Property()

    # Number of bands
    num_channels = Property()

    @abstractmethod
    def _get_sos(self):
        """Returns a list of second order section coefficients."""

    @abstractmethod
    def _get_bands(self):
        """Returns a list of labels for the bands."""

    @abstractmethod
    def _get_num_bands(self):
        """Returns the number of bands."""

    def _get_num_channels(self):
        return self.num_bands * self.source.num_channels

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            Delivers the bandpass filtered output of source.
            The last block may be shorter than num.

        """
        numbands = self.num_bands
        snumch = self.source.num_channels
        sos = self.sos
        zi = [zeros((sos[0].shape[0], 2, snumch)) for _ in range(numbands)]
        res = zeros((num, self.num_channels), dtype='float')
        for block in self.source.result(num):
            for i in range(numbands):
                res[:, i * snumch : (i + 1) * snumch], zi[i] = sosfilt(sos[i], block, axis=0, zi=zi[i])
            yield res


class OctaveFilterBank(FilterBank):
    """Octave or third-octave filter bank."""

    # Lowest band center frequency index; defaults to 21 (=125 Hz).
    lband = Int(21, desc='lowest band center frequency index')

    # Lowest band center frequency index + 1; defaults to 40 (=8000 Hz).
    hband = Int(40, desc='lowest band center frequency index')

    # Octave fraction: 'Octave' or 'Third octave'; defaults to 'Octave'.
    fraction = Map({'Octave': 1, 'Third octave': 3}, default_value='Octave', desc='fraction of octave')

    # List of filter coefficients for all filters
    ba = Property(depends_on=['lband', 'hband', 'fraction', 'source.digest'])

    # List of labels for bands
    bands = Property(depends_on=['lband', 'hband', 'fraction'])

    # Number of bands
    num_bands = Property(depends_on=['lband', 'hband', 'fraction'])

    # internal identifier
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
        of = FiltOctave(source=self.source, fraction=self.fraction)
        sos = []
        for i in range(self.lband, self.hband, 4 - self.fraction_):
            of.band = 10 ** (i / 10)
            sos_ = of.sos
            sos.append(sos_)
        return sos


@deprecated_alias({'name': 'file'})
class WriteWAV(TimeOut):
    """Saves time signal from one or more channels as mono/stereo/multi-channel
    `*.wav` file.
    """

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Name of the file to be saved. If none is given, the name will be
    # automatically generated from the sources.
    file = File(filter=['*.wav'], desc='name of wave file')

    # Basename for cache, readonly.
    basename = Property(depends_on=['digest'])

    # Channel(s) to save. List can only contain one or two channels.
    channels = List(int, desc='channel to save')

    # internal identifier
    digest = Property(depends_on=['source.digest', 'channels'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        warn(
            (
                f'The basename attribute of a {self.__class__.__name__} object is deprecated'
                ' and will be removed in a future release!'
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return find_basename(self.source)

    def save(self):
        """Saves source output to one- or multiple-channel `*.wav` file."""
        nc = len(self.channels)
        if nc == 0:
            msg = 'No channels given for output.'
            raise ValueError(msg)
        if nc > 2:
            warn(f'More than two channels given for output, exported file will have {nc:d} channels', stacklevel=1)
        if self.file == '':
            name = self.basename
            for nr in self.channels:
                name += f'{nr:d}'
            name += '.wav'
        else:
            name = self.file
        with wave.open(name, 'w') as wf:
            wf.setnchannels(nc)
            wf.setsampwidth(2)
            wf.setframerate(self.source.sample_freq)
            wf.setnframes(self.source.num_samples)
            mx = 0.0
            ind = array(self.channels)
            for data in self.source.result(1024):
                mx = max(abs(data[:, ind]).max(), mx)
            scale = 0.9 * 2**15 / mx
            for data in self.source.result(1024):
                wf.writeframesraw(array(data[:, ind] * scale, dtype=int16).tostring())

    def result(self, num):
        msg = 'result method not implemented yet! Data from source will be passed without transformation.'
        warn(msg, Warning, stacklevel=2)
        yield from self.source.result(num)


@deprecated_alias({'name': 'file', 'numsamples_write': 'num_samples_write', 'writeflag': 'write_flag'})
class WriteH5(TimeOut):
    """Saves time signal as `*.h5` file."""

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Name of the file to be saved. If none is given, the name will be
    # automatically generated from a time stamp.
    file = File(filter=['*.h5'], desc='name of data file')

    # Number of samples to write to file by `result` method.
    # defaults to -1 (write as long as source yields data).
    num_samples_write = Int(-1)

    # flag that can be raised to stop file writing
    write_flag = Bool(True)

    # internal identifier
    digest = Property(depends_on=['source.digest'])

    # The floating-number-precision of entries of H5 File corresponding
    # to numpy dtypes. Default is 32 bit.
    precision = Enum('float32', 'float64', desc='precision of H5 File')

    # Metadata to be stored in HDF5 file object
    metadata = Dict(desc='metadata to be stored in .h5 file')

    @cached_property
    def _get_digest(self):
        return digest(self)

    def create_filename(self):
        if self.file == '':
            name = datetime.now(tz=timezone.utc).isoformat('_').replace(':', '-').replace('.', '_')
            self.file = path.join(config.td_dir, name + '.h5')

    def get_initialized_file(self):
        file = _get_h5file_class()
        self.create_filename()
        f5h = file(self.file, mode='w')
        f5h.create_extendable_array('time_data', (0, self.num_channels), self.precision)
        ac = f5h.get_data_by_reference('time_data')
        f5h.set_node_attribute(ac, 'sample_freq', self.sample_freq)
        self.add_metadata(f5h)
        return f5h

    def save(self):
        """Saves source output to `*.h5` file."""
        f5h = self.get_initialized_file()
        ac = f5h.get_data_by_reference('time_data')
        for data in self.source.result(4096):
            f5h.append_data(ac, data)
        f5h.close()

    def add_metadata(self, f5h):
        """Adds metadata to .h5 file."""
        nitems = len(self.metadata.items())
        if nitems > 0:
            f5h.create_new_group('metadata', '/')
            for key, value in self.metadata.items():
                if isinstance(value, str):
                    value = array(value, dtype='S')
                f5h.create_array('/metadata', key, value)

    def result(self, num):
        """Python generator that saves source output to `*.h5` file and
        yields the source output block-wise.


        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.
            Echos the source output, but reads it from cache
            when available and prevents unnecessary recalculation.

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
    """Fast frequency domain convolution with the Uniformly partitioned overlap-save method (UPOLS).

    See :cite:`Wefers2015` for details.
    """

    # Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Convolution kernel in the time domain. The second dimension of the kernel array has to be
    # either 1 or match :attr:`~SamplesGenerator.num_channels`. If only a single kernel is supplied,
    # it is applied to all channels.
    kernel = CArray(dtype=float, desc='Convolution kernel.')

    _block_size = Int(desc='Block size')

    _kernel_blocks = Property(
        depends_on=['kernel', '_block_size'],
        desc='Frequency domain Kernel blocks',
    )

    # internal identifier
    digest = Property(depends_on=['source.digest', 'kernel'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _validate_kernel(self):
        # reshape kernel for broadcasting
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
        [L, N] = self.kernel.shape
        num = self._block_size
        P = int(ceil(L / num))
        trim = num * (P - 1)
        blocks = zeros([P, num + 1, N], dtype='complex128')

        if P > 1:
            for i, block in enumerate(split(self.kernel[:trim], P - 1, axis=0)):
                blocks[i] = rfft(concatenate([block, zeros([num, N])], axis=0), axis=0)

        blocks[-1] = rfft(
            concatenate([self.kernel[trim:], zeros([2 * num - L + trim, N])], axis=0),
            axis=0,
        )
        return blocks

    def result(self, num=128):
        """Python generator that yields the output block-wise.
        The source output is convolved with the kernel.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

        """
        self._validate_kernel()
        # initialize variables
        self._block_size = num
        L = self.kernel.shape[0]
        N = self.source.num_channels
        M = self.source.num_samples
        numblocks_kernel = int(ceil(L / num))  # number of kernel blocks
        Q = int(ceil(M / num))  # number of signal blocks
        R = int(ceil((L + M - 1) / num))  # number of output blocks
        last_size = (L + M - 1) % num  # size of final block

        idx = 0
        fdl = zeros([numblocks_kernel, num + 1, N], dtype='complex128')
        buff = zeros([2 * num, N])  # time-domain input buffer
        spec_sum = zeros([num + 1, N], dtype='complex128')

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
            buff = concatenate(
                [buff[num:], zeros([num, N])],
                axis=0,
            )  # shift input buffer to the left
            buff[num : num + temp.shape[0]] = temp  # append new time-data

        for _ in range(R - Q):
            _append_to_fdl(fdl, idx, numblocks_kernel, rfft(buff, axis=0))
            spec_sum = _spectral_sum(spec_sum, fdl, self._kernel_blocks)
            yield irfft(spec_sum, axis=0)[num:]
            buff = concatenate(
                [buff[num:], zeros([num, N])],
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


class MaskedTimeInOut(MaskedTimeOut):
    """Signal processing block for channel and sample selection.

    .. deprecated:: 24.10
        Using :class:`~acoular.tprocess.MaskedTimeInOut` is deprecated and will be removed in
        Acoular version 25.07. Use :class:`~acoular.tprocess.MaskedTimeOut` instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(
            'Using MaskedTimeInOut is deprecated and will be removed in Acoular version 25.07. \
            Use class MaskedTimeOut instead.',
            DeprecationWarning,
            stacklevel=2,
        )
