# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2017, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements processing in the time domain.

.. autosummary::
    :toctree: generated/

    TimeInOut
    MaskedTimeInOut
    Trigger
    EngineOrderAnalysis
    SpatialInterpolator
    SpatialInterpolatorConstantRotation
    Mixer
    TimePower
    TimeAverage
    TimeReverse
    FiltFiltOctave
    FiltOctave
    TimeCache
    WriteWAV
    WriteH5
"""

# imports from other packages
from six import next
from numpy import array, empty, empty_like, pi, sin, sqrt, zeros, newaxis, unique, \
int16, cross, isclose, zeros_like, dot, nan, concatenate, isnan, nansum, float64, \
identity, argsort, interp, arange, append, linspace, flatnonzero, argmin, argmax, \
delete, mean, inf, ceil, log2, logical_and, asarray
from numpy.linalg import norm
from numpy.matlib import repmat
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from traits.api import Float, Int, CLong, \
File, Property, Instance, Trait, Delegate, \
cached_property, on_trait_change, List, ListInt, CArray
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons
from datetime import datetime
from os import path
import tables
import wave
from scipy.signal import butter, lfilter, filtfilt
from warnings import warn

# acoular imports
from .internal import digest
from .h5cache import H5cache, td_dir
from .sources import SamplesGenerator
from .environments import cartToCyl
from .microphones import MicGeom


class TimeInOut( SamplesGenerator ):
    """
    Base class for any time domain signal processing block, 
    gets samples from :attr:`source` and generates output via the 
    generator :meth:`result`
    """

    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Trait(SamplesGenerator)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')
    
    #: Number of channels in output, as given by :attr:`source`.
    numchannels = Delegate('source')
               
    #: Number of samples in output, as given by :attr:`source`.
    numsamples = Delegate('source')
            
    # internal identifier
    digest = Property( depends_on = ['source.digest', '__class__'])

    traits_view = View(
        Item('source', style='custom')
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    def result(self, num):
        """ 
        Python generator: dummy function, just echoes the output of source,
        yields samples in blocks of shape (num, :attr:`numchannels`), the last block
        may be shorter than num.
        """
        for temp in self.source.result(num):
            # effectively no processing
            yield temp


class MaskedTimeInOut ( TimeInOut ):
    """
    Signal processing block for channel and sample selection.
    
    This class serves as intermediary to define (in)valid 
    channels and samples for any 
    :class:`~acoular.sources.SamplesGenerator` (or derived) object.
    It gets samples from :attr:`~acoular.tprocess.TimeInOut.source` 
    and generates output via the generator :meth:`result`.
    """
        
    #: Index of the first sample to be considered valid.
    start = CLong(0, 
        desc="start of valid samples")
    
    #: Index of the last sample to be considered valid.
    stop = Trait(None, None, CLong, 
        desc="stop of valid samples")
    
    #: Channels that are to be treated as invalid.
    invalid_channels = List(
        desc="list of invalid channels")
    
    #: Channel mask to serve as an index for all valid channels, is set automatically.
    channels = Property(depends_on = ['invalid_channels', 'source.numchannels'], 
        desc="channel mask")
    
    #: Number of channels in input, as given by :attr:`~acoular.tprocess.TimeInOut.source`.
    numchannels_total = Delegate('source', 'numchannels')
               
    #: Number of samples in input, as given by :attr:`~acoular.tprocess.TimeInOut.source`.
    numsamples_total = Delegate('source', 'numsamples')

    #: Number of valid channels, is set automatically.
    numchannels = Property(depends_on = ['invalid_channels', \
        'source.numchannels'], desc="number of valid input channels")

    #: Number of valid time samples, is set automatically.
    numsamples = Property(depends_on = ['start', 'stop', 'source.numsamples'], 
        desc="number of valid samples per channel")

    #: Name of the cache file without extension, readonly.
    basename = Property( depends_on = 'source.digest', 
        desc="basename for cache file")

    # internal identifier
    digest = Property( depends_on = ['source.digest', 'start', 'stop', \
        'invalid_channels'])

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_basename( self ):
        if 'basename' in self.source.all_trait_names():
            return self.source.basename
        else: 
            return self.source.__class__.__name__ + self.source.digest
    
    @cached_property
    def _get_channels( self ):
        if len(self.invalid_channels)==0:
            return slice(0, None, None)
        allr=[i for i in range(self.numchannels_total) if not (i in self.invalid_channels)]
        return array(allr)
    
    @cached_property
    def _get_numchannels( self ):
        if len(self.invalid_channels)==0:
            return self.numchannels_total
        return len(self.channels)
    
    @cached_property
    def _get_numsamples( self ):
        sli = slice(self.start, self.stop).indices(self.numsamples_total)
        return sli[1]-sli[0]

    def result(self, num):
        """ 
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`). 
            The last block may be shorter than num.
        """
        sli = slice(self.start, self.stop).indices(self.numsamples_total)
        start = sli[0]
        stop = sli[1]
        if start >= stop:
            raise IOError("no samples available")
        
        if start != 0 or stop != self.numsamples_total:

            stopoff = -stop % num
            offset = -start % num
            if offset == 0: offset = num      
            buf = empty((num + offset , self.numchannels), dtype=float) # buffer array
            i = 0
            for block in self.source.result(num):
                i += num
                if i > start and i <= stop+stopoff:
                    ns = block.shape[0] # numbers of samples
                    buf[offset:offset+ns] = block[:, self.channels]
                    if i > start + num:
                        yield buf[:num]
                    buf[:offset] = buf[num:num+offset]
            if offset-stopoff != 0:
                yield buf[:(offset-stopoff)]
        
        else: # if no start/stop given, don't do the resorting thing
            for block in self.source.result(num):
                yield block[:, self.channels]
                

class Trigger(TimeInOut):
    """
    Class for identifying trigger signals.
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
    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)
    
    #: Threshold of trigger. Has different meanings for different 
    #: :attr:`~acoular.tprocess.Trigger.trigger_type`. The sign is relevant.
    #: If a sample of the signal is above/below the positive/negative threshold, 
    #: it is assumed to be a peak.
    #: Default is None, in which case a first estimate is used: The threshold
    #: is assumed to be 75% of the max/min difference between all extremums and the 
    #: mean value of the trigger signal. E.g: the mean value is 0 and there are positive
    #: extremums at 400 and negative extremums at -800. Then the estimated threshold would be 
    #: 0.75 * -800 = -600.
    threshold = Float(None)
    
    #: Maximum allowable variation of length of each revolution duration. Default is
    #: 2%. A warning is thrown, if any revolution length surpasses this value:
    #: abs(durationEachRev - meanDuration) > 0.02 * meanDuration
    max_variation_of_duration = Float(0.02)
    
    #: Defines the length of hunks via lenHunk = hunk_length * maxOncePerRevDuration.
    #: If there are multiple peaks within lenHunk, then the algorithm will 
    #: cancel all but one out (see :attr:`~acoular.tprocess.Trigger.multiple_peaks_in_hunk`).
    #: Default is to 0.1.
    hunk_length = Float(0.1)
    
    #: Type of trigger.
    #: - 'Dirac': a single puls is assumed (sign of 
    #:      :attr:`~acoular.tprocess.Trigger.trigger_type` is important).
    #:      Sample will trigger if its value is above/below the pos/neg threshold.
    #: - 'Rect' : repeating rectangular functions. Only every second 
    #:      edge is assumed to be a trigger. The sign of 
    #:      :attr:`~acoular.tprocess.Trigger.trigger_type` gives information
    #:      on which edge should be used (+ for rising edge, - for falling edge).
    #:      Sample will trigger if the difference between its value and its predecessors value
    #:      is above/below the pos/neg threshold.
    #: Default is to 'Dirac'.
    trigger_type = Trait('Dirac', 'Rect')
    
    #: Identifier which peak to consider, if there are multiple peaks in one hunk
    #: (see :attr:`~acoular.tprocess.Trigger.hunk_length`). Default is to 'extremum', 
    #: in which case the extremal peak (maximum if threshold > 0, minimum if threshold < 0) is considered.
    multiple_peaks_in_hunk = Trait('extremum', 'first')
    
    #: Tuple consisting of 3 entries: 
    #: 1.: -Vector with the sample indices of the 1/Rev trigger samples
    #: 2.: -maximum of number of samples between adjacent trigger samples
    #: 3.: -minimum of number of samples between adjacent trigger samples
    trigger_data = Property(depends_on=['source.digest', 'threshold', 'max_variation_of_duration', \
                                        'hunk_length', 'trigger_type', 'multiple_peaks_in_hunk'])
    
    # internal identifier
    digest = Property(depends_on=['source.digest', 'threshold', 'max_variation_of_duration', \
                                        'hunk_length', 'trigger_type', 'multiple_peaks_in_hunk'])
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_trigger_data(self):
        self._check_trigger_existence()
        triggerFunc = {'Dirac' : self._trigger_dirac,
                       'Rect' : self._trigger_rect}[self.trigger_type]
        nSamples = 2048  # number samples for result-method of source
        threshold = self._threshold(nSamples)
        
        # get all samples which surpasse the threshold
        peakLoc = array([], dtype='int')  # all indices which surpasse the threshold
        triggerData = array([])
        x0 = []
        dSamples = 0
        for triggerSignal in self.source.result(nSamples):
            localTrigger = flatnonzero(triggerFunc(x0, triggerSignal, threshold))
            if not len(localTrigger) == 0:
                peakLoc = append(peakLoc, localTrigger + dSamples)
                triggerData = append(triggerData, triggerSignal[localTrigger])
            dSamples += nSamples
            x0 = triggerSignal[-1]
        if len(peakLoc) <= 1:
            raise Exception('Not enough trigger info. Check *threshold* sign and value!')

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
                values = triggerData[indHelp]
                deleteInd = indHelp[argmin(abs(values))]
            elif self.multiple_peaks_in_hunk == 'first':
                deleteInd = indHelp[1]
            peakLoc = delete(peakLoc, deleteInd)
            triggerData = delete(triggerData, deleteInd)
            peakDist = peakLoc[1:] - peakLoc[:-1]
            multiplePeaksWithinHunk = flatnonzero(peakDist < self.hunk_length * maxPeakDist)
        
        # check whether distances between peaks are evenly distributed
        meanDist = mean(peakDist)
        diffDist = abs(peakDist - meanDist)
        faultyInd = flatnonzero(diffDist > self.max_variation_of_duration * meanDist)
        if faultyInd.size != 0:
            warn('In Trigger-Identification: The distances between the peaks (and therefor the lengths of the revolutions) vary too much (check samples %s).' % str(peakLoc[faultyInd] + self.source.start), Warning, stacklevel = 2)
        return peakLoc, max(peakDist), min(peakDist)
    
    def _trigger_dirac(self, x0, x, threshold):
        # x0 not needed here, but needed in _trigger_rect
        return self._trigger_value_comp(x, threshold)
    
    def _trigger_rect(self, x0, x, threshold):
        # x0 stores the last value of the the last generator cycle
        xNew = append(x0, x)
#        indPeakHunk = abs(xNew[1:] - xNew[:-1]) > abs(threshold)  # with this line: every edge would be located
        indPeakHunk = self._trigger_value_comp(xNew[1:] - xNew[:-1], threshold)
        return indPeakHunk
    
    def _trigger_value_comp(self, triggerData, threshold):
        if threshold > 0.0:
            indPeaks= triggerData > threshold
        else:
            indPeaks = triggerData < threshold
        return indPeaks
    
    def _threshold(self, nSamples):
        if self.threshold == None:  # take a guessed threshold
            # get max and min values of whole trigger signal
            maxVal = -inf
            minVal = inf
            meanVal = 0
            cntMean = 0
            for triggerData in self.source.result(nSamples):
                maxVal = max(maxVal, triggerData.max())
                minVal = min(minVal, triggerData.min())
                meanVal += triggerData.mean()
                cntMean += 1
            meanVal /= cntMean
            
            # get 75% of maximum absolute value of trigger signal
            maxTriggerHelp = [minVal, maxVal] - meanVal
            argInd = argmax(abs(maxTriggerHelp))
            thresh = maxTriggerHelp[argInd] * 0.75  # 0.75 for 75% of max trigger signal
            warn('No threshold was passed. An estimated threshold of %s is assumed.' % thresh, Warning, stacklevel = 2)
        else:  # take user defined  threshold
            thresh = self.threshold
        return thresh
    
    def _check_trigger_existence(self):
        for triggerData in self.source.result(1):
            nChannels = triggerData.shape[1]
            if not nChannels == 1:
                raise Exception('Trigger signal must consist of ONE channel, instead %s channels are given!' % nChannels)
        return 0


class EngineOrderAnalysis(TimeInOut):
    """
    Signal processing block for Engine-Order-Analysis or Order-Tracking of 
    rotating sound sources.
    
    If a signal with 1/Rev triggers is provided, this class upsamples its 
    :attr:`~acoular.tprocess.TimeInOut.source` samples, s.t. each revolution 
    consists of the same number of samples N.
    Here N is the next power of 2 which contains all original samples of the 
    slowest revolution.
    
    Because of the common samples per revolution, possible fluctuations of the
    rotational speed still result in the same phase of frequency bins, when 
    performing a subsequent FFT. Averaging those FFTs would mean that 
    non-rotor-coherent sources decay, whereas rotor-coherent sources don't.
    
    The output is delivered via the generator :meth:`result`.
    """
    #: Upsampled sample frequency. Is calculated via 
    #: round(mean(:attr:`rpm`) / 60 * :attr:`samples_per_rev`).
    sample_freq = Property(depends_on = ['source.sample_freq', 'trigger.digest'])
    
    #: Number of samples 
    numsamples = Property(depends_on = ['source.sample_freq', 'trigger.digest'])
    
    #: Trigger source; :class:`~acoular.tprocess.Trigger` object.
    trigger = Instance(Trigger)
    
    #: Integer containing the samples per revolution for the upsampled signal.
    #: Is the next power of 2 which contains all original samples of the 
    #: slowest revolution.
    samples_per_rev = Property(depends_on = 'trigger.digest')
    
    #: Rotational speed in rpm for each revolution. A vector of floats of 
    #: length n-1, where n are the trigger-peaks delivered by :attr:`trigger`.
    rpm = Property(depends_on = ['source.sample_freq', 'trigger.digest'])
    
    # internal identifier
    digest = Property(depends_on = ['source.digest', 'trigger.digest'])
    
    @cached_property
    def _get_sample_freq(self):
        rps = mean(self.rpm) / 60
        fs = int(round(rps * self.samples_per_rev))
        return fs
    
    @cached_property
    def _get_numsamples(self):
        return self.samples_per_rev * len(self.rpm)
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_samples_per_rev(self):
        maxLen = self.trigger.trigger_data[1]
        nResample = int(2 ** ceil(log2(maxLen)))  # round to next power of 2
        return nResample
    
    @cached_property
    def _get_rpm(self):
        fsMeasured = self.source.sample_freq
        peakLoc = self.trigger.trigger_data[0]
        samplesPerRev = peakLoc[1:] - peakLoc[:-1]
        durationPerRev = samplesPerRev / fsMeasured  # in seconds
        return 60. / durationPerRev
    
    def _block_size(self, bs):
        """
        Iternally used function, not for users.
        """
        # needed in PowerSpectra
        nSamples = self.samples_per_rev * bs
        return int(nSamples)
    
    def result(self, num=1):
        """ 
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded. In the 
            EO context one block means one revolution. So num=2 yields all
            the samples of 2 revolutions.
        
        Returns
        -------
        Samples in blocks of shape (:attr:`samples_per_rev` * num, :attr:`numchannels`). 
            This generator only yields full revolutions, which means that all
            samples following the last trigger peak are cropped.
        """
        nMics = self.numchannels
        oncePerRevInd, maxLen, minLen = self.trigger.trigger_data
        nRev = len(oncePerRevInd) - 1
        nResample = self.samples_per_rev
        tNew = linspace(0, 1, nResample, endpoint=False)
        indStart = cntRev = cntAdjacentRevs = 0
        samplesIDPerRev = arange(oncePerRevInd[0], oncePerRevInd[1])
        valuesPerRev = zeros((len(samplesIDPerRev), nMics))
        valuesPerRev.fill(nan)
        resampled = zeros((len(tNew), nMics))
        resamplesStack = zeros((len(tNew) * num, nMics))

        for timeDataRaw in self.source.result(minLen):
            # check which entries of timeDataRaw can be used for current Revolution
            indEnd = indStart + timeDataRaw.shape[0]
            correctSamplesPerRev = logical_and(samplesIDPerRev >= indStart, samplesIDPerRev < indEnd)
            samplesIDRawInput = arange(indStart, indEnd)
            correctSamplesRawInput = logical_and(samplesIDRawInput >= oncePerRevInd[cntRev], samplesIDRawInput < oncePerRevInd[cntRev + 1])
            valuesPerRev[correctSamplesPerRev, :] = timeDataRaw[correctSamplesRawInput]
            
            # check whether current Revolution is filled completely
            if not isnan(valuesPerRev).any():
                cntAdjacentRevs += 1
                cntRev += 1
                tOld = linspace(0, 1, len(samplesIDPerRev), endpoint=False)
                
                # actual resampling
                resampled = asarray([interp(tNew, tOld, valuesPerRev[:, cntMic]) for cntMic in range(nMics)]).T
                stackStart = int((cntAdjacentRevs - 1) * nResample)
                stackEnd = int(stackStart + nResample)
                resamplesStack[stackStart : stackEnd, :] = resampled
                
                # if num resamples are calculated -> yield them
                if cntAdjacentRevs == num:
                    yield resamplesStack
                    cntAdjacentRevs = 0
            
                if cntRev < nRev:
                    # fill all entries of next Revolution with the left over stuff of current timeDataRaw
                    samplesIDPerRev = arange(oncePerRevInd[cntRev], oncePerRevInd[cntRev + 1])
                    correctSamplesPerRev = logical_and(samplesIDPerRev >= indStart, samplesIDPerRev < indEnd)
                    valuesPerRev = zeros((len(samplesIDPerRev), nMics))
                    valuesPerRev.fill(nan)
                    valuesPerRev[correctSamplesPerRev, :] = timeDataRaw[~correctSamplesRawInput]
                else:
                    # stop if the last revolution with beginning and end trigger peak is done
                    break
            indStart = indEnd


class SpatialInterpolator(TimeInOut):
    """
    Base class for spatial linear Interpolation of microphone data.
    Gets samples from :attr:`source` and generates output via the 
    generator :meth:`result`
    """
    #: :class:`~acoular.microphones.MicGeom` object that provides the real microphone locations.
    mpos_real = Instance(MicGeom, 
        desc="microphone geometry")
    
    #: :class:`~acoular.microphones.MicGeom` object that provides the virtual microphone locations.
    mpos_virtual = Instance(MicGeom, 
        desc="microphone geometry")
    
    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)
    
    #: Identifier of sub-arrays. E.g. if the array consists of two subarrays, then
    #: ind_subarray=[0,0,0,0, ... ,1,1,1,1] would identify which entries of m.mpos
    #: belong to the sub-arrays 0 and 1. The entries can be any Integers.
    #: Default is [], which assumes that there are no sub-arrays included.
    ind_subarray = ListInt()
    
    #: rtol for numpys isclose when comparing if all mics are on an plain or line 
    #: via cross/dot-products
    eps = Float(1e-4)
    
    #: Stores the output of :meth:`_reduced_interp_dim_core_func`; Read-Only
    reduced_interp_dim = Property(depends_on=['mpos_real.digest', 'mpos_virtual.digest', 'ind_subarray', 'eps'])
    
    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source', 'sample_freq')
    
    #: Number of channels in output.
    numchannels = Property()
    
    #: Number of samples in output, as given by :attr:`source`.
    numsamples = Delegate('source', 'numsamples')
    
    # internal identifier
    digest = Property(depends_on=['mpos_real.digest', 'mpos_virtual.digest', 'source.digest', 'ind_subarray', 'eps'])
    
    def _get_numchannels(self):
        return self.mpos_virtual.num_mics
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_reduced_interp_dim(self):
        return self._reduced_interp_dim_core_func(self.mpos_real.mpos, self.mpos_virtual.mpos)
    
    def _block_size(self, bs):
        """
        Iternally used function, not for users.
        """
        # needed in PowerSpectra
        return self.source._block_size(bs)
    
    def _reduced_interp_dim_core_func(self, mic, micVirt, basisVectors=[]):
        """ 
        Core functionality for getting the reduced interpolation dimensions.
        
        Parameters
        ----------
        mic : float[3, nPhysicalMics]
            The mic positions of the physical (really existing) mics
        micVirt : float[3, nVirtualMics]
            The mic positions of the virtual mics
        basisVectors : list of float[3]
            If passed, the entries of basisVectors contain the basis vectors of
            the new coordinate system. In this case, the code assumes rotational periodicity.
        
        Returns
        -------
        indCorOut : int64[nVirtualMics]
            Array of indices, to which predefined subarrays (see SpatialInterpolator.ind_subarray) each virtual mic belongs. 
        mesh : List[nSubarrays]
            For each subarray (see unique entries of SpatialInterpolator.ind_subarray) a list is passed.
            The items of these lists are dependent of the reduced interpolation dimension of each subarray.
            If the subarray is 1D the list items are:
                1. item : float64[nMicsInSpecificSubarray]
                    Ordered positions of the real mics (of the subarray) on the new 1d axis, to be used as inputs for numpys interp.
                2. item : int64[nMicsInSpecificSubarray]
                    Indices identifying how the measured pressures must be evaluated, s.t. the entries of the previous item (see last line)
                    correspond to their initial pressure values
            If the subarray is 2D or 3d the list items are:
                1. item : Delaunay mesh object
                    Delauney mesh (see scipy.spatial.Delaunay) for the specific subarray
                2. item : int64[nMicsInSpecificSubarray]
                    same as 1d case, BUT with the difference, that here the rotational periodicy is handled, when constructing the mesh.
                    Therefor the mesh could have more vertices than the actual subarray mics.
        virtNewCoord : float64[3, nVirtualMics]
            Projection of each virtual mic onto its new reduced coordinates.
            If indeed a dimension reduction could be performed, the reduced dimensions are NaN. E.g. if a virtual mic
            lies on a 1d subarray the items 1 and 2 of virtNewCoord[:, virtual mic] are Nan whereas item 0 contains the projection.
            For rotational cases, the columns of virtNewCoord correspond to [phi, rho, z]
        """

        # check whether sub arrays are specified or not
        if self.ind_subarray == []:  # no sub arrays
            sub = zeros((mic.shape[1]), dtype='int64')
            subUnique = [0]
            nSubArrays = 1
            self.ind_subarray = sub.tolist()
            warn('No sub arrays were defined (see SpatialInterpolator.ind_subarray). This may lead to much worse performance.')
        else:
            sub = array(self.ind_subarray)
            subUnique = unique(sub)
            if not len(sub) == mic.shape[1]:
                raise Exception('mpos_real.mpos and ind_subarray must correspond in dimension!')
            nSubArrays = len(subUnique)
        
        # init
        normalizeVec = lambda vec : vec / norm(vec)
        nVirtMics = micVirt.shape[1]
        indCorrespSub = zeros((nSubArrays, nVirtMics))
        indCorrespSub.fill(nan)
        virtNewCoord = zeros((3, nVirtMics))
        virtNewCoord.fill(nan)
        mesh = []
        
        # check for each sub array whether dimension reduction can be done
        for cntSub in range(nSubArrays):
            indSub = sub == subUnique[cntSub]
            subArrayMics = mic[:, indSub]
            
            # check whether all current subArray mics are on a line (cross prod == 0)
            micRef = subArrayMics[:, 0]
            distMic = (subArrayMics[:, 1:].T - micRef.T).T  # distance between all real mics (except the first) and the first real mic
            if basisVectors == []:
                basisVec1 = normalizeVec(distMic[:, 0])  # first basis vector in sub array coordinates
            else:
                basisVec1 = basisVectors[0]
                
            crossMicsOnBasisVec = cross(distMic, basisVec1, axisa=0, axisc=0)
            micsOnLine = isclose(crossMicsOnBasisVec, zeros_like(crossMicsOnBasisVec), rtol=self.eps)
            if micsOnLine.all():  # 1d sub array (line)
                # check which virtual mics are in line with basisVec1
                distVirtToRef = (micVirt.T - micRef.T).T
                crossVirtOnBasis = cross(distVirtToRef, basisVec1, axisa=0, axisc=0)
                virtOnLine = isclose(crossVirtOnBasis, zeros_like(crossVirtOnBasis), rtol=self.eps)
                virtInSub = virtOnLine.all(axis=0)
                indCorrespSub[cntSub, virtInSub] = subUnique[cntSub]
                
                # get projections onto new coordinate, for real mics and virtual mics
                projectionOnNewAxis = dot(basisVec1, subArrayMics)
                indReorderHelp = argsort(projectionOnNewAxis)
                virtNewCoord[0, virtInSub] = dot(basisVec1, micVirt[:, virtInSub])
                mesh.append([projectionOnNewAxis[indReorderHelp], indReorderHelp])
            else:  # try 2d case
                if basisVectors == []:
                    # find a real mic which spans a plain with basisVec1 and obtain its normal vector
                    helpForPlain = micsOnLine.all(axis=0) == False
                    indPlain = helpForPlain.nonzero()[0][0]
                    normalOfPlain = crossMicsOnBasisVec[:, indPlain]
                    basisVec2 = normalizeVec(cross(basisVec1, normalOfPlain))  # get second basis Vector in sub array coordinates
                else:
                    basisVec2 = basisVectors[1]
                    normalOfPlain = cross(basisVec1, basisVec2)
                
                # check whether all subarray mics are on that plain
                scalarProdMicsOnNormal = dot(normalOfPlain, subArrayMics)
                micsOnPlain = isclose(scalarProdMicsOnNormal, zeros_like(scalarProdMicsOnNormal), rtol=self.eps)
                if micsOnPlain.all():  # 2d sub array (plain)
                    # get all virtual mics on that plain
                    scalarProdVirtOnNormal = dot(normalOfPlain, micVirt)
                    virtInSub = isclose(scalarProdVirtOnNormal, zeros_like(scalarProdVirtOnNormal), rtol=self.eps)
                    indCorrespSub[cntSub, virtInSub] = subUnique[cntSub]
                    
                    # get mic projections on new coord system
                    projectionOnNewAxis1 = dot(basisVec1, subArrayMics)[:,newaxis]
                    projectionOnNewAxis2 = dot(basisVec2, subArrayMics)[:,newaxis]
                    newCoordinates = concatenate((projectionOnNewAxis1, projectionOnNewAxis2), axis=1)
                    
                    # get virtual mic projections on new coord system
                    projectionOnNewAxisVirt1 = dot(basisVec1, micVirt[:, virtInSub])[newaxis]
                    projectionOnNewAxisVirt2 = dot(basisVec2, micVirt[:, virtInSub])[newaxis]
                    virtNewCoord[:2, virtInSub] = concatenate((projectionOnNewAxisVirt1, projectionOnNewAxisVirt2), axis=0)
                    
                    tri = Delaunay(newCoordinates, incremental=True)
                else:  # 3d case
                    newCoordinates = subArrayMics.T
                    tri = Delaunay(newCoordinates, incremental=True)
                    virtInSub = tri.find_simplex(micVirt.T) != -1
                    indCorrespSub[cntSub, virtInSub] = subUnique[cntSub]
                    virtNewCoord[:, virtInSub] = micVirt[:, virtInSub]
                    
                if not (tri.points == newCoordinates).all():
                    # even though nothing is said about that in the docu, tri.points seems to have the same order as the
                    # mics given as input in tri = Delaunay(mics). This behaviour is assumed in the code, therefor any contradiction
                    # to that behaviour must be raised.
                    raise Exception('Unexpected behaviour in Delaunay-Triangulation. Please contact the developer!')
                        
                if not basisVectors == []:
                    # extend mesh with closest boundary points of repeating mesh 
                    # (both left and right of original array)
                    pointsOriginal = arange(tri.points.shape[0])
                    hull = tri.convex_hull
                    hullPoints = unique(hull)
                    
                    addRight = tri.points[hullPoints]
                    addRight[:, 0] += 2 * pi
                    addLeft= tri.points[hullPoints]
                    addLeft[:, 0] -= 2 * pi
                    indOrigPoints = concatenate((pointsOriginal, pointsOriginal[hullPoints], pointsOriginal[hullPoints]))
        
                    # add all hull vertices to original mesh and check which of those 
                    # are actual neighbors of the original array. Cancel out all others.
                    tri.add_points(concatenate([addLeft, addRight]))
                    indices, indptr = tri.vertex_neighbor_vertices
                    hullNeighbor = empty((0), dtype='int32')
                    for currHull in hullPoints:
                        neighborOfHull = indptr[indices[currHull]:indices[currHull + 1]]
                        hullNeighbor = append(hullNeighbor, neighborOfHull)
                    hullNeighborUnique = unique(hullNeighbor)
                    pointsNew = unique(append(pointsOriginal, hullNeighborUnique))
                    tri = Delaunay(tri.points[pointsNew])  # re-meshing
                    mesh.append([tri, indOrigPoints[pointsNew]])
                else:
                    mesh.append([tri, arange(tri.points.shape[0])])

        # check whether sub arrays make sense
        correspSubarrays = ~isnan(indCorrespSub)
        correspSubarraysHelp = correspSubarrays.sum(axis=0)
        multplSubs = correspSubarraysHelp > 1
        if multplSubs.any():
            multplSubsNonZero = multplSubs.nonzero()[0]
            indMultp = [subUnique[correspSubarrays[:, multplSubs[cnt]]].tolist() for cnt in range(len(multplSubsNonZero))]
            raise Exception('mpos_virtual (entries %s) correspond to multiple sub arrays with indices %s (one list of indices for each virual mic).' % (multplSubsNonZero, indMultp))
        elif (correspSubarraysHelp == 0).any():
            raise Exception('mpos_virtual (entries %s) do not correspond to (sub-) array. Extrapolation is not allowed!' % (correspSubarraysHelp == 0).nonzero()[0])
        else:  # everything is fine
            indCorOut = nansum(indCorrespSub, axis=0, dtype='int')
        return indCorOut, mesh, virtNewCoord

                     
    def _result_core_func(self, p, phiDelay=[], period=None):
        """
        Performs the actual Interpolation.
        
        Parameters
        ----------
        p : float[nSamples, nMicsReal]
            The pressure field of the yielded sample at real mics.
        phiDelay : empty list (default) or float[nSamples] 
            If passed (rotational case), this list contains the angular delay 
            of each sample in rad.
        period : None (default) or float
            If periodicity can be assumed (rotational case) and the array is 1D 
            (see :attr:`reduced_interp_dim`) this parameter contains the periodicity length (2pi)
        
        Returns
        -------
        pInterp : float[nSamples, nMicsVirtual]
            The interpolated time data at the virtual mics
        """
        nTime = p.shape[0]
        indSubArray, meshList, virtProj = self.reduced_interp_dim
        subArrays = unique(indSubArray)
        pInterp = zeros((nTime, len(indSubArray)))
        for cntSub in range(len(subArrays)):
            subMicsReal = self.ind_subarray == subArrays[cntSub]
            subMicsVirt = indSubArray == subArrays[cntSub]
            pHelp = p[:, subMicsReal][:, meshList[cntSub][1]]
            
            # get interpolation dim
            interpDimHelp = unique((~isnan(virtProj[:, subMicsVirt])).sum(axis=0))
            if len(interpDimHelp) == 1:
                interpDim = interpDimHelp[0]
            else:
                raise Exception('Interpolation dim is not the same for all mics in (sub-) array! Something went wrong in reduced_interp_dim')
            wantedCoord = virtProj[:interpDim, subMicsVirt]
            
            if not phiDelay == []:
                xInterpHelp = repmat(wantedCoord[0, :], nTime, 1) + repmat(phiDelay, wantedCoord.shape[1], 1).T
                xInterp = ((xInterpHelp + pi) % (2 * pi)) - pi  # shifting phi cootrdinate into feasible area [-pi, pi]
            else:
                xInterp = repmat(wantedCoord[0, :], nTime, 1)
            
            # interpolating
            for cntTime in range(nTime):
                if interpDim == 1:  # 1D
                    x = meshList[cntSub][0]
                    pInterp[cntTime, subMicsVirt] = interp(xInterp[cntTime, :], x, pHelp[cntTime, :], period=period, left=nan, right=nan)
                else:  # 2D and 3D
                    mesh = meshList[cntSub][0]
                    newPoint = concatenate((xInterp[cntTime, :][:, newaxis], wantedCoord[1:, :].T), axis=1)
                    f = LinearNDInterpolator(mesh, pHelp[cntTime, :])
                    pInterp[cntTime, subMicsVirt] = f(newPoint)
        return pInterp
        
    def result(self, num=128):
        """ 
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`). 
            The last block may be shorter than num.
        """
        for timeData in self.source.result(num):
            interpVal = self._result_core_func(timeData)
            yield interpVal

    
class SpatialInterpolatorConstantRotation(SpatialInterpolator):
    """
    Spatial linear Interpolation for constantly rotating sources.
    Gets samples from :attr:`source` and generates output via the 
    generator :meth:`result`
    """
    #: Rotational speed in rps. Positive, if rotation is around positive z-axis sense,
    #: which means from x to y axis.
    rotational_speed = Float(0.0)
    
    #: The rotation must be around the z-axis, which means from x to y axis.
    #: If the coordinates are not build like that, than this 3x3 orthogonal 
    #: transformation matrix Q can be used to modify the coordinates.
    #: It is assumed that with the modified coordinates the rotation is around the z-axis. 
    #: The transformation is done via [x,y,z]_mod = Q * [x,y,z]. (default is Identity).
    Q = CArray(dtype=float64, shape=(3, 3), value=identity(3))
    
    #: Stores the output of :meth:`_reduced_interp_dim_core_func`; Read-Only
    reduced_interp_dim = Property(depends_on=['mpos_real.digest', 'ind_subarray', 'eps', 'mpos_virtual.digest', 'Q'])
    
    # internal identifier
    digest = Property( depends_on = ['source.digest', 'rotational_speed', 'Q', 'mpos_real.digest', 'ind_subarray', 'eps', 'mpos_virtual.digest'])
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_reduced_interp_dim(self):
        if not isclose(dot(self.Q.T, self.Q), identity(3)).all():
            raise Exception('Transformation matrix SpatialInterpolatorRotation.Q must be orthogonal!')
        mic = cartToCyl(self.mpos_real.mpos, self.Q)
        micVirt = cartToCyl(self.mpos_virtual.mpos, self.Q)
        basisVecs = [array([1,0,0]), array([0,1,0])]
        return self._reduced_interp_dim_core_func(mic, micVirt, basisVecs)
    
    def result(self, num=1):
        """ 
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`). 
            The last block may be shorter than num.
        """
        omega = 2 * pi * self.rotational_speed
        period = 2 * pi
        phiOffset = 0.0
        for timeData in self.source.result(num):
            nTime = timeData.shape[0]
            phiDelay = phiOffset + linspace(0, nTime / self.sample_freq * omega, nTime, endpoint=False)
            interpVal = self._result_core_func(timeData, phiDelay, period)
            phiOffset = phiDelay[-1] + omega / self.sample_freq
            yield interpVal
   
    
class Mixer( TimeInOut ):
    """
    Mixes the signals from several sources.
    """

    #: Data source; :class:`~acoular.sources.SamplesGenerator` object.
    source = Trait(SamplesGenerator)

    #: List of additional :class:`~acoular.sources.SamplesGenerator` objects
    #: to be mixed.
    sources = List( Instance(SamplesGenerator, ()) ) 

    #: Sampling frequency of the signal as given by :attr:`source`.
    sample_freq = Delegate('source')
    
    #: Number of channels in output as given by :attr:`source`.
    numchannels = Delegate('source')
               
    #: Number of samples in output as given by :attr:`source`.
    numsamples = Delegate('source')

    # internal identifier
    ldigest = Property( depends_on = ['sources.digest', ])

    # internal identifier
    digest = Property( depends_on = ['source.digest', 'ldigest', '__class__'])

    traits_view = View(
        Item('source', style='custom')
                    )

    @cached_property
    def _get_ldigest( self ):
        res = ''
        for s in self.sources:
            res += s.digest
        return res

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @on_trait_change('sources,source')
    def validate_sources( self ):
        """ validates if sources fit together """
        if self.source:
            for s in self.sources:
                if self.sample_freq != s.sample_freq:
                    raise ValueError("Sample frequency of %s does not fit" % s)
                if self.numchannels != s.numchannels:
                    raise ValueError("Channel count of %s does not fit" % s)

    def result(self, num):
        """
        Python generator that yields the output block-wise.
        The output from the source and those in the list 
        sources are being added.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        gens = [i.result(num) for i in self.sources]
        for temp in self.source.result(num):
            sh = temp.shape[0]
            for g in gens:
                temp1 = next(g)
                if temp.shape[0] > temp1.shape[0]:
                    temp = temp[:temp1.shape[0]]
                temp += temp1[:temp.shape[0]]
            yield temp
            if sh > temp.shape[0]:
                break


class TimePower( TimeInOut ):
    """
    Calculates time-depended power of the signal.
    """

    def result(self, num):
        """
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Squared output of source. 
            Yields samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        for temp in self.source.result(num):
            yield temp*temp
    
class TimeAverage( TimeInOut ) :
    """
    Calculates time-depended average of the signal
    """
    #: Number of samples to average over, defaults to 64.
    naverage = Int(64, 
        desc = "number of samples to average over")
        
    #: Sampling frequency of the output signal, is set automatically.
    sample_freq = Property( depends_on = 'source.sample_freq, naverage')
    
    #: Number of samples of the output signal, is set automatically.
    numsamples = Property( depends_on = 'source.numsamples, naverage')
    
    # internal identifier
    digest = Property( depends_on = ['source.digest', '__class__', 'naverage'])

    traits_view = View(
        [Item('source', style='custom'), 
         'naverage{Samples to average}', 
            ['sample_freq~{Output sampling frequency}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Linear average', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    @cached_property
    def _get_sample_freq ( self ):
        if self.source:
            return 1.0 * self.source.sample_freq / self.naverage

    @cached_property
    def _get_numsamples ( self ):
        if self.source:
            return self.source.numsamples / self.naverage

    def result(self, num):
        """
        Python generator that yields the output block-wise.

        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Average of the output of source. 
            Yields samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        nav = self.naverage
        for temp in self.source.result(num*nav):
            ns, nc = temp.shape
            nso = int(ns/nav)
            if nso > 0:
                yield temp[:nso*nav].reshape((nso, -1, nc)).mean(axis=1)
                
class TimeReverse( TimeInOut ):
    """
    Calculates the time-reversed signal of a source. 
    """
    def result(self, num):
        """
        Python generator that yields the output block-wise.

        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Yields samples in blocks of shape (num, numchannels). 
            Time-reversed output of source. 
            The last block may be shorter than num.
        """
        l = []
        l.extend(self.source.result(num))
        temp = empty_like(l[0])
        h = l.pop()
        nsh = h.shape[0]
        temp[:nsh] = h[::-1]
        for h in l[::-1]:
            temp[nsh:] = h[:nsh-1:-1]
            yield temp
            temp[:nsh] = h[nsh-1::-1]
        yield temp[:nsh]
        
class FiltFiltOctave( TimeInOut ):
    """
    Octave or third-octave filter with zero phase delay.
    
    This filter can be applied on time signals.
    It requires large amounts of memory!   
    """
    #: Band center frequency; defaults to 1000.
    band = Float(1000.0, 
        desc = "band center frequency")
        
    #: Octave fraction: 'Octave' or 'Third octave'; defaults to 'Octave'.
    fraction = Trait('Octave', {'Octave':1, 'Third octave':3}, 
        desc = "fraction of octave")
        
    # internal identifier
    digest = Property( depends_on = ['source.digest', '__class__', \
        'band', 'fraction'])

    traits_view = View(
        [Item('source', style='custom'), 
         'band{Center frequency}', 
         'fraction{Bandwidth}', 
            ['sample_freq~{Output sampling frequency}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Linear average', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    def ba(self, order):
        """ 
        Internal Butterworth filter design routine.
        
        Parameters
        ----------
        order : integer
            The order of the filter.
        
        Returns
        -------
            b, a : ndarray, ndarray
                Filter coefficients.
        """
        # filter design
        fs = self.sample_freq
        # adjust filter edge frequencies
        beta = pi/(4*order)
        alpha = pow(2.0, 1.0/(2.0*self.fraction_))
        beta = 2 * beta / sin(beta) / (alpha-1/alpha)
        alpha = (1+sqrt(1+beta*beta))/beta
        fr = 2*self.band/fs
        if fr > 1/sqrt(2):
            raise ValueError("band frequency too high:%f,%f" % (self.band, fs))
        om1 = fr/alpha 
        om2 = fr*alpha
#        print om1, om2
        return butter(order, [om1, om2], 'bandpass') 
        
    def result(self, num):
        """
        Python generator that yields the output block-wise.

        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            Delivers the zero-phase bandpass filtered output of source.
            The last block may be shorter than num.
        """
        b, a = self.ba(3) # filter order = 3
        data = empty((self.source.numsamples, self.source.numchannels))
        j = 0
        for block in self.source.result(num):
            ns, nc = block.shape
            data[j:j+ns] = block
            j += ns
        for j in range(self.source.numchannels):
            data[:, j] = filtfilt(b, a, data[:, j])
        j = 0
        ns = data.shape[0]
        while j < ns:
            yield data[j:j+num]
            j += num

class FiltOctave( FiltFiltOctave ):
    """
    Octave or third-octave filter (not zero-phase).
    """

    def result(self, num):
        """ 
        Python generator that yields the output block-wise.

        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            Delivers the bandpass filtered output of source.
            The last block may be shorter than num.
        """
        b, a = self.ba(3) # filter order = 3
        zi = zeros((max(len(a), len(b))-1, self.source.numchannels))
        for block in self.source.result(num):
            block, zi = lfilter(b, a, block, axis=0, zi=zi)
            yield block

                       
class TimeCache( TimeInOut ):
    """
    Caches time signal in cache file.
    """
    # basename for cache
    basename = Property( depends_on = 'digest')
    
    # hdf5 cache file
    h5f = Instance(tables.File,  transient = True)
    
    # internal identifier
    digest = Property( depends_on = ['source.digest', '__class__'])

    traits_view = View(
        [Item('source', style='custom'), 
            ['basename~{Cache file name}', 
            '|[Properties]'], 
            '|'
        ], 
        title='TimeCache', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_basename ( self ):
        obj = self.source # start with source
        basename = 'void' # if no file source is found
        while obj:
            if 'basename' in obj.all_trait_names(): # at original source?
                basename = obj.basename # get the name
                break
            else:
                try:
                    obj = obj.source # traverse down until original data source
                except AttributeError:
                    obj = None
        return basename

    # result generator: delivers input, possibly from cache
    def result(self, num):
        """ 
        Python generator that yields the output from cache block-wise.

        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
            Echos the source output, but reads it from cache
            when available and prevents unnecassary recalculation.
        """
        name = 'tc_' + self.digest
        H5cache.get_cache( self, self.basename )
        if not name in self.h5f.root:
            ac = self.h5f.create_earray(self.h5f.root, name, \
                                       tables.atom.Float32Atom(), \
                                        (0, self.numchannels))
            ac.set_attr('sample_freq', self.sample_freq)
            for data in self.source.result(num):
                ac.append(data)
                yield data
        else:
            ac = self.h5f.get_node('/', name)
            i = 0
            while i < ac.shape[0]:
                yield ac[i:i+num]
                i += num

class WriteWAV( TimeInOut ):
    """
    Saves time signal from one or more channels as mono/stereo/multi-channel
    `*.wav` file.
    """
    
    #: Basename for cache, readonly.
    basename = Property( depends_on = 'digest')
       
    #: Channel(s) to save. List can only contain one or two channels.
    channels = List(desc="channel to save")
       
    # internal identifier
    digest = Property( depends_on = ['source.digest', 'channels', '__class__'])

    traits_view = View(
        [Item('source', style='custom'), 
            ['basename~{File name}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Write wav file', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_basename ( self ):
        obj = self.source # start with source
        try:
            while obj:
                if 'basename' in obj.all_trait_names(): # at original source?
                    basename = obj.basename # get the name
                    break
                else:
                    obj = obj.source # traverse down until original data source
            else:
                basename = 'void'
        except AttributeError:
            basename = 'void' # if no file source is found
        return basename

    def save(self):
        """ 
        Saves source output to one- or multiple-channel `*.wav` file. 
        """
        nc = len(self.channels)
        if nc == 0:
            raise ValueError("No channels given for output.")
        if nc > 2:
            warn("More than two channels given for output, exported file will have %i channels" % nc)
        name = self.basename
        for nr in self.channels:
            name += '_%i' % nr
        name += '.wav'
        wf = wave.open(name,'w')
        wf.setnchannels(nc)
        wf.setsampwidth(2)
        wf.setframerate(self.source.sample_freq)
        wf.setnframes(self.source.numsamples)
        mx = 0.0
        ind = array(self.channels)
        for data in self.source.result(1024):
            mx = max(abs(data[:, ind]).max(), mx)
        scale = 0.9*2**15/mx
        for data in self.source.result(1024):
            wf.writeframesraw(array(data[:, ind]*scale, dtype=int16).tostring())
        wf.close()

class WriteH5( TimeInOut ):
    """
    Saves time signal as `*.h5` file
    """
    #: Name of the file to be saved. If none is given, the name will be
    #: automatically generated from a time stamp.
    name = File(filter=['*.h5'], 
        desc="name of data file")    
      
    # internal identifier
    digest = Property( depends_on = ['source.digest', '__class__'])

    traits_view = View(
        [Item('source', style='custom'), 
            ['name{File name}', 
            '|[Properties]'], 
            '|'
        ], 
        title='write .h5', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)


    def save(self):
        """ 
        Saves source output to `*.h5` file 
        """
        if self.name == '':
            name = datetime.now().isoformat('_').replace(':','-').replace('.','_')
            self.name = path.join(td_dir,name+'.h5')
        f5h = tables.open_file(self.name, mode = 'w')
        ac = f5h.create_earray(f5h.root, 'time_data', \
            tables.atom.Float32Atom(), (0, self.numchannels))
        ac.set_attr('sample_freq', self.sample_freq)
        for data in self.source.result(4096):
            ac.append(data)
        f5h.close()
        
