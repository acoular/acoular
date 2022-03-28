# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements signal generators for the simulation of acoustic sources.

.. autosummary::
    :toctree: generated/

    SignalGenerator
    WNoiseGenerator
    PNoiseGenerator
    FiltWNoiseGenerator
    SineGenerator
    GenericSignalGenerator

"""

# imports from other packages
from __future__ import print_function, division
from numpy import pi, arange, sin, sqrt, repeat, tile, log, zeros, array
from numpy.random import RandomState
from traits.api import HasPrivateTraits, Trait, Float, Int, CLong, Bool, \
Property, cached_property, Delegate, CArray
from scipy.signal import resample, sosfilt, tf2sos
from warnings import warn

# acoular imports
from .tprocess import SamplesGenerator
from .internal import digest

class SignalGenerator( HasPrivateTraits ):
    """
    Virtual base class for a simple one-channel signal generator.
    
    Defines the common interface for all SignalGenerator classes. This class
    may be used as a base for specialized SignalGenerator implementations. It
    should not be used directly as it contains no real functionality.
    """

    #: RMS amplitude of source signal (for point source: in 1 m distance).
    rms = Float(1.0, 
        desc="rms amplitude")
    
    #: Sampling frequency of the signal.
    sample_freq = Float(1.0, 
        desc="sampling frequency")
                   
    #: Number of samples to generate.
    numsamples = CLong
    
    # internal identifier
    digest = Property

    def _get_digest( self ):
        return ''

    def signal(self):
        """
        Deliver the signal.
        """
        pass
    
    def usignal(self, factor):
        """
        Delivers the signal resampled with a multiple of the sampling freq.
        
        Uses fourier transform method for resampling (from scipy.signal).
        
        Parameters
        ----------
        factor : integer
            The factor defines how many times the new sampling frequency is
            larger than :attr:`sample_freq`.
        
        Returns
        -------
        array of floats
            The resulting signal of length `factor` * :attr:`numsamples`.
        """
        return resample(self.signal(), factor*self.numsamples)

class WNoiseGenerator( SignalGenerator ):
    """
    White noise signal generator. 
    """

    #: Seed for random number generator, defaults to 0.
    #: This parameter should be set differently for different instances
    #: to guarantee statistically independent (non-correlated) outputs.
    seed = Int(0, 
        desc="random seed value")

    # internal identifier
    digest = Property( 
        depends_on = ['rms', 'numsamples', \
        'sample_freq', 'seed', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)

    def signal(self):
        """
        Deliver the signal.

        Returns
        -------
        Array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.numsamples`.
        """
        rnd_gen = RandomState(self.seed)
        return self.rms*rnd_gen.standard_normal(self.numsamples)
    

class PNoiseGenerator( SignalGenerator ):
    """
    Pink noise signal generator.
    
    Simulation of pink noise is based on the Voss-McCartney algorithm.
    Ref.:
        
      * S.J. Orfanidis: Signal Processing (2010), pp. 729-733
      * online discussion: http://www.firstpr.com.au/dsp/pink-noise/
      
    The idea is to iteratively add larger-wavelength noise to get 1/f 
    characteristic.
    """
    
    #: Seed for random number generator, defaults to 0.
    #: This parameter should be set differently for different instances
    #: to guarantee statistically independent (non-correlated) outputs.
    seed = Int(0, 
        desc="random seed value")
    
    #: "Octave depth" -- higher values for 1/f spectrum at low frequencies, 
    #: but longer calculation, defaults to 16.
    depth = Int(16,
        desc="octave depth")

    # internal identifier
    digest = Property( 
        depends_on = ['rms', 'numsamples', \
        'sample_freq', 'seed', 'depth', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)

    def signal(self):
        nums = self.numsamples
        depth = self.depth
        # maximum depth depending on number of samples
        max_depth = int( log(nums) / log(2) )
        
        if depth > max_depth:
            depth = max_depth
            print("Pink noise filter depth set to maximum possible value of %d." % max_depth)
        
        rnd_gen = RandomState(self.seed)
        s = rnd_gen.standard_normal(nums)
        for _ in range(depth):
            ind = 2**_-1
            lind = nums-ind
            dind = 2**(_+1)
            s[ind:] += repeat( rnd_gen.standard_normal(nums // dind+1 ), dind)[:lind]
        # divide by sqrt(depth+1.5) to get same overall level as white noise
        return self.rms/sqrt(depth+1.5) * s


class FiltWNoiseGenerator(WNoiseGenerator):
    """
    Filtered white noise signal following an autoregressive (AR), moving-average
    (MA) or autoregressive moving-average (ARMA) process.
    
    The desired frequency response of the filter can be defined by specifying 
    the filter coefficients :attr:`ar` and :attr:`ma`. 
    The RMS value specified via the :attr:`rms` attribute belongs to the white noise 
    signal and differs from the RMS value of the filtered signal.  
    For numerical stability at high orders, the filter is a combination of second order 
    sections (sos). 
    """

    ar = CArray(value=array([]),dtype=float,
        desc="autoregressive coefficients (coefficients of the denominator)")
    
    ma = CArray(value=array([]),dtype=float,
        desc="moving-average coefficients (coefficients of the numerator)")
    
    # internal identifier
    digest = Property( 
        depends_on = [
            'ar', 'ma', 'rms', 'numsamples', \
        'sample_freq', 'seed', '__class__'
        ], 
        )
        
    @cached_property
    def _get_digest( self ):
        return digest(self)

    def handle_empty_coefficients(self,coefficients):
        if coefficients.size == 0:
            return array([1.0])
        else:
            return coefficients
      
    def signal(self):
        """
        Deliver the signal.

        Returns
        -------
        Array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.numsamples`.
        """
        rnd_gen = RandomState(self.seed)
        ma = self.handle_empty_coefficients(self.ma)
        ar = self.handle_empty_coefficients(self.ar)
        sos = tf2sos(ma, ar)
        ntaps = ma.shape[0]
        sdelay = round(0.5*(ntaps-1)) 
        wnoise = self.rms*rnd_gen.standard_normal(self.numsamples+sdelay) # create longer signal to compensate delay
        return sosfilt(sos, x=wnoise)[sdelay:]


class SineGenerator( SignalGenerator ):
    """
    Sine signal generator with adjustable frequency and phase.
    """

    #: Sine wave frequency, float, defaults to 1000.0.
    freq = Float(1000.0, 
        desc="Frequency")

    #: Sine wave phase (in radians), float, defaults to 0.0.
    phase = Float(0.0, 
        desc="Phase")

    # Internal shadow trait for rms/amplitude values.
    # Do not set directly.
    _amp = Float(1.0)
    
    #: RMS of source signal (for point source: in 1 m distance).
    #: Deprecated. For amplitude use :attr:`amplitude`.
    rms = Property(desc='rms amplitude')
    
    def _get_rms(self):
        return self._amp/2**0.5
    
    def _set_rms(self, rms):
        warn("Up to Acoular 20.02, rms is interpreted as sine amplitude. "
             "This has since been corrected (rms now is 1/sqrt(2) of amplitude). "
             "Use 'amplitude' trait to directly set the ampltiude.", 
             Warning, stacklevel = 2)
        self._amp = rms * 2**0.5
    
    #: Amplitude of source signal (for point source: in 1 m distance). 
    #: Defaults to 1.0.
    amplitude = Property(desc='amplitude')
    
    def _get_amplitude(self):
        return self._amp
    
    def _set_amplitude(self, amp):
        self._amp = amp
    
    # internal identifier
    digest = Property( 
        depends_on = ['_amp', 'numsamples', \
        'sample_freq', 'freq', 'phase', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)

    def signal(self):
        """
        Deliver the signal.

        Returns
        -------
        array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.numsamples`.
        """
        t = arange(self.numsamples, dtype=float)/self.sample_freq
        return self.amplitude * sin(2*pi*self.freq * t + self.phase)


class GenericSignalGenerator( SignalGenerator ):
    """
    Generate signal from output of :class:`~acoular.tprocess.SamplesGenerator` object.
    """
    #: Data source; :class:`~acoular.tprocess.SamplesGenerator` or derived object.
    source = Trait(SamplesGenerator)
    
    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')
    
    _numsamples = CLong(0)
   
    #: Number of samples to generate. Is set to source.numsamples by default.
    numsamples = Property()
    
    def _get_numsamples( self ):
        if self._numsamples:
            return self._numsamples
        else:
            return self.source.numsamples
    
    def _set_numsamples( self, numsamples ):
        self._numsamples = numsamples

    #: Boolean flag, if 'True' (default), signal track is repeated if requested 
    #: :attr:`numsamples` is higher than available sample number
    loop_signal = Bool(True)
    
    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 'loop_signal', 'numsamples', \
        'rms', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    def signal(self):
        """
        Deliver the signal.

        Returns
        -------
        array of floats
            The resulting signal as an array of length :attr:`~GenericSignalGenerator.numsamples`.
        """
        block = 1024
        if self.source.numchannels > 1:
            warn("Signal source has more than one channel. Only channel 0 will be used for signal.", Warning, stacklevel = 2)
        nums = self.numsamples
        track = zeros(nums)
        
        # iterate through source generator to fill signal track
        for i, temp in enumerate(self.source.result(block)):
            start = block*i
            stop = start + len(temp[:,0])
            if nums > stop:
                track[start:stop] = temp[:,0]
            else: # exit loop preliminarily if wanted signal samples are reached
                track[start:nums] = temp[:nums-start,0]
                break
        
        # if the signal should be repeated after finishing and there are still samples open
        if self.loop_signal and (nums > stop):
            
            # fill up empty track with as many full source signals as possible
            nloops = nums // stop
            if nloops>1: track[stop:stop*nloops] = tile(track[:stop], nloops-1)
            # fill up remaining empty track
            res = nums % stop # last part of unfinished loop
            if res > 0: track[stop*nloops:] = track[:res]
        
        # The rms value is just an amplification here
        return self.rms*track
    