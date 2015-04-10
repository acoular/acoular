# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements signal generators for the simulation of acoustic sources.

.. autosummary::
    :toctree: generated/

    SignalGenerator
    WNoiseGenerator
    PNoiseGenerator
    SineGenerator

"""

# imports from other packages
from numpy import pi, arange, sin, sqrt, repeat, log
from numpy.random import RandomState
from traits.api import HasPrivateTraits, Float, Int, Long, \
Property, cached_property
from traitsui.api import View
from scipy.signal import resample

# acoular imports
from .internal import digest

class SignalGenerator( HasPrivateTraits ):
    """
    Virtual base class for a simple one-channel signal generator.
    
    Defines the common interface for all SignalGenerator classes. This class
    may be used as a base for specialized SignalGenerator implementations. It
    should not be used directly as it contains no real functionality.
    """

    #: RMS amplitude of source signal (for point source: in 1 m  distance).
    rms = Float(1.0, 
        desc="rms amplitude")
    
    #: Sampling frequency of the signal.
    sample_freq = Float(1.0, 
        desc="sampling frequency")
                   
    #: Number of samples to generate.
    numsamples = Long
    
    # internal identifier
    digest = Property

    # no view necessary
    traits_view = View()

    def _get_digest( self ):
        return ''

    def signal(self):
        """
        Deliver the signal.
        """
        pass
    
    def usignal(self, factor):
        """
        Delivers the signal resampled with multiple of the sampling freq.
        
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
    #: to guarantee statistically independent (non-correlated) outputs
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
    #: to guarantee statistically independent (non-correlated) outputs
    seed = Int(0, 
        desc="random seed value")
    
    #: "Octave depth" -- higher values for 1/f spectrum at low frequencies, 
    #: but longer calculation    
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
            print 'Pink noise filter depth set to maximum possible value of %d.' % max_depth
        
        rnd_gen = RandomState(self.seed)
        s = rnd_gen.standard_normal(nums)
        for _ in range(depth):
            ind = 2**_-1
            lind = nums-ind
            dind = 2**(_+1)
            s[ind:] += repeat( rnd_gen.standard_normal(nums / dind+1 ), dind)[:lind]
        # divide by sqrt(depth+1.5) to get same overall level as white noise
        return self.rms/sqrt(depth+1.5) * s


class SineGenerator( SignalGenerator ):
    """
    Sine signal generator with adjustable frequency and phase.
    """

    #: Sine wave frequency, float, defaults to 1000.0
    freq = Float(1000.0, 
        desc="Frequency")

    #: Sine wave phase (in radians), float, defaults to 0.0
    phase = Float(0.0, 
        desc="Phase")
        
    # internal identifier
    digest = Property( 
        depends_on = ['rms', 'numsamples', \
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
            The resulting signal as an array of length *numsamples*.
        """
        t = arange(self.numsamples, dtype=float)/self.sample_freq
        return self.rms*sin(2*pi*self.freq*t+self.phase)
