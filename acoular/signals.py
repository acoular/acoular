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
    SineGenerator

"""

# imports from other packages
from numpy import pi, arange, sin
from numpy.random import normal, seed
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
    """White noise signal generator.
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
        seed(self.seed)
        return self.rms*normal(scale=1.0, size=self.numsamples)
    
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
