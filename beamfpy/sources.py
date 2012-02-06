# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103
#pylint: disable-msg=R0901, R0902, R0903, R0904, R0914, W0232
"""
sources.py: classes for simulated signals and sources

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming

(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""

# imports from other packages
from numpy import array, pi, arange, sin, sqrt, ones, empty
from numpy.random import normal, seed
from enthought.traits.api import HasPrivateTraits, Float, Int, Long, \
Property, Trait, Delegate, cached_property, Tuple
from scipy.signal import resample

# beamfpy imports
from timedomain import SamplesGenerator, Trajectory
from internal import digest
from grids import MicGeom, Environment

class SignalGenerator( HasPrivateTraits ):
    """
    Base class for a simple one-channel signal generator,
    generates output via the function 'signal'
    """

    # rms amplitude of source signal in 1 m  distance
    rms = Float(1.0, 
        desc="rms amplitude")
    
    # sample_freq of signal
    sample_freq = Float(1.0, 
        desc="sampling frequency")
                   
    # number of samples 
    numsamples = Long
    
    # internal identifier
    digest = ''
               
    def signal(self):
        """delivers the signal as an array of length numsamples"""
        pass
    
    def usignal(self, factor):
        """
        delivers the upsampled signal as an array of length factor*numsamples
        """
        return resample(self.signal(), factor*self.numsamples)

class WNoiseGenerator( SignalGenerator ):
    """
    Simple one-channel white noise signal generator
    """
    # seed for random number generator, defaults to 0
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
        """delivers the signal as an array of length numsamples"""
        seed(self.seed)
        return self.rms*normal(scale=1.0, size=self.numsamples)
    
class SineGenerator( SignalGenerator ):
    """
    Simple one-channel sine signal generator
    """

    # Sine wave frequency
    freq = Float(1000.0, 
        desc="Frequency")

    # Sine wave phase
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
        """delivers the signal as an array of length numsamples"""
        t = arange(self.numsamples, dtype=float)/self.sample_freq
        return self.rms*sin(2*pi*self.freq*t+self.phase)

class PointSource( SamplesGenerator ):
    """
    fixed point source class for simulations
    generates output via the generator 'result'
    """
    
    # signal generator
    signal = Trait(SignalGenerator)
    
    # location of source 
    loc = Tuple((0.0, 0.0, 1.0),
        desc="source location")
               
    # number of channels in output
    numchannels = Delegate('mpos', 'num_mics')

    # MicGeom object that provides the microphone locations
    mpos = Trait(MicGeom, 
        desc="microphone geometry")
        
    # Environment object that provides grid-mic distances
    env = Trait(Environment(), Environment)

    # the speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")
        
    # the start time of the signal, in seconds
    start_t = Float(0.0,
        desc="signal start time")
    
    # the start time of the data aquisition at microphones, in seconds
    start = Float(0.0,
        desc="sample start time")

    # upsampling factor, internal use
    up = Int(16, 
        desc="upsampling factor")        
    
    # number of samples 
    numsamples = Delegate('signal')
    
    # sample_freq of signal
    sample_freq = Delegate('signal') 

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'signal.digest', 'loc', 'c', \
         'env.digest', 'start_t', 'start', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)
           
    def result(self, num=128):
        """ 
        python generator: yields source output at microphones in blocks of 
        shape (num, numchannels), the last block may be shorter than num
        if signal samples are needed for te < t_start, then samples are taken
        
        """       
        signal = self.signal.usignal(self.up)
        out = empty((num, self.numchannels))
        # distances
        rm = self.env.r(self.c, array(self.loc).reshape((3, 1)), self.mpos.mpos)
        # emission time relative to start_t (in samples) for first sample
        ind = (-rm/self.c-self.start_t+self.start)*self.sample_freq   
        i = 0
        n = self.numsamples        
        while n:
            n -= 1
            try:
                out[i] = signal[array(0.5+ind*self.up, dtype=long)]/rm
                ind += 1.
                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError:
                break
        yield out[:i]            

class MovingPointSource( PointSource ):
    """
    point source class for simulations that moves along a given trajectory
    generates output via the generator 'result'
    """

    # trajectory, start time is assumed to be the same as for the samples
    trajectory = Trait(Trajectory, 
        desc="trajectory of the source")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'signal.digest', 'loc', 'c', \
         'env.digest', 'start_t', 'start', 'trajectory.digest', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)

    def result(self, num=128):
        """ 
        python generator: yields source output at microphones in blocks of 
        shape (num, numchannels), the last block may be shorter than num
        """       
        signal = self.signal.usignal(self.up)
        out = empty((num, self.numchannels))
        m = self.mpos
        t = self.start*ones(m.num_mics)
        i = 0
        epslim = 0.1/self.up/self.sample_freq
        c0 = self.c
        tr = self.trajectory
        n = self.numsamples
        while n:
            n -= 1
            eps = ones(m.num_mics)
            te = t.copy()
            j = 0
            while abs(eps).max()>epslim and j<100:
                loc = array(tr.location(te))
                rm = loc-m.mpos
                rm = sqrt((rm*rm).sum(0))
                loc /= sqrt((loc*loc).sum(0))
                der = array(tr.location(te, der=1))
                Mr = (der*loc).sum(0)/c0
                eps = (te + rm/c0 - t)/(1+Mr)
                te -= eps
                j += 1
            t += 1./self.sample_freq
            ind = (te-self.start_t+self.start)*self.sample_freq
            try:
                out[i] = signal[array(0.5+ind*self.up, dtype=long)]/rm
                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError:
                break
        yield out[:i]
