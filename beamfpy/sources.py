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
from numpy import array, newaxis, pi, fft, exp, arange, sin, zeros
from numpy.random import normal, seed
from enthought.traits.api import HasPrivateTraits, Float, Int, Long, \
Property, Trait, Delegate, cached_property, Tuple
from scipy.signal import cheby1, lfilter

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
        
    # Environment object that provides speed of sound and grid-mic distances
    env = Trait(Environment(), Environment)

    # the speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")
    
    # number of samples 
    numsamples = Delegate('signal')
    
    # sample_freq of signal
    sample_freq = Delegate('signal') 

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'signal.digest', 'loc', 'c', \
         'env.digest', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    def result(self, num=128):
        """ 
        python generator: yields source output at microphones in blocks of 
        shape (num, numchannels), the last block may be shorter than num
        """
        signal = fft.fft(self.signal.signal())
        pos = array(self.loc, dtype=float).reshape(3, 1)
        rm = self.env.r(self.c, pos, self.mpos.mpos)
        delays = exp(-2j*pi*(rm/self.c)*\
            fft.fftfreq(int(self.numsamples),1.0/self.sample_freq)[:,newaxis])
        out = fft.ifft(signal[:, newaxis]*delays, axis=0).real/rm
        i = 0
        while i < self.numsamples:
            yield out[i:i+num]
            i += num
        
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
        depends_on = ['mpos.digest', 'signal.digest', 'c', \
         'env.digest', 'trajectory.digest', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    def result(self, num=128):
        """ 
        python generator: yields source output at microphones in blocks of 
        shape (num, numchannels), the last block may be shorter than num
        """
        c = self.c/self.signal.sample_freq
        maxrm = self.env.r( self.c, array(self.trajectory.points.values()).T,
                              self.mpos.mpos).max()
        offset = int(1.1*maxrm/c)
        signal = self.signal.signal()
        upsample = 16
        # TODO: fix numerical problems with filter design
        (b, a) = cheby1(8, 0.05, 0.8/upsample)
        out = zeros(((self.numsamples+offset)*upsample, self.numchannels))
        d_index2 = arange(self.numchannels, dtype=int)
        g = self.trajectory.traj(1/self.signal.sample_freq)
        for i in xrange(self.numsamples):
            tpos = array(g.next())[:, newaxis]
            rm = self.env.r( self.c, tpos, self.mpos.mpos)
            d_index = array(upsample*rm/c, dtype=int)+i*upsample # integer index
            out[d_index, d_index2] += signal[i, newaxis]/rm
        i = 0
        while out[i].sum() == 0:
            i += 1
        out = lfilter(b, a, out[i:], axis=0)[::upsample][:self.numsamples]
        i = 0
        while i < self.numsamples:
            yield out[i:i+num]
            i += num
