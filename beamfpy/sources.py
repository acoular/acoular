# -*- coding: utf-8 -*-
"""
Created on Tue May 11 10:04:22 2010

@author: sarradj
"""
#pylint: disable-msg=E0611, E1101, C0103, C0111, R0901, R0902, R0903, R0904, W0232
# imports from other packages
from numpy import array, newaxis, pi, fft, exp
from numpy.random import normal, seed
from enthought.traits.api import Float, Int, Long, \
Property, Trait, Delegate, \
cached_property, Tuple
from enthought.traits.ui.api import View, Item
from enthought.traits.ui.menu import OKCancelButtons
from os import path
import tables
from scipy.signal import butter, cheby1, lfilter, filtfilt

# beamfpy imports
from timedomain import TimeSamples, TimeInOut
from internal import digest
from h5cache import H5cache
from grids import RectGrid, MicGeom, Environment

class PointSource( TimeSamples ):
    """
    point source class for simulations
    generates output via the generator 'result'
    """

    # rms amplitude of source signal in 1 m  distance
    rms = Float(1.0, 
        desc="rms amplitude")
    
    # seed for random number generator, defaults to 0
    seed = Int(0, 
        desc="random seed value")
    
    # location of source 
    loc = Tuple((0.0,0.0,1.0),
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
    numsamples = Long
    
    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'rms', 'loc', 'c', 'numsamples', \
        'sample_freq', 'seed', 'env.digest', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)

    # result generator: delivers output in blocks of num samples
    def result(self, num=128):
        seed(self.seed)
        signal = fft.fft(normal(scale=self.rms, size=self.numsamples))
        pos = array(self.loc, dtype=float).reshape(3,1)
        rm = self.env.r(self.c,pos,self.mpos.mpos)
        delays = exp(-2j*pi*(rm/self.c)*\
            fft.fftfreq(int(self.numsamples),1.0/self.sample_freq)[:,newaxis])
        out = fft.ifft(signal[:,newaxis]*delays,axis=0).real/rm
        i = 0
        while i < self.numsamples:
            yield out[i:i+num]
            i += num
        
