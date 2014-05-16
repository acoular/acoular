# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Beamfpy Development Team.
#------------------------------------------------------------------------------
"""Measured multichannel data managment and simulation of acoustic sources.

.. autosummary::
    :toctree: generated/

    SamplesGenerator
    TimeSamples
    MaskedTimeSamples
    PointSource
    MovingPointSource
"""

# imports from other packages
from numpy import array, sqrt, ones, empty, newaxis
from traits.api import Float, Int, Property, Trait, Delegate, \
cached_property, Tuple, HasPrivateTraits, CLong, File, Instance, Any, \
on_trait_change, List
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons
import tables
from os import path

# beamfpy imports
from .calib import Calib
from .trajectory import Trajectory
from .internal import digest
from .microphones import MicGeom
from .environments import Environment
from .signals import SignalGenerator

class SamplesGenerator( HasPrivateTraits ):
    """
    Base class for any generating signal processing block, 
    generates output via the generator 'result'
    """

    #: sample_freq of signal
    sample_freq = Float(1.0, 
        desc="sampling frequency")
    
    #: number of channels 
    numchannels = CLong
               
    #: number of samples 
    numsamples = CLong
    
    # internal identifier
    digest = ''
               
    def result(self, num):
        """ python generator: yields output in blocks of num samples """
        pass

class TimeSamples( SamplesGenerator ):
    """
    Container for time data, loads time data
    and provides information about this data
    """

    #: full name of the .h5 file with data
    name = File(filter=['*.h5'], 
        desc="name of data file")

    #: basename of the .h5 file with data
    basename = Property( depends_on = 'name', #filter=['*.h5'], 
        desc="basename of data file")
    
    #: calibration data
    calib = Trait( Calib, 
        desc="Calibration data")
    
    #: number of channels, is set automatically
    numchannels = CLong(0L, 
        desc="number of input channels")

    #: number of time data samples, is set automatically
    numsamples = CLong(0L, 
        desc="number of samples")

    #: the time data as (numsamples, numchannels) array of floats
    data = Any( transient = True, 
        desc="the actual time data array")

    #: hdf5 file object
    h5f = Instance(tables.File, transient = True)
    
    # internal identifier
    digest = Property( depends_on = ['basename', 'calib.digest'])

    traits_view = View(
        ['name{File name}', 
            ['sample_freq~{Sampling frequency}', 
            'numchannels~{Number of channels}', 
            'numsamples~{Number of samples}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Time data', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.name))[0]
    
    @on_trait_change('basename')
    def load_data( self ):
        """ open the .h5 file and setting attributes
        """
        if not path.isfile(self.name):
            # no file there
            self.numsamples = 0
            self.numchannels = 0
            self.sample_freq = 0
            raise IOError("No such file: %s" % self.name)
        if self.h5f != None:
            try:
                self.h5f.close()
            except IOError:
                pass
        self.h5f = tables.openFile(self.name)
        self.data = self.h5f.root.time_data
        self.sample_freq = self.data.getAttr('sample_freq')
        (self.numsamples, self.numchannels) = self.data.shape

    def result(self, num=128):
        """ 
        python generator: yields samples in blocks of shape (num, numchannels), 
        the last block may be shorter than num
        """
        if self.numsamples == 0:
            raise IOError("no samples available")
        i = 0
        if self.calib:
            if self.calib.num_mics == self.numchannels:
                cal_factor = self.calib.data[newaxis]
            else:
                raise ValueError("calibration data not compatible: %i, %i" % \
                            (self.calib.num_mics, self.numchannels))
            while i < self.numsamples:
                yield self.data[i:i+num]*cal_factor
                i += num
        else:
            while i < self.numsamples:
                yield self.data[i:i+num]
                i += num

class MaskedTimeSamples( TimeSamples ):
    """
    Container for time data, loads time data
    and provides information about this data, 
    stores information about valid samples and
    valid channels
    """
    
    #: start of valid samples
    start = CLong(0L, 
        desc="start of valid samples")
    
    #: stop of valid samples
    stop = Trait(None, None, CLong, 
        desc="stop of valid samples")
    
    #: invalid channels  
    invalid_channels = List(
        desc="list of invalid channels")
    
    #: channel mask to serve as an index for all valid channels
    channels = Property(depends_on = ['invalid_channels', 'numchannels_total'], 
        desc="channel mask")
        
    #: number of channels, is set automatically
    numchannels_total = CLong(0L, 
        desc="total number of input channels")

    #: number of time data samples, is set automatically
    numsamples_total = CLong(0L, 
        desc="total number of samples per channel")

    #: number of channels, is set automatically
    numchannels = Property(depends_on = ['invalid_channels', \
        'numchannels_total'], desc="number of valid input channels")

    #: number of time data samples, is set automatically
    numsamples = Property(depends_on = ['start', 'stop', 'numsamples_total'], 
        desc="number of valid samples per channel")

    # internal identifier
    digest = Property( depends_on = ['basename', 'start', 'stop', \
        'calib.digest', 'invalid_channels'])

    traits_view = View(
        ['name{File name}', 
         ['start{From sample}', Item('stop', label='to', style='text'), '-'], 
         'invalid_channels{Invalid channels}', 
            ['sample_freq~{Sampling frequency}', 
            'numchannels~{Number of channels}', 
            'numsamples~{Number of samples}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Time data', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.name))[0]
    
    @cached_property
    def _get_channels( self ):
        if len(self.invalid_channels)==0:
            return slice(0, None, None)
        allr = range(self.numchannels_total)
        for channel in self.invalid_channels:
            if channel in allr:
                allr.remove(channel)
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

    @on_trait_change('basename')
    def load_data( self ):
        """ open the .h5 file and setting attributes
        """
        if not path.isfile(self.name):
            # no file there
            self.numsamples_total = 0
            self.numchannels_total = 0
            self.sample_freq = 0
            raise IOError("No such file: %s" % self.name)
        if self.h5f != None:
            try:
                self.h5f.close()
            except IOError:
                pass
        self.h5f = tables.openFile(self.name)
        self.data = self.h5f.root.time_data
        self.sample_freq = self.data.getAttr('sample_freq')
        (self.numsamples_total, self.numchannels_total) = self.data.shape

    def result(self, num=128):
        """ 
        python generator: yields samples in blocks of shape (num, numchannels), 
        the last block may be shorter than num
        """
        sli = slice(self.start, self.stop).indices(self.numsamples_total)
        i = sli[0]
        stop = sli[1]
        cal_factor = 1.0
        if i >= stop:
            raise IOError("no samples available")
        if self.calib:
            if self.calib.num_mics == self.numchannels_total:
                cal_factor = self.calib.data[self.channels][newaxis]
            elif self.calib.num_mics == self.numchannels:
                cal_factor = self.calib.data[newaxis]
            else:
                raise ValueError("calibration data not compatible: %i, %i" % \
                            (self.calib.num_mics, self.numchannels))
        while i < stop:
            yield self.data[i:min(i+num, stop)][:, self.channels]*cal_factor
            i += num


class PointSource( SamplesGenerator ):
    """
    fixed point source class for simulations
    generates output via the generator 'result'
    """
    
    #: signal generator
    signal = Trait(SignalGenerator)
    
    #: location of source 
    loc = Tuple((0.0, 0.0, 1.0),
        desc="source location")
               
    #: number of channels in output
    numchannels = Delegate('mpos', 'num_mics')

    #: MicGeom object that provides the microphone locations
    mpos = Trait(MicGeom, 
        desc="microphone geometry")
        
    #: Environment object that provides grid-mic distances
    env = Trait(Environment(), Environment)

    #: the speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")
        
    #: the start time of the signal, in seconds
    start_t = Float(0.0,
        desc="signal start time")
    
    #: the start time of the data aquisition at microphones, in seconds
    start = Float(0.0,
        desc="sample start time")

    #: upsampling factor, internal use
    up = Int(16, 
        desc="upsampling factor")        
    
    #: number of samples 
    numsamples = Delegate('signal')
    
    #: sample_freq of signal
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

    #: trajectory, start time is assumed to be the same as for the samples
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
        # shortcuts and intial values
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
            te = t.copy() # init emission time = receiving time
            j = 0
            # Newton-Rhapson iteration
            while abs(eps).max()>epslim and j<100:
                loc = array(tr.location(te))
                rm = loc-m.mpos# distance vectors to microphones
                rm = sqrt((rm*rm).sum(0))# absolute distance
                loc /= sqrt((loc*loc).sum(0))# distance unit vector
                der = array(tr.location(te, der=1))
                Mr = (der*loc).sum(0)/c0# radial Mach number
                eps = (te + rm/c0 - t)/(1+Mr)# discrepancy in time 
                te -= eps
                j += 1 #iteration count
            t += 1./self.sample_freq
            # emission time relative to start time
            ind = (te-self.start_t+self.start)*self.sample_freq
            try:
                out[i] = signal[array(0.5+ind*self.up, dtype=long)]/rm
                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError: #if no ore samples available from the source 
                break
        yield out[:i]
