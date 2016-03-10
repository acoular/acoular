# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------
"""Measured multichannel data managment and simulation of acoustic sources.

.. autosummary::
    :toctree: generated/

    SamplesGenerator
    TimeSamples
    MaskedTimeSamples
    PointSource
    PointSourceDipole
    MovingPointSource
    UncorrelatedNoiseSource
"""

# imports from other packages
from numpy import array, sqrt, ones, empty, newaxis, uint32, arange, dot
from traits.api import Float, Int, Property, Trait, Delegate, \
cached_property, Tuple, HasPrivateTraits, CLong, File, Instance, Any, \
on_trait_change, List, CArray
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons
import tables
from os import path

# acoular imports
from .calib import Calib
from .trajectory import Trajectory
from .internal import digest
from .microphones import MicGeom
from .environments import Environment
from .signals import SignalGenerator, WNoiseGenerator, PNoiseGenerator

class SamplesGenerator( HasPrivateTraits ):
    """
    Base class for any generating signal processing block
    
    It provides a common interface for all SamplesGenerator classes, which
    generate an output via the generator :meth:`result`.
    This class has no real functionality on its own and should not be 
    used directly.
    """

    #: Sampling frequency of the signal, defaults to 1.0
    sample_freq = Float(1.0, 
        desc="sampling frequency")
    
    #: Number of channels 
    numchannels = CLong
               
    #: Number of samples 
    numsamples = CLong
    
    # internal identifier
    digest = ''
               
    def result(self, num):
        """
        Python generator that yields the output block-wise.
                
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        No output since SamplesGenerator only represents a base class to derive
        other classes from
        """
        pass

class TimeSamples( SamplesGenerator ):
    """
    Container for time data in *.h5 format
    
    This class loads measured data from h5 files and
    and provides information about this data.
    It also serves as an interface where the data can be accessed
    (e.g. for use in a block chain) via the :meth:`result` generator.
    """

    #: Full name of the .h5 file with data
    name = File(filter=['*.h5'], 
        desc="name of data file")

    #: Basename of the .h5 file with data, is set automatically
    basename = Property( depends_on = 'name', #filter=['*.h5'], 
        desc="basename of data file")
    
    #: Calibration data, instance of :class:`~acoular.calib.Calib` class, optional 
    calib = Trait( Calib, 
        desc="Calibration data")
    
    #: Number of channels, is set automatically / read from file
    numchannels = CLong(0L, 
        desc="number of input channels")

    #: Number of time data samples, is set automatically / read from file
    numsamples = CLong(0L, 
        desc="number of samples")

    #: The time data as array of floats with dimension (numsamples, numchannels)
    data = Any( transient = True, 
        desc="the actual time data array")

    #: HDF5 file object
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
        #""" open the .h5 file and set attributes
        #"""
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
        self.h5f = tables.open_file(self.name)
        self.data = self.h5f.root.time_data
        self.sample_freq = self.data.get_attr('sample_freq')
        (self.numsamples, self.numchannels) = self.data.shape

    def result(self, num=128):
        """
        Python generator that yields the output block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
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
    Container for time data in *.h5 format
    
    This class loads measured data from h5 files and
    and provides information about this data.
    It supports storing information about (in)valid samples and (in)valid channels
    It also serves as an interface where the data can be accessed
    (e.g. for use in a block chain) via the :meth:`result` generator.
    
    """
    
    #: Index of the first sample to be considered valid
    start = CLong(0L, 
        desc="start of valid samples")
    
    #: Index of the last sample to be considered valid
    stop = Trait(None, None, CLong, 
        desc="stop of valid samples")
    
    #: Channels that are to be treated as invalid
    invalid_channels = List(
        desc="list of invalid channels")
    
    #: Channel mask to serve as an index for all valid channels, is set automatically
    channels = Property(depends_on = ['invalid_channels', 'numchannels_total'], 
        desc="channel mask")
        
    #: Number of channels (including invalid channels), is set automatically
    numchannels_total = CLong(0L, 
        desc="total number of input channels")

    #: Number of time data samples (including invalid samples), is set automatically
    numsamples_total = CLong(0L, 
        desc="total number of samples per channel")

    #: Number of valid channels, is set automatically
    numchannels = Property(depends_on = ['invalid_channels', \
        'numchannels_total'], desc="number of valid input channels")

    #: Number of valid time data samples, is set automatically
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
        #""" open the .h5 file and set attributes
        #"""
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
        self.h5f = tables.open_file(self.name)
        self.data = self.h5f.root.time_data
        self.sample_freq = self.data.get_attr('sample_freq')
        (self.numsamples_total, self.numchannels_total) = self.data.shape

    def result(self, num=128):
        """
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
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
    Class to define a fixed point source with an arbitrary signal.
    This can be used in simulations.
    
    The output is being generated via the :meth:`result` generator.
    """
    
    #:  Emitted signal, instance of the :class:`~acoular.signals.SignalGenerator` class
    signal = Trait(SignalGenerator)
    
    #: Location of source in (x, y, z) coordinates (left-oriented system)
    loc = Tuple((0.0, 0.0, 1.0),
        desc="source location")
               
    #: Number of channels in output, is automatically set / 
    #: depends on used microphone geometry
    numchannels = Delegate('mpos', 'num_mics')

    #: Microphone locations as provided by a 
    #: :class:`~acoular.microphones.MicGeom`-derived object
    mpos = Trait(MicGeom, 
        desc="microphone geometry")
        
    #: :class:`~acoular.environments.Environment` object 
    #: that provides distances from grid points to microphone positions
    env = Trait(Environment(), Environment)

    #: Speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")
        
    #: Start time of the signal in seconds, defaults to 0 s
    start_t = Float(0.0,
        desc="signal start time")
    
    #: Start time of the data aquisition at microphones in seconds, 
    #: defaults to 0 s
    start = Float(0.0,
        desc="sample start time")

    #: Upsampling factor, internal use, defaults to 16
    up = Int(16, 
        desc="upsampling factor")        
    
    #: Number of samples, is set automatically / 
    #: depends on :attr:`signal`
    numsamples = Delegate('signal')
    
    #: Sampling frequency of the signal, is set automatically / 
    #: depends on :attr:`signal`
    sample_freq = Delegate('signal') 

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'signal.digest', 'loc', 'c', \
         'env.digest', 'start_t', 'start', 'up', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)
           
    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        #If signal samples are needed for te < t_start, then samples are taken
        #from the end of the calculated signal.
        
   
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
            except IndexError: #if no more samples available from the source
                break
        if i > 0: # if there are still samples to yield
            yield out[:i]         


class MovingPointSource( PointSource ):
    """
    Class to define a point source with an arbitrary 
    signal moving along a given trajectory.
    This can be used in simulations.
    
    The output is being generated via the :meth:`result` generator.
    """

    #: Trajectory of the source, 
    #: instance of the :class:`~acoular.trajectory.Trajectory` class.
    #: The start time is assumed to be the same as for the samples
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
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """   
        #If signal samples are needed for te < t_start, then samples are taken
        #from the end of the calculated signal.
        
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
            except IndexError: #if no more samples available from the source 
                break
        if i > 0: # if there are still samples to yield
            yield out[:i]

class PointSourceDipole ( PointSource ):
    """
    Class to define a fixed point source with an arbitrary signal and
    dipole characteristics via superposition of two nearby inversely
    phased monopoles.
    This can be used in simulations.
    
    The output is being generated via the :meth:`result` generator.
    """
    
    #: Vector to define the orientation of the dipole lobes. Its magnitude
    #: governs the distance between the monopoles
    #: (dist = [lowest wavelength in spectrum] x [magnitude] x 1e-5).
    #: Note: Use vectors with order of magnitude around 1.0 or less 
    #: for good results.
    direction = Tuple((0.0, 0.0, 1.0),
        desc="dipole orientation and distance of the inversely phased monopoles")
    
    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'signal.digest', 'loc', 'c', \
         'env.digest', 'start_t', 'start', 'up', 'direction', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)
        
        
    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        #If signal samples are needed for te < t_start, then samples are taken
        #from the end of the calculated signal.
        
        mpos = self.mpos.mpos
        
        # position of the dipole as (3,1) vector
        loc = array(self.loc, dtype = float).reshape((3, 1)) 
        # direction vector from tuple
        direc = array(self.direction, dtype = float) * 1e-5
        direc_mag =  sqrt(dot(direc,direc))
        
        # normed direction vector
        direc_n = direc / direc_mag
        
        c = self.c
        
        # distance between monopoles as function of c, sample freq, direction vector
        dist = c / self.sample_freq * direc_mag
        
        # vector from dipole center to one of the monopoles
        dir2 = (direc_n * dist / 2.0).reshape((3, 1))
        
        signal = self.signal.usignal(self.up)
        out = empty((num, self.numchannels))
        
        # distance from dipole center to microphones
        rm = self.env.r(c, loc, mpos)
        
        # distances from monopoles to microphones
        rm1 = self.env.r(c, loc + dir2, mpos)
        rm2 = self.env.r(c, loc - dir2, mpos)
        
        # emission time relative to start_t (in samples) for first sample
        ind1 = (-rm1 / c - self.start_t + self.start) * self.sample_freq   
        ind2 = (-rm2 / c - self.start_t + self.start) * self.sample_freq
        
        i = 0
        n = self.numsamples        
        while n:
            n -= 1
            try:
                # subtract the second signal b/c of phase inversion
                out[i] = rm / dist * \
                         (signal[array(0.5 + ind1 * self.up, dtype=long)] / rm1 - \
                          signal[array(0.5 + ind2 * self.up, dtype=long)] / rm2)
                ind1 += 1.
                ind2 += 1.
                
                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError:
                break
            
        yield out[:i]


class UncorrelatedNoiseSource( SamplesGenerator ):
    """
    Class to simulate white or pink noise as uncorrelated signal at each
    channel.
    
    The output is being generated via the :meth:`result` generator.
    """
    
    #: Type of noise to generate at the channels. 
    #: The `~acoular.signals.SignalGenerator`-derived class has to 
    # feature the parameter "seed" (i.e. white or pink noise).
    signal = Trait(SignalGenerator,
                   desc = "type of noise")

    #: Array with seeds for random number generator.
    #: When left empty, arange(:attr:`numchannels`)+:attr:`signal`.seed  
    #: will be used
    seed = CArray(dtype = uint32,
                  desc = "random seed values")
    
    #: Number of channels in output, is automatically set / 
    #: depends on used microphone geometry
    numchannels = Delegate('mpos', 'num_mics')

    #: Microphone locations as provided by a 
    #: :class:`~acoular.microphones.MicGeom`-derived object
    mpos = Trait(MicGeom, 
        desc="microphone geometry")
        

    #: Speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")
        
    #: Start time of the signal in seconds, defaults to 0 s
    start_t = Float(0.0,
        desc="signal start time")
    
    #: Start time of the data aquisition at microphones in seconds, 
    #: defaults to 0 s
    start = Float(0.0,
        desc="sample start time")

    
    #: Number of samples, is set automatically / 
    #: depends on :attr:`signal`
    numsamples = Delegate('signal')
    
    #: Sampling frequency of the signal, is set automatically / 
    #: depends on :attr:`signal`
    sample_freq = Delegate('signal') 
    
    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'signal.rms', 'signal.numsamples', \
        'signal.sample_freq', 'signal.__class__' , 'seed', 'loc', 'c', \
         'start_t', 'start', '__class__'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    def result ( self, num=128 ):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """

        Noise = self.signal.__class__
        # create or get the array of random seeds
        if not self.seed:            
            seed = arange(self.numchannels) + self.signal.seed
        elif self.seed.shape == (self.numchannels,):
            seed = self.seed
        else:
            raise ValueError(\
               "Seed array expected to be of shape (%i,), but has shape %s." \
                % (self.numchannels, str(self.seed.shape)) )
        
        # create array with [numchannels] noise signal tracks
        signal = array([Noise(seed = s, 
                              numsamples = self.numsamples,
                              sample_freq = self.sample_freq,
                              rms = self.signal.rms).signal() \
                        for s in seed]).T

        n = num        
        while n <= self.numsamples:
            yield signal[n-num:n,:]
            n += num
        else:
            yield signal[n-num:,:]



class SourceMixer( SamplesGenerator ):
    """
    Mixes the signals from several sources. 
    """

    #: List of :class:`~beamfpy.sources.SamplesGenerator` objects
    #: to be mixed.
    sources = List( Instance(SamplesGenerator, ()) ) 

    #: Sampling frequency of the signal
    sample_freq = Trait( SamplesGenerator().sample_freq )
    
    #: Number of channels 
    numchannels = Trait( SamplesGenerator().numchannels )
               
    #: Number of samples 
    numsamples = Trait( SamplesGenerator().numsamples )
    
    # internal identifier
    ldigest = Property( depends_on = ['sources.digest', ])

    # internal identifier
    digest = Property( depends_on = ['ldigest', '__class__'])

    traits_view = View(
        Item('sources', style='custom')
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

    @on_trait_change('sources')
    def validate_sources( self ):
        """ validates if sources fit together """
        if self.sources:
            self.sample_freq = self.sources[0].sample_freq
            self.numchannels = self.sources[0].numchannels
            self.numsamples = self.sources[0].numsamples
            for s in self.sources[1:]:
                if self.sample_freq != s.sample_freq:
                    raise ValueError("Sample frequency of %s does not fit" % s)
                if self.numchannels != s.numchannels:
                    raise ValueError("Channel count of %s does not fit" % s)
                if self.numsamples != s.numsamples:
                    raise ValueError("Number of samples of %s does not fit" % s)


    def result(self, num):
        """
        Python generator that yields the output block-wise.
        The outputs from the sources in the list are being added.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        gens = [i.result(num) for i in self.sources[1:]]
        for temp in self.sources[0].result(num):
            sh = temp.shape[0]
            for g in gens:
                temp1 = g.next()
                if temp.shape[0] > temp1.shape[0]:
                    temp = temp[:temp1.shape[0]]
                temp += temp1[:temp.shape[0]]
            yield temp
            if sh > temp.shape[0]:
                break