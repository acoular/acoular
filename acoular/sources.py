# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Measured multichannel data managment and simulation of acoustic sources.

.. autosummary::
    :toctree: generated/

    TimeSamples
    MaskedTimeSamples
    PointSource
    PointSourceDipole
    SphericalHarmonicSource
    LineSource
    MovingPointSource
    MovingPointSourceDipole
    MovingLineSource
    UncorrelatedNoiseSource
    SourceMixer
    PointSourceConvolve
"""

# imports from other packages

from numpy import array, sqrt, ones, empty, newaxis, uint32, arange, dot, int64 ,real, pi, tile,\
cross, zeros, ceil, repeat
from numpy import min as npmin
from numpy import any as npany

from numpy.fft import ifft, fft
from numpy.linalg import norm

import numba as nb

from traits.api import Float, Int, Property, Trait, Delegate, \
cached_property, Tuple, CLong, File, Instance, Any, Str, \
on_trait_change, observe, List, ListInt, CArray, Bool, Dict, Enum
from os import path
from warnings import warn

# acoular imports
from .calib import Calib
from .trajectory import Trajectory
from .internal import digest, ldigest
from .microphones import MicGeom
from .environments import Environment
from .tprocess import SamplesGenerator, TimeConvolve
from .signals import SignalGenerator
from .h5files import H5FileBase, _get_h5file_class
from .tools import get_modes


@nb.njit(cache=True, error_model="numpy") # jit with nopython        
def _fill_mic_signal_block(out,signal,rm,ind,blocksize,numchannels,up,prepadding):
    if prepadding:
        for b in range(blocksize):
            for m in range(numchannels):
                if ind[0,m]<0:
                    out[b,m] = 0
                else:
                    out[b,m] = signal[int(0.5+ind[0,m])]/rm[0,m]
            ind += up
    else:
        for b in range(blocksize):
            for m in range(numchannels):
                out[b,m] = signal[int(0.5+ind[0,m])]/rm[0,m]
            ind += up
    return out


class TimeSamples( SamplesGenerator ):
    """
    Container for time data in `*.h5` format.
    
    This class loads measured data from h5 files and
    and provides information about this data.
    It also serves as an interface where the data can be accessed
    (e.g. for use in a block chain) via the :meth:`result` generator.
    """

    #: Full name of the .h5 file with data.
    name = File(filter=['*.h5'], 
        desc="name of data file")

    #: Basename of the .h5 file with data, is set automatically.
    basename = Property( depends_on = 'name', #filter=['*.h5'], 
        desc="basename of data file")
    
    #: Calibration data, instance of :class:`~acoular.calib.Calib` class, optional .
    calib = Trait( Calib, 
        desc="Calibration data")
    
    #: Number of channels, is set automatically / read from file.
    numchannels = CLong(0, 
        desc="number of input channels")

    #: Number of time data samples, is set automatically / read from file.
    numsamples = CLong(0, 
        desc="number of samples")

    #: The time data as array of floats with dimension (numsamples, numchannels).
    data = Any( transient = True, 
        desc="the actual time data array")

    #: HDF5 file object
    h5f = Instance(H5FileBase, transient = True)
    
    #: Provides metadata stored in HDF5 file object
    metadata = Dict(
        desc="metadata contained in .h5 file")
    
    # Checksum over first data entries of all channels
    _datachecksum = Property()
    
    # internal identifier
    digest = Property( depends_on = ['basename', 'calib.digest', '_datachecksum'])

    def _get__datachecksum( self ):
        return self.data[0,:].sum()
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.name))[0]
    
    @on_trait_change('basename')
    def load_data( self ):
        """ 
        Open the .h5 file and set attributes.
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
        file = _get_h5file_class()
        self.h5f = file(self.name)
        self.load_timedata()
        self.load_metadata()

    def load_timedata( self ):
        """ loads timedata from .h5 file. Only for internal use. """
        self.data = self.h5f.get_data_by_reference('time_data')    
        self.sample_freq = self.h5f.get_node_attribute(self.data,'sample_freq')
        (self.numsamples, self.numchannels) = self.data.shape

    def load_metadata( self ):
        """ loads metadata from .h5 file. Only for internal use. """
        self.metadata = {}
        if '/metadata' in self.h5f:
            for nodename, nodedata in self.h5f.get_child_nodes('/metadata'):
                self.metadata[nodename] = nodedata

    def result(self, num=128):
        """
        Python generator that yields the output block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        if self.numsamples == 0:
            raise IOError("no samples available")
        self._datachecksum # trigger checksum calculation
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
    Container for time data in `*.h5` format.
    
    This class loads measured data from h5 files 
    and provides information about this data.
    It supports storing information about (in)valid samples and (in)valid channels
    It also serves as an interface where the data can be accessed
    (e.g. for use in a block chain) via the :meth:`result` generator.
    
    """
    
    #: Index of the first sample to be considered valid.
    start = CLong(0, 
        desc="start of valid samples")
    
    #: Index of the last sample to be considered valid.
    stop = Trait(None, None, CLong, 
        desc="stop of valid samples")
    
    #: Channels that are to be treated as invalid.
    invalid_channels = ListInt(
        desc="list of invalid channels")
    
    #: Channel mask to serve as an index for all valid channels, is set automatically.
    channels = Property(depends_on = ['invalid_channels', 'numchannels_total'], 
        desc="channel mask")
        
    #: Number of channels (including invalid channels), is set automatically.
    numchannels_total = CLong(0, 
        desc="total number of input channels")

    #: Number of time data samples (including invalid samples), is set automatically.
    numsamples_total = CLong(0, 
        desc="total number of samples per channel")

    #: Number of valid channels, is set automatically.
    numchannels = Property(depends_on = ['invalid_channels', \
        'numchannels_total'], desc="number of valid input channels")

    #: Number of valid time data samples, is set automatically.
    numsamples = Property(depends_on = ['start', 'stop', 'numsamples_total'], 
        desc="number of valid samples per channel")

    # internal identifier
    digest = Property( depends_on = ['basename', 'start', 'stop', \
        'calib.digest', 'invalid_channels','_datachecksum'])

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
        allr=[i for i in range(self.numchannels_total) if i not in self.invalid_channels]
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
        file = _get_h5file_class()
        self.h5f = file(self.name)
        self.load_timedata()
        self.load_metadata()

    def load_timedata( self ):
        """ loads timedata from .h5 file. Only for internal use. """
        self.data = self.h5f.get_data_by_reference('time_data')    
        self.sample_freq = self.h5f.get_node_attribute(self.data,'sample_freq')
        (self.numsamples_total, self.numchannels_total) = self.data.shape

    def result(self, num=128):
        """
        Python generator that yields the output block-wise.
        
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
        sli = slice(self.start, self.stop).indices(self.numsamples_total)
        i = sli[0]
        stop = sli[1]
        cal_factor = 1.0
        if i >= stop:
            raise IOError("no samples available")
        self._datachecksum # trigger checksum calculation
        if self.calib:
            if self.calib.num_mics == self.numchannels_total:
                cal_factor = self.calib.data[self.channels][newaxis]
            elif self.calib.num_mics == self.numchannels:
                cal_factor = self.calib.data[newaxis]
            elif self.calib.num_mics == 0:
                warn("No calibration data used.", Warning, stacklevel = 2)    
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
    
    #:  Emitted signal, instance of the :class:`~acoular.signals.SignalGenerator` class.
    signal = Trait(SignalGenerator)
    
    #: Location of source in (`x`, `y`, `z`) coordinates (left-oriented system).
    loc = Tuple((0.0, 0.0, 1.0),
        desc="source location")
               
    #: Number of channels in output, is set automatically / 
    #: depends on used microphone geometry.
    numchannels = Delegate('mics', 'num_mics')

    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Trait(MicGeom, 
        desc="microphone geometry")

    def _validate_locations(self):
        dist = self.env._r(array(self.loc).reshape((3, 1)), self.mics.mpos)
        if npany(dist < 1e-7):
            warn("Source and microphone locations are identical.", Warning, stacklevel = 2)
    
    #: :class:`~acoular.environments.Environment` or derived object, 
    #: which provides information about the sound propagation in the medium.
    env = Trait(Environment(), Environment)

    # --- List of backwards compatibility traits and their setters/getters -----------

    # Microphone locations.
    # Deprecated! Use :attr:`mics` trait instead.
    mpos = Property()
    
    def _get_mpos(self):
        return self.mics
    
    def _set_mpos(self, mpos):
        warn("Deprecated use of 'mpos' trait. ", Warning, stacklevel = 2)
        self.mics = mpos
        
    # The speed of sound.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`env` trait.
    c = Property()
    
    def _get_c(self):
        return self.env.c
    
    def _set_c(self, c):
        warn("Deprecated use of 'c' trait. ", Warning, stacklevel = 2)
        self.env.c = c

    # --- End of backwards compatibility traits --------------------------------------
        
    #: Start time of the signal in seconds, defaults to 0 s.
    start_t = Float(0.0,
        desc="signal start time")
    
    #: Start time of the data aquisition at microphones in seconds, 
    #: defaults to 0 s.
    start = Float(0.0,
        desc="sample start time")

    #: Signal behaviour for negative time indices, i.e. if :attr:`start` < :attr:start_t.
    #: `loop` take values from the end of :attr:`signal.signal()` array.
    #: `zeros` set source signal to zero, advisable for deterministic signals.
    #: defaults to `loop`.
    prepadding = Trait('loop','zeros', desc="Behaviour for negative time indices.")

    #: Upsampling factor, internal use, defaults to 16.
    up = Int(16, 
        desc="upsampling factor")        
    
    #: Number of samples, is set automatically / 
    #: depends on :attr:`signal`.
    numsamples = Delegate('signal')
    
    #: Sampling frequency of the signal, is set automatically / 
    #: depends on :attr:`signal`.
    sample_freq = Delegate('signal') 

    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', \
         'env.digest', 'start_t', 'start', 'up', 'prepadding', '__class__'], 
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
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        
        self._validate_locations()
        N = int(ceil(self.numsamples/num)) # number of output blocks
        signal = self.signal.usignal(self.up)
        out = empty((num, self.numchannels))
        # distances
        rm = self.env._r(array(self.loc).reshape((3, 1)), self.mics.mpos).reshape(1,-1)
        # emission time relative to start_t (in samples) for first sample
        ind = (-rm/self.env.c-self.start_t+self.start)*self.sample_freq*self.up

        if self.prepadding == 'zeros':
            # number of blocks where signal behaviour is amended
            pre = -int(npmin(ind[0])//(self.up*num))
            # amend signal for first blocks
            # if signal stops during prepadding, terminate
            if N <= pre:
                for nb in range(N-1):
                    out = _fill_mic_signal_block(out,signal,rm,ind,num,self.numchannels,self.up,True)
                    yield out

                blocksize = self.numsamples%num or num
                out = _fill_mic_signal_block(out,signal,rm,ind,blocksize,self.numchannels,self.up,True)
                yield out[:blocksize]
                return
            else:
                for nb in range(pre):
                    out = _fill_mic_signal_block(out,signal,rm,ind,num,self.numchannels,self.up,True)
                    yield out

        else:
            pre = 0

        # main generator
        for nb in range(N-pre-1):
            out = _fill_mic_signal_block(out,signal,rm,ind,num,self.numchannels,self.up,False)
            yield out

        # last block of variable size
        blocksize = self.numsamples%num or num
        out = _fill_mic_signal_block(out,signal,rm,ind,blocksize,self.numchannels,self.up,False)
        yield out[:blocksize]


class SphericalHarmonicSource( PointSource ):
    """
    Class to define a fixed Spherical Harmonic Source with an arbitrary signal.
    This can be used in simulations.
    
    The output is being generated via the :meth:`result` generator.
    """
    
    #: Order of spherical harmonic source
    lOrder = Int(0,
                   desc ="Order of spherical harmonic")
    
    alpha = CArray(desc="coefficients of the (lOrder,) spherical harmonic mode")
    
    
    #: Vector to define the orientation of the SphericalHarmonic. 
    direction = Tuple((1.0, 0.0, 0.0),
        desc="Spherical Harmonic orientation")

    prepadding = Enum('loop', desc="Behaviour for negative time indices.")
    
    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', \
         'env.digest', 'start_t', 'start', 'up', '__class__', 'alpha','lOrder', 'prepadding'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)

    def transform(self,signals):
        Y_lm = get_modes(lOrder = self.lOrder, direction= self.direction, mpos = self.mics.mpos,sourceposition = array(self.loc))
        return real(ifft(fft(signals,axis=0) * (Y_lm @ self.alpha),axis=0))

    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        #If signal samples are needed for te < t_start, then samples are taken
        #from the end of the calculated signal.
        
        signal = self.signal.usignal(self.up)
        # emission time relative to start_t (in samples) for first sample
        rm = self.env._r(array(self.loc).reshape((3, 1)), self.mics.mpos)
        ind = (-rm/self.env.c-self.start_t+self.start)*self.sample_freq   + pi/30 
        i = 0
        n = self.numsamples
        out = empty((num, self.numchannels))
        while n:
            n -= 1
            try:
                out[i] = signal[array(0.5+ind*self.up, dtype=int64)]/rm
                ind += 1
                i += 1
                if i == num:
                    yield self.transform(out)
                    i = 0
            except IndexError: #if no more samples available from the source
                break
        if i > 0: # if there are still samples to yield
            yield self.transform(out[:i])
            
class MovingPointSource( PointSource ):
    """
    Class to define a point source with an arbitrary 
    signal moving along a given trajectory.
    This can be used in simulations.
    
    The output is being generated via the :meth:`result` generator.
    """

    #: Considering of convective amplification
    conv_amp = Bool(False, 
        desc="determines if convective amplification is considered")

    #: Trajectory of the source, 
    #: instance of the :class:`~acoular.trajectory.Trajectory` class.
    #: The start time is assumed to be the same as for the samples.
    trajectory = Trait(Trajectory, 
        desc="trajectory of the source")

    prepadding = Enum('loop', desc="Behaviour for negative time indices.")

    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', 'conv_amp', \
         'env.digest', 'start_t', 'start', 'trajectory.digest', 'prepadding', '__class__'], 
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
        m = self.mics
        t = self.start*ones(m.num_mics)
        i = 0
        epslim = 0.1/self.up/self.sample_freq
        c0 = self.env.c
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
            if self.conv_amp: rm *= (1-Mr)**2
            try:
                out[i] = signal[array(0.5+ind*self.up, dtype=int64)]/rm
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

    prepadding = Enum('loop', desc="Behaviour for negative time indices.")
    
    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', \
         'env.digest', 'start_t', 'start', 'up', 'direction', 'prepadding', '__class__'], 
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
            (i.e. the number of samples per block) .
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        #If signal samples are needed for te < t_start, then samples are taken
        #from the end of the calculated signal.
        
        mpos = self.mics.mpos
        # position of the dipole as (3,1) vector
        loc = array(self.loc, dtype = float).reshape((3, 1)) 
        # direction vector from tuple
        direc = array(self.direction, dtype = float) * 1e-5
        direc_mag =  sqrt(dot(direc,direc))
        
        # normed direction vector
        direc_n = direc / direc_mag
        
        c = self.env.c
        
        # distance between monopoles as function of c, sample freq, direction vector
        dist = c / self.sample_freq * direc_mag
        
        # vector from dipole center to one of the monopoles
        dir2 = (direc_n * dist / 2.0).reshape((3, 1))
        
        signal = self.signal.usignal(self.up)
        out = empty((num, self.numchannels))
        
        # distance from dipole center to microphones
        rm = self.env._r(loc, mpos)
        
        # distances from monopoles to microphones
        rm1 = self.env._r(loc + dir2, mpos)
        rm2 = self.env._r(loc - dir2, mpos)
        
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
                         (signal[array(0.5 + ind1 * self.up, dtype=int64)] / rm1 - \
                          signal[array(0.5 + ind2 * self.up, dtype=int64)] / rm2)
                ind1 += 1.
                ind2 += 1.
                
                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError:
                break
            
        yield out[:i]

class MovingPointSourceDipole(PointSourceDipole, MovingPointSource):
    
    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', \
         'env.digest', 'start_t', 'start', 'up', 'direction', '__class__'], 
        )
               
    #: Reference vector, perpendicular to the x and y-axis of moving source.
    #: rotation source directivity around this axis
    rvec = CArray( dtype=float, shape=(3, ), value=array((0, 0, 0)), 
        desc="reference vector")
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    

    def get_emission_time(self,t,direction):
        eps = ones(self.mics.num_mics)
        epslim = 0.1/self.up/self.sample_freq
        te = t.copy() # init emission time = receiving time
        j = 0
        # Newton-Rhapson iteration
        while abs(eps).max()>epslim and j<100:
            xs = array(self.trajectory.location(te))
            loc = xs.copy()
            loc += direction
            rm = loc-self.mics.mpos# distance vectors to microphones
            rm = sqrt((rm*rm).sum(0))# absolute distance
            loc /= sqrt((loc*loc).sum(0))# distance unit vector
            der = array(self.trajectory.location(te, der=1))
            Mr = (der*loc).sum(0)/self.env.c# radial Mach number
            eps = (te + rm/self.env.c - t)/(1+Mr)# discrepancy in time 
            te -= eps
            j += 1 #iteration count
        return te, rm, Mr, xs
           
                
    def get_moving_direction(self,direction,time=0):
        """
        function that yields the moving coordinates along the trajectory  
        """

        trajg1 = array(self.trajectory.location( time, der=1))[:,0][:,newaxis]
        rflag = (self.rvec == 0).all() #flag translation vs. rotation
        if rflag:
            return direction 
        else:
            dx = array(trajg1.T) #direction vector (new x-axis)
            dy = cross(self.rvec, dx) # new y-axis
            dz = cross(dx, dy) # new z-axis
            RM = array((dx, dy, dz)).T # rotation matrix
            RM /= sqrt((RM*RM).sum(0)) # column normalized
            newdir = dot(RM, direction)
            return cross(newdir[:,0].T,self.rvec.T).T

    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        #If signal samples are needed for te < t_start, then samples are taken
        #from the end of the calculated signal.
        mpos = self.mics.mpos
        
        # direction vector from tuple
        direc = array(self.direction, dtype = float) * 1e-5
        direc_mag =  sqrt(dot(direc,direc))   
        # normed direction vector
        direc_n = direc / direc_mag
        c = self.env.c
        # distance between monopoles as function of c, sample freq, direction vector
        dist = c / self.sample_freq * direc_mag * 2
        
        # vector from dipole center to one of the monopoles
        dir2 = (direc_n * dist / 2.0).reshape((3, 1))
        
        signal = self.signal.usignal(self.up)
        out = empty((num, self.numchannels))
        # shortcuts and intial values
        m = self.mics
        t = self.start*ones(m.num_mics)

        i = 0
        n = self.numsamples        
        while n:
            n -= 1
            te, rm, Mr, locs = self.get_emission_time(t,0)                
            t += 1./self.sample_freq
            #location of the center
            loc = array(self.trajectory.location(te), dtype = float)[:,0][:,newaxis] 
            #distance of the dipoles from the center
            diff = self.get_moving_direction(dir2,te)
 
            # distance of sources
            rm1 = self.env._r(loc + diff, mpos) 
            rm2 = self.env._r(loc - diff, mpos)
                                   
            ind = (te-self.start_t+self.start)*self.sample_freq
            if self.conv_amp: 
                rm *= (1-Mr)**2
                rm1 *= (1-Mr)**2 # assume that Mr is the same for both poles
                rm2 *= (1-Mr)**2
            try:
                # subtract the second signal b/c of phase inversion
                out[i] = rm / dist * \
                         (signal[array(0.5 + ind * self.up, dtype=int64)] / rm1 - \
                          signal[array(0.5 + ind * self.up, dtype=int64)] / rm2)
                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError:
                break
        yield out[:i]        



class LineSource( PointSource ):
    """
    Class to define a fixed Line source with an arbitrary signal.
    This can be used in simulations.
    
    The output is being generated via the :meth:`result` generator.
    """
    
    #: Vector to define the orientation of the line source
    direction = Tuple((0.0, 0.0, 1.0),
        desc="Line orientation ")
    
    #: Vector to define the length of the line source in m
    length = Float(1,desc="length of the line source")
    
    #: number of monopol sources in the line source
    num_sources = Int(1)
    
    #: source strength for every monopole
    source_strength = CArray(desc="coefficients of the source strength")
    
    #:coherence
    coherence = Trait( 'coherent', 'incoherent', 
        desc="coherence mode")
       
    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', \
         'env.digest', 'start_t', 'start', 'up', 'direction',\
             'source_strength','coherence','__class__'], 

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
            (i.e. the number of samples per block) .
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        #If signal samples are needed for te < t_start, then samples are taken
        #from the end of the calculated signal.

        mpos = self.mics.mpos
        
        # direction vector from tuple
        direc = array(self.direction, dtype = float)         
        # normed direction vector
        direc_n = direc/norm(direc)
        c = self.env.c
        
        # distance between monopoles in the line 
        dist = self.length / self.num_sources 
        
        #blocwise output
        out = zeros((num, self.numchannels))
        
        # distance from line start position to microphones   
        loc = array(self.loc, dtype = float).reshape((3, 1)) 
        
        # distances from monopoles in the line to microphones
        rms = empty(( self.numchannels,self.num_sources))
        inds = empty((self.numchannels,self.num_sources))
        signals = empty((self.num_sources, len(self.signal.usignal(self.up))))
        #for every source - distances
        for s in range(self.num_sources):
            rms[:,s] = self.env._r((loc.T+direc_n*dist*s).T, mpos)
            inds[:,s] = (-rms[:,s]  / c - self.start_t + self.start) * self.sample_freq 
            #new seed for every source
            if self.coherence == 'incoherent':
                self.signal.seed = s + abs(int(hash(self.digest)//10e12))
            self.signal.rms = self.signal.rms * self.source_strength[s]
            signals[s] = self.signal.usignal(self.up)
        i = 0
        n = self.numsamples        
        while n:
            n -= 1
            try:
                for s in range(self.num_sources):
                # sum sources
                    out[i] += (signals[s,array(0.5 + inds[:,s].T * self.up, dtype=int64)] / rms[:,s])
                
                inds += 1.
                i += 1
                if i == num:
                    yield out
                    out = zeros((num, self.numchannels))
                    i = 0
            except IndexError:
                break
            
        yield out[:i]
        
class MovingLineSource(LineSource,MovingPointSource):
    
    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', \
         'env.digest', 'start_t', 'start', 'up', 'direction', '__class__'], 
        )
        

    #: Reference vector, perpendicular to the x and y-axis of moving source.
    #: rotation source directivity around this axis
    rvec = CArray( dtype=float, shape=(3, ), value=array((0, 0, 0)), 
        desc="reference vector")
    
    @cached_property
    def _get_digest( self ):
        return digest(self) 
    
    def get_moving_direction(self,direction,time=0):
        """
        function that yields the moving coordinates along the trajectory  
        """
    
        trajg1 = array(self.trajectory.location( time, der=1))[:,0][:,newaxis]
        rflag = (self.rvec == 0).all() #flag translation vs. rotation
        if rflag:
            return direction 
        else:
            dx = array(trajg1.T) #direction vector (new x-axis)
            dy = cross(self.rvec, dx) # new y-axis
            dz = cross(dx, dy) # new z-axis
            RM = array((dx, dy, dz)).T # rotation matrix
            RM /= sqrt((RM*RM).sum(0)) # column normalized
            newdir = dot(RM, direction)
            return cross(newdir[:,0].T,self.rvec.T).T

    def get_emission_time(self,t,direction):
        eps = ones(self.mics.num_mics)
        epslim = 0.1/self.up/self.sample_freq
        te = t.copy() # init emission time = receiving time
        j = 0
        # Newton-Rhapson iteration
        while abs(eps).max()>epslim and j<100:
            xs = array(self.trajectory.location(te))
            loc = xs.copy()
            loc += direction
            rm = loc-self.mics.mpos# distance vectors to microphones
            rm = sqrt((rm*rm).sum(0))# absolute distance
            loc /= sqrt((loc*loc).sum(0))# distance unit vector
            der = array(self.trajectory.location(te, der=1))
            Mr = (der*loc).sum(0)/self.env.c# radial Mach number
            eps = (te + rm/self.env.c - t)/(1+Mr)# discrepancy in time 
            te -= eps
            j += 1 #iteration count
        return te, rm, Mr, xs
        
    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        
        #If signal samples are needed for te < t_start, then samples are taken
        #from the end of the calculated signal.
        mpos = self.mics.mpos
        
        # direction vector from tuple
        direc = array(self.direction, dtype = float)         
        # normed direction vector
        direc_n = direc/norm(direc)
        
        # distance between monopoles in the line 
        dist = self.length / self.num_sources 
        dir2 = (direc_n * dist).reshape((3, 1))
        
        #blocwise output
        out = zeros((num, self.numchannels))
        
        # distances from monopoles in the line to microphones
        rms = empty(( self.numchannels,self.num_sources))
        inds = empty((self.numchannels,self.num_sources))
        signals = empty((self.num_sources, len(self.signal.usignal(self.up))))
        #coherence
        for s in range(self.num_sources):
            #new seed for every source
            if self.coherence == 'incoherent':
                self.signal.seed = s + abs(int(hash(self.digest)//10e12))
            self.signal.rms = self.signal.rms * self.source_strength[s]
            signals[s] = self.signal.usignal(self.up)
        mpos = self.mics.mpos
    
        # shortcuts and intial values
        m = self.mics
        t = self.start*ones(m.num_mics)
        i = 0
        n = self.numsamples        
        while n:
            n -= 1                         
            t += 1./self.sample_freq
            te1, rm1, Mr1, locs1 = self.get_emission_time(t,0)         
            #trajg1 = array(self.trajectory.location( te1, der=1))[:,0][:,newaxis]
            
            # get distance and ind for every source in the line
            for s in range(self.num_sources):
                diff = self.get_moving_direction(dir2,te1)
                te, rm, Mr, locs = self.get_emission_time(t,tile((diff*s).T,(self.numchannels,1)).T)
                loc = array(self.trajectory.location(te), dtype = float)[:,0][:,newaxis] 
                diff = self.get_moving_direction(dir2,te)
                rms[:,s] = self.env._r((loc+diff*s), mpos)
                inds[:,s] = (te-self.start_t+self.start)*self.sample_freq      
            
            if self.conv_amp: 
                rm *= (1-Mr)**2
                rms[:,s] *= (1-Mr)**2 # assume that Mr is the same 
            try:
                # subtract the second signal b/c of phase inversion
                for s in range(self.num_sources):
                # sum sources
                    out[i] += (signals[s,array(0.5 + inds[:,s].T * self.up, dtype=int64)] / rms[:,s])
                                    
                i += 1
                if i == num:
                    yield out
                    out = zeros((num, self.numchannels))
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
    #: When left empty, arange(:attr:`numchannels`) + :attr:`signal`.seed  
    #: will be used.
    seed = CArray(dtype = uint32,
                  desc = "random seed values")
    
    #: Number of channels in output; is set automatically / 
    #: depends on used microphone geometry.
    numchannels = Delegate('mics', 'num_mics')

    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Trait(MicGeom, 
        desc="microphone geometry")

    # --- List of backwards compatibility traits and their setters/getters -----------

    # Microphone locations.
    # Deprecated! Use :attr:`mics` trait instead.
    mpos = Property()
    
    def _get_mpos(self):
        return self.mics
    
    def _set_mpos(self, mpos):
        warn("Deprecated use of 'mpos' trait. ", Warning, stacklevel = 2)
        self.mics = mpos

    # --- End of backwards compatibility traits --------------------------------------
        
    #: Start time of the signal in seconds, defaults to 0 s.
    start_t = Float(0.0,
        desc="signal start time")
    
    #: Start time of the data aquisition at microphones in seconds, 
    #: defaults to 0 s.
    start = Float(0.0,
        desc="sample start time")

    
    #: Number of samples is set automatically / 
    #: depends on :attr:`signal`.
    numsamples = Delegate('signal')
    
    #: Sampling frequency of the signal; is set automatically / 
    #: depends on :attr:`signal`.
    sample_freq = Delegate('signal') 
    
    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.rms', 'signal.numsamples', \
        'signal.sample_freq', 'signal.__class__' , 'seed', 'loc', \
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
            (i.e. the number of samples per block) .
        
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
            if (n-num) < self.numsamples:
                yield signal[n-num:,:]
            else:
                return



class SourceMixer( SamplesGenerator ):
    """
    Mixes the signals from several sources. 
    """

    #: List of :class:`~acoular.tprocess.SamplesGenerator` objects
    #: to be mixed.
    sources = List( Instance(SamplesGenerator, ()) ) 

    #: Sampling frequency of the signal.
    sample_freq = Property( depends_on=['sdigest'] )
    
    #: Number of channels.
    numchannels = Property( depends_on=['sdigest'] )
               
    #: Number of samples.
    numsamples = Property( depends_on=['sdigest'] )
    
    #: Amplitude weight(s) for the sources as array. If not set, 
    #: all source signals are equally weighted.
    #: Must match the number of sources in :attr:`sources`.
    weights = CArray(desc="channel weights")

    # internal identifier    
    sdigest = Str()

    @observe('sources.items.digest')
    def _set_sources_digest( self, event ):
        self.sdigest = ldigest(self.sources) 

    # internal identifier
    digest = Property( depends_on = ['sdigest', 'weights'])

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_sample_freq( self ):
        if self.sources:
            sample_freq = self.sources[0].sample_freq
        else:
            sample_freq = 0
        return sample_freq

    @cached_property
    def _get_numchannels( self ):
        if self.sources:
            numchannels = self.sources[0].numchannels
        else:
            numchannels = 0
        return numchannels

    @cached_property
    def _get_numsamples( self ):
        if self.sources:
            numsamples = self.sources[0].numsamples
        else:
            numsamples = 0
        return numsamples

    def validate_sources( self ):
        """ Validates if sources fit together. """
        if len(self.sources) < 1:
            raise ValueError("Number of sources in SourceMixer should be at least 1.")
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
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        # check whether all sources fit together
        self.validate_sources()

        gens = [i.result(num) for i in self.sources[1:]]
        weights = self.weights.copy()
        if weights.size == 0:
            weights = array([1. for j in range(len( self.sources))])
        assert weights.shape[0] == len(self.sources)
        for temp in self.sources[0].result(num):
            temp *= weights[0]
            sh = temp.shape[0]
            for j,g in enumerate(gens):
                temp1 = next(g)*weights[j+1]
                if temp.shape[0] > temp1.shape[0]:
                    temp = temp[:temp1.shape[0]]
                temp += temp1[:temp.shape[0]]
            yield temp
            if sh > temp.shape[0]:
                break


class PointSourceConvolve( PointSource ):
    """
    Class to blockwise convolve an arbitrary source signal with a spatial room impulse response
    """

    #: Convolution kernel in the time domain.
    #: The second dimension of the kernel array has to be either 1 or match :attr:`~SamplesGenerator.numchannels`.
    #: If only a single kernel is supplied, it is applied to all channels.
    kernel = CArray(dtype=float, desc="Convolution kernel.")

    # ------------- overwrite traits that are not supported by this class -------------
    
    #: Start time of the signal in seconds, defaults to 0 s.
    start_t = Enum(0.0,
        desc="signal start time")
    
    #: Start time of the data aquisition at microphones in seconds, 
    #: defaults to 0 s.
    start = Enum(0.0,
        desc="sample start time")

    #: Signal behaviour for negative time indices, i.e. if :attr:`start` < :attr:start_t.
    #: `loop` take values from the end of :attr:`signal.signal()` array.
    #: `zeros` set source signal to zero, advisable for deterministic signals.
    #: defaults to `loop`.
    prepadding = Enum(None, desc="Behaviour for negative time indices.")

    #: Upsampling factor, internal use, defaults to 16.
    up = Enum(None, desc="upsampling factor") 
            
    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', 'kernel', '__class__'], 
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
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        data = repeat(
            self.signal.signal()[:,newaxis],self.mics.num_mics,axis=1)
        source = TimeSamples(
            data=data,
            sample_freq=self.sample_freq,
            numsamples=self.numsamples,
            numchannels=self.mics.num_mics,
        )
        time_convolve = TimeConvolve(
            source = source,
            kernel = self.kernel,
        )
        for block in time_convolve.result(num):
            yield block