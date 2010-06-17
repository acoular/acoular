# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:25:10 2010

@author: sarradj
"""
#pylint: disable-msg=E0611, E1101, C0103, C0111, R0901, R0902, R0903, R0904, W0232
# imports from other packages
from numpy import array, newaxis, empty, empty_like, pi, sin, sqrt, arange, \
clip, sort, r_, s_, zeros, int16, histogram, unique1d, log10, where
from scipy.interpolate import splprep, splev
from enthought.traits.api import HasPrivateTraits, Float, Int, Long, \
File, CArray, Property, Instance, Trait, Bool, Delegate, Any, \
cached_property, on_trait_change, property_depends_on, List, Dict, Tuple
from enthought.traits.ui.api import View, Item
from enthought.traits.ui.menu import OKCancelButtons
from os import path
import tables
from scipy.signal import butter, lfilter, filtfilt

# beamfpy imports
from internal import digest
from h5cache import H5cache
from grids import RectGrid, MicGeom, Environment

class Calib( HasPrivateTraits ):
    """
    container for calibration data that is loaded from
    an .xml-file
    """

    # name of the .xml file
    from_file = File(filter=['*.xml'], 
        desc="name of the xml file to import")

    # basename of the .xml-file
    basename = Property( depends_on = 'from_file', 
        desc="basename of xml file")
    
    # number of microphones in the calibration data 
    num_mics = Int( 1, 
        desc="number of microphones in the geometry")

    # array of calibration factors
    data = CArray(
        desc="calibration data")

    # internal identifier
    digest = Property( depends_on = ['basename', ] )

    test = Any
    traits_view = View(
        ['from_file{File name}', 
            ['num_mics~{Number of microphones}', 
                '|[Properties]'
            ]
        ], 
        title='Calibration data', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_basename( self ):
        if not path.isfile(self.from_file):
            return ''
        return path.splitext(path.basename(self.from_file))[0]
    
    @on_trait_change('basename')
    def import_data( self ):
        "loads the calibration data from .xml file"
        if not path.isfile(self.from_file):
            # no file there
            self.data = array([1.0, ], 'd')
            self.num_mics = 1
            return
        import xml.dom.minidom
        doc = xml.dom.minidom.parse(self.from_file)
        names = []
        data = []
        for element in doc.getElementsByTagName('pos'):
            names.append(element.getAttribute('Name'))
            data.append(float(element.getAttribute('factor')))
        self.data = array(data, 'd')
        self.num_mics = self.data.shape[0]

class SamplesGenerator( HasPrivateTraits ):
    """
    Base class for any generating signal processing block, 
    generates output via the generator 'result'
    """

    # sample_freq of signal
    sample_freq = Float(1.0, 
        desc="sampling frequency")
    
    # number of channels 
    numchannels = Long
               
    # number of samples 
    numsamples = Long
    
    # internal identifier
    digest = ''
               
    # result generator: delivers output in blocks of num samples
    def result(self, num):
        pass

class TimeSamples( SamplesGenerator ):
    """
    Container for time data, loads time data
    and provides information about this data
    """

    # full name of the .h5 file with data
    name = File(filter=['*.h5'], 
        desc="name of data file")

    # basename of the .h5 file with data
    basename = Property( depends_on = 'name', #filter=['*.h5'], 
        desc="basename of data file")
    
    # number of channels, is set automatically
    numchannels = Long(0L, 
        desc="number of input channels")

    # number of time data samples, is set automatically
    numsamples = Long(0L, 
        desc="number of samples")

    # the time data as (numsamples, numchannels) array of floats
    data = Any(
        desc="the actual time data array")

    # hdf5 file object
    h5f = Instance(tables.File)
    
    # internal identifier
    digest = Property( depends_on = ['basename', ])

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
            return None
        if self.h5f != None:
            try:
                self.h5f.close()
            except IOError:
                pass
        self.h5f = tables.openFile(self.name)
        self.data = self.h5f.root.time_data
        self.sample_freq = self.data.getAttr('sample_freq')
        (self.numsamples, self.numchannels) = self.data.shape

    # generator function
    def result(self, num=128):
        i = 0
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
    
    # start of valid samples
    start = Long(0L, 
        desc="start of valid samples")
    
    # stop of valid samples
    stop = Trait(None, None, Long, 
        desc="stop of valid samples")
    
    # invalid channels  
    invalid_channels = List(
        desc="list of invalid channels")
    
    # channel mask to serve as an index for all valid channels
    channels = Property(depends_on = ['invalid_channels', 'numchannels_total'], 
        desc="channel mask")
        
    # calibration data
    calib = Trait( Calib, 
        desc="Calibration data")
    
    # number of channels, is set automatically
    numchannels_total = Long(0L, 
        desc="total number of input channels")

    # number of time data samples, is set automatically
    numsamples_total = Long(0L, 
        desc="total number of samples per channel")

    # number of channels, is set automatically
    numchannels = Property(depends_on = ['invalid_channels', \
        'numchannels_total'], desc="number of valid input channels")

    # number of time data samples, is set automatically
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
            return None
        if self.h5f != None:
            try:
                self.h5f.close()
            except IOError:
                pass
        self.h5f = tables.openFile(self.name)
        self.data = self.h5f.root.time_data
        self.sample_freq = self.data.getAttr('sample_freq')
        (self.numsamples_total, self.numchannels_total) = self.data.shape
        
    # generator function
    def result(self, num=128):
        sli = slice(self.start, self.stop).indices(self.numsamples_total)
        i = sli[0]
        stop = sli[1]
        cal_factor = 1.0
        if self.calib:
            if self.calib.num_mics == self.numchannels_total:
                cal_factor = self.calib.data[self.channels][newaxis]
            elif self.calib.num_mics == self.numchannels:
                cal_factor = self.calib.data[newaxis]
            else:
                cal_factor = 1.0
        while i < stop:
            yield self.data[i:min(i+num, stop)][:, self.channels]*cal_factor
            i += num

class TimeInOut( SamplesGenerator ):
    """
    Base class for any time domain signal processing block, 
    gets samples from 'source' and generates output via the generator 'result'
    """

    # data source, object that has a property sample_freq and 
    # a python generator result(N) that will generate data blocks of N samples
    source = Trait(SamplesGenerator)

    # sample_freq of output signal
    sample_freq = Delegate('source')
    
    # number of channels in output
    numchannels = Delegate('source')
               
    # number of channels in output
    numsamples = Delegate('source')
            
    # internal identifier
    digest = Property( depends_on = ['source.digest', '__class__'])

    traits_view = View(
        Item('source', style='custom')
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    # result generator: delivers output in blocks of N
    def result(self, n):
        for temp in self.source.result(n):
            # effectively no processing
            yield temp

class TimePower( TimeInOut ):
    """
    calculates time-depend power of the signal
    """
    # result generator: delivers squared input
    def result(self, n):
        for temp in self.source.result(n):
            yield temp*temp
    
class TimeAverage( TimeInOut ) :
    """
    calculates time-depend power of the signal
    """
    # number of samples to average over
    naverage = Int(64, 
        desc = "number of samples to average over")
        
    # sample_freq of output signal
    sample_freq = Property( depends_on = 'source.sample_freq, naverage')
    
    # number of samples in output signal
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

    # result generator: delivers average of input
    def result(self, n):
        nav = self.naverage
        for temp in self.source.result(n*nav):
            ns, nc = temp.shape
            nso = ns/nav
            if nso > 0:
                yield temp[:nso*nav].reshape((nso, -1, nc)).mean(axis=1)
                
class TimeReverse( TimeInOut ):
    """
    reverses time 
    """
    # result generator: delivers output in blocks of N
    def result(self, n):
        l = []
        l.extend(self.source.result(n))
        temp = empty_like(l[0])
        nst = temp.shape[0]
        h = l.pop()
        nsh = h.shape[0]
        temp[:nsh] = h[::-1]
        for h in l[::-1]:
#            print nsh, nst
            temp[nsh:] = h[:nsh-1:-1]
            yield temp
            temp[:nsh] = h[nsh-1::-1]
        yield temp[:nsh]
        
class FiltFiltOctave( TimeInOut ):
    """
    octave or fractional octave filter with zero phase delay
    requires large amounts of memory !   
    """
    # band center frequency
    band = Float(1000.0, 
        desc = "band center frequency")
        
    # octave fraction
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
        # filter design
        fs = self.sample_freq
        # adjust filter edge frequencies
        beta = pi/(4*order)
        alpha = pow(2.0, 1.0/(2.0*self.fraction_))
        beta = 2 * beta / sin(beta) / (alpha-1/alpha)
        alpha = (1+sqrt(1+beta*beta))/beta
        fr = 2*self.band/fs
        if fr > 1/sqrt(2):
            raise ValueError("band frequency too high")
        om1 = fr/alpha 
        om2 = fr*alpha
        return butter(order, [om1, om2], 'bandpass') 
        
    def result(self, n):
        b, a = self.ba(3) # filter order = 3
        data = empty((self.source.numsamples, self.source.numchannels))
        j = 0
        for block in self.source.result(n):
            ns, nc = block.shape
            data[j:j+ns] = block
            j += ns
        for j in range(nc):
            data[:, j] = filtfilt(b, a, data[:, j])
        j = 0
        ns = data.shape[0]
        while j < ns:
            yield data[j:j+n]
            j += n

class FiltOctave( FiltFiltOctave ):
    """
    octave or fractional octave filter (not zero-phase)
    """

    def result(self, n):
        b, a = self.ba(3) # filter order = 3
        zi = zeros((max(len(a), len(b))-1, self.source.numchannels))
        for block in self.source.result(n):
            block, zi = lfilter(b, a, block, axis=0, zi=zi)
            yield block

def const_power_weight( bf ):
    """
    provides microphone weighting for BeamformerTime
    to make the power per unit area of the
    microphone array geometry constant 
    """
    r = bf.env.r( bf.c, zeros((3,1)), bf.mpos.mpos) # distances to center
    # round the relative distances to one decimal place
    r = (r/r.max()).round(decimals=1)
    ru,ind = unique1d(r,return_inverse=True)
    ru = (ru[1:]+ru[:-1])/2
    count,bins = histogram(r,r_[0,ru,1.5*r.max()-0.5*ru[-1]])
    bins *= bins
    weights = sqrt((bins[1:]-bins[:-1])/count)
    weights /= weights.mean()
    return weights[ind]

possible_weights = {'none':None, 
                    'power':const_power_weight}


class BeamformerTime( TimeInOut ):
    """
    Provides a basic time domain beamformer with time signal output
    for a spatially fixed grid
    """

    # RectGrid object that provides the grid locations
    grid = Trait(RectGrid, 
        desc="beamforming grid")

    # number of channels in output
    numchannels = Delegate('grid', 'size')

    # MicGeom object that provides the microphone locations
    mpos = Trait(MicGeom, 
        desc="microphone geometry")
        
    # Environment object that provides speed of sound and grid-mic distances
    env = Trait(Environment(), Environment)

    # spatial weighting function (from timedomain.possible_weights)
    weights = Trait('none', possible_weights, 
        desc="spatial weighting function")

    # the speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")
    
    # sound travel distances from microphone array center to grid points
    r0 = Property(
        desc="array center to grid distances")

    # sound travel distances from array microphones to grid points
    rm = Property(
        desc="array center to grid distances")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'source.digest', 'c', \
        'env.digest', 'weights', '__class__'], 
        )

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
            [Item('weights{}', style='custom')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    #@cached_property
    def _get_r0 ( self ):
        return self.env.r( self.c, self.grid.pos())

    #@cached_property
    def _get_rm ( self ):
        return self.env.r( self.c, self.grid.pos(), self.mpos.mpos)

    # generator, delivers the beamformer result
    def result( self, n=2048 ):
        if self.weights_:
            w = self.weights_(self)[newaxis]
        else:
            w = 1.0
        c = self.c/self.sample_freq
        delays = self.rm/c
        d_index = array(delays, dtype=int) # integer index
        d_interp1 = delays % 1 # 1st coeff for lin interpolation between samples
        d_interp2 = 1-d_interp1 # 2nd coeff for lin interpolation 
        d_index2 = arange(self.mpos.num_mics)
#        amp = (self.rm/self.r0[:, newaxis]) # multiplication factor
        amp = (w/(self.rm*self.rm)).sum(1) * self.r0
        amp = 1.0/(amp[:, newaxis]*self.rm) # multiplication factor
        d_interp1 *= amp # premultiplication, to save later ops
        d_interp2 *= amp
        dmin = d_index.min() # minimum index
        dmax = d_index.max()+1 # maximum index
        aoff = dmax-dmin # index span
        #working copy of data:
        zi = empty((aoff+n, self.source.numchannels), dtype=float) 
        o = empty((n, self.grid.size), dtype=float) # output array
        offset = aoff # start offset for working array
        ooffset = 0 # offset for output array
        for block in self.source.result(n):
            ns = block.shape[0] # numbers of samples and channels
            maxoffset = ns-dmin # ns - aoff +aoff -dmin
            zi[aoff:aoff+ns] = block * w # copy data to working array
            # loop over data samples 
            while offset < maxoffset:
                # yield output array if full
                if ooffset == n:
                    yield o
                    ooffset = 0
                # the next line needs to be implemented faster
                o[ooffset] = (zi[offset+d_index, d_index2]*d_interp1 + \
                        zi[offset+d_index+1, d_index2]*d_interp2).sum(-1)
                offset += 1
                ooffset += 1
            # copy remaining samples in front of next block
            zi[0:aoff] = zi[-aoff:]
            offset -= n
        # remaining data chunk 
        yield o[:ooffset]
            
class BeamformerTimeSq( BeamformerTime ):
    """
    Provides a time domain beamformer with time-dependend
    power signal output and possible autopower removal
    for a spatially fixed grid
    """
    
    # flag, if true (default), the auto power is removed 
    r_diag = Bool(True, 
        desc="removal of diagonal")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'source.digest', 'r_diag', \
        'c', 'env.digest', 'weights', '__class__'], 
        )

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('r_diag', label='diagonal removed')], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
            [Item('weights{}', style='custom')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    # generator, delivers the beamformer result
    def result( self, n=2048 ):
        if self.weights_:
            w = self.weights_(self)[newaxis]
        else:
            w = 1.0
        c = self.c/self.source.sample_freq
        delays = self.rm/c
        d_index = array(delays, dtype=int) # integer index
        d_interp1 = delays % 1 # 1st coeff for lin interpolation between samples
        d_interp2 = 1-d_interp1 # 2nd coeff for lin interpolation 
        d_index2 = arange(self.mpos.num_mics)
#        amp = (self.rm/self.r0[:, newaxis]) # multiplication factor
        amp = (w/(self.rm*self.rm)).sum(1) * self.r0
        amp = 1.0/(amp[:, newaxis]*self.rm) # multiplication factor
        d_interp1 *= amp # premultiplication, to save later ops
        d_interp2 *= amp
        dmin = d_index.min() # minimum index
        dmax = d_index.max()+1 # maximum index
#        print dmin, dmax
        aoff = dmax-dmin # index span
        #working copy of data:
        zi = empty((aoff+n, self.source.numchannels), dtype=float)
        o = empty((n, self.grid.size), dtype=float) # output array
        temp = empty((self.grid.size, self.source.numchannels), dtype=float)
        offset = aoff # start offset for working array
        ooffset = 0 # offset for output array
        for block in self.source.result(n):
            ns = block.shape[0] # numbers of samples and channels
            maxoffset = ns-dmin # ns - aoff +aoff -dmin
            zi[aoff:aoff+ns] = block * w # copy data to working array
            # loop over data samples 
            while offset < maxoffset:
                # yield output array if full
                if ooffset == n:
                    yield o
                    ooffset = 0
                # the next line needs to be implemented faster
                temp[:, :] = (zi[offset+d_index, d_index2]*d_interp1 \
                    + zi[offset+d_index+1, d_index2]*d_interp2)
                if self.r_diag:
                    # simple sum and remove autopower
                    o[ooffset] = clip(temp.sum(-1)**2 - \
                            (temp**2).sum(-1), 1e-100, 1e+100)
                else:
                    # simple sum
                    o[ooffset] = temp.sum(-1)**2
                offset += 1
                ooffset += 1
            # copy remaining samples in front of next block
            zi[0:aoff] = zi[-aoff:]
            offset -= n
        # remaining data chunk 
        yield o[:ooffset]

class Trajectory( HasPrivateTraits ):
    """
    Base class for any time domain signal processing block, 
    gets samples from 'source' and generates output via the generator 'result'
    """
    # dictionary; keys: time, values: sampled (x, y, z) positions along the 
    # trajectory
    points = Dict(key_trait = Float, value_trait = Tuple(Float, Float, Float), 
        desc = "sampled positions along the trajectory")
    
    # t_min, t_max tuple
    interval = Property()
    
    # spline data, internal use
    tck = Property()
    
    # internal identifier
    digest = Property( 
        depends_on = ['points'], 
        )

    traits_view = View(
        [Item('points', style='custom')
        ], 
        title='Grid center trajectory', 
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    @property_depends_on('points')
    def _get_interval( self ):
        return sort(self.points.keys())[r_[0, -1]]

    @property_depends_on('points')
    def _get_tck( self ):
        t = sort(self.points.keys())
        xp = array([self.points[i] for i in t]).T
        k = min(3, len(self.points)-1)
        tcku = splprep(xp, u=t, s=0, k=k)
        return tcku[0]
    
    # returns (x, y, z) for t, x, y and z have the same shape as t
    def location(self, t):
        return splev(t, self.tck)
    
    # generator, gives locations along the trajectory
    # x.traj(0.1)  every 0.1s within self.interval
    # x.traj(2.5, 4.5, 0.1)  every 0.1s between 2.5s and 4.5s
    def traj(self, t_start, t_end=None, delta_t=None):
        if not delta_t:
            delta_t = t_start
            t_start, t_end = self.interval
        for t in arange(t_start, t_end, delta_t):
            yield self.location(t)
        
class BeamformerTimeSqTraj( BeamformerTimeSq ):
    """
    Provides a time domain beamformer with time-dependend
    power signal output and possible autopower removal
    for a grid moving along a trajectory
    """
    
    # trajectory, start time is assumed to be the same as for the samples
    trajectory = Trait(Trajectory, 
        desc="trajectory of the grid center")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'source.digest', 'r_diag', \
            'c', 'weights', 'env.digest', 'trajectory.digest', '__class__'], 
        )

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('trajectory{}', style='custom')], 
            [Item('r_diag', label='diagonal removed')], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
            [Item('weights{}', style='custom')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    # generator, delivers the beamformer result
    def result( self, n=2048 ):
        if self.weights_:
            w = self.weights_(self)[newaxis]
        else:
            w = 1.0
        c = self.c/self.source.sample_freq
        gpos = self.grid.pos()
        # determine the maximum delay
        tmin, tmax = self.trajectory.interval
        tpos = gpos + array(self.trajectory.location(tmin))[:, newaxis]
        dmax1 = array(self.env.r( self.c, tpos, self.mpos.mpos)/c, \
            dtype=int).max()
        tpos = gpos + array(self.trajectory.location(tmax))[:, newaxis]
        dmax2 = array(self.env.r( self.c, tpos, self.mpos.mpos)/c, \
            dtype=int).max()
        dmax = max(dmax1,dmax2)
        #max1 = self.r0.max() # maximum grid point distance to center
        # maximum microphone distance to center:
        #max2 = (self.mpos.mpos**2).sum(-1).max() 
        #trpos = array(self.trajectory.location(self.trajectory.interval))**2
#        print trpos.shape
        # + maximum trajectory displacement:
        #dmax = int((max1+max2+max(sqrt(trpos.sum(0))))/c)+1 
        dmin = 0 # minimum: no delay
#        print dmin, dmax
        aoff = dmax-dmin # index span
        #working copy of data:
        zi = empty((aoff+n, self.source.numchannels), dtype=float) 
        o = empty((n, self.grid.size), dtype=float) # output array
        temp = empty((self.grid.size, self.source.numchannels), dtype=float)
        d_index2 = arange(self.mpos.num_mics, dtype=int)
        offset = aoff # start offset for working array
        ooffset = 0 # offset for output array
        # generator for trajectory:
        g = self.trajectory.traj(1/self.source.sample_freq) 
        for block in self.source.result(n):
            ns = block.shape[0] # numbers of samples and channels
            maxoffset = ns-dmin # ns - aoff +aoff -dmin
            zi[aoff:aoff+ns] = block * w# copy data to working array
            # loop over data samples 
            while offset < maxoffset:
                # yield output array if full
                if ooffset == n:
                    yield o
                    ooffset = 0
                #
                #~ print gpos.shape, array(g.next()).shape
                tpos = gpos + array(g.next())[:, newaxis]
                rm = self.env.r( self.c, tpos, self.mpos.mpos)
                r0 = self.env.r( self.c, tpos)
                delays = rm/c
                d_index = array(delays, dtype=int) # integer index
                d_interp1 = delays % 1 # 1st coeff for lin interpolation
                d_interp2 = 1-d_interp1 # 2nd coeff for lin interpolation
#                amp = rm/r0[:, newaxis] # multiplication factor
                amp = (w/(rm*rm)).sum(1) * r0
                amp = 1.0/(amp[:, newaxis]*rm) # multiplication factor
                # the next line needs to be implemented faster
                temp[:, :] = (zi[offset+d_index, d_index2]*d_interp1 \
                            + zi[offset+d_index+1, d_index2]*d_interp2)*amp
                if self.r_diag:
                    # simple sum and remove autopower
                    o[ooffset] = clip(temp.sum(-1)**2 - \
                        (temp**2).sum(-1), 1e-100, 1e+100)
                else:
                    # simple sum
                    o[ooffset] = temp.sum(-1)**2
                offset += 1
                ooffset += 1
            # copy remaining samples in front of next block
            zi[0:aoff] = zi[-aoff:]
            offset -= n
        # remaining data chunk 
        yield o[:ooffset]
                
        
class TimeCache( TimeInOut ):
    """
    caches time signal in cache file
    """
    # basename for cache
    basename = Property( depends_on = 'digest')
    
    # hdf5 cache file
    h5f = Instance(tables.File)
    
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
        obj = self.source # start width source
        basename = 'void' # if no file source is found
        while obj:
#            print obj
            if 'basename' in obj.all_trait_names(): # at original source?
                basename = obj.basename # get the name
                break
            else:
                obj = obj.source # traverse down until original data source
        return basename

    # result generator: delivers input, possibly from cache
    def result(self, n):
        name = 'tc_' + self.digest
        H5cache.get_cache( self, self.basename )
        if not name in self.h5f.root:
            ac = self.h5f.createEArray(self.h5f.root, name, \
                                       tables.atom.Float32Atom(), \
                                        (0, self.numchannels))
            ac.setAttr('sample_freq', self.sample_freq)
            for data in self.source.result(n):
                ac.append(data)
                yield data
        else:
            ac = self.h5f.getNode('/', name)
            i = 0
            while i < ac.shape[0]:
                yield ac[i:i+n]
                i += n

class WriteWAV( TimeInOut ):
    """
    saves time signal from one or two channels as mono or stereo wav file
    """
    # basename for cache
    basename = Property( depends_on = 'digest')
       
    # channel(s) to save
    channels = List(desc="channel to save")
       
    # internal identifier
    digest = Property( depends_on = ['source.digest', 'channels', '__class__'])

    traits_view = View(
        [Item('source', style='custom'), 
            ['name~{Cache file name}', 
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
    def _get_basename ( self ):
        obj = self.source # start width source
        basename = 'void' # if no file source is found
        while obj:
#            print obj
            if 'basename' in obj.all_trait_names(): # at original source?
                basename = obj.basename # get the name
                break
            else:
                obj = obj.source # traverse down until original data source
        return basename

    # result generator: delivers nothing
    def result(self, n):
        pass

    def save(self):
        nc = len(self.channels)
        if not (0 < nc < 3):
            raise ValueError("one or two channels allowed, %i channels given" %\
            nc)
        name = self.basename
        for nr in self.channels:
            name += '_%i' % nr
        name += '.wav'
        import wave
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
        
class IntegratorSectorTime( TimeInOut ):
    """
    Provides an Integrator
    """

    # RectGrid object that provides the grid locations
    grid = Trait(RectGrid, 
        desc="beamforming grid")
        
    # List of sectors in grid
    sectors = List()

    # Clipping, in Dezibel relative to maximum (negative values)
    clip = Float(-1000.0)

    # number of channels in output
    numchannels = Property( depends_on = ['sectors',])

    # internal identifier
    digest = Property( 
        depends_on = ['sectors', 'clip', 'grid.digest', 'source.digest', \
        '__class__'], 
        )

    traits_view = View(
        [
            [Item('sectors', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            '|'
        ], 
        title='Integrator', 
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    @cached_property
    def _get_numchannels ( self ):
        return len(self.sectors)

    # generator, delivers the beamformer result
    def result( self, n=1 ):
        inds = [self.grid.indices(*sector) for sector in self.sectors]
        gshape = self.grid.shape
        o = empty((n, self.numchannels), dtype=float) # output array
        for r in self.source.result(n):
            ns, nc = r.shape
            mapshape = (ns,) + gshape
            rmax = r.max()
            rmin = rmax * 10**(self.clip/10.0)
            r = where(r>rmin,r,0.0)
            i = 0
            for ind in inds:
                h = r[:].reshape(mapshape)[ (s_[:],) + ind ]
                o[:ns,i] = h.reshape(h.shape[0],-1).sum(axis=1)
                i += 1
            yield o[:ns]


