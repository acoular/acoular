# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
"""
beamfpy.py: classes for calculations in the time domain

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming
 
(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""

# imports from other packages
from numpy import array, newaxis, empty, empty_like, pi, sin, sqrt, arange, \
clip, sort, r_, s_, zeros, int16, histogram, unique, where, cross, dot
from scipy.interpolate import splprep, splev
from traits.api import HasPrivateTraits, Float, Int, CLong, \
File, CArray, Property, Instance, Trait, Bool, Delegate, Any, \
cached_property, on_trait_change, property_depends_on, List, Dict, Tuple
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons
from datetime import datetime
from os import path
import tables
import wave
from scipy.signal import butter, lfilter, filtfilt

# beamfpy imports
from internal import digest
from h5cache import H5cache, td_dir
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
    num_mics = CLong( 0, 
        desc="number of microphones in the geometry")

    # array of calibration factors
    data = CArray(
        desc="calibration data")

    # internal identifier
    digest = Property( depends_on = ['basename', ] )

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
            # empty calibration
            if self.basename=='':
                self.data = None 
                self.num_mics = 0
            # no file there
            else:
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
    numchannels = CLong
               
    # number of samples 
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

    # full name of the .h5 file with data
    name = File(filter=['*.h5'], 
        desc="name of data file")

    # basename of the .h5 file with data
    basename = Property( depends_on = 'name', #filter=['*.h5'], 
        desc="basename of data file")
    
    # calibration data
    calib = Trait( Calib, 
        desc="Calibration data")
    
    # number of channels, is set automatically
    numchannels = CLong(0L, 
        desc="number of input channels")

    # number of time data samples, is set automatically
    numsamples = CLong(0L, 
        desc="number of samples")

    # the time data as (numsamples, numchannels) array of floats
    data = Any( transient = True, 
        desc="the actual time data array")

    # hdf5 file object
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
    
    # start of valid samples
    start = CLong(0L, 
        desc="start of valid samples")
    
    # stop of valid samples
    stop = Trait(None, None, CLong, 
        desc="stop of valid samples")
    
    # invalid channels  
    invalid_channels = List(
        desc="list of invalid channels")
    
    # channel mask to serve as an index for all valid channels
    channels = Property(depends_on = ['invalid_channels', 'numchannels_total'], 
        desc="channel mask")
        
    # number of channels, is set automatically
    numchannels_total = CLong(0L, 
        desc="total number of input channels")

    # number of time data samples, is set automatically
    numsamples_total = CLong(0L, 
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

    def result(self, num):
        """ 
        python generator: dummy function, just echoes the output of source,
        yields samples in blocks of shape (num, numchannels), the last block
        may be shorter than num
        """
        for temp in self.source.result(num):
            # effectively no processing
            yield temp

class Mixer( TimeInOut ):
    """
    mixes the signals from several sources
    """

    # data source, object that has a property sample_freq and 
    # a python generator result(N) that will generate data blocks of N samples
    source = Trait(SamplesGenerator)

    # list of additional data source objects
    sources = List( Instance(SamplesGenerator, ()) ) 

    # sample_freq of output signal
    sample_freq = Delegate('source')
    
    # number of channels in output
    numchannels = Delegate('source')
               
    # number of channels in output
    numsamples = Delegate('source')

    # internal identifier
    ldigest = Property( depends_on = ['sources.digest', ])

    # internal identifier
    digest = Property( depends_on = ['source.digest', 'ldigest', '__class__'])

    traits_view = View(
        Item('source', style='custom')
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

    @on_trait_change('sources,source')
    def validate_sources( self ):
        """ validates if sources fit together """
        if self.source:
            for s in self.sources:
                if self.sample_freq != s.sample_freq:
                    raise ValueError("Sample frequency of %s does not fit" % s)
                if self.numchannels != s.numchannels:
                    raise ValueError("Channel count of %s does not fit" % s)

    def result(self, num):
        """ 
        python generator: adds the output from the source and those in the list 
        sources, yields samples in blocks of shape (num, numchannels), 
        the last block may be shorter than num
        """
        gens = [i.result(num) for i in self.sources]
        for temp in self.source.result(num):
            sh = temp.shape[0]
            for g in gens:
                temp1 = g.next()
                if temp.shape[0] > temp1.shape[0]:
                    temp = temp[:temp1.shape[0]]
                temp += temp1[:temp.shape[0]]
            yield temp
            if sh > temp.shape[0]:
                break


class TimePower( TimeInOut ):
    """
    calculates time-depended power of the signal
    """

    def result(self, num):
        """ 
        python generator: echoes the squared output of source, yields samples 
        in blocks of shape (num, numchannels), the last block may be shorter 
        than num
        """
        for temp in self.source.result(num):
            yield temp*temp
    
class TimeAverage( TimeInOut ) :
    """
    calculates time-depended average of the signal
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

    def result(self, num):
        """ 
        python generator: delivers the blockwise average of the output of 
        source, yields samples in blocks of shape (num, numchannels), the last 
        block may be shorter than num
        """
        nav = self.naverage
        for temp in self.source.result(num*nav):
            ns, nc = temp.shape
            nso = ns/nav
            if nso > 0:
                yield temp[:nso*nav].reshape((nso, -1, nc)).mean(axis=1)
                
class TimeReverse( TimeInOut ):
    """
    reverses time 
    """
    def result(self, num):
        """ 
        python generator: delivers the time-reversed output of source, yields 
        samples in blocks of shape (num, numchannels), the last block may be 
        shorter than num
        """
        l = []
        l.extend(self.source.result(num))
        temp = empty_like(l[0])
        h = l.pop()
        nsh = h.shape[0]
        temp[:nsh] = h[::-1]
        for h in l[::-1]:
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
        """ internal filter design routine, return filter coeffs """
        # filter design
        fs = self.sample_freq
        # adjust filter edge frequencies
        beta = pi/(4*order)
        alpha = pow(2.0, 1.0/(2.0*self.fraction_))
        beta = 2 * beta / sin(beta) / (alpha-1/alpha)
        alpha = (1+sqrt(1+beta*beta))/beta
        fr = 2*self.band/fs
        if fr > 1/sqrt(2):
            raise ValueError("band frequency too high:%f,%f" % (self.band, fs))
        om1 = fr/alpha 
        om2 = fr*alpha
#        print om1, om2
        return butter(order, [om1, om2], 'bandpass') 
        
    def result(self, num):
        """ 
        python generator: delivers the zero-phase bandpass filtered output of 
        source, yields samples in blocks of shape (num, numchannels), the last 
        block may be shorter than num
        """
        b, a = self.ba(3) # filter order = 3
        data = empty((self.source.numsamples, self.source.numchannels))
        j = 0
        for block in self.source.result(num):
            ns, nc = block.shape
            data[j:j+ns] = block
            j += ns
        for j in range(self.source.numchannels):
            data[:, j] = filtfilt(b, a, data[:, j])
        j = 0
        ns = data.shape[0]
        while j < ns:
            yield data[j:j+num]
            j += num

class FiltOctave( FiltFiltOctave ):
    """
    octave or fractional octave filter (not zero-phase)
    """

    def result(self, num):
        """ 
        python generator: delivers the bandpass filtered output of source,
        yields samples in blocks of shape (num, numchannels), the last 
        block may be shorter than num
        """
        b, a = self.ba(3) # filter order = 3
        zi = zeros((max(len(a), len(b))-1, self.source.numchannels))
        for block in self.source.result(num):
            block, zi = lfilter(b, a, block, axis=0, zi=zi)
            yield block

def const_power_weight( bf ):
    """
    provides microphone weighting for BeamformerTime
    to make the power per unit area of the
    microphone array geometry constant 
    """
    r = bf.env.r( bf.c, zeros((3, 1)), bf.mpos.mpos) # distances to center
    # round the relative distances to one decimal place
    r = (r/r.max()).round(decimals=1)
    ru, ind = unique(r, return_inverse=True)
    ru = (ru[1:]+ru[:-1])/2
    count, bins = histogram(r, r_[0, ru, 1.5*r.max()-0.5*ru[-1]])
    bins *= bins
    weights = sqrt((bins[1:]-bins[:-1])/count)
    weights /= weights.mean()
    return weights[ind]

# possible choices for spatial weights
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

    def result( self, num=2048 ):
        """ 
        python generator: delivers the beamformer result, yields samples in 
        blocks of shape (num, numchannels), the last block may be shorter 
        than num; numchannels is usually very large (number of grid points)
        """
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
        zi = empty((aoff+num, self.source.numchannels), dtype=float) 
        o = empty((num, self.grid.size), dtype=float) # output array
        offset = aoff # start offset for working array
        ooffset = 0 # offset for output array
        for block in self.source.result(num):
            ns = block.shape[0] # numbers of samples and channels
            maxoffset = ns-dmin # ns - aoff +aoff -dmin
            zi[aoff:aoff+ns] = block * w # copy data to working array
            # loop over data samples 
            while offset < maxoffset:
                # yield output array if full
                if ooffset == num:
                    yield o
                    ooffset = 0
                # the next line needs to be implemented faster
                o[ooffset] = (zi[offset+d_index, d_index2]*d_interp1 + \
                        zi[offset+d_index+1, d_index2]*d_interp2).sum(-1)
                offset += 1
                ooffset += 1
            # copy remaining samples in front of next block
            zi[0:aoff] = zi[-aoff:]
            offset -= num
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
    def result( self, num=2048 ):
        """ 
        python generator: delivers the _squared_ beamformer result with optional
        removal of autocorrelation, yields samples in blocks of shape 
        (num, numchannels), the last block may be shorter than num; 
        numchannels is usually very large (number of grid points)
        """
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
        zi = empty((aoff+num, self.source.numchannels), dtype=float)
        o = empty((num, self.grid.size), dtype=float) # output array
        temp = empty((self.grid.size, self.source.numchannels), dtype=float)
        offset = aoff # start offset for working array
        ooffset = 0 # offset for output array
        for block in self.source.result(num):
            ns = block.shape[0] # numbers of samples and channels
            maxoffset = ns-dmin # ns - aoff +aoff -dmin
            zi[aoff:aoff+ns] = block * w # copy data to working array
            # loop over data samples 
            while offset < maxoffset:
                # yield output array if full
                if ooffset == num:
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
            offset -= num
        # remaining data chunk 
        yield o[:ooffset]

class Trajectory( HasPrivateTraits ):
    """
    describes the trajectory from sampled points
    does spline interpolation of positions between samples
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
        depends_on = ['points[]'], 
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
        
    @property_depends_on('points[]')
    def _get_interval( self ):
        return sort(self.points.keys())[r_[0, -1]]

    @property_depends_on('points[]')
    def _get_tck( self ):
        t = sort(self.points.keys())
        xp = array([self.points[i] for i in t]).T
        k = min(3, len(self.points)-1)
        tcku = splprep(xp, u=t, s=0, k=k)
        return tcku[0]
    
    def location(self, t, der=0):
        """ returns (x, y, z) for t, x, y and z have the same shape as t """
        return splev(t, self.tck, der)
    
    def traj(self, t_start, t_end=None, delta_t=None, der=0):
        """
        python generator, yields locations along the trajectory
        x.traj(0.1)  every 0.1s within self.interval
        x.traj(2.5, 4.5, 0.1)  every 0.1s between 2.5s and 4.5s
        x.traj(0.1, der=1)  1st derivative every 0.1s within self.interval
        """
        if not delta_t:
            delta_t = t_start
            t_start, t_end = self.interval
        if not t_end:
            t_end = self.interval[1]
        for t in arange(t_start, t_end, delta_t):
            yield self.location(t, der)
        
class BeamformerTimeSqTraj( BeamformerTimeSq ):
    """
    Provides a time domain beamformer with time-dependent
    power signal output and possible autopower removal
    for a grid moving along a trajectory
    """
    
    # trajectory, start time is assumed to be the same as for the samples
    trajectory = Trait(Trajectory, 
        desc="trajectory of the grid center")

    # reference vector, perpendicular to the y-axis of moving grid
    rvec = CArray( dtype=float, shape=(3, ), value=array((0, 0, 0)), 
        desc="reference vector")
    
    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'source.digest', 'r_diag', \
            'c', 'weights', 'rvec', 'env.digest', 'trajectory.digest', \
            '__class__'], 
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
        
    def result( self, num=2048 ):
        """ 
        python generator: delivers the _squared_ beamformer result with optional
        removal of autocorrelation for a moving (translated and optionally 
        rotated grid), yields samples in blocks of shape (num, numchannels), 
        the last block may be shorter than num; numchannels is usually very 
        large (number of grid points)
        
        the output starts for signals that were emitted from the grid at t=0
        """
        if self.weights_:
            w = self.weights_(self)[newaxis]
        else:
            w = 1.0
        c = self.c/self.source.sample_freq
        # temp array for the grid co-ordinates
        gpos = self.grid.pos()
        # max delay span = sum of
        # max diagonal lengths of circumscribing cuboids for grid and micarray
        dmax = sqrt(((gpos.max(1)-gpos.min(1))**2).sum())
        dmax += sqrt(((self.mpos.mpos.max(1)-self.mpos.mpos.min(1))**2).sum())
        dmax = int(dmax/c)+1 # max index span
        zi = empty((dmax+num, self.source.numchannels), \
            dtype=float) #working copy of data
        o = empty((num, self.grid.size), dtype=float) # output array
        temp = empty((self.grid.size, self.source.numchannels), dtype=float)
        d_index2 = arange(self.mpos.num_mics, dtype=int) # second index (static)
        offset = dmax+num # start offset for working array
        ooffset = 0 # offset for output array      
        # generators for trajectory, starting at time zero
        start_t = 0.0
        g = self.trajectory.traj( start_t, delta_t=1/self.source.sample_freq)
        g1 = self.trajectory.traj( start_t, delta_t=1/self.source.sample_freq, 
                                  der=1)
        rflag = (self.rvec == 0).all() #flag translation vs. rotation
        data = self.source.result(num)
        flag = True
        while flag:
            # yield output array if full
            if ooffset == num:
                yield o
                ooffset = 0
            if rflag:
                # grid is only translated, not rotated
                tpos = gpos + array(g.next())[:, newaxis]
            else:
                # grid is both translated and rotated
                loc = array(g.next()) #translation
                dx = array(g1.next()) #direction vector (new x-axis)
                dy = cross(self.rvec, dx) # new y-axis
                dz = cross(dx, dy) # new z-axis
                RM = array((dx, dy, dz)).T # rotation matrix
                RM /= sqrt((RM*RM).sum(0)) # column normalized
                tpos = dot(RM, gpos)+loc[:, newaxis] # rotation+translation
            rm = self.env.r( self.c, tpos, self.mpos.mpos)
            r0 = self.env.r( self.c, tpos)
            delays = rm/c
            d_index = array(delays, dtype=int) # integer index
            d_interp1 = delays % 1 # 1st coeff for lin interpolation
            d_interp2 = 1-d_interp1 # 2nd coeff for lin interpolation
            amp = (w/(rm*rm)).sum(1) * r0
            amp = 1.0/(amp[:, newaxis]*rm) # multiplication factor
            # now, we have to make sure that the needed data is available                 
            while offset+d_index.max()+2>dmax+num:
                # copy remaining samples in front of next block
                zi[0:dmax] = zi[-dmax:]
                # the offset is adjusted by one block length
                offset -= num
                # test if data generator is exhausted
                try:
                    # get next data
                    block = data.next()
                except StopIteration:
                    flag = False
                    break
                # samples in the block, equals to num except for the last block
                ns = block.shape[0]                
                zi[dmax:dmax+ns] = block * w# copy data to working array
            else:
                # the next line needs to be implemented faster
                # it eats half of the time
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
        # remaining data chunk
        yield o[:ooffset]
                       
class TimeCache( TimeInOut ):
    """
    caches time signal in cache file
    """
    # basename for cache
    basename = Property( depends_on = 'digest')
    
    # hdf5 cache file
    h5f = Instance(tables.File,  transient = True)
    
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
        obj = self.source # start with source
        basename = 'void' # if no file source is found
        while obj:
            if 'basename' in obj.all_trait_names(): # at original source?
                basename = obj.basename # get the name
                break
            else:
                try:
                    obj = obj.source # traverse down until original data source
                except AttributeError:
                    obj = None
        return basename

    # result generator: delivers input, possibly from cache
    def result(self, num):
        """ 
        python generator: just echos the source output, but reads it from cache
        when available and prevents unnecassary recalculation, yields samples 
        in blocks of shape (num, numchannels), the last block may be shorter 
        than num
        """
        name = 'tc_' + self.digest
        H5cache.get_cache( self, self.basename )
        if not name in self.h5f.root:
            ac = self.h5f.createEArray(self.h5f.root, name, \
                                       tables.atom.Float32Atom(), \
                                        (0, self.numchannels))
            ac.setAttr('sample_freq', self.sample_freq)
            for data in self.source.result(num):
                ac.append(data)
                yield data
        else:
            ac = self.h5f.getNode('/', name)
            i = 0
            while i < ac.shape[0]:
                yield ac[i:i+num]
                i += num

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
            ['basename~{File name}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Write wav file', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_basename ( self ):
        obj = self.source # start width source
        try:
            while obj:
    #            print obj
                if 'basename' in obj.all_trait_names(): # at original source?
                    basename = obj.basename # get the name
                    break
                else:
                    obj = obj.source # traverse down until original data source
        except AttributeError:
            basename = 'void' # if no file source is found
        return basename

    def save(self):
        """ saves source output to one- or two-channel .wav file """
        nc = len(self.channels)
        if not (0 < nc < 3):
            raise ValueError("one or two channels allowed, %i channels given" %\
            nc)
        name = self.basename
        for nr in self.channels:
            name += '_%i' % nr
        name += '.wav'
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

class WriteH5( TimeInOut ):
    """
    saves time signal as h5 file
    """
    # basename for cache
    name = File(filter=['*.h5'], 
        desc="name of data file")    
      
    # internal identifier
    digest = Property( depends_on = ['source.digest', '__class__'])

    traits_view = View(
        [Item('source', style='custom'), 
            ['name{File name}', 
            '|[Properties]'], 
            '|'
        ], 
        title='write .h5', 
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)


    def save(self):
        """ saves source output h5 file """
        if self.name == '':
            name = datetime.now().isoformat('_').replace(':','-').replace('.','_')
            self.name = path.join(td_dir,name+'.h5')
        f5h = tables.openFile(self.name, mode = 'w')
        ac = f5h.createEArray(f5h.root, 'time_data', \
            tables.atom.Float32Atom(), (0, self.numchannels))
        ac.setAttr('sample_freq', self.sample_freq)
        for data in self.source.result(4096):
            ac.append(data)
        f5h.close()
        
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
    numchannels = Property( depends_on = ['sectors', ])

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

    def result( self, num=1 ):
        """ 
        python generator: delivers the source output integrateds over the given 
        sectors, yields samples in blocks of shape (num, numchannels), the last 
        block may be shorter than num; numchannels is the number of sectors
        """
        inds = [self.grid.indices(*sector) for sector in self.sectors]
        gshape = self.grid.shape
        o = empty((num, self.numchannels), dtype=float) # output array
        for r in self.source.result(num):
            ns = r.shape[0]
            mapshape = (ns,) + gshape
            rmax = r.max()
            rmin = rmax * 10**(self.clip/10.0)
            r = where(r>rmin, r, 0.0)
            i = 0
            for ind in inds:
                h = r[:].reshape(mapshape)[ (s_[:],) + ind ]
                o[:ns, i] = h.reshape(h.shape[0], -1).sum(axis=1)
                i += 1
            yield o[:ns]


