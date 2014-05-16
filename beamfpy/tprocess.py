# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Beamfpy Development Team.
#------------------------------------------------------------------------------
"""Implements processing in the time domain.

.. autosummary::
    :toctree: generated/

    TimeInOut
    Mixer
    TimePower
    TimeAverage
    TimeReverse
    FiltFiltOctave
    FiltOctave
    TimeCache
    WriteWAV
    WriteH5
"""

# imports from other packages
from numpy import array, newaxis, empty, empty_like, pi, sin, sqrt, arange, \
clip, r_, zeros, int16, histogram, unique, cross, dot
from traits.api import Float, Int, \
File, CArray, Property, Instance, Trait, Bool, Delegate, \
cached_property, on_trait_change, List
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons
from datetime import datetime
from os import path
import tables
import wave
from scipy.signal import butter, lfilter, filtfilt

# beamfpy imports
from .internal import digest
from h5cache import H5cache, td_dir
from .grids import RectGrid
from .microphones import MicGeom
from .environments import Environment
from .sources import SamplesGenerator
from .trajectory import Trajectory


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
        
