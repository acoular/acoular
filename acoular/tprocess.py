# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements processing in the time domain.

.. autosummary::
    :toctree: generated/

    TimeInOut
    MaskedTimeInOut
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
from traits.api import Float, Int, CLong, \
File, CArray, Property, Instance, Trait, Bool, Delegate, \
cached_property, on_trait_change, List
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons
from datetime import datetime
from os import path
import tables
import wave
from scipy.signal import butter, lfilter, filtfilt
from warnings import warn

# acoular imports
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
    gets samples from :attr:`source` and generates output via the 
    generator :meth:`result`
    """

    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Trait(SamplesGenerator)

    #: Sampling frequency of output signal, as given by :attr:`source`
    sample_freq = Delegate('source')
    
    #: Number of channels in output, as given by :attr:`source`
    numchannels = Delegate('source')
               
    #: Number of samples in output, as given by :attr:`source`
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
        Python generator: dummy function, just echoes the output of source,
        yields samples in blocks of shape (num, :attr:`numchannels`), the last block
        may be shorter than num.
        """
        for temp in self.source.result(num):
            # effectively no processing
            yield temp


class MaskedTimeInOut ( TimeInOut ):
    """
    Signal processing block for channel and sample selection.
    
    This class serves as intermediary to define (in)valid 
    channels and samples for any 
    :class:`~acoular.sources.SamplesGenerator` (or derived) object.
    It gets samples from :attr:`~acoular.tprocess.TimeInOut.source` 
    and generates output via the generator :meth:`result`.
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
    channels = Property(depends_on = ['invalid_channels', 'source.numchannels'], 
        desc="channel mask")
    
    #: Number of channels in input, as given by :attr:`~acoular.tprocess.TimeInOut.source`
    numchannels_total = Delegate('source', 'numchannels')
               
    #: Number of samples in input, as given by :attr:`~acoular.tprocess.TimeInOut.source`
    numsamples_total = Delegate('source', 'numsamples')

    #: Number of valid channels, is set automatically
    numchannels = Property(depends_on = ['invalid_channels', \
        'source.numchannels'], desc="number of valid input channels")

    #: Number of valid time samples, is set automatically
    numsamples = Property(depends_on = ['start', 'stop', 'source.numsamples'], 
        desc="number of valid samples per channel")

    #: Name of the cache file without extension, readonly.
    basename = Property( depends_on = 'source.digest', 
        desc="basename for cache file")

    # internal identifier
    digest = Property( depends_on = ['source.digest', 'start', 'stop', \
        'invalid_channels'])

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_basename( self ):
        if 'basename' in self.source.all_trait_names():
            return self.source.basename
        else: 
            return self.source.__class__.__name__ + self.source.digest
    
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
        Samples in blocks of shape (num, :attr:`numchannels`). 
            The last block may be shorter than num.
        """
        sli = slice(self.start, self.stop).indices(self.numsamples_total)
        start = sli[0]
        stop = sli[1]
        if start >= stop:
            raise IOError("no samples available")
        
        if start != 0 or stop != self.numsamples_total:

            stopoff = -stop % num
            offset = -start % num
            if offset == 0: offset = num      
            buf = empty((num + offset , self.numchannels), dtype=float) # buffer array
            i = 0
            for block in self.source.result(num):
                i += num
                if i > start and i <= stop+stopoff:
                    ns = block.shape[0] # numbers of samples
                    buf[offset:offset+ns] = block[:, self.channels]
                    if i > start + num:
                        yield buf[:num]
                    buf[:offset] = buf[num:num+offset]
            if offset-stopoff != 0:
                yield buf[:(offset-stopoff)]
        
        else: # if no start/stop given, don't do the resorting thing
            for block in self.source.result(num):
                yield block[:, self.channels]
                


class Mixer( TimeInOut ):
    """
    Mixes the signals from several sources.
    """

    #: Data source; :class:`~acoular.sources.SamplesGenerator` object.
    source = Trait(SamplesGenerator)

    #: List of additional :class:`~acoular.sources.SamplesGenerator` objects
    #: to be mixed.
    sources = List( Instance(SamplesGenerator, ()) ) 

    #: Sampling frequency of the signal as given by :attr:`source`.
    sample_freq = Delegate('source')
    
    #: Number of channels in output as given by :attr:`source`.
    numchannels = Delegate('source')
               
    #: Number of samples in output as given by :attr:`source`.
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
        Python generator that yields the output block-wise.
        The output from the source and those in the list 
        sources are being added.
        
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
    Calculates time-depended power of the signal
    """

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
        Squared output of source. 
            Yields samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        for temp in self.source.result(num):
            yield temp*temp
    
class TimeAverage( TimeInOut ) :
    """
    Calculates time-depended average of the signal
    """
    #: Number of samples to average over, defaults to 64.
    naverage = Int(64, 
        desc = "number of samples to average over")
        
    #: Sampling frequency of the output signal, is set automatically.
    sample_freq = Property( depends_on = 'source.sample_freq, naverage')
    
    #: Number of samples of the output signal, is set automatically.
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
        Python generator that yields the output block-wise.

        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Average of the output of source. 
            Yields samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        nav = self.naverage
        for temp in self.source.result(num*nav):
            ns, nc = temp.shape
            nso = ns/nav
            if nso > 0:
                yield temp[:nso*nav].reshape((nso, -1, nc)).mean(axis=1)
                
class TimeReverse( TimeInOut ):
    """
    Calculates the time-reversed signal of a source. 
    """
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
        Yields samples in blocks of shape (num, numchannels). 
            Time-reversed output of source. 
            The last block may be shorter than num.
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
    Octave or third-octave filter with zero phase delay.
    
    This filter can be applied on time signals.
    It requires large amounts of memory!   
    """
    #: Band center frequency; defaults to 1000.
    band = Float(1000.0, 
        desc = "band center frequency")
        
    #: Octave fraction: 'Octave' or 'Third octave'; defaults to 'Octave'.
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
        """ 
        Internal Butterworth filter design routine.
        
        Parameters
        ----------
        order : integer
            The order of the filter.
        
        Returns
        -------
            b, a : ndarray, ndarray
                Filter coefficients
        """
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
        Python generator that yields the output block-wise.

        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            Delivers the zero-phase bandpass filtered output of source.
            The last block may be shorter than num.
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
    Octave or third-octave filter (not zero-phase).
    """

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
        Samples in blocks of shape (num, numchannels). 
            Delivers the bandpass filtered output of source.
            The last block may be shorter than num.
        """
        b, a = self.ba(3) # filter order = 3
        zi = zeros((max(len(a), len(b))-1, self.source.numchannels))
        for block in self.source.result(num):
            block, zi = lfilter(b, a, block, axis=0, zi=zi)
            yield block

                       
class TimeCache( TimeInOut ):
    """
    Caches time signal in cache file.
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
        Python generator that yields the output from cache block-wise.

        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
            Echos the source output, but reads it from cache
            when available and prevents unnecassary recalculation.
        """
        name = 'tc_' + self.digest
        H5cache.get_cache( self, self.basename )
        if not name in self.h5f.root:
            ac = self.h5f.create_earray(self.h5f.root, name, \
                                       tables.atom.Float32Atom(), \
                                        (0, self.numchannels))
            ac.set_attr('sample_freq', self.sample_freq)
            for data in self.source.result(num):
                ac.append(data)
                yield data
        else:
            ac = self.h5f.get_node('/', name)
            i = 0
            while i < ac.shape[0]:
                yield ac[i:i+num]
                i += num

class WriteWAV( TimeInOut ):
    """
    Saves time signal from one or more channels as mono/stereo/multi-channel
    *.wav file
    """
    # basename for cache
    basename = Property( depends_on = 'digest')
       
    #: Channel(s) to save. List can only contain one or two channels.
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
            else:
                basename = 'void'
        except AttributeError:
            basename = 'void' # if no file source is found
        return basename

    def save(self):
        """ 
        Saves source output to one- or multiple-channel *.wav file. 
        """
        nc = len(self.channels)
        if nc == 0:
            raise ValueError("No channels given for output")
        if nc > 2:
            warn("More than two channels given for output, exported file will have %i channels" % nc)
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
    Saves time signal as *.h5 file
    """
    #: Name of the file to be saved. If none is given, the name will be
    #: automatically generated from a time stamp.
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
        f5h = tables.open_file(self.name, mode = 'w')
        ac = f5h.create_earray(f5h.root, 'time_data', \
            tables.atom.Float32Atom(), (0, self.numchannels))
        ac.set_attr('sample_freq', self.sample_freq)
        for data in self.source.result(4096):
            ac.append(data)
        f5h.close()
        
