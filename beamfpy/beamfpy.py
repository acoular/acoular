# coding=UTF-8
"""
Several classes for the implemetation of acoustic beamforming

A minimal usage example would be:

>>    m=MicGeom(from_file='mic_geom.xml')
>>    g=RectGrid(x_min=-0.8,x_max=-0.2,y_min=-0.1,y_max=0.3,z=0.8,increment=0.01)
>>    t1=TimeSamples(name='measured_data.h5')
>>    cal=Calib(from_file='calibration_data.xml')
>>    f1=EigSpectra(time_data=t1,block_size=256,window="Hanning",overlap='75%',calib=cal)
>>    e1=BeamformerBase(freq_data=f1,grid=g,mpos=m,r_diag=False)
>>    fr=4000
>>    L1=L_p(e1.synthetic(fr,0))

The classes in the module possess a number of automatic data update
capabilities. That is, only the traits must be set to get the results.
The calculation need not be triggered explictely.
    BEWARE: sometimes this gives problems with timing and results will not
    be available immediately
The classes are also GUI-aware, they know how to display a graphical user
interface. So by calling
>>    object_name.configure_traits()
on object "object_name" the relevant traits of each instance object may
be edited graphically.
The traits could also be set explicitely in the program, either in the
constructor of an object:
>>    m=MicGeom(from_file='mic_geom.xml')
or at a later time
>>    m.from_file='another_mic_geom.xml'
where all objects that depend upon the specific trait will update their
output if necessary.

beamfpy.py (c) Ennes Sarradj 2007-2008, all rights reserved
"""

__author__ = "Ennes Sarradj, ennes.sarradj@gmx.de"
__date__ = "22 Dez 2008"
__version__ = "3.0beta"

from scipy import io
from numpy import *
from threading import Thread, Lock
from enthought.traits.api import HasTraits, HasPrivateTraits, Float, Int, Long,\
File, CArray, Property, Instance, Trait, Bool, Range, Delegate, Any, Str,\
cached_property, on_trait_change, property_depends_on
from enthought.traits.ui.api import View, Item, Group
from enthought.traits.ui.menu import OKCancelButtons
from beamformer import * # ok to use *
from os import path, mkdir, environ
from string import join
from time import sleep, time
import md5
import cPickle
import tables
from weakref import WeakValueDictionary, getweakrefcount
import ConfigParser
import struct

from internal import digest


# path to cache directory, possibly in temp
try:
    cache_dir=path.join(environ['TEMP'],'beamfpy_cache')
except:
    cache_dir=path.join(path.curdir,'cache')
if not path.exists(cache_dir):
    mkdir(cache_dir)

# path to td directory (used for import to *.h5 files)
try:
    td_dir=path.join(environ['HOMEDRIVE'],environ['HOMEPATH'],'beamfpy_td')
except:
    td_dir=path.join(path.curdir,'td')
if not path.exists(td_dir):
    mkdir(td_dir)


#import pp
#~ ppservers = ()
#ppservers = ("141.43.128.41",)
#job_server = pp.Server(ncpus=1,ppservers=ppservers)
#~ job_server = pp.Server(ncpus=4,ppservers=ppservers)
#~ job_server = pp.Server(ppservers=ppservers)

#print "Starting pp with", job_server.get_ncpus(), "workers"



class H5cache_class(HasPrivateTraits):
    """
    cache class that handles opening and closing tables.File objects
    """
    # cache directory
    cache_dir = Str
    
    busy = Bool(False)
    
    open_files = WeakValueDictionary()
    
    open_count = dict()
    
    def get_cache( self, object, name, mode='a' ):
        while self.busy:
            pass
        self.busy = True
        cname = name + '_cache.h5'
        if isinstance(object.h5f,tables.File):
            oname = path.basename(object.h5f.filename)
            if oname == cname:
                self.busy = False
                return
            else:
                self.open_count[oname] = self.open_count[oname] - 1
                # close if no references to file left
                if not self.open_count[oname]:
                    object.h5f.close()
        # open each file only once
        if not self.open_files.has_key(cname):
            object.h5f = tables.openFile(path.join(self.cache_dir,cname),mode)
            self.open_files[cname] = object.h5f
        else:
            object.h5f = self.open_files[cname]
            object.h5f.flush()
        self.open_count[cname] = self.open_count.get(cname,0) + 1
        print self.open_count.items()
        self.busy = False
        
        
H5cache = H5cache_class(cache_dir=cache_dir)

class TimeSamples( HasPrivateTraits ):
    """
    Container for time data, loads time data
    and provides information about this data
    """

    # full name of the .h5 file with data
    name = File(filter=['*.h5'],
        desc="name of data file")

    # basename of the .h5 file with data
    basename = Property( depends_on = 'name',#filter=['*.h5'],
        desc="basename of data file")
    
    # sampling frequency of the data, is set automatically
    sample_freq = Float(1.0,
        desc="sampling frequency")

    # number of channels, is set automatically
    numchannels = Long(0L,
        desc="number of input channels")

    # number of time data samples, is set automatically
    numsamples = Long(0L,
        desc="number of samples")

    # the time data as (numsamples,numchannels) array of floats
    data = Any(
        desc="the actual time data array")

    # hdf5 file object
    h5f = Instance(tables.File)
    
    # internal identifier
    digest = Property( depends_on = ['basename',])

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
        if self.h5f!=None:
            try:
                self.h5f.close()
            except:
                pass
        self.h5f = tables.openFile(self.name)
        self.data = self.h5f.root.time_data
        self.sample_freq = self.data.getAttr('sample_freq')
        (self.numsamples,self.numchannels) = self.data.shape
#        self.basename = path.basename(self.name)

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
    digest = Property( depends_on = ['basename',] )

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
            self.data=array([1.0,],'d')
            self.num_mics=1
            return
        import xml.dom.minidom
        doc=xml.dom.minidom.parse(self.from_file)
        names=[]
        data=[]
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            data.append(float(el.getAttribute('factor')))
        self.data=array(data,'d')
        self.num_mics=shape(self.data)[0]

class PowerSpectra( HasPrivateTraits ):
    """
    efficient calculation of full cross spectral matrix
    container for data and properties of this matrix
    """

    # the TimeSamples object that provides the data
    time_data = Trait(TimeSamples,
        desc="time data object")

    # the Calib object that provides the calibration data,
    # defaults to no calibration, i.e. the raw time data is used
    calib = Instance(Calib)

    # FFT block size, one of: 128,256,512,1024,2048 ... 16384
    # defaults to 1024
    block_size = Trait(1024,128,256,512,1024,2048,4096,8192,16384,
        desc="number of samples per FFT block")

    # index of lowest frequency line
    # defaults to 0
    ind_low = Range(1,
        desc="index of lowest frequency line")

    # index of highest frequency line
    # defaults to -1 (last possible line for default block_size)
    ind_high = Int(-1,
        desc="index of highest frequency line")

    # window function for FFT, one of:
    # 'Rectangular' (default),'Hanning','Hamming','Bartlett','Blackman'
    window = Trait('Rectangular',
        {'Rectangular':ones,
        'Hanning':hanning,
        'Hamming':hamming,
        'Bartlett':bartlett,
        'Blackman':blackman},
        desc="type of window for FFT")

    # overlap factor for averaging: 'None'(default),'50%','75%','87.5%'
    overlap = Trait('None',{'None':1,'50%':2,'75%':4,'87.5%':8},
        desc="overlap of FFT blocks")

    # number of FFT blocks to average (auto-set from block_size and overlap)
    num_blocks = Property(
        desc="overall number of FFT blocks")

    # frequency range
    freq_range = Property(
        desc = "frequency range" )
        
    # frequency range
    indices = Property(
        desc = "index range" )
        
    # the cross spectral matrix as
    # (number of frequencies,numchannels,numchannels) array of complex
    csm = Property( 
        desc="cross spectral matrix")

    # internal identifier
    digest = Property( 
        depends_on = ['time_data.digest','calib.digest','block_size',
            'window','overlap'],
        )

    # hdf5 cache file
    h5f = Instance(tables.File)
    
    traits_view = View(
        ['time_data@{}',
         'calib@{}',
            ['block_size',
                'window',
                'overlap',
                    ['ind_low{Low Index}',
                    'ind_high{High Index}',
                    '-[Frequency range indices]'],
                    ['num_blocks~{Number of blocks}',
                    'freq_range~{Frequency range}',
                    '-'],
                '[FFT-parameters]'
            ],
        ],
        buttons = OKCancelButtons
        )
    
    @property_depends_on('time_data.numsamples,block_size,overlap')
    def _get_num_blocks ( self ):
        return self.overlap_*self.time_data.numsamples/self.block_size-self.overlap_+1

    @property_depends_on( 'time_data.sample_freq,block_size,ind_low,ind_high' )
    def _get_freq_range ( self ):
            try:
                return self.fftfreq()[[ self.ind_low, self.ind_high ]]
            except IndexError:
                return array([0.,0])

    @property_depends_on( 'block_size,ind_low,ind_high' )
    def _get_indices ( self ):
            try:
                return range(self.block_size/2+1)[ self.ind_low: self.ind_high ]
            except IndexError:
                return range(0)

    @cached_property
    def _get_digest( self ):
        return digest( self )

    @property_depends_on('digest')
    def _get_csm ( self ):
        """main work is done here:
        cross spectral matrix is either loaded from cache file or
        calculated and then additionally stored into cache
        """
  #      try:
        name = 'csm_' + self.digest
        H5cache.get_cache( self, self.time_data.basename )
        if not name in self.h5f.root:
            t = self.time_data
            td = t.data
            wind = self.window_( self.block_size )
            weight = dot( wind, wind )
            wind = wind[newaxis,:].swapaxes( 0, 1 )
            numfreq = self.block_size/2 + 1
            csm_shape = (numfreq,t.numchannels,t.numchannels)
            csm = zeros(csm_shape,'D')
            print "num blocks",self.num_blocks
            if self.calib:
                if self.calib.num_mics==t.numchannels:
                    wind = wind * self.calib.data[newaxis,:]
                else:
                    print "warning: calibration data not compatible:",self.calib.num_mics,t.numchannels
            for block in range(self.num_blocks):
                pos = block*self.block_size/self.overlap_
                ft = fft.rfft(self.time_data.data[pos:(pos+self.block_size)]*wind,None,0)
                faverage(csm,ft)
            csm=csm*(2.0/self.block_size/weight/self.num_blocks) #2.0=sqrt(2)^2 wegen der halbseitigen FFT
            atom = tables.ComplexAtom(8)
            #filters = tables.Filters(complevel=5, complib='zlib')
            ac = self.h5f.createCArray(self.h5f.root, name, atom, csm_shape)#, filters=filters)
            ac[:] = csm
            return ac
        else:
            return self.h5f.getNode('/',name)
#        except:
 #           return None

    def fftfreq ( self ):
        """
        returns an array of the frequencies for
        the spectra in the cross spectral matrix
        """
        return abs(fft.fftfreq(self.block_size,1./self.time_data.sample_freq)[:self.block_size/2+1])
#        if self.time_data.sample_freq>0:
#            return fft.fftfreq(self.block_size,1./self.time_data.sample_freq)[:self.block_size/2+1][self.ind_low:self.ind_high]
#        else:
#            return array([0.],'d')

class EigSpectra( PowerSpectra ):
    """
    efficient calculation of full cross spectral matrix
    container for data and properties of this matrix
    and its eigenvalues and eigenvectors
    """

    # eigenvalues of the cross spectral matrix
    eva = Property( 
        desc="eigenvalues of cross spectral matrix")

    # eigenvectors of the cross spectral matrix
    eve = Property( 
        desc="eigenvectors of cross spectral matrix")

    @property_depends_on('digest')
    def _get_eva ( self ):
        return self.calc_ev()[0]

    @property_depends_on('digest')
    def _get_eve ( self ):
        return self.calc_ev()[1]

    def calc_ev ( self ):
        """
        eigenvalues / eigenvectors calculation
        """
        name_eva = 'eva_' + self.digest
        name_eve = 'eve_' + self.digest
        csm = self.csm #trigger calculation
        if (not name_eva in self.h5f.root) or (not name_eve in self.h5f.root):
            csm_shape = self.csm.shape
            eva = empty(csm_shape[0:2],float32)
            eve = empty(csm_shape,complex64)
            for i in range(csm_shape[0]):
                (eva[i],eve[i])=linalg.eigh(self.csm[i])
            atom_eva = tables.Float32Atom()
            atom_eve = tables.ComplexAtom(8)
            #filters = tables.Filters(complevel=5, complib='zlib')
            ac_eva = self.h5f.createCArray(self.h5f.root, name_eva, atom_eva, eva.shape)#, filters=filters)
            ac_eve = self.h5f.createCArray(self.h5f.root, name_eve, atom_eve, eve.shape)#, filters=filters)
            ac_eva[:] = eva
            ac_eve[:] = eve
        return (self.h5f.getNode('/',name_eva),self.h5f.getNode('/',name_eve))
            

    def synthetic_ev( self, freq, num=0):
        """
        returns synthesized frequency band values of the eigenvalues
        num = 0: single frequency line
        num = 1: octave band
        num = 3: third octave band
        etc.
        """
        f=self.fftfreq()
        if num==0:
            # single frequency line
            return self.eva[searchsorted(f,freq)]
        else:
            f1=searchsorted(f,freq*2.**(-0.5/num))
            f2=searchsorted(f,freq*2.**(0.5/num))
            if f1==f2:
                return self.eva[f1]
            else:
                return sum(self.eva[f1:f2],0)

#TODO: construct a base class for this
class RectGrid( HasPrivateTraits ):
    """
    constructs a quadratic 2D grid for the beamforming results
    that is on a plane perpendicular to the z-axis
    """

    x_min = Float(-1.0,
        desc="minimum  x-value")

    x_max = Float(1.0,
        desc="maximum  x-value")

    y_min = Float(-1.0,
        desc="minimum  y-value")

    y_max = Float(1.0,
        desc="maximum  y-value")

    z = Float(1.0,
        desc="position on z-axis")

    # increment in x- and y- direction
    increment = Float(0.1,
        desc="step size")

    # overall number of grid points (auto-set)
    size = Property( 
        desc="overall number of grid points")
        
    # shape of grid
    shape = Property(
        desc="grid shape as tuple")
    
    # number of grid points alog x-axis (auto-set)
    nxsteps = Property( 
        desc="number of grid points alog x-axis")

    # number of grid points alog y-axis (auto-set)
    nysteps = Property( 
        desc="number of grid points alog y-axis")

    # internal identifier
    digest = Property( 
        depends_on = ['x_min', 'x_max', 'y_min', 'y_max', 'z', 'increment']
        )

    traits_view = View(
            [
                ['x_min','y_min','|'],
                ['x_max','y_max','z','increment','size~{grid size}','|'],
                '-[Map extension]'
            ]
        )

    @property_depends_on('nxsteps,nysteps')
    def _get_size ( self ):
        return self.nxsteps*self.nysteps

    @property_depends_on('nxsteps,nysteps')
    def _get_shape ( self ):
        return (self.nxsteps,self.nysteps)

    @property_depends_on('x_min,x_max,increment')
    def _get_nxsteps ( self ):
        i=abs(self.increment)
        if i!=0:
            return int(round((abs(self.x_max-self.x_min)+i)/i))
        return 1

    @property_depends_on('y_min,y_max,increment')
    def _get_nysteps ( self ):
        i=abs(self.increment)
        if i!=0:
            return int(round((abs(self.y_max-self.y_min)+i)/i))
        return 1

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def pos ( self ):
        """
        returns an (3,size) array with the grid point x,y,z-coordinates
        """
        i=self.increment
        xi=1j*round((self.x_max-self.x_min+i)/i)
        yi=1j*round((self.y_max-self.y_min+i)/i)
        bpos=mgrid[self.x_min:self.x_max:xi,self.y_min:self.y_max:yi,self.z:self.z+1]
        bpos.resize((3,self.size))
        return bpos

    def index ( self,x,y ):
        """
        returns the indices for a certain x,y co-ordinate
        """
        if x<self.x_min or x>self.x_max:
            raise ValueError, "x-value out of range"
        if y<self.y_min or y>self.y_max:
            raise ValueError, "y-value out of range"
        xi=round((x-self.x_min)/self.increment)
        yi=round((y-self.y_min)/self.increment)
        return xi,yi

    def indices ( self,x1,y1,x2,y2 ):
        """
        returns the slices to index a recangular subdomain,
        useful for inspecting subdomains in a result already calculated
        """
        xi1,yi1 = self.index(min(x1,x2),min(y1,y2))
        xi2,yi2 = self.index(max(x1,x2),max(y1,y2))
        return s_[xi1:xi2+1],s_[yi1:yi2+1]

    def extend (self) :
        """
        returns the x,y extension of the grid,
        useful for the imshow function from pylab
        """
        return (self.x_min,self.x_max,self.y_min,self.y_max)

class MicGeom( HasPrivateTraits ):
    """
    container for the geometric arrangement of microphones
    reads data from xml-source with element tag names 'pos'
    and attributes Name,x,y and z
    """

    # name of the .xml-file
    from_file = File(filter=['*.xml'],
        desc="name of the xml file to import")

    # basename of the .xml-file
    basename = Property( depends_on = 'from_file',
        desc="basename of xml file")
    
    # number of mics
    num_mics = Int( 0,
        desc="number of microphones in the geometry")

    # positions as (3,num_mics) array
    mpos = Any(
        desc="x,y,z position of microphones")

    # internal identifier
    digest = Property( depends_on = ['mpos',])

    traits_view = View(
        ['from_file',
        'num_mics~',
        '|[Microphone geometry]'
        ],
#        title='Microphone geometry',
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.from_file))[0]

    @on_trait_change('basename')
    def import_mpos( self ):
        """import the microphone positions from .xml file,
        called when basename changes
        """
        if not path.isfile(self.from_file):
            # no file there
            self.mpos=array([],'d')
            self.num_mics=0
            return
        import xml.dom.minidom
        doc=xml.dom.minidom.parse(self.from_file)
        names=[]
        xyz=[]
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            xyz.append(map(lambda a : float(el.getAttribute(a)),'xyz'))
        self.mpos=array(xyz,'d').swapaxes(0,1)
        self.num_mics=self.mpos.shape[1]


class BeamformerBase( HasPrivateTraits ):
    """
    beamforming using the basic delay-and-sum algorithm
    """

    # PowerSpectra object that provides the cross spectral matrix
    freq_data = Trait(PowerSpectra,
        desc="freq data object")

    # RectGrid object that provides the grid locations
    grid = Trait(RectGrid,
        desc="beamforming grid")

    # MicGeom object that provides the microphone locations
    mpos = Trait(MicGeom,
        desc="microphone geometry")

    # the speed of sound, defaults to 343 m/s
    c = Float(343.,
        desc="speed of sound")

    # flag, if true (default), the main diagonal is removed before beamforming
    r_diag = Bool(True,
        desc="removal of diagonal")

    # hdf5 cache file
    h5f = Instance(tables.File)
    
    # the result, sound pressure squared in all grid locations
    # as (number of frequencies, nxsteps,nysteps) array of float
    result = Property(
        desc="beamforming result")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', 'r_diag'],
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest','freq_data.ind_low','freq_data.ind_high'],
        )

    traits_view = View(
        [
            [Item('mpos{}',style='custom')],
            [Item('grid',style='custom'),'-<>'],
            [Item('r_diag',label='diagonal removed')],
            '|'
        ],
        title='Beamformer options',
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @cached_property
    def _get_ext_digest( self ):
        return digest( self, 'ext_digest' )

    @property_depends_on('ext_digest')
    def _get_result ( self ):
        """
        beamforming result is either loaded or calculated
        """
        #try:
        digest=''
        while self.digest!=digest:
            digest = self.digest
            name = self.__class__.__name__ + self.digest
            print 1,name
            numchannels = self.freq_data.time_data.numchannels
            print "nch",numchannels
            if  numchannels != self.mpos.num_mics or numchannels==0:
                return None#zeros(1)
                raise ValueError()
            numfreq = self.freq_data.block_size/2 + 1
            H5cache.get_cache( self, self.freq_data.time_data.basename)
            if not name in self.h5f.root:
                group = self.h5f.createGroup(self.h5f.root, name)
                shape = (numfreq,self.grid.size)
                atom = tables.Float32Atom()
                #filters = tables.Filters(complevel=5, complib='zlib')
                ac = self.h5f.createCArray(group, 'result', atom, shape)#, filters=filters)
                shape = (numfreq,)
                atom = tables.BoolAtom()
                fr = self.h5f.createCArray(group, 'freqs', atom, shape)#, filters=filters)
            else:
                ac = self.h5f.getNode('/'+name,'result')
                fr = self.h5f.getNode('/'+name,'freqs')
            # print self.freq_data.h5f#csm[5,1,1]
            if not fr[self.freq_data.ind_low:self.freq_data.ind_high].all():
                self.calc(ac,fr)
                self.h5f.flush()
            print 2,name
        return ac
     #   except:
      #      import sys
       #     print sys.exc_info()
       #     return None

    def calc(self, ac, fr):
        """
        calculation of delay-and-sum beamforming result 
        for all missing frequencies
        """
        # prepare calculation
        kj = 2j*pi*self.freq_data.fftfreq()/self.c
        numchannels = self.freq_data.time_data.numchannels
        e = zeros((numchannels),'D')
        bpos = self.grid.pos()
        h = zeros((1,bpos.shape[1]),'d')
        # function
        if self.r_diag:
            beamfunc = beamdiag
            adiv = 1.0/(numchannels*numchannels-numchannels)
            scalefunc = lambda h : adiv*multiply(h,(sign(h)+1-1e-35)/2)
        else:
            beamfunc = beamfull
            adiv= 1.0/(numchannels*numchannels)
            scalefunc = lambda h : adiv*h
   #     frange=range(self.freq_data.ind_low,self.freq_data.ind_high)
  #      from time import time
       # t=time()
        #~ jobs = [
            #~ (i, job_server.submit(cfunc, 
            #~ (array(self.freq_data.csm[i][newaxis],dtype='complex128'),kj[i,newaxis],bpos,self.mpos.mpos,numchannels), ( ),
            #~ ("beamfpy.beamformer", "numpy",))) 
            #~ for i in frange if not fr[i]]
        #~ for i, h in jobs:
            #~ ac[i] = scalefunc(h())
            #~ fr[i] = True
        #~ print time()-t
        #~ job_server.print_stats()
        for i in self.freq_data.indices:
            if not fr[i]:
                csm = array(self.freq_data.csm[i][newaxis],dtype='complex128')
                kji = kj[i,newaxis]
                beamfunc(csm,e,h,bpos,self.mpos.mpos,kji)
                ac[i] = scalefunc(h)
                fr[i] = True
#        print time()-t
    
    def synthetic( self, freq, num=0):
        """
        returns synthesized frequency band values of beamforming result
        num = 0: single frequency line
        num = 1: octave band
        num = 3: third octave band
        etc.
        """
        r = self.result # trigger calculation
        #print "synth",num
        f=self.freq_data.fftfreq()
        if len(f)==0:
            return None#array([[1,],],'d')
        try:
            if num==0:
                # single frequency line
                h = self.result[searchsorted(f,freq)]
            else:
                f1=searchsorted(f,freq*2.**(-0.5/num))
                f2=searchsorted(f,freq*2.**(0.5/num))
                if f1==f2:
                    h = self.result[f1]
                else:
                    h = sum(self.result[f1:f2],0)
            return h.reshape(self.grid.shape)
        except:
            return None#ones((1,1),'d')

    def integrate(self,sector):
        """
        integrates result map over the given sector
        where sector is a tuple with arguments for grid.indices
        e.g. (xmin,ymin,xmin,ymax)
        returns spectrum
        """
        ind = self.grid.indices(*sector)
        gshape = self.grid.shape
        r = self.result
        rshape = r.shape
        mapshape = (rshape[0],) + gshape
        h = r[:].reshape(mapshape)[ (s_[:],) + ind ]
        return h.reshape(h.shape[0],prod(h.shape[1:])).sum(axis=1)

def cfunc(csm,kji,bpos,mpos,numchannels):
    e = numpy.zeros((numchannels),'D')
    h = numpy.zeros((1,bpos.shape[1]),'d')
    beamfpy.beamformer.beamdiag(csm,e,h,bpos,mpos,kji)
    return h

class BeamformerCapon( BeamformerBase ):
    """
    beamforming using the minimum variance or Capon algorithm
    """
    traits_view = View(
        [
            [Item('mpos{}',style='custom')],
            [Item('grid',style='custom'),'-<>'],
            '|'
        ],
        title='Beamformer options',
        buttons = OKCancelButtons
        )

    def calc(self, ac, fr):
        """
        calculation of Capon (Mininimum Variance) beamforming result 
        for all missing frequencies
        """
        # prepare calculation
        kj = 2j*pi*self.freq_data.fftfreq()/self.c
        numchannels = self.freq_data.time_data.numchannels
        e = zeros((numchannels),'D')
        bpos = self.grid.pos()
        h = zeros((1,bpos.shape[1]),'d')
        adiv= 1.0/(numchannels*numchannels)
        for i in self.freq_data.indices:
            if not fr[i]:
                csm = linalg.inv(array(self.freq_data.csm[i],dtype='complex128'))[newaxis]
                kji = kj[i,newaxis]
                beamfull(csm,e,h,bpos,self.mpos.mpos,kji)
                ac[i] = adiv/h
                fr[i] = True

class BeamformerEig( BeamformerBase ):
    """
    beamforming using eigenvalue and eigenvector techniques
    """

    # EigSpectra object that provides the cross spectral matrix and eigenvalues
    freq_data = Trait(EigSpectra,
        desc="freq data object")

    # no of component to calculate 0 (smallest) ... numchannels-1
    # defaults to -1, i.e. numchannels-1
    n = Int(-1,
        desc="no of eigenvalue")

    # actual component to calculate
    na = Property(
        desc="no of eigenvalue")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', 'r_diag', 'na'],
        )

    traits_view = View(
        [
            [Item('mpos{}',style='custom')],
            [Item('grid',style='custom'),'-<>'],
            [Item('n',label='component no',style='text')],
            [Item('r_diag',label='diagonal removed')],
            '|'
        ],
        title='Beamformer options',
        buttons = OKCancelButtons
        )
    
    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @property_depends_on('n')
    def _get_na( self ):
        na = self.n
        nm = self.mpos.num_mics
        if na<0:
            na = max(nm + na,0)
        return min(nm - 1, na)

    def calc(self, ac, fr):
        """
        calculation of eigenvalue beamforming result 
        for all missing frequencies
        """
        # prepare calculation
        kj = 2j*pi*self.freq_data.fftfreq()/self.c
        na = self.na
        numchannels = self.freq_data.time_data.numchannels
        e = zeros((numchannels),'D')
        bpos = self.grid.pos()
        h = zeros((1,bpos.shape[1]),'d')
        # function
        if self.r_diag:
            beamfunc = beamortho_sum_diag
            adiv = 1.0/(numchannels*numchannels-numchannels)
            scalefunc = lambda h : adiv*multiply(h,(sign(h)+1-1e-35)/2)
        else:
            beamfunc = beamortho_sum
            adiv= 1.0/(numchannels*numchannels)
            scalefunc = lambda h : adiv*h
        for i in self.freq_data.indices:        
            if not fr[i]:
                eva = array(self.freq_data.eva[i][newaxis],dtype='float64')
                eve = array(self.freq_data.eve[i][newaxis],dtype='complex128')
                kji = kj[i,newaxis]
                #TODO: change beamfuncs to work with nonzero h array
                h *= 0.0
                beamfunc(e,h,bpos,self.mpos.mpos,kji,eva,eve,na,na+1)
                ac[i] = scalefunc(h)
                fr[i] = True

class BeamformerMusic( BeamformerEig ):
    """
    beamforming using MUSIC algoritm
    """

    # assumed number of sources, should be set to a value not too small
    # defaults to 1
    n = Int(1,
        desc="assumed number of sources")

    traits_view = View(
        [
            [Item('mpos{}',style='custom')],
            [Item('grid',style='custom'),'-<>'],
            [Item('n',label='no of sources',style='text')],
            '|'
        ],
        title='Beamformer options',
        buttons = OKCancelButtons
        )

    def calc(self, ac, fr):
        """
        calculation of MUSIC beamforming result 
        for all missing frequencies
        """
        # prepare calculation
        kj = 2j*pi*self.freq_data.fftfreq()/self.c
        n = self.mpos.num_mics-self.na
        numchannels = self.freq_data.time_data.numchannels
        e = zeros((numchannels),'D')
        bpos = self.grid.pos()
        h = zeros((1,bpos.shape[1]),'d')
        # function
        for i in self.freq_data.indices:        
            if not fr[i]:
                eva = array(self.freq_data.eva[i][newaxis],dtype='float64')
                eve = array(self.freq_data.eve[i][newaxis],dtype='complex128')
                kji = kj[i,newaxis]
                #TODO: change beamfuncs to work with nonzero h array
                h *= 0.0
                beamortho_sum(e,h,bpos,self.mpos.mpos,kji,eva,eve,0,n)
                ac[i] = 4e-10*h.min()/h
                fr[i] = True

class PointSpreadFunction (HasPrivateTraits):
    """
    Array point spread function
    """
    # RectGrid object that provides the grid locations
    grid = Trait(RectGrid,
        desc="beamforming grid")

    # MicGeom object that provides the microphone locations
    mpos = Trait(MicGeom,
        desc="microphone geometry")

    # the speed of sound, defaults to 343 m/s
    c = Float(343.,
        desc="speed of sound")

    # frequency 
    freq = Float(1.0,
        desc="frequency")
        
    # the actual point spread function
    psf = Property(
        desc="point spread function")

    # hdf5 cache file
    h5f = Instance(tables.File)
    
    # internal identifier
    digest = Property( depends_on = ['mpos.digest', 'grid.digest', 'c'],
        cached = True)

    @cached_property
    def _get_digest( self ):
        return digest( self )

    @property_depends_on('digest,freq')
    def _get_psf ( self ):
        """
        point spread function is either calculated or loaded from cache
        """
        #try:           
        name = 'psf' + self.digest
        H5cache.get_cache( self, name)
        fr = ('Hz_%.2f' % self.freq).replace('.','_')
        if not fr in self.h5f.root:
            kj = array((2j*pi*self.freq/self.c,))
            gs = self.grid.size
            bpos = self.grid.pos()
            hh = ones((1,gs,gs),'d')
            e = zeros((self.mpos.num_mics),'D')
            e1 = e.copy()
            beam_psf(e,e1,hh,bpos,self.mpos.mpos,kj)
            ac = self.h5f.createArray('/', fr, hh[0]/diag(hh[0]))
        else:
            ac = self.h5f.getNode('/',fr)
        return ac

class BeamformerDamas (BeamformerBase):
    """
    DAMAS Deconvolution
    """

    # BeamformerBase object that provides data for deconvolution
    beamformer = Trait(BeamformerBase)

    # PowerSpectra object that provides the cross spectral matrix
    freq_data = Delegate('beamformer')

    # RectGrid object that provides the grid locations
    grid = Delegate('beamformer')

    # MicGeom object that provides the microphone locations
    mpos = Delegate('beamformer')

    # the speed of sound, defaults to 343 m/s
    c =  Delegate('beamformer')

    # flag, if true (default), the main diagonal is removed before beamforming
    r_diag =  Delegate('beamformer')

    # number of iterations
    n_iter = Int(100,
        desc="number of iterations")

    # internal identifier
    digest = Property( 
        depends_on = ['beamformer.digest','n_iter'],
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest','beamformer.ext_digest'],
        )
    
    traits_view = View(
        [
            [Item('beamformer{}',style='custom')],
            [Item('n_iter{Number of iterations}')],
            '|'
        ],
        title='Beamformer denconvolution options',
        buttons = OKCancelButtons
        )
    
    @cached_property
    def _get_digest( self ):
        return digest( self )
      
    @cached_property
    def _get_ext_digest( self ):
        return digest( self, 'ext_digest' )
    
    def calc(self, ac, fr):
        """
        calculation of DAMAS result 
        for all missing frequencies
        """
        freqs = self.freq_data.fftfreq()
        p = PointSpreadFunction(mpos=self.mpos, grid=self.grid, c=self.c)
        for i in self.freq_data.indices:        
            if not fr[i]:
                p.freq = freqs[i]
                y = array(self.beamformer.result[i],dtype=float64)
                x = y.copy()
                psf = p.psf[:]
                gseidel(psf,y,x,self.n_iter,1.0)
                ac[i] = x
                fr[i] = True

class BeamformerOrth (BeamformerBase):
    """
    Estimation using orthogonal beamforming
    """

    # BeamformerEig object that provides data for deconvolution
    beamformer = Trait(BeamformerEig)

    # EigSpectra object that provides the cross spectral matrix and Eigenvalues
    freq_data = Delegate('beamformer')

    # RectGrid object that provides the grid locations
    grid = Delegate('beamformer')

    # MicGeom object that provides the microphone locations
    mpos = Delegate('beamformer')

    # the speed of sound, defaults to 343 m/s
    c =  Delegate('beamformer')

    # flag, if true (default), the main diagonal is removed before beamforming
    r_diag =  Delegate('beamformer')

    # list of components to consider
    eva_list = CArray(
        desc="components")
        
    # helper: number of components to consider
    n = Int(1)

    # internal identifier
    digest = Property( 
        depends_on = ['beamformer.digest','eva_list'],
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest','beamformer.ext_digest'],
        )
    
    traits_view = View(
        [
            [Item('mpos{}',style='custom')],
            [Item('grid',style='custom'),'-<>'],
            [Item('n',label='number of components',style='text')],
            [Item('r_diag',label='diagonal removed')],
            '|'
        ],
        title='Beamformer options',
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )

    @cached_property
    def _get_ext_digest( self ):
        return digest( self, 'ext_digest' )
    
    @on_trait_change('n')
    def set_eva_list(self):
        self.eva_list = arange(-1,-1-self.n,-1)

    def calc(self, ac, fr):
        """
        calculation of orthogonal beamforming result 
        for all missing frequencies
        """
        # prepare calculation
        ii = []
        for i in self.freq_data.indices:        
            if not fr[i]:
                ii.append(i)
        numchannels = self.freq_data.time_data.numchannels
        e = self.beamformer
        for n in self.eva_list:
            e.n = n
            for i in ii:
                ac[i,e.result[i].argmax()]+=e.freq_data.eva[i,n]/numchannels
        for i in ii:
            fr[i] = True
    
class BeamformerCleansc( BeamformerBase ):
    """
    beamforming using CLEAN-SC (Sijtsma)
    """

    # no of CLEAN-SC iterations
    # defaults to 0, i.e. automatic (max 2*numchannels)
    n = Int(0,
        desc="no of iterations")

    # iteration damping factor
    # defaults to 0.6
    damp = Range(0.01,1.0,0.6,
        desc="damping factor")

    # iteration stop criterion for automatic detection
    # iteration stops if power[i]>power[i-stopn]
    # defaults to 3
    stopn = Int(3,
        desc="stop criterion index")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', 'n'],
        )

    traits_view = View(
        [
            [Item('mpos{}',style='custom')],
            [Item('grid',style='custom'),'-<>'],
            [Item('n',label='no of iterations',style='text')],
            [Item('r_diag',label='diagonal removed')],
            '|'
        ],
        title='Beamformer options',
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def calc(self, ac, fr):
        """
        calculation of orthogonal beamforming result 
        for all missing frequencies
        """
        # prepare calculation
        numchannels = self.freq_data.time_data.numchannels
        f = self.freq_data.fftfreq()
        kjall = 2j*pi*f/self.c
        bpos = self.grid.pos()
        mpos = self.mpos.mpos
        e = zeros((numchannels),'D')
        result = zeros((self.grid.size),'f')
        if self.r_diag:
           adiv = 1.0/(numchannels*numchannels-numchannels)
           fullbeamfunc = beamdiag
           orthbeamfunc = beamortho_sum_diag1
        else:
           adiv = 1.0/(numchannels*numchannels)
           fullbeamfunc = beamfull
           orthbeamfunc = beamortho_sum
        if not self.n:
            J = numchannels*2
        else:
            J = self.n
        powers = zeros(J,'d')
        h = zeros((1,self.grid.size),'d')
        h1 = h.copy()
        # loop over frequencies
        for i in self.freq_data.indices:        
            if not fr[i]:
                kj = kjall[i,newaxis]
                csm = array(self.freq_data.csm[i][newaxis],dtype='complex128',copy=1)
                fullbeamfunc(csm,e,h,bpos,mpos,kj)
                h = h*adiv
                # CLEANSC Iteration
                result *= 0.0
                for j in range(J):
                    xi_max = h.argmax() #index of maximum
                    powers[j] = hmax = h[0,xi_max] #maximum
                    result[xi_max] += hmax
                    if  j>2 and hmax>powers[j-self.stopn]:
                        #print j
                        break
                    rm = bpos[:,xi_max,newaxis]-mpos
                    rm = sum(rm*rm,0)
                    rm = sqrt(rm)
                    r0 = sum(bpos[:,xi_max]*bpos[:,xi_max],0)
                    r0 = sqrt(r0)
                    rs = (r0*(1/(rm*rm)).sum(0))
                    wmax = numchannels*sqrt(adiv)*exp(-kj[0]*(r0-rm))/(rm*rs)
                    hh = wmax.copy()
                    D1 = dot(csm[0]-diag(diag(csm[0])),wmax)/hmax
                    ww = wmax.conj()*wmax
                    for m in range(20):
                        H = hh.conj()*hh
                        hh = (D1+H*wmax)/sqrt(1+dot(ww,H))
                    hh = hh[:,newaxis]
                    csm1 = hmax*(hh*hh.conj().T)[newaxis,:,:]
                    h1 = zeros((1,shape(bpos)[1]),'d')
                    orthbeamfunc(e,h1,bpos,mpos,kj,array((hmax,))[newaxis,:],hh[newaxis,:],0,1)
                    h -= self.damp*h1*adiv
                    csm -= self.damp*csm1
                ac[i] = result
                fr[i] = True

def L_p ( x ):
    """
    calculates the sound pressure level from the sound pressure squared:

    L_p = 10 lg x/4e-10

    if x<0, return -1000. dB
    """
    return where(x>0, 10*log10(x/4e-10), -1000.)

def synthetic (data, freqs, f, num=3):
    """
    returns synthesized frequency band values of data
    num = 0: function simply returns the unaltered data, no processing
    num = 1: octave bands
    num = 3: third octave bands
    etc.

    freqs: the frequencies that correspond to the input data

    f: band center frequencies

    """
    if num==0:
        return data
    find1 = searchsorted(freqs, f*2.**(-0.5/num))
    find2 = searchsorted(freqs, f*2.**(+0.5/num))
    return array(map(lambda i,j : data[i:j].sum(),find1,find2))


