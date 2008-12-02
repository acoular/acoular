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
__date__ = "26 Sept 2008"
__version__ = "1.3beta"

from scipy import io
from numpy import *
from threading import Thread, Lock
from enthought.traits.api import HasTraits, HasPrivateTraits, Float, Int, Long, File, CArray, Property, Instance, Trait, Bool, Range, Delegate, Any, Str
from enthought.traits.ui.api import View, Item, Group
from enthought.traits.ui.menu import OKCancelButtons
#from enthought.pyface.api import GUI # unnecessary import
from beamformer import * # ok to use *
from os import path, mkdir, environ
from string import join
from time import sleep, time
import md5
import cPickle
import tables
import ConfigParser
import struct

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

class time_data_import( HasPrivateTraits ):
    """
    base class for import of time data
    """

    def get_data (self,td):
        """
        imports the data into time_data object td
        (this is a dummy function)
        """
        td.data = None
        td.numsamples = 0
        td.numchannels = 0
        td.sample_freq = 0

class csv_import( time_data_import ):
    """
    import of CSV data as saved by NI VI Logger
    """

    # name of the comma delimited file to import
    from_file = File(filter=['*.txt'],
        desc="name of the comma delimited file to import")

    # header length, defaults to 6
    header_length =  Int(6,
        desc="length of the header to ignore during import")

    # number of leading columns, defaults to 1
    dummy_columns = Int(1,
        desc="number of leading columns to ignore during import")

    traits_view = View(
        ['from_file',
            ['header_length','dummy_columns','-'],
            '|[Import]'
        ],
        title='Time data',
        buttons = OKCancelButtons
                    )

    def get_data (self,td):
        """
        main work is done here: imports the data from CSV file into
        TimeSamples object td and saves also a '*.h5' file so this import
        need not be performed every time the data is needed
        """
        if not path.isfile(self.from_file):
            # no file there
            time_data_import.getdata(self,td)
            return
        #import data
        c=self.header_length
        d=self.dummy_columns
        f=file(self.from_file)
        #read header
        for line in f:
            c-=1
            h=line.split(':')
            if h[0]=='Scan rate':
                sample_freq = int(1./float(h[1].split(' ')[1]))
            if c==0:
                break
        line=f.next()
        data=fromstring(line,dtype=float32,sep=',')[d:]
        numchannels=len(data)
        name = td.name
        if name=="":
            name = path.join(td_dir,path.splitext(path.basename(self.from_file))[0]+'.h5')
        else:
            if td.h5f != None:
                td.h5f.close()
        # TODO problems with already open h5 files from other instances
        f5h=tables.openFile(name,mode='w')
        ac=f5h.createEArray(f5h.root,'time_data',tables.atom.Float32Atom(),(0,numchannels))
        ac.setAttr('sample_freq',sample_freq)
        ac.append(data[newaxis,:])
        for line in f:
            ac.append(fromstring(line,dtype=float32,sep=',')[newaxis,d:])
        f5h.close()
        td.name = name
        td.load_data()

class td_import( time_data_import ):
    """
    import of *.td data as saved by earlier versions
    """

    # name of the comma delimited file to import
    from_file = File(filter=['*.td'],
        desc="name of the *.td file to import")

    traits_view = View(
        ['from_file',
            '|[Import]'
        ],
        title='Time data',
        buttons = OKCancelButtons
                    )

    def get_data (self,td):
        """
        main work is done here: imports the data from *.td file into
        TimeSamples object td and saves also a '*.h5' file so this import
        need not be performed only once
        """
        if not path.isfile(self.from_file):
            # no file there
            time_data_import.getdata(self,td)
            return
        f=file(self.from_file,'rb')
        h=cPickle.load(f)
        f.close()
        sample_freq = h['sample_freq']
        data = h['data']
        (numsamples,numchannels)=data.shape
        name = td.name
        if name=="":
            name = path.join(td_dir,path.splitext(path.basename(self.from_file))[0]+'.h5')
        else:
            if td.h5f != None:
                td.h5f.close()
        # TODO problems with already open h5 files from other instances
        f5h=tables.openFile(name,mode='w')
        ac=f5h.createEArray(f5h.root,'time_data',tables.atom.Float32Atom(),(0,numchannels))
        ac.setAttr('sample_freq',sample_freq)
        ac.append(data)
        f5h.close()
        td.name = name
        td.load_data()


class bk_mat_import( time_data_import ):
    """
    import of BK pulse matlab data
    """

    # name of the mat file to import
    from_file = File(filter=['*.mat'],
        desc="name of the BK pulse mat file to import")

    traits_view = View(
        ['from_file',
            '|[Import]'
        ],
        title='Time data',
        buttons = OKCancelButtons
                    )

    def get_data (self,td):
        """
        main work is done here: imports the data from pulse .mat file into
        time_data object td and saves also a '*.h5' file so this import
        need not be performed every time the data is needed
        """
        if not path.isfile(self.from_file):
            # no file there
            time_data_import.getdata(self,td)
            return
        #import data
        from scipy.io import loadmat
        m=loadmat(self.from_file)
        fh=m['File_Header']
        numchannels=int(fh.NumberOfChannels)
        l=int(fh.NumberOfSamplesPerChannel)
        sample_freq=float(fh.SampleFrequency.replace(',','.'))
        data=empty((l,numchannels),'f')
        for i in range(numchannels):
            # map SignalName "Point xx" to channel xx-1
            ii=int(m["Channel_%i_Header" % (i+1)].SignalName[-2:])-1
            data[:,ii]=m["Channel_%i_Data" % (i+1)]
        name = td.name
        if name=="":
            name = path.join(td_dir,path.splitext(path.basename(self.from_file))[0]+'.h5')
        else:
            if td.h5f != None:
                td.h5f.close()
        # TODO problems with already open h5 files from other instances
        f5h=tables.openFile(name,mode='w')
        ac=f5h.createEArray(f5h.root,'time_data',tables.atom.Float32Atom(),(0,numchannels))
        ac.setAttr('sample_freq',sample_freq)
        ac.append(data)
        f5h.close()
        td.name = name
        td.load_data()

class datx_d_file(HasPrivateTraits):
    """
    helper class for import of .datx data, represents
    datx data file
    """
    # file name
    name = File(filter=['*.datx'],
        desc="name of datx data file")

    # file object
    f = Any()

    # properties
    data_offset = Int()
    channel_count = Int()
    num_samples_per_block = Int()
    bytes_per_sample = Int()
    block_size = Property()

    # number of blocks to read in one pull
    blocks = Int()
    # the actual block data
    data = CArray()

    def _get_block_size( self ):
        return self.channel_count*self.num_samples_per_block*self.bytes_per_sample

    def get_next_blocks( self ):
        s=self.f.read(self.blocks*self.block_size)
        ls=len(s)
        if ls==0:
            return -1
        bl_no=ls/self.block_size
        self.data=fromstring(s, dtype='Int16').reshape((bl_no,self.channel_count,self.num_samples_per_block)).swapaxes(0,1).reshape((self.channel_count,bl_no*self.num_samples_per_block))

    def __init__(self,name,blocks=128):
        self.name=name
        self.f=file(self.name,'rb')
        s=self.f.read(32)
        # header
        s0=struct.unpack('IIIIIIHHf',s)        # Getting information about Properties of data-file 3=Offset to data 4=channel count 5=number of samples per block 6=bytes per sample
        self.data_offset = s0[3]
        self.channel_count = s0[4]
        self.num_samples_per_block = s0[5]
        self.bytes_per_sample = s0[6]
        self.blocks = blocks
        self.f.seek(self.data_offset)

class datx_channel(HasPrivateTraits):
    """
    helper class for import of .datx data, represents
    one channel
    """

    label = Str()
    d_file = Str()
    ch_no = Int()
    ch_K = Str()
    volts_per_count = Float()
    msl_ccf = Float()
    cal_corr_factor = Float()
    internal_gain = Float()
    external_gain = Float()
    tare_volts = Float()
    cal_coeff_2 = Float()
    cal_coeff_1 = Float()
    tare_eu = Float()
    z0 = Float()


    def __init__(self,config,channel):
        d_file,ch_no,ch_K=config.get('channels',channel).split(',') # Extraction and Splitting of Channel information
        self.d_file=d_file
        self.ch_no=int(ch_no)
        self.label=config.get(ch_K,'channel_label')
        self.ch_K=ch_K
        # V                                                     # Reading conversion factors
        self.volts_per_count=float(config.get(ch_K,'volts_per_count'))
        self.msl_ccf=float(config.get(ch_K,'msl_ccf'))
        self.cal_corr_factor=float(config.get(ch_K,'cal_corr_factor'))
        self.internal_gain=float(config.get(ch_K,'internal_gain'))
        self.external_gain=float(config.get(ch_K,'external_gain'))
        self.tare_volts=float(config.get(ch_K,'tare_volts'))
        # EU
        self.cal_coeff_2 = float(config.get(ch_K,'cal_coeff_2'))
        self.cal_coeff_1 = float(config.get(ch_K,'cal_coeff_1'))
        self.tare_eu = float(config.get(ch_K,'tare_eu'))
        self.z0 = (self.volts_per_count * self.msl_ccf * self.cal_corr_factor) / (self.internal_gain * self.external_gain)

    def scale(self,x):
        return (x * self.z0 - self.tare_volts) * self.cal_coeff_2 + self.cal_coeff_1 - self.tare_eu

class datx_import(time_data_import):
    """
    import of .datx data
    """

    # name of the mat file to import
    from_file = File(filter=['*.datx_index'],
        desc="name of the datx index file to import")

    traits_view = View(
        ['from_file',
            '|[Import]'
        ],
        title='Time data',
        buttons = OKCancelButtons
                    )

    def get_data (self,td):
        """
        main work is done here: imports the data from datx files into
        time_data object td and saves also a '*.h5' file so this import
        need not be performed every time the data is needed
        """
        if not path.isfile(self.from_file):
            # no file there
            time_data_import.getdata(self,td)
            return
        #browse datx information
        f0=open(self.from_file)
        config=ConfigParser.ConfigParser()
        config.readfp(f0)
        sample_rate=float(config.get('keywords','sample_rate'))           # reading sample-rate from index-file
        channels=[]
        d_files={}
        for channel in sort(config.options('channels')):                                         # Loop over all channels assigned in index-file
            ch=datx_channel(config,channel)
            if ch.label.find('Mic')>=0:
                channels.append(ch)
                if not d_files.has_key(ch.d_file):
                    d_files[ch.d_file]=datx_d_file(path.join(path.dirname(self.from_file),config.get(ch.d_file,'fn')),32)
        numchannels=len(channels)
        # prepare hdf5
        name = td.name
        if name=="":
            name = path.join(td_dir,path.splitext(path.basename(self.from_file))[0]+'.h5')
        else:
            if td.h5f != None:
                td.h5f.close()
        # TODO problems with already open h5 files from other instances
        f5h=tables.openFile(name,mode='w')
        ac=f5h.createEArray(f5h.root,'time_data',tables.atom.Float32Atom(),(0,numchannels))
        ac.setAttr('sample_freq',sample_rate)
        block_data=zeros((128*d_files[channels[0].d_file].num_samples_per_block,numchannels),'Float32')
        flag=0
        while(not flag):
            for i in d_files.values():
                flag=i.get_next_blocks()
            if flag:
                continue
            for i in range(numchannels):
                data=d_files[channels[i].d_file].data[channels[i].ch_no]
                block_data[:data.size,i]=channels[i].scale(data)
            ac.append(block_data[:data.size])
        f5h.close()
        f0.close()
        for i in d_files.values():
            i.f.close()
        td.name = name
        td.load_data()

class TimeSamples( HasPrivateTraits ):
    """
    Container for time data, loads time data
    and provides information about this data
    """

    # name of the .h5 file with data
    name = File(filter=['*.h5'],
        desc="name of data file")

    # sampling frequency of the data, is set automatically
    sample_freq = Float(1.0,
        desc="sampling frequency")

    # number of channels, is set automatically
    numchannels = Long(1,
        desc="number of input channels")

    # number of time data samples, is set automatically
    numsamples = Long(1,
        desc="number of samples")

    # the time data as (numsamples,numchannels) array of floats
    data = Property( depends_on = [],
        cached     = True,
        desc="the actual time data array")

    # internal identifier
    digest = Property( depends_on = ['name',],
        cached = True)

    # thread (internal use)
    load_thread = Instance( Thread )

    # hdf5 file object
    h5f = Instance(tables.File)

    traits_view = View(
        ['name{File name}',
            ['sample_freq~{Sampling frequency}','numchannels~{Number of channels}','numsamples~{Number of samples}','|[Properties]'],
            '|'
        ],
        title='Time data',
        buttons = OKCancelButtons
                    )

    def _name_changed ( self ):
        self.load_data()
        #~ if self.load_thread and self.load_thread.isAlive():
            #~ self.load_thread.join()
        #~ self.load_thread = Thread( target=TimeSamples.load_data,args=(self,) )
        #~ print "start load thread"
        #~ self.load_thread.start()
        #~ print "load thread started"

    def _get_data ( self ):
        #~ print "get data"
        #~ if self.load_thread and self.load_thread.isAlive():
            #~ self.load_thread.join()
        return self._data

    def _get_digest( self ):
        if self._digest is None:
            s=[path.basename(self.name),
                ]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def load_data ( self ):
        """ loads the data from .h5 file,
        will be started in an extra thread (this must be changed)
        """
        if not path.isfile(self.name):
            # no file there
            self._data=array([],'d')
            self.numsamples = 0
            self.numchannels = 0
            self.sample_freq = 0
            return
        if self.h5f!=None:
            try:
                self.h5f.close()
            except:
                pass
        self.h5f=tables.openFile(self.name)
        self._data=self.h5f.root.time_data
        self.sample_freq = self._data.getAttr('sample_freq')
        (self.numsamples,self.numchannels)=self._data.shape

    #~ def samples ( self, tslice=s_[:], cslice=s_[:]):
        #~ pass

class Calib( HasPrivateTraits ):
    """
    container for calibration data that is loaded from
    an .xml-file
    """

    # name of the .xml file
    from_file = File(filter=['*.xml'],
        desc="name of the xml file to import")

    # number of microphones in the calibration data (auto-set)
    num_mics = Int( 0,
        desc="number of microphones in the geometry")

    # array of calibration factors
    data = Property( depends_on = [ 'from_file' ],
        cached     = True,
        desc="calibration data")

    # internal identifier
    digest = Property( depends_on = ['data',],
        cached = True)

    traits_view = View(
        ['from_file{File name}',
            ['num_mics~{Number of microphones}',
                '|[Properties]'
            ]
        ],
        title='Calibration data',
        buttons = OKCancelButtons
                    )

    def _get_data ( self ):
        if self._data is None:
            self.import_data()
        return self._data

    def _get_digest( self ):
        if self._digest is None:
            s=[str(self.data),
                ]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def import_data( self ):
        "loads the calibration data from .xml file"
        if not path.isfile(self.from_file):
            # no file there
            self._data=array([1.0,],'d')
            self.num_mics=1
            return
        import xml.dom.minidom
        doc=xml.dom.minidom.parse(self.from_file)
        names=[]
        data=[]
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            data.append(float(el.getAttribute('factor')))
        self._data=array(data,'d')
        self.num_mics=shape(self._data)[0]

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

    # FFT block size, one of: 128,256,512,1024,2048 or 4096
    # defaults to 1024
    block_size = Trait(1024,128,256,512,1024,2048,4096,
        desc="number of samples per FFT block")

    # index of lowest frequency line
    # defaults to 0
    ind_low = Range(0,
        desc="index of lowest frequency line")

    # index of highest frequency line
    # defaults to -1 (last possible line for default block_size)
    ind_high = Int(-1,
        desc="index of highest frequency line")

    # window function for FFT, one of:
    # 'Rectangular' (default),'Hanning','Hamming','Bartlett','Blackman'
    window = Trait('Rectangular',{'Rectangular':ones,'Hanning':hanning,'Hamming':hamming,'Bartlett':bartlett,'Blackman':blackman},
        desc="type of window for FFT")

    # overlap factor for averaging: 'None'(default),'50%','75%','87.5%'
    overlap = Trait('None',{'None':1,'50%':2,'75%':4,'87.5%':8},
        desc="overlap of FFT blocks")

    # number of FFT blocks to average (auto-set from block_size and overlap)
    num_blocks = Property( depends_on = [ 'time_data.numsamples', 'block_size','overlap'],
        cached     = True,
        desc="overall number of FFT blocks")

    # frequency range
    freq_range = Property( depends_on = [ 'time_data.digest','block_size','ind_low','ind_high'],
        cached = True,
        desc = "frequency range" )

    # flag for internal use
    csm_flag = Property( depends_on = [ 'digest'],
        cached     = True,
        desc="flag=1 if csm is invalid")

    # the cross spectral matrix as
    # (number of frequencies,numchannels,numchannels) array of complex
    csm = Property( depends_on = [],
        cached= True,
        desc="cross spectral matrix")

    # internal identifier
    digest = Property( depends_on = ['time_data.digest','calib.digest','block_size','window','overlap','ind_low','ind_high'],
        cached = True)

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

    def _get_num_blocks ( self ):
        if self._num_blocks is None and not self.time_data is None:
            self._num_blocks=self.overlap_*self.time_data.numsamples/self.block_size-self.overlap_+1
        return self._num_blocks

    def _get_freq_range ( self ):
        if self._freq_range is None and not self.time_data is None:
            try:
                self._freq_range=self.fftfreq()[[0,-1]]
            except IndexError:
                self._freq_range=array([0.,0])
        return self._freq_range

    def _get_csm_flag ( self ):
        if self._csm_flag is None:
            self._csm_flag=1
            self._csm=None
        return self._csm_flag

    def _get_csm ( self ):
        if self._csm_flag==1:
            self.calc_csm()
            self._csm_flag=2
        return self._csm

    def _get_digest( self ):
        if self._digest is None:
            if self.calib:
                cdigest=self.calib.digest
            else:
                cdigest=''
            try:
                s=[self.time_data.digest,
                    cdigest,
                    str(self.block_size),
                    str(self.window),
                    str(self.overlap),
                    str(self.ind_low),
                    str(self.ind_high)
                    ]
            except AttributeError:
                s=['',]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def calc_csm ( self ):
        """main work is done here:
        cross spectral matrix is either loaded from cache file or
        calculated and then additionally stored into cache
        !! is called automatically
        """
        if self.num_blocks==0:
            self._csm = zeros(0)
            return
        cache_name = path.join(cache_dir,'f_'+self.digest+'.cache')
        if path.isfile(cache_name):
            self._csm=load(cache_name)
            return
        t=self.time_data
        td=t.data
        wind=self.window_(self.block_size)
        weight=dot(wind,wind)
        wind=wind[newaxis,:].swapaxes(0,1)
        numfreq=len(self.fftfreq())
        csm=zeros((numfreq,t.numchannels,t.numchannels),'D')
        print "num blocks",self.num_blocks
        calib=1.0
        if self.calib:
            if self.calib.num_mics==t.numchannels:
                calib=self.calib.data[newaxis,:]
            else:
                print "warning: calibration data not compatible:",self.calib.num_mics,t.numchannels
        for block in range(self.num_blocks):
            pos=block*self.block_size/self.overlap_
            ft=fft.rfft(self.time_data.data[pos:(pos+self.block_size)]*wind*calib,None,0)[self.ind_low:self.ind_high]
            faverage(csm,ft)
        csm=csm*(2.0/self.block_size/weight/self.num_blocks) #2.0=sqrt(2)^2 wegen der halbseitigen FFT
        self._csm=csm#[:numfreq-1]
        self._csm.dump(cache_name)

    def fftfreq ( self ):
        """
        returns an array of the frequencies for
        the spectra in the cross spectral matrix
        """
        if self.time_data.sample_freq>0:
            return fft.fftfreq(self.block_size,1./self.time_data.sample_freq)[:self.block_size/2+1][self.ind_low:self.ind_high]
        else:
            return array([0.],'d')

class EigSpectra( PowerSpectra ):
    """
    efficient calculation of full cross spectral matrix
    container for data and properties of this matrix
    and its eigenvalues and eigenvectors
    """

    # eigenvalues of the cross spectral matrix
    eva = Property( depends_on = [],
        cached= True,
        desc="eigenvalues of cross spectral matrix")

    # eigenvectors of the cross spectral matrix
    eve = Property( depends_on = [],
        cached= True,
        desc="eigenvectors of cross spectral matrix")

    def _get_eva ( self ):
        if self._csm_flag==1:
            self.calc_csm()
            self._csm_flag=2
        return self._eva

    def _get_eve ( self ):
        if self._csm_flag==1:
            self.calc_csm()
            self._csm_flag=2
        return self._eve

    def calc_csm ( self ):
        """main work is done here:
        cross spectral matrix is either loaded from cache file or
        calculated and then additionally stored into cache
        the same is done for eigenvalues / eigenvectors
        !! is called automatically
        """
        PowerSpectra.calc_csm( self )
        if shape(self._csm)[0]==0:
            self._eva = self._csm
            self._eve = self._csm
            return
        cache_name = path.join(cache_dir,'e_'+self.digest+'.cache')
        if path.isfile(cache_name):
            f=file(cache_name,'rb')
            (self._eva,self._eve)=cPickle.load(f)
            f.close()
            return
        csm=self._csm
        self._eva=empty(shape(csm)[0:2],'d')
        self._eve=empty_like(csm)
        for i in range(shape(csm)[0]):
            (self._eva[i],self._eve[i])=linalg.eigh(csm[i])
        f=open(cache_name,'wb')
        cPickle.dump((self._eva,self._eve),f,-1)
        f.close()

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

    # number of grid points alog x-axis (auto-set)
    nxsteps = Property( depends_on = [ 'x_min', 'x_max','increment'],
        desc="number of grid points alog x-axis")

    # number of grid points alog y-axis (auto-set)
    nysteps = Property( depends_on = [ 'y_min', 'y_max','increment'],
        desc="number of grid points alog y-axis")

    # overall number of grid points (auto-set)
    size = Property( depends_on = [ 'nxsteps', 'nysteps'],
        desc="overall number of grid points")

    # internal identifier
    digest = Property( depends_on = ['x_min', 'x_max', 'y_min', 'y_max', 'z', 'increment'],
        cached = True)

    traits_view = View(
            [
                ['x_min','y_min','|'],
                ['x_max','y_max','z','increment','size~{grid size}','|'],
                '-[Map extension]'
            ]
        )

    def _get_size ( self ):
        return self.nxsteps*self.nysteps

    def _get_nxsteps ( self ):
        i=abs(self.increment)
        if i!=0:
            return int(round((abs(self.x_max-self.x_min)+i)/i))
        return 1

    def _get_nysteps ( self ):
        i=abs(self.increment)
        if i!=0:
            return int(round((abs(self.y_max-self.y_min)+i)/i))
        return 1

    def _get_digest( self ):
        if self._digest is None:
            s=[str(self.x_min),
                str(self.x_max),
                str(self.y_min),
                str(self.y_max),
                str(self.z),
                str(self.increment)
                ]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

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

    # number of mics
    num_mics = Int( 0,
        desc="number of microphones in the geometry")

    # positions as (3,num_mics) array
    mpos = Property( depends_on = [ 'from_file' ],
        cached     = True,
        desc="x,y,z position of microphones")

    # internal identifier
    digest = Property( depends_on = ['mpos',],
        cached = True)

    traits_view = View(
        ['from_file',
        'num_mics~',
        '|[Microphone geometry]'
        ],
#        title='Microphone geometry',
        buttons = OKCancelButtons
                    )

    def _get_mpos ( self ):
        if self._mpos is None:
            self.import_mpos()
        return self._mpos

    def _get_digest( self ):
        if self._digest is None:
            s=[str(self.mpos),
                ]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def import_mpos( self ):
        """import the microphone positions from .xml file,
        called automatically
        """
        if not path.isfile(self.from_file):
            # no file there
            self._mpos=array([],'d')
            self.num_mics=0
            return
        import xml.dom.minidom
        doc=xml.dom.minidom.parse(self.from_file)
        names=[]
        xyz=[]
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            xyz.append(map(lambda a : float(el.getAttribute(a)),'xyz'))
        self._mpos=array(xyz,'d').swapaxes(0,1)
        self.num_mics=shape(self._mpos)[1]


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

    # internal use
    result_flag = Property( depends_on = [ 'digest' ],
        cached     = True,
        desc="flag=1 if result is invalid")

    # the result, sound pressure squared in all grid locations
    # as (number of frequencies, nxsteps,nysteps) array of float
    result = Property( depends_on = [],
        cached     = True,
        desc="beamforming result")

    # internal identifier
    digest = Property( depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', 'r_diag'],
        cached = True)

    # thread
    calc_thread = Instance( Thread )

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

    def _get_result_flag ( self ):
        #print "get result flag",self.r_diag
        if self._result_flag is None:
            self._result_flag=1
            self._result=None
        return self._result_flag

    def _get_result ( self ):
        #print "get_result",self.r_diag
        if self._result_flag==1:
            #print "calc_result",self.r_diag
            self.calc_result()
            self._result_flag=2
        return self._result

    def _get_digest( self ):
        if self._digest is None:
            try:
                s=[self.mpos.digest,
                    self.grid.digest,
                    self.freq_data.digest,
                    str(self.c),
                    str(self.r_diag)
                    ]
            except AttributeError:
                s=['',]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def calc_result ( self ):
        """main work is done here:
        beamform result is either loaded from cache or
        calculated and additionally saved to cache
        this runs automatically and may take some time
        the result is discarded if the digest changes during the calculation
        """
        # check validity of input data
        if self.freq_data is None or self.grid is None or self.mpos is None:
            return
        numchannels=self.freq_data.time_data.numchannels
        if  numchannels != self.mpos.num_mics:
            print "channel counts in time data (%i) and mic geometry (%i) do not fit" % (numchannels,self.mpos.num_mics)
            return
        # store digest value for later comparison
        digest=self.digest
        # check for HD-cached values
        cache_name = path.join(cache_dir,'b_'+self.digest+'.cache')
        if path.isfile(cache_name):
            self._result=load(cache_name)
            return
        # prepare calculation
        f=self.freq_data.fftfreq()
        kj=2j*pi*f/self.c
        e=zeros((numchannels),'D')
        bpos=self.grid.pos()
        h=zeros((len(f),shape(bpos)[1]),'d')
        # ! this may take a while
        if self.r_diag:
            beamdiag(self.freq_data.csm,e,h,bpos,self.mpos.mpos,kj)
            adiv=(numchannels*numchannels-numchannels)
            h=multiply(h,(sign(h)+1-1e-35)/2)
        else:
            beamfull(self.freq_data.csm,e,h,bpos,self.mpos.mpos,kj)
            adiv=(numchannels*numchannels)
        # all data still the same
        if digest==self.digest:
            self._result=reshape(h,(len(f),self.grid.nxsteps,self.grid.nysteps))/adiv
            self._result.dump(cache_name)
##            self._result_flag=2

    def synthetic( self, freq, num=0):
        """
        returns synthesized frequency band values of beamforming result
        num = 0: single frequency line
        num = 1: octave band
        num = 3: third octave band
        etc.
        """
        #print "synth",num
        f=self.freq_data.fftfreq()
        if len(f)==0:
            return array([[1,],],'d')
        try:
            if num==0:
                # single frequency line
                return self.result[searchsorted(f,freq)]
            else:
                f1=searchsorted(f,freq*2.**(-0.5/num))
                f2=searchsorted(f,freq*2.**(0.5/num))
                if f1==f2:
                    return self.result[f1]
                else:
                    return sum(self.result[f1:f2],0)
        except:
            return ones((1,1),'d')

    def integrate(self,sector):
        """
        integrates result map over the given sector
        where sector is a tuple with arguments for grid.indices
        e.g. (xmin,ymin,xmin,ymax)
        returns spectrum
        """
        ind = self.grid.indices(*sector)
        h = self.result[ (s_[:],) + ind ]
        return h.reshape(h.shape[0],prod(h.shape[1:])).sum(axis=1)

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

    def calc_result ( self ):
        """main work is done here:
        beamform result is either loaded from cache or
        calculated and additionally saved to cache
        this runs automatically and may take some time
        the result is discarded if the digest changes during the calculation
        """
        # check validity of input data
        if self.freq_data is None or self.grid is None or self.mpos is None:
            return
        numchannels=self.freq_data.time_data.numchannels
        if  numchannels != self.mpos.num_mics:
            print "channel counts in time data (%i) and mic geometry (%i) do not fit" % (numchannels,self.mpos.num_mics)
            return
        # store digest value for later comparison
        digest=self.digest
        # check for HD-cached values
        cache_name = path.join(cache_dir,'w_'+self.digest+'.cache')
        if path.isfile(cache_name):
            self._result=load(cache_name)
            return
        # prepare calculation
        f=self.freq_data.fftfreq()
        kj=2j*pi*f/self.c
        e=zeros((numchannels),'D')
        h=zeros((len(f),self.grid.size),'d')
        csm=zeros_like(self.freq_data.csm)
        for i in range(shape(self.freq_data.csm)[0]):
            csm[i]=linalg.inv(self.freq_data.csm[i])
        #~ if self.r_diag:
            #~ beamdiag(self.freq_data.csm,e,h,self.grid.pos(),self.mpos.mpos,kj)
            #~ self._result=reshape(h.real,(len(f),self.grid.nxsteps,self.grid.nysteps))/(numchannels*numchannels-numchannels)
            #~ self._result=multiply(self._result,(sign(self._result)+1-1e-35)/2)
        #~ else:
        # all data still the same
        beamfull(csm,e,h,self.grid.pos(),self.mpos.mpos,kj)
        if digest==self.digest:
            self._result=reshape(1./h,(len(f),self.grid.nxsteps,self.grid.nysteps))/(numchannels*numchannels)
            self._result.dump(cache_name)

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

    # internal use
    result_flag = Property( depends_on = [ 'digest' ],
        cached     = True,
        desc="flag=1 if result is invalid")

    # internal identifier
    digest = Property( depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', 'n'],
        cached = True)

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

    def on_n_changed ( self, new ):
        if new>=self.mpos.num_mics:
            self.n=self.mpos.num_mics-1
        elif new<-1:
            self.n=-1

    def _get_result_flag ( self ):
        if self._result_flag is None:
            self._result_flag=1
            self._result=None
        return self._csm_flag

    def _get_digest( self ):
        if self._digest is None:
            try:
                s=[self.mpos.digest,
                    self.grid.digest,
                    self.freq_data.digest,
                    str(self.c),
                    str(self.n),
                    str(self.r_diag)
                    ]
            except AttributeError:
                s=['',]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def calc_result ( self ):
        """main work is done here:
        beamform result is either loaded from cache or
        calculated and additionally saved to cache
        this runs automatically and may take some time
        the result is discarded if the digest changes during the calculation
        """
        # check validity of input data
        if self.freq_data is None or self.grid is None or self.mpos is None:
            return
        numchannels=self.freq_data.time_data.numchannels
        if  numchannels != self.mpos.num_mics:
            print "channel counts in time data (%i) and mic geometry (%i) do not fit" % (numchannels,self.mpos.num_mics)
            return
        # store digest value for later comparison
        digest=self.digest
        # check for HD-cached values
        cache_name = path.join(cache_dir,'v_'+self.digest+'.cache')
        if path.isfile(cache_name):
            self._result = load(cache_name)
            return
        # prepare calculation
        f=self.freq_data.fftfreq()
        kj=2j*pi*f/self.c
        if self.n<0:
            n = int(numchannels-1)
        else:
            n = self.n
        e = zeros((numchannels),'D')
        h = zeros((len(f),self.grid.size),'d')
        eva = self.freq_data.eva
        eve = self.freq_data.eve
        if self.r_diag:
            beamortho_sum_diag(e,h,self.grid.pos(),self.mpos.mpos,kj,eva,eve,n,n+1)
            adiv=(numchannels*numchannels-numchannels)
            h=multiply(h,(sign(h)+1-1e-35)/2)
        else:
            beamortho_sum(e,h,self.grid.pos(),self.mpos.mpos,kj,eva,eve,n,n+1)
            adiv=(numchannels*numchannels)
        # all data still the same
        if digest==self.digest:
            self._result=reshape(h,(len(f),self.grid.nxsteps,self.grid.nysteps))/adiv
            self._result.dump(cache_name)

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

    def on_n_changed ( self, new ):
        if new>=self.mpos.num_mics:
            self.n=self.mpos.num_mics-2
        elif new<1:
            self.n=1

    def calc_result ( self ):
        """main work is done here:
        beamform result is either loaded from cache or
        calculated and additionally saved to cache
        this runs automatically and may take some time
        the result is discarded if the digest changes during the calculation
        """
        # check validity of input data
        if self.freq_data is None or self.grid is None or self.mpos is None:
            return
        numchannels=self.freq_data.time_data.numchannels
        if  numchannels != self.mpos.num_mics:
            print "channel counts in time data (%i) and mic geometry (%i) do not fit" % (numchannels,self.mpos.num_mics)
            return
        # store digest value for later comparison
        digest=self.digest
        # check for HD-cached values
        cache_name = path.join(cache_dir,'m_'+self.digest+'.cache')
        if path.isfile(cache_name):
            self._result=load(cache_name)
            return
        # prepare calculation
        n = self.mpos.num_mics-self.n
        f=self.freq_data.fftfreq()
        kj=2j*pi*f/self.c
        e = zeros((numchannels),'D')
        h = zeros((len(f),self.grid.size),'d')
        eva = self.freq_data.eva
        eve = self.freq_data.eve
        beamortho_sum(e,h,self.grid.pos(),self.mpos.mpos,kj,eva,eve,0,n)
        # all data still the same
        if digest==self.digest:
            h=1./h
            h=(h.T/h.max(1)).T
            self._result=reshape(h,(len(f),self.grid.nxsteps,self.grid.nysteps))*4e-10
            self._result.dump(cache_name)

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

    # frequency list
    freqs = CArray(desc="frequencies")

    # internal identifier
    digest = Property( depends_on = ['mpos.digest', 'grid.digest', 'freqs', 'c'],
        cached = True)

    def _get_digest( self ):
        if self._digest is None:
            try:
                s=[self.mpos.digest,
                   self.grid.digest,
                   str(self.freqs),
                    str(self.c)
                    ]
            except AttributeError:
                s=['',]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def psf ( self ):
        """
        point spread function as calculated or loaded from cache
        """
        # check validity of input data
        if self.freqs is None or self.grid is None or self.mpos is None:
            return
        numchannels = self.mpos.num_mics
        # store digest value for later comparison
        digest=self.digest
        # check for HD-cached values
        cache_name = path.join(cache_dir,'p_'+self.digest+'.cache')
        if path.isfile(cache_name):
            return load(cache_name)
        # prepare calculation
        f = self.freqs
        numfreq = len(f)
        kj = 2j*pi*f/self.c
        gs = self.grid.size
        bpos = self.grid.pos()
        hh = ones((numfreq,gs,gs),'d')
        e = zeros((numchannels),'D')
        e1 = e.copy()
        beam_psf(e,e1,hh,bpos,self.mpos.mpos,kj)
        for i in range(numfreq):
            h = hh[i]
            hh[i] = h/diag(h)[:,newaxis]
        hh.dump(cache_name)
        return hh

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

    # iherited
    # result_flag, result

    # number of iterations
    n_iter = Int(100,
        desc="number of iterations")

    # internal identifier
    digest = Property( depends_on = ['beamformer.digest','n_iter'],
        cached = True)

    # thread
    calc_thread = Instance( Thread )

    traits_view = View(
        [
            [Item('beamformer{}',style='custom')],
            [Item('n_iter{Number of iterations}')],
            '|'
        ],
        title='Beamformer denconvolution options',
        buttons = OKCancelButtons
        )

    def _get_result_flag ( self ):
        #print "get result flag",self.r_diag
        if self._result_flag is None:
            self._result_flag=1
            self._result=None
        return self._result_flag

    def _get_result ( self ):
        #print "get_result",self.r_diag
        if self._result_flag==1:
            #print "calc_result",self.r_diag
            self.calc_result()
            self._result_flag=2
        return self._result

    def _get_digest( self ):
        if self._digest is None:
            try:
                s=[self.beamformer.digest,
                    str(self.n_iter)
                    ]
            except AttributeError:
                s=['',]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def calc_result ( self ):
        """main work is done here:
        beamform result is either loaded from cache or
        calculated and additionally saved to cache
        this runs automatically and may take some time
        the result is discarded if the digest changes during the calculation
        """
        # check validity of input data
        if self.freq_data is None or self.grid is None or self.mpos is None or self.beamformer.result is None:
            return
        numchannels=self.freq_data.time_data.numchannels
        if  numchannels != self.mpos.num_mics:
            print "channel counts in time data (%i) and mic geometry (%i) do not fit" % (numchannels,self.mpos.num_mics)
            return
        # store digest value for later comparison
        digest=self.digest
        # check for HD-cached values
        cache_name = path.join(cache_dir,'x_'+self.digest+'.cache')
        if path.isfile(cache_name):
            self._result=load(cache_name)
            return
        # prepare calculation
        f = self.freq_data.fftfreq()
        numfreq = len(f)
        bresult = self.beamformer.result
        result = empty_like(bresult)
        for i in range(numfreq):
            y = bresult[i].flatten()
            x = y.copy()
            hh = PointSpreadFunction(mpos=self.mpos, grid=self.grid, c=self.c, freqs=f[i,newaxis]).psf()
            gseidel(hh[0],y,x,self.n_iter,1.0)
            result[i] = x.reshape((self.grid.nxsteps,self.grid.nysteps))
        # all data still the same
        if digest==self.digest:
            self._result=result #reshape(h,(len(f),self.grid.nxsteps,self.grid.nysteps))/adiv
            self._result.dump(cache_name)

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

    # iherited
    # result_flag, result

    # list of components to consider
    eva_list = CArray(
        desc="components")

    # internal identifier
    digest = Property( depends_on = ['beamformer.digest','eva_list'],
        cached = True)

    # thread
    calc_thread = Instance( Thread )

    traits_view = View(
        [
            [Item('beamformer{}',style='custom')],
            [Item('eva_list{Components}')],
            '|'
        ],
        title='Options',
        buttons = OKCancelButtons
        )

    def _get_result_flag ( self ):
        #print "get result flag",self.r_diag
        if self._result_flag is None:
            self._result_flag=1
            self._result=None
        return self._result_flag

    def _get_result ( self ):
        #print "get_result",self.r_diag
        if self._result_flag==1:
            #print "calc_result",self.r_diag
            self.calc_result()
            self._result_flag=2
        return self._result

    def _get_digest( self ):
        if self._digest is None:
            try:
                s=[self.mpos.digest,
                    self.grid.digest,
                    self.freq_data.digest,
                    str(self.c),
                    str(self.r_diag),
                    str(self.eva_list)
                    ]
            except AttributeError:
                s=['',]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def calc_result ( self ):
        """main work is done here:
        result is either loaded from cache or
        calculated and additionally saved to cache
        this runs automatically and may take some time
        the result is discarded if the digest changes during the calculation
        """
        # check validity of input data
        if self.freq_data is None or self.grid is None or self.mpos is None or self.beamformer.result is None:
            return
        numchannels=self.freq_data.time_data.numchannels
        if  numchannels != self.mpos.num_mics:
            print "channel counts in time data (%i) and mic geometry (%i) do not fit" % (numchannels,self.mpos.num_mics)
            return
        # store digest value for later comparison
        digest=self.digest
        # check for HD-cached values
        cache_name = path.join(cache_dir,'o_'+self.digest+'.cache')
        if path.isfile(cache_name):
            self._result=load(cache_name)
            return
        # prepare calculation
        f = self.freq_data.fftfreq()
        numfreq = len(f)
        e = self.beamformer
        result = zeros_like(e.result)
        for i in self.eva_list:
            self.beamformer.n = i
            for j in range(numfreq):
                result[j].flat[e.result[j].argmax()]+=e.freq_data.eva[j,i]/numchannels
        if digest==self.digest:
            self._result=result
            self._result.dump(cache_name)

class BeamformerCleansc( BeamformerBase ):
    """
    beamforming using CLEAN-SC (Sijtsma)
    """

    # EigSpectra object that provides the cross spectral matrix and eigenvalues
    freq_data = Trait(PowerSpectra,
        desc="freq data object")

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

    # internal use
    result_flag = Property( depends_on = [ 'digest' ],
        cached     = True,
        desc="flag=1 if result is invalid")

    # internal identifier
    digest = Property( depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', 'n'],
        cached = True)

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

    def on_n_changed ( self, new ):
        if new>=self.mpos.num_mics:
            self.n=self.mpos.num_mics-1
        elif new<-1:
            self.n=-1

    def _get_result_flag ( self ):
        if self._result_flag is None:
            self._result_flag=1
            self._result=None
        return self._csm_flag

    def _get_digest( self ):
        if self._digest is None:
            try:
                s=[self.mpos.digest,
                    self.grid.digest,
                    self.freq_data.digest,
                    str(self.c),
                    str(self.n),
                    str(self.damp),
                    str(self.stopn),
                    str(self.r_diag)
                    ]
            except AttributeError:
                s=['',]
            self._digest = md5.new(join(s)).hexdigest()
        return self._digest

    def calc_result ( self ):
        """main work is done here:
        beamform result is either loaded from cache or
        calculated and additionally saved to cache
        this runs automatically and may take some time
        the result is discarded if the digest changes during the calculation
        """
        # check validity of input data
        if self.freq_data is None or self.grid is None or self.mpos is None:
            return
        numchannels=self.freq_data.time_data.numchannels
        if  numchannels != self.mpos.num_mics:
            print "channel counts in time data (%i) and mic geometry (%i) do not fit" % (numchannels,self.mpos.num_mics)
            return
        # store digest value for later comparison
        digest=self.digest
        # check for HD-cached values
        cache_name = path.join(cache_dir,'s_'+self.digest+'.cache')
        if path.isfile(cache_name):
            self._result = load(cache_name)
            return
        # prepare calculation
        f=self.freq_data.fftfreq()
        kjall=2j*pi*f/self.c
        bpos=self.grid.pos()
        mpos=self.mpos.mpos
        e=zeros((numchannels),'D')
        result = zeros((f.size,self.grid.size),'d')
        if self.r_diag:
           adiv=1.0*(numchannels*numchannels-numchannels)
           fullbeamfunc=beamdiag
           orthbeamfunc=beamortho_sum_diag1
        else:
           adiv=1.0*(numchannels*numchannels)
           fullbeamfunc=beamfull
           orthbeamfunc=beamortho_sum
        if not self.n:
            J=numchannels*2
        else:
            J=self.n
        powers=zeros(J,'d')
        #ind=zeros(J)
        h=zeros((1,shape(bpos)[1]),'d')
        h1=h.copy()
        # loop over frequencies
        for nf in range(f.size):
            kj=kjall[nf,newaxis]
            csm=self.freq_data.csm[nf,newaxis,:,:].copy()
            fullbeamfunc(csm,e,h,bpos,mpos,kj)
            h=h/adiv
            # CLEANSC Iteration
            for j in range(J):
                xi_max=h.argmax()
                powers[j]=hmax=h[0,xi_max]
                result[nf,xi_max]+=hmax
                if  j>2 and hmax>powers[j-3]:
                    #print j
                    break
                rm=bpos[:,xi_max,newaxis]-mpos
                rm=sum(rm*rm,0)
                rm=sqrt(rm)
                r0=sum(bpos[:,xi_max]*bpos[:,xi_max],0)
                r0=sqrt(r0)
                rs=(r0*(1/(rm*rm)).sum(0))
                wmax=numchannels/sqrt(adiv)*exp(-kj[0]*(r0-rm))/(rm*rs)
                hh=wmax.copy()
                D1=dot(csm[0]-diag(diag(csm[0])),wmax)/hmax
                ww=wmax.conj()*wmax
                for i in range(20):
                    H=hh.conj()*hh
                    hh=(D1+H*wmax)/sqrt(1+dot(ww,H))
                hh=hh[:,newaxis]
                csm1=hmax*(hh*hh.conj().T)[newaxis,:,:]
                h1=zeros((1,shape(bpos)[1]),'d')
                orthbeamfunc(e,h1,bpos,mpos,kj,array((hmax,))[newaxis,:],hh[newaxis,:],0,1)
                h-=self.damp*h1/adiv
                csm-=self.damp*csm1
        # all data still the same
        if digest==self.digest:
            self._result=reshape(result,(len(f),self.grid.nxsteps,self.grid.nysteps))
            self._result.dump(cache_name)

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


