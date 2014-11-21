# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------
"""
Contains classes for importing time data in several file formats.

.. autosummary::
    :toctree: generated/

    time_data_import
    td_import
    bk_mat_import
    datx_d_file
    datx_channel
    datx_import
"""

from h5cache import td_dir
from numpy import fromstring, float32, newaxis, empty, sort, zeros
from traits.api import HasPrivateTraits, Float, Int, \
File, CArray, Property, Any, Str
from traitsui.api import View
from traitsui.menu import OKCancelButtons
from os import path
import cPickle
import tables
import ConfigParser
import struct

class time_data_import( HasPrivateTraits ):
    """
    Base class for import of time data.
    """

    def get_data (self, td):
        """
        Imports the data into an arbitrary time_data object td.
        This is a dummy function and should not be used directly.
        """
        td.data = None
        td.numsamples = 0
        td.numchannels = 0
        td.sample_freq = 0

class csv_import( time_data_import ):
    """
    Class that supports the import of CSV data as saved by NI VI Logger.
    """

    #: Name of the comma delimited file to import.
    from_file = File(filter = ['*.txt'], 
        desc = "name of the comma delimited file to import")

    #: Header length, defaults to 6.
    header_length =  Int(6, 
        desc = "length of the header to ignore during import")

    #: Number of leading columns (will be ignored during import), defaults to 1.
    dummy_columns = Int(1, 
        desc = "number of leading columns to ignore during import")

    traits_view = View(
        ['from_file', 
            ['header_length', 'dummy_columns', '-'], 
            '|[Import]'
        ], 
        title = 'Time data', 
        buttons = OKCancelButtons
                    )

    def get_data (self, td):
        """
        Imports the data from CSV file into a
        :class:`~acoular.sources.TimeSamples` object td.
        Also, a '*.h5' file will be written, so this import
        need not be performed every time the data is needed
        """
        if not path.isfile(self.from_file):
            # no file there
            time_data_import.getdata(self, td)
            return
        #import data
        c = self.header_length
        d = self.dummy_columns
        f = file(self.from_file)
        #read header
        for line in f:
            c -= 1
            h = line.split(':')
            if h[0] == 'Scan rate':
                sample_freq = int(1./float(h[1].split(' ')[1]))
            if c == 0:
                break
        line = f.next()
        data = fromstring(line, dtype = float32, sep = ', ')[d:]
        numchannels = len(data)
        name = td.name
        if name == "":
            name = path.join(td_dir, \
                path.splitext(path.basename(self.from_file))[0]+'.h5')
        else:
            if td.h5f !=  None:
                td.h5f.close()
        # TODO problems with already open h5 files from other instances
        f5h = tables.open_file(name, mode = 'w')
        ac = f5h.create_earray(f5h.root, 'time_data', \
            tables.atom.Float32Atom(), (0, numchannels))
        ac.set_attr('sample_freq', sample_freq)
        ac.append(data[newaxis, :])
        for line in f:
            ac.append(fromstring(line, dtype=float32, sep=', ')[newaxis, d:])
        f5h.close()
        td.name = name
        td.load_data()

class td_import( time_data_import ):
    """
    import of *.td data as saved by earlier versions
    """

    # name of the comma delimited file to import
    from_file = File(filter = ['*.td'], 
        desc = "name of the *.td file to import")

    traits_view = View(
        ['from_file', 
            '|[Import]'
        ], 
        title  = 'Time data', 
        buttons = OKCancelButtons
                    )

    def get_data (self, td):
        """
        main work is done here: imports the data from *.td file into
        TimeSamples object td and saves also a '*.h5' file so this import
        need not be performed only once
        """
        if not path.isfile(self.from_file):
            # no file there
            time_data_import.getdata(self, td)
            return
        f = file(self.from_file, 'rb')
        h = cPickle.load(f)
        f.close()
        sample_freq = h['sample_freq']
        data = h['data']
        numchannels = data.shape[1]
        name = td.name
        if name == "":
            name = path.join(td_dir, \
                        path.splitext(path.basename(self.from_file))[0]+'.h5')
        else:
            if td.h5f !=  None:
                td.h5f.close()
        # TODO problems with already open h5 files from other instances
        f5h = tables.open_file(name, mode = 'w')
        ac = f5h.create_earray(f5h.root, 'time_data', \
            tables.atom.Float32Atom(), (0, numchannels))
        ac.set_attr('sample_freq', sample_freq)
        ac.append(data)
        f5h.close()
        td.name = name
        td.load_data()


class bk_mat_import( time_data_import ):
    """
    import of BK pulse matlab data
    """

    # name of the mat file to import
    from_file = File(filter = ['*.mat'], 
        desc = "name of the BK pulse mat file to import")

    traits_view = View(
        ['from_file', 
            '|[Import]'
        ], 
        title = 'Time data', 
        buttons = OKCancelButtons
                    )

    def get_data (self, td):
        """
        main work is done here: imports the data from pulse .mat file into
        time_data object td and saves also a '*.h5' file so this import
        need not be performed every time the data is needed
        """
        if not path.isfile(self.from_file):
            # no file there
            time_data_import.getdata(self, td)
            return
        #import data
        from scipy.io import loadmat
        m = loadmat(self.from_file)
        fh = m['File_Header']
        numchannels = int(fh.NumberOfChannels)
        l = int(fh.NumberOfSamplesPerChannel)
        sample_freq = float(fh.SampleFrequency.replace(', ', '.'))
        data = empty((l, numchannels), 'f')
        for i in range(numchannels):
            # map SignalName "Point xx" to channel xx-1
            ii = int(m["Channel_%i_Header" % (i+1)].SignalName[-2:])-1
            data[:, ii] = m["Channel_%i_Data" % (i+1)]
        name = td.name
        if name == "":
            name = path.join(td_dir, \
                path.splitext(path.basename(self.from_file))[0]+'.h5')
        else:
            if td.h5f !=  None:
                td.h5f.close()
        # TODO problems with already open h5 files from other instances
        f5h = tables.open_file(name, mode = 'w')
        ac = f5h.create_earray(f5h.root, 'time_data', \
            tables.atom.Float32Atom(), (0, numchannels))
        ac.set_attr('sample_freq', sample_freq)
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
    name = File(filter = ['*.datx'], 
        desc = "name of datx data file")

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
        return self.channel_count*self.num_samples_per_block*\
                self.bytes_per_sample

    def get_next_blocks( self ):
        """ pulls next blocks """
        s = self.f.read(self.blocks*self.block_size)
        ls = len(s)
        if ls == 0:
            return -1
        bl_no = ls/self.block_size
        self.data = fromstring(s, dtype = 'Int16').reshape((bl_no, \
            self.channel_count, self.num_samples_per_block)).swapaxes(0, \
            1).reshape((self.channel_count, bl_no*self.num_samples_per_block))

    def __init__(self, name, blocks = 128):
        self.name = name
        self.f = file(self.name, 'rb')
        s = self.f.read(32)
        # header
        s0 = struct.unpack('IIIIIIHHf', s)        
        # Getting information about Properties of data-file 
        # 3 = Offset to data 4 = channel count 
        # 5 = number of samples per block 6 = bytes per sample
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


    def __init__(self, config, channel):
        d_file, ch_no, ch_K = config.get('channels', channel).split(', ') 
        # Extraction and Splitting of Channel information
        self.d_file = d_file
        self.ch_no = int(ch_no)
        self.label = config.get(ch_K, 'channel_label')
        self.ch_K = ch_K
        # V                                                     
        # Reading conversion factors
        self.volts_per_count = float(config.get(ch_K, 'volts_per_count'))
        self.msl_ccf = float(config.get(ch_K, 'msl_ccf'))
        self.cal_corr_factor = float(config.get(ch_K, 'cal_corr_factor'))
        self.internal_gain = float(config.get(ch_K, 'internal_gain'))
        self.external_gain = float(config.get(ch_K, 'external_gain'))
        self.tare_volts = float(config.get(ch_K, 'tare_volts'))
        # EU
        self.cal_coeff_2 = float(config.get(ch_K, 'cal_coeff_2'))
        self.cal_coeff_1 = float(config.get(ch_K, 'cal_coeff_1'))
        self.tare_eu = float(config.get(ch_K, 'tare_eu'))
        self.z0 = (self.volts_per_count * self.msl_ccf * self.cal_corr_factor) \
                    / (self.internal_gain * self.external_gain)

    def scale(self, x):
        """ scale function to produce output in engineering units """
        return (x * self.z0 - self.tare_volts) * self.cal_coeff_2 + \
                self.cal_coeff_1 - self.tare_eu

class datx_import(time_data_import):
    """
    import of .datx data
    """

    # name of the mat file to import
    from_file = File(filter = ['*.datx_index'], 
        desc = "name of the datx index file to import")

    traits_view = View(
        ['from_file', 
            '|[Import]'
        ], 
        title = 'Time data', 
        buttons = OKCancelButtons
                    )

    def get_data (self, td):
        """
        main work is done here: imports the data from datx files into
        time_data object td and saves also a '*.h5' file so this import
        need not be performed every time the data is needed
        """
        if not path.isfile(self.from_file):
            # no file there
            time_data_import.getdata(self, td)
            return
        #browse datx information
        f0 = open(self.from_file)
        config = ConfigParser.ConfigParser()
        config.readfp(f0)
        sample_rate = float(config.get('keywords', 'sample_rate'))
        # reading sample-rate from index-file
        channels = []
        d_files = {}
        # Loop over all channels assigned in index-file
        for channel in sort(config.options('channels')):
            ch = datx_channel(config, channel)
            if ch.label.find('Mic') >= 0:
                channels.append(ch)
                if not d_files.has_key(ch.d_file):
                    d_files[ch.d_file] = \
                        datx_d_file(path.join(path.dirname(self.from_file), \
                            config.get(ch.d_file, 'fn')), 32)
        numchannels = len(channels)
        # prepare hdf5
        name = td.name
        if name == "":
            name = path.join(td_dir, \
                path.splitext(path.basename(self.from_file))[0]+'.h5')
        else:
            if td.h5f !=  None:
                td.h5f.close()
        # TODO problems with already open h5 files from other instances
        f5h = tables.open_file(name, mode = 'w')
        ac = f5h.create_earray(f5h.root, 'time_data', \
            tables.atom.Float32Atom(), (0, numchannels))
        ac.set_attr('sample_freq', sample_rate)
        block_data = \
            zeros((128*d_files[channels[0].d_file].num_samples_per_block, \
                numchannels), 'Float32')
        flag = 0
        while(not flag):
            for i in d_files.values():
                flag = i.get_next_blocks()
            if flag:
                continue
            for i in range(numchannels):
                data = d_files[channels[i].d_file].data[channels[i].ch_no]
                block_data[:data.size, i] = channels[i].scale(data)
            ac.append(block_data[:data.size])
        f5h.close()
        f0.close()
        for i in d_files.values():
            i.f.close()
        td.name = name
        td.load_data()
