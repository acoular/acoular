# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, E1103, C0103, R0901, R0902, R0903, R0904
#pylint: disable-msg=W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Beamfpy Development Team.
#------------------------------------------------------------------------------
"""Estimation of power spectra and related tools

.. autosummary::
    :toctree: generated/

    PowerSpectra
    EigSpectra
    synthetic
"""

from numpy import array, ones, hanning, hamming, bartlett, blackman, \
dot, newaxis, zeros, empty, fft, float32, complex64, linalg, \
searchsorted
import tables
from traits.api import HasPrivateTraits, Int, Property, Instance, Trait, \
Range, Bool, cached_property, property_depends_on
from traitsui.api import View
from traitsui.menu import OKCancelButtons

from .beamformer import faverage

from .h5cache import H5cache
from .internal import digest
from .timedomain import SamplesGenerator, Calib


class PowerSpectra( HasPrivateTraits ):
    """
    efficient calculation of full cross spectral matrix
    container for data and properties of this matrix
    """

    # the SamplesGenerator object that provides the data
    time_data = Trait(SamplesGenerator, 
        desc="time data object")

    # the Calib object that provides the calibration data, 
    # defaults to no calibration, i.e. the raw time data is used
    calib = Instance(Calib)

    # FFT block size, one of: 128, 256, 512, 1024, 2048 ... 16384
    # defaults to 1024
    block_size = Trait(1024, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 
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
    # 'Rectangular' (default), 'Hanning', 'Hamming', 'Bartlett', 'Blackman'
    window = Trait('Rectangular', 
        {'Rectangular':ones, 
        'Hanning':hanning, 
        'Hamming':hamming, 
        'Bartlett':bartlett, 
        'Blackman':blackman}, 
        desc="type of window for FFT")

    # overlap factor for averaging: 'None'(default), '50%', '75%', '87.5%'
    overlap = Trait('None', {'None':1, '50%':2, '75%':4, '87.5%':8}, 
        desc="overlap of FFT blocks")
        
    # flag, if true (default), the result is cached in h5 files
    cached = Bool(True, 
        desc="cached flag")   

    # number of FFT blocks to average (auto-set from block_size and overlap)
    num_blocks = Property(
        desc="overall number of FFT blocks")

    # frequency range
    freq_range = Property(
        desc = "frequency range" )
        
    # frequency range
    indices = Property(
        desc = "index range" )
        
    # basename for cache file
    basename = Property( depends_on = 'time_data.digest', 
        desc="basename for cache file")

    # the cross spectral matrix as
    # (number of frequencies, numchannels, numchannels) array of complex
    csm = Property( 
        desc="cross spectral matrix")

    # internal identifier
    digest = Property( 
        depends_on = ['time_data.digest', 'calib.digest', 'block_size', 
            'window', 'overlap'], 
        )

    # hdf5 cache file
    h5f = Instance(tables.File, transient = True )
    
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
    
    @property_depends_on('time_data.numsamples, block_size, overlap')
    def _get_num_blocks ( self ):
        return self.overlap_*self.time_data.numsamples/self.block_size-\
        self.overlap_+1

    @property_depends_on('time_data.sample_freq, block_size, ind_low, ind_high')
    def _get_freq_range ( self ):
        try:
            return self.fftfreq()[[ self.ind_low, self.ind_high ]]
        except IndexError:
            return array([0., 0])

    @property_depends_on( 'block_size, ind_low, ind_high' )
    def _get_indices ( self ):
        try:
            return range(self.block_size/2+1)[ self.ind_low: self.ind_high ]
        except IndexError:
            return range(0)

    @cached_property
    def _get_digest( self ):
        return digest( self )

    @cached_property
    def _get_basename( self ):
        if 'basename' in self.time_data.all_trait_names():
            return self.time_data.basename
        else: 
            return self.time_data.__class__.__name__ + self.time_data.digest

    @property_depends_on('digest')
    def _get_csm ( self ):
        """main work is done here:
        cross spectral matrix is either loaded from cache file or
        calculated and then additionally stored into cache
        """
        # test for dual calibration
        obj = self.time_data # start with time_data obj
        while obj:
            #print obj
            if 'calib' in obj.all_trait_names(): # at original source?
                if obj.calib and self.calib:
                    if obj.calib.digest == self.calib.digest:
                        self.calib = None # ignore it silently
                    else:
                        raise ValueError("nonidentical dual calibration for "\
                                    "both TimeSamples and PowerSpectra object")
                obj = None
            else:
                try:
                    obj = obj.source # traverse down until original data source
                except AttributeError:
                    obj = None
        name = 'csm_' + self.digest
        H5cache.get_cache( self, self.basename )
        #print self.basename
        if not self.cached  or not name in self.h5f.root:
            t = self.time_data
            wind = self.window_( self.block_size )
            weight = dot( wind, wind )
            wind = wind[newaxis, :].swapaxes( 0, 1 )
            numfreq = self.block_size/2 + 1
            csm_shape = (numfreq, t.numchannels, t.numchannels)
            csm = zeros(csm_shape, 'D')
            #print "num blocks", self.num_blocks
            # for backward compatibility
            if self.calib and self.calib.num_mics > 0:
                if self.calib.num_mics == t.numchannels:
                    wind = wind * self.calib.data[newaxis, :]
                else:
                    raise ValueError(
                            "calibration data not compatible: %i, %i" % \
                            (self.calib.num_mics, t.numchannels))
            bs = self.block_size
            temp = empty((2*bs, t.numchannels))
            pos = bs
            posinc = bs/self.overlap_
            for data in t.result(bs):
                ns = data.shape[0]
                temp[bs:bs+ns] = data
                while pos+bs <= bs+ns:
                    ft = fft.rfft(temp[pos:(pos+bs)]*wind, None, 0)
                    faverage(csm, ft)
                    pos += posinc
                temp[0:bs] = temp[bs:]
                pos -= bs
            # onesided spectrum: multiplication by 2.0=sqrt(2)^2
            csm = csm*(2.0/self.block_size/weight/self.num_blocks)
            if self.cached:
                atom = tables.ComplexAtom(8)
                ac = self.h5f.createCArray(self.h5f.root, name, atom, csm_shape)
                ac[:] = csm
                return ac
            else:
                return csm
        else:
            return self.h5f.getNode('/', name)

    def fftfreq ( self ):
        """
        returns an array of the frequencies for the spectra in the 
        cross spectral matrix from 0 to fs/2
        """
        return abs(fft.fftfreq(self.block_size, 1./self.time_data.sample_freq)\
                    [:self.block_size/2+1])


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
            eva = empty(csm_shape[0:2], float32)
            eve = empty(csm_shape, complex64)
            for i in range(csm_shape[0]):
                (eva[i], eve[i])=linalg.eigh(self.csm[i])
            atom_eva = tables.Float32Atom()
            atom_eve = tables.ComplexAtom(8)
            #filters = tables.Filters(complevel=5, complib='zlib')
            ac_eva = self.h5f.createCArray(self.h5f.root, name_eva, atom_eva, \
                eva.shape)#, filters=filters)
            ac_eve = self.h5f.createCArray(self.h5f.root, name_eve, atom_eve, \
                eve.shape)#, filters=filters)
            ac_eva[:] = eva
            ac_eve[:] = eve
        return (self.h5f.getNode('/', name_eva), \
                    self.h5f.getNode('/', name_eve))
            
    def synthetic_ev( self, freq, num=0):
        """
        returns synthesized frequency band values of the eigenvalues
        num = 0: single frequency line
        num = 1: octave band
        num = 3: third octave band
        etc.
        """
        f = self.fftfreq()
        if num == 0:
            # single frequency line
            return self.eva[searchsorted(f, freq)]
        else:
            f1 = searchsorted(f, freq*2.**(-0.5/num))
            f2 = searchsorted(f, freq*2.**(0.5/num))
            if f1 == f2:
                return self.eva[f1]
            else:
                return sum(self.eva[f1:f2], 0)


def synthetic (data, freqs, f, num=3):
    """
    returns synthesized frequency band values of data
    num = 0: single frequency line
    num = 1: octave bands
    num = 3: third octave bands
    etc.

    freqs: the frequencies that correspond to the input data

    f: band center frequencies

    """
    if num == 0:
        res = [ data[searchsorted(freqs, i)] for i in f]        
    else:
        res = [ data[searchsorted(freqs, i*2.**(-0.5/num)):\
                    searchsorted(freqs, i*2.**(+0.5/num))].sum(0) for i in f]
    return array(res)
        