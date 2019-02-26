# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, E1103, C0103, R0901, R0902, R0903, R0904
#pylint: disable-msg=W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2017, Acoular Development Team.
#------------------------------------------------------------------------------
"""Estimation of power spectra and related tools

.. autosummary::
    :toctree: generated/

    PowerSpectra
    synthetic
"""
from warnings import warn
from six.moves import xrange  # solves the xrange/range issue for python2/3: in py3 'xrange' is now treated as 'range' and in py2 nothing changes

from numpy import array, ones, hanning, hamming, bartlett, blackman, \
dot, newaxis, zeros, empty, fft, linalg, \
searchsorted, isscalar, fill_diagonal, arange, zeros_like, sum
import tables
from traits.api import HasPrivateTraits, Int, Property, Instance, Trait, \
Range, Bool, cached_property, property_depends_on, Delegate
from traitsui.api import View
from traitsui.menu import OKCancelButtons

from .fastFuncs import calcCSM

from .h5cache import H5cache
from .internal import digest
from .sources import SamplesGenerator
from .calib import Calib


def _precision(idString):
    """
    Internal method, needed for allocation of memory.
    """
    # Create dictionary: third value is needed as Argument for tables.ComplexAtoms.
    # Quite ugly creation of dictionary needed because several keys have same value.
    precDict = dict.fromkeys(['float32', 'complex64'], ('float32', 'complex64', 8, tables.Float32Atom))
    precDict.update(dict.fromkeys(['float64', 'complex128'], ('float64', 'complex128', 16, tables.Float64Atom)))
    return precDict[idString]

class PowerSpectra( HasPrivateTraits ):
    """Provides the cross spectral matrix of multichannel time data
     and its eigen-decomposition.
    
    This class includes the efficient calculation of the full cross spectral
    matrix using the Welch method with windows and overlap. It also contains 
    the CSM's eigenvalues and eigenvectors and additional properties. 
    
    The result is computed only when needed, that is when the :attr:`csm`,
    :attr:`eva`, or :attr:`eve` attributes are acturally read.
    Any change in the input data or parameters leads to a new calculation, 
    again triggered when an attribute is read. The result may be 
    cached on disk in HDF5 files and need not to be recomputed during
    subsequent program runs with identical input data and parameters. The
    input data is taken to be identical if the source has identical parameters
    and the same file name in case of that the data is read from a file.
    """

    #: The :class:`~acoular.sources.SamplesGenerator` object that provides the data.
    time_data = Trait(SamplesGenerator, 
        desc="time data object")

    #: Number of samples 
    numchannels = Delegate('time_data')

    #: The :class:`~acoular.calib.Calib` object that provides the calibration data, 
    #: defaults to no calibration, i.e. the raw time data is used.
    #:
    #: **deprecated**:      use :attr:`~acoular.sources.TimeSamples.calib` property of 
    #: :class:`~acoular.sources.TimeSamples` objects
    calib = Instance(Calib)

    #: FFT block size, one of: 128, 256, 512, 1024, 2048 ... 16384,
    #: defaults to 1024.
    block_size = Trait(1024, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 
        desc="number of samples per FFT block")

    #: Index of lowest frequency line to compute, integer, defaults to 1,
    #: is used only by objects that fetch the csm, PowerSpectra computes every
    #: frequency line.
    ind_low = Range(1, 
        desc="index of lowest frequency line")

    #: Index of highest frequency line to compute, integer, 
    #: defaults to -1 (last possible line for default block_size).
    ind_high = Int(-1, 
        desc="index of highest frequency line")

    #: Window function for FFT, one of:
    #:   * 'Rectangular' (default)
    #:   * 'Hanning'
    #:   * 'Hamming'
    #:   * 'Bartlett'
    #:   * 'Blackman'
    window = Trait('Rectangular', 
        {'Rectangular':ones, 
        'Hanning':hanning, 
        'Hamming':hamming, 
        'Bartlett':bartlett, 
        'Blackman':blackman}, 
        desc="type of window for FFT")

    #: Overlap factor for averaging: 'None'(default), '50%', '75%', '87.5%'.
    overlap = Trait('None', {'None':1, '50%':2, '75%':4, '87.5%':8}, 
        desc="overlap of FFT blocks")
        
    #: Flag, if true (default), the result is cached in h5 files and need not
    #: to be recomputed during subsequent program runs.
    cached = Bool(True, 
        desc="cached flag")   

    #: Number of FFT blocks to average, readonly
    #: (set from block_size and overlap).
    num_blocks = Property(
        desc="overall number of FFT blocks")

    #: 2-element array with the lowest and highest frequency, readonly.
    freq_range = Property(
        desc = "frequency range" )
        
    #: Array with a sequence of indices for all frequencies 
    #: between :attr:`ind_low` and :attr:`ind_high` within the result, readonly.
    indices = Property(
        desc = "index range" )
        
    #: Name of the cache file without extension, readonly.
    basename = Property( depends_on = 'time_data.digest', 
        desc="basename for cache file")

    #: The cross spectral matrix, 
    #: (number of frequencies, numchannels, numchannels) array of complex;
    #: readonly.
    csm = Property( 
        desc="cross spectral matrix")
    
    #: The floating-number-precision of entries of csm, eigenvalues and 
    #: eigenvectors, corresponding to numpy dtypes. Default is 64 bit.
    precision = Trait('complex128', 'complex64', 
                      desc="precision csm, eva, eve")

    #: Eigenvalues of the cross spectral matrix as an
    #: (number of frequencies) array of floats, readonly.
    eva = Property( 
        desc="eigenvalues of cross spectral matrix")

    #: Eigenvectors of the cross spectral matrix as an
    #: (number of frequencies, numchannels, numchannels) array of floats,
    #: readonly.
    eve = Property( 
        desc="eigenvectors of cross spectral matrix")



    # internal identifier
    digest = Property( 
        depends_on = ['time_data.digest', 'calib.digest', 'block_size', 
            'window', 'overlap', 'precision'], 
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
            return arange(self.block_size/2+1,dtype=int)[ self.ind_low: self.ind_high ]
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
        """
        Main work is done here:
        Cross spectral matrix is either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        # test for dual calibration
        obj = self.time_data # start with time_data obj
        while obj:
            if 'calib' in obj.all_trait_names(): # at original source?
                if obj.calib and self.calib:
                    if obj.calib.digest == self.calib.digest:
                        self.calib = None # ignore it silently
                    else:
                        raise ValueError("Non-identical dual calibration for "\
                                    "both TimeSamples and PowerSpectra object")
                obj = None
            else:
                try:
                    obj = obj.source # traverse down until original data source
                except AttributeError:
                    obj = None
        name = 'csm_' + self.digest
        H5cache.get_cache( self, self.basename )
        #print(self.basename)
        if not self.cached  or not name in self.h5f.root:
            t = self.time_data
            wind = self.window_( self.block_size )
            weight = dot( wind, wind )
            wind = wind[newaxis, :].swapaxes( 0, 1 )
            numfreq = int(self.block_size/2 + 1)
            csm_shape = (numfreq, t.numchannels, t.numchannels)
            csmUpper = zeros(csm_shape, dtype=self.precision)
            #print "num blocks", self.num_blocks
            # for backward compatibility
            if self.calib and self.calib.num_mics > 0:
                if self.calib.num_mics == t.numchannels:
                    wind = wind * self.calib.data[newaxis, :]
                else:
                    raise ValueError(
                            "Calibration data not compatible: %i, %i" % \
                            (self.calib.num_mics, t.numchannels))
            bs = self.block_size
            temp = empty((2*bs, t.numchannels))
            pos = bs
            posinc = bs/self.overlap_
            for data in t.result(bs):
                ns = data.shape[0]
                temp[bs:bs+ns] = data
                while pos+bs <= bs+ns:
                    ft = fft.rfft(temp[int(pos):int(pos+bs)]*wind, None, 0).astype(self.precision)
                    calcCSM(csmUpper, ft)  # only upper triangular part of matrix is calculated (for speed reasons)
                    pos += posinc
                temp[0:bs] = temp[bs:]
                pos -= bs
            
            # create the full csm matrix via transposingand complex conj.
            csmLower = csmUpper.conj().transpose(0,2,1)
            [fill_diagonal(csmLower[cntFreq, :, :], 0) for cntFreq in xrange(csmLower.shape[0])]
            csm = csmLower + csmUpper

            # onesided spectrum: multiplication by 2.0=sqrt(2)^2
            csm = csm*(2.0/self.block_size/weight/self.num_blocks)
            
            if self.cached:
                precisionTuple = _precision(self.precision)
                atom = tables.ComplexAtom(precisionTuple[2])
                filters = tables.Filters(complevel=5, complib='blosc')
                ac = self.h5f.create_carray(self.h5f.root, name, atom,
                                            csm_shape, filters=filters)
                ac[:] = csm
                return ac
            else:
                return csm
        else:
            return self.h5f.get_node('/', name)



    @property_depends_on('digest')
    def _get_eva ( self ):
        return self._calc_ev()[0]

    @property_depends_on('digest')
    def _get_eve ( self ):
        return self._calc_ev()[1]

    def _calc_ev ( self ):
        """
        eigenvalues / eigenvectors calculation
        """
        
        name_eva = 'eva_' + self.digest
        name_eve = 'eve_' + self.digest
        csm = self.csm #trigger calculation
        
        if not self.cached  or  (not name_eva in self.h5f.root) or (not name_eve in self.h5f.root):
            csm_shape = self.csm.shape
            precisionTuple = _precision(self.precision)
            eva = empty(csm_shape[0:2], dtype=precisionTuple[0])
            eve = empty(csm_shape, dtype=precisionTuple[1])
            for i in range(csm_shape[0]):
                (eva[i], eve[i])=linalg.eigh(self.csm[i])
                
            if self.cached:   
                atom_eva = precisionTuple[3]()
                atom_eve = tables.ComplexAtom(precisionTuple[2])
                filters = tables.Filters(complevel=5, complib='blosc')
                ac_eva = self.h5f.create_carray(self.h5f.root, name_eva, atom_eva, \
                                                eva.shape, filters=filters)
                ac_eve = self.h5f.create_carray(self.h5f.root, name_eve, atom_eve, \
                                                eve.shape, filters=filters)
                ac_eva[:] = eva
                ac_eve[:] = eve
                return (ac_eva,ac_eve)
            else:
                return (eva,eve)
            
        return (self.h5f.get_node('/', name_eva), \
                    self.h5f.get_node('/', name_eve))
            
    def synthetic_ev( self, freq, num=0):
        """Return synthesized frequency band values of the eigenvalues.
        
        Parameters
        ----------
        freq : float 
            Band center frequency for which to return the results.
        num : integer
            Controls the width of the frequency bands considered; defaults to
            3 (third-octave band).
            
            ===  =====================
            num  frequency band width
            ===  =====================
            0    single frequency line
            1    octave band
            3    third-octave band
            n    1/n-octave band
            ===  =====================

        Returns
        -------
        float 
            Synthesized frequency band value of the eigenvalues (the sum of
            all values that are contained in the band).
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


    def fftfreq ( self ):
        """
        Return the Discrete Fourier Transform sample frequencies.
        
        Returns
        -------
        f : ndarray
            Array of length *block_size/2+1* containing the sample frequencies.
        """
        return abs(fft.fftfreq(self.block_size, 1./self.time_data.sample_freq)\
                    [:int(self.block_size/2+1)])



def synthetic (data, freqs, f, num=3):
    """
    Returns synthesized frequency band values of spectral data.
    
    If used with :meth:`Beamformer.result()<acoular.fbeamform.BeamformerBase.result>` 
    and only one frequency band, the output is identical to the result of the intrinsic 
    :meth:`Beamformer.synthetic<acoular.fbeamform.BeamformerBase.synthetic>` method.
    It can, however, also be used with the 
    :meth:`Beamformer.integrate<acoular.fbeamform.BeamformerBase.integrate>`
    output and more frequency bands.
    
    Parameters
    ----------
    data : array of floats
        The spectral data (sound pressures in Pa) in an array with one value 
        per frequency line.
        The number of entries must be identical to the number of
        grid points.
    freq : array of floats
        The frequencies that correspon to the input *data* (as yielded by
        the :meth:`PowerSpectra.fftfreq<acoular.spectra.PowerSpectra.fftfreq>`
        method).
    f : float or list of floats
        Band center frequency/frequencies for which to return the results.
    num : integer
        Controls the width of the frequency bands considered; defaults to
        3 (third-octave band).
        
        ===  =====================
        num  frequency band width
        ===  =====================
        0    single frequency line
        1    octave band
        3    third-octave band
        n    1/n-octave band
        ===  =====================

    Returns
    -------
    array of floats
        Synthesized frequency band values of the beamforming result at 
        each grid point (the sum of all values that are contained in the band).
        Note that the frequency resolution and therefore the bandwidth 
        represented by a single frequency line depends on 
        the :attr:`sampling frequency<acoular.sources.SamplesGenerator.sample_freq>` 
        and used :attr:`FFT block size<acoular.spectra.PowerSpectra.block_size>`.
    """
    if isscalar(f):
        f = (f,)
    if num == 0:
        # single frequency lines
        res = list()
        for i in f:
            ind = searchsorted(freqs, i)
            if ind >= len(freqs):
                warn('Queried frequency (%g Hz) not in resolved '
                     'frequency range. Returning zeros.' % i, 
                     Warning, stacklevel = 2)
                h = zeros_like(data[0])
            else:
                if freqs[ind] != i:
                    warn('Queried frequency (%g Hz) not in set of '
                         'discrete FFT sample frequencies. '
                         'Using frequency %g Hz instead.' % (i,freqs[ind]), 
                         Warning, stacklevel = 2)
                h = data[ind]
            res += [h]      
    else:
        # fractional octave bands
        res = list()
        for i in f:
            f1 = i*2.**(-0.5/num)
            f2 = i*2.**(+0.5/num)
            ind1 = searchsorted(freqs, f1)
            ind2 = searchsorted(freqs, f2)
            if ind1 == ind2:
                warn('Queried frequency band (%g to %g Hz) does not '
                     'include any discrete FFT sample frequencies. '
                     'Returning zeros.' % (f1,f2), 
                     Warning, stacklevel = 2)
                h = zeros_like(data[0])
            else:
                h = sum(data[ind1:ind2], 0)
            res += [h]
    return array(res)
        
