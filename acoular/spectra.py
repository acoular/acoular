# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, E1103, C0103, R0901, R0902, R0903, R0904
#pylint: disable-msg=W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Estimation of power spectra and related tools

.. autosummary::
    :toctree: generated/

    BaseSpectra
    FFTSpectra    
    PowerSpectra
    synthetic
    PowerSpectraImport
"""
from warnings import warn

from numpy import array, ones, hanning, hamming, bartlett, blackman, \
dot, newaxis, zeros, empty, linalg, sqrt,real, imag,\
searchsorted, isscalar, fill_diagonal, arange, zeros_like, sum, ndarray
from scipy import fft
from traits.api import HasPrivateTraits, Int, Property, Instance, Trait, \
Bool, cached_property, property_depends_on, Delegate, Float, Enum, \
    CArray

from .fastFuncs import calcCSM
from .h5cache import H5cache
from .h5files import H5CacheFileBase
from .internal import digest
from .tprocess import SamplesGenerator, TimeInOut
from .calib import Calib
from .configuration import config


class BaseSpectra( HasPrivateTraits ):

    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Trait(SamplesGenerator)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of time data channels 
    numchannels = Delegate('source')

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
    
    #: FFT block size, one of: 128, 256, 512, 1024, 2048 ... 65536,
    #: defaults to 1024.
    block_size = Trait(1024, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        desc="number of samples per FFT block")

    #: The floating-number-precision of entries of csm, eigenvalues and 
    #: eigenvectors, corresponding to numpy dtypes. Default is 64 bit.
    precision = Trait('complex128', 'complex64', 
                      desc="precision of the fft")
    
    # internal identifier
    digest = Property( depends_on = ['precision','block_size',
                                    'window','overlap'])

    @cached_property
    def _get_digest( self ):
        return digest(self)

    def fftfreq ( self ):
        """
        Return the Discrete Fourier Transform sample frequencies.
        
        Returns
        -------
        f : ndarray
            Array of length *block_size/2+1* containing the sample frequencies.
        """
        if self.source is not None:
            return abs(fft.fftfreq(self.block_size, 1./self.source.sample_freq)\
                        [:int(self.block_size/2+1)])
        else:
            return None

    #generator that yields the time data blocks for every channel (with optional overlap)
    def get_source_data(self):
        bs = self.block_size
        temp = empty((2*bs, self.numchannels))
        pos = bs
        posinc = bs/self.overlap_
        for data_block in self.source.result(bs):
            ns = data_block.shape[0]
            temp[bs:bs+ns] = data_block # fill from right
            while pos+bs <= bs+ns:
                yield temp[int(pos):int(pos+bs)]
                pos += posinc
            else:
                temp[0:bs] = temp[bs:] # copy to left
                pos -= bs


class FFTSpectra( BaseSpectra,TimeInOut ):
    """Provides the spectra of multichannel time data. 
    
    Returns Spectra per block over a Generator.       
    """
    
    # internal identifier
    digest = Property( depends_on = ['source.digest','precision','block_size',
                                    'window','overlap'])

    @cached_property
    def _get_digest( self ):
        return digest(self)

    #generator that yields the fft for every channel
    def result(self):
        """ 
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (numfreq, :attr:`numchannels`). 
            The last block may be shorter than num.
            """
        wind = self.window_( self.block_size )
        weight=sqrt(2)/self.block_size*sqrt(self.block_size/dot(wind,wind))*wind[:, newaxis]
        for data in self.get_source_data():
            ft = fft.rfft(data*weight, None, 0).astype(self.precision)
            yield ft


class PowerSpectra( BaseSpectra ):
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

    # Shadow trait, should not be set directly, for internal use.
    _source = Trait(SamplesGenerator)

    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object. 
    source = Property(_source,
        desc="time data object")

    #: The :class:`~acoular.tprocess.SamplesGenerator` object that provides the data.
    time_data = Property(_source, 
        desc="deprecated attribute holding the time data object. Use PowerSpectra.source instead!")

    #: The :class:`~acoular.calib.Calib` object that provides the calibration data, 
    #: defaults to no calibration, i.e. the raw time data is used.
    #:
    #: **deprecated**:      use :attr:`~acoular.sources.TimeSamples.calib` property of 
    #: :class:`~acoular.sources.TimeSamples` objects
    calib = Instance(Calib)

    # Shadow trait, should not be set directly, for internal use.
    _ind_low = Int(1,
        desc="index of lowest frequency line")

    # Shadow trait, should not be set directly, for internal use.
    _ind_high = Trait(-1,(Int,None),
        desc="index of highest frequency line")

    #: Index of lowest frequency line to compute, integer, defaults to 1,
    #: is used only by objects that fetch the csm, PowerSpectra computes every
    #: frequency line.
    ind_low = Property(_ind_low,
        desc="index of lowest frequency line")

    #: Index of highest frequency line to compute, integer, 
    #: defaults to -1 (last possible line for default block_size).
    ind_high = Property(_ind_high,
        desc="index of lowest frequency line")

    # Stores the set lower frequency, for internal use, should not be set directly.
    _freqlc = Float(0)

    # Stores the set higher frequency, for internal use, should not be set directly.
    _freqhc = Trait(0,(Float,None))

    # Saves whether the user set indices or frequencies last, for internal use only,
    # not to be set directly, if True (default), indices are used for setting
    # the freq_range interval.
    _index_set_last = Bool(True)
      
    #: Flag, if true (default), the result is cached in h5 files and need not
    #: to be recomputed during subsequent program runs.
    cached = Bool(True, 
        desc="cached flag")   

    #: Number of FFT blocks to average, readonly
    #: (set from block_size and overlap).
    num_blocks = Property(
        desc="overall number of FFT blocks")

    #: 2-element array with the lowest and highest frequency. If set, 
    #: will overwrite :attr:`_freqlc` and :attr:`_freqhc` according to
    #: the range. 
    #: The freq_range interval will be the smallest discrete frequency
    #: inside the half-open interval [_freqlc, _freqhc[ and the smallest
    #: upper frequency outside of the interval.
    #: If user chooses the higher frequency larger than the max frequency,
    #: the max frequency will be the upper bound.
    freq_range = Property(
        desc = "frequency range" )
        
    #: Array with a sequence of indices for all frequencies 
    #: between :attr:`ind_low` and :attr:`ind_high` within the result, readonly.
    indices = Property(
        desc = "index range" )
        
    #: Name of the cache file without extension, readonly.
    basename = Property( depends_on = '_source.digest', 
        desc="basename for cache file")

    #: The cross spectral matrix, 
    #: (number of frequencies, numchannels, numchannels) array of complex;
    #: readonly.
    csm = Property( 
        desc="cross spectral matrix")
    
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
        depends_on = ['_source.digest', 'calib.digest', 'block_size', 
            'window', 'overlap', 'precision'], 
        )

    # hdf5 cache file
    h5f = Instance( H5CacheFileBase, transient = True )
    
    @property_depends_on('_source.numsamples, block_size, overlap')
    def _get_num_blocks ( self ):
        return self.overlap_*self._source.numsamples/self.block_size-\
        self.overlap_+1

    @property_depends_on('_source.sample_freq, block_size, ind_low, ind_high')
    def _get_freq_range ( self ):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._ind_high is None:
                return array([fftfreq[self.ind_low],None])
            else:
                return fftfreq[[ self.ind_low, self.ind_high ]]

    def _set_freq_range( self, freq_range ):# by setting this the user sets _freqlc and _freqhc
        self._index_set_last = False
        self._freqlc = freq_range[0]
        self._freqhc = freq_range[1]

    @property_depends_on( '_source.sample_freq, block_size, _ind_low, _freqlc' )
    def _get_ind_low( self ):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._index_set_last:
                return min(self._ind_low, fftfreq.shape[0]-1)
            else:
                return searchsorted(fftfreq[:-1], self._freqlc)

    @property_depends_on( '_source.sample_freq, block_size, _ind_high, _freqhc' )
    def _get_ind_high( self ):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            if self._index_set_last:
                if self._ind_high is None: 
                    return None
                else:
                    return min(self._ind_high, fftfreq.shape[0]-1)
            else:
                if self._freqhc is None:
                    return None
                else:
                    return searchsorted(fftfreq[:-1], self._freqhc)

    def _set_ind_high(self, ind_high):# by setting this the user sets the lower index
        self._index_set_last = True
        self._ind_high = ind_high

    def _set_ind_low( self, ind_low):# by setting this the user sets the higher index
        self._index_set_last = True
        self._ind_low = ind_low

    def _set_time_data(self, time_data):
        self._source = time_data

    def _set_source(self, source):
        self._source = source

    def _get_time_data(self):
        return self._source

    def _get_source(self):
        return self._source

    @property_depends_on( 'block_size, ind_low, ind_high' )
    def _get_indices ( self ):
        fftfreq = self.fftfreq()
        if fftfreq is not None:
            try:
                indices = arange(fftfreq.shape[0],dtype=int)
                if self.ind_high is None:
                    return indices[ self.ind_low:]
                else:
                    return indices[ self.ind_low: self.ind_high ]
            except IndexError:
                return range(0)

    @cached_property
    def _get_digest( self ):
        return digest( self )

    @cached_property
    def _get_basename( self ):
        if 'basename' in self._source.all_trait_names():
            return self._source.basename
        else: 
            return self._source.__class__.__name__ + self._source.digest

    def calc_csm( self ):
        """ csm calculation """
        t = self.source
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
        # get time data blockwise
        for data in self.get_source_data():
            ft = fft.rfft(data*wind, None, 0).astype(self.precision)
            calcCSM(csmUpper, ft)  # only upper triangular part of matrix is calculated (for speed reasons)
        # create the full csm matrix via transposing and complex conj.
        csmLower = csmUpper.conj().transpose(0,2,1)
        [fill_diagonal(csmLower[cntFreq, :, :], 0) for cntFreq in range(csmLower.shape[0])]
        csm = csmLower + csmUpper
        # onesided spectrum: multiplication by 2.0=sqrt(2)^2
        csm = csm*(2.0/self.block_size/weight/self.num_blocks)
        return csm

    def calc_ev ( self ):
        """ eigenvalues / eigenvectors calculation """
        if self.precision == 'complex128': eva_dtype = 'float64'
        elif self.precision == 'complex64': eva_dtype = 'float32'
#        csm = self.csm #trigger calculation
        csm_shape = self.csm.shape
        eva = empty(csm_shape[0:2], dtype=eva_dtype)
        eve = empty(csm_shape, dtype=self.precision)
        for i in range(csm_shape[0]):
            (eva[i], eve[i])=linalg.eigh(self.csm[i])
        return (eva,eve)

    def calc_eva( self ):
        """ calculates eigenvalues of csm """
        return self.calc_ev()[0]
    
    def calc_eve( self ):
        """ calculates eigenvectors of csm """
        return self.calc_ev()[1]
                
    def _handle_dual_calibration(self):
        obj = self.source # start with time_data obj
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

    def _get_filecache( self, traitname ):
        """
        function handles result caching of csm, eigenvectors and eigenvalues
        calculation depending on global/local caching behaviour.  
        """
        if traitname == 'csm':
            func = self.calc_csm
            numfreq = int(self.block_size/2 + 1)
            shape = (numfreq, self._source.numchannels, self._source.numchannels)
            precision = self.precision
        elif traitname == 'eva':
            func = self.calc_eva
            shape = self.csm.shape[0:2]
            if self.precision == 'complex128': precision = 'float64'
            elif self.precision == 'complex64': precision = 'float32'
        elif traitname == 'eve':
            func = self.calc_eve
            shape = self.csm.shape
            precision = self.precision

        H5cache.get_cache_file( self, self.basename ) 
        if not self.h5f: # in case of global caching readonly
            return func() 

        nodename = traitname + '_' + self.digest 
        if config.global_caching == 'overwrite' and self.h5f.is_cached(nodename):
            #print("remove existing node",nodename)
            self.h5f.remove_data(nodename) # remove old data before writing in overwrite mode
        
        if not self.h5f.is_cached(nodename): 
            if config.global_caching == 'readonly': 
                return func()
#            print("create array, data not cached for",nodename)
            self.h5f.create_compressible_array(nodename,shape,precision)
            
        ac = self.h5f.get_data_by_reference(nodename)
        if ac[:].sum() == 0: # only initialized
#            print("write {} to:".format(traitname),nodename)
            ac[:] = func()
            self.h5f.flush()
        return ac
             
    @property_depends_on('digest')
    def _get_csm ( self ):
        """
        Main work is done here:
        Cross spectral matrix is either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        self._handle_dual_calibration()
        if (
                config.global_caching == 'none' or 
                (config.global_caching == 'individual' and self.cached is False)
            ):
            return self.calc_csm()
        else:
            return self._get_filecache('csm')
                          
    @property_depends_on('digest')
    def _get_eva ( self ):
        """
        Eigenvalues of cross spectral matrix are either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        if (
                config.global_caching == 'none' or 
                (config.global_caching == 'individual' and self.cached is False)
            ):
            return self.calc_eva()
        else:
            return self._get_filecache('eva')

    @property_depends_on('digest')
    def _get_eve ( self ):
        """
        Eigenvectors of cross spectral matrix are either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        if (
                config.global_caching == 'none' or 
                (config.global_caching == 'individual' and self.cached is False)
            ):
            return self.calc_eve()
        else:
            return self._get_filecache('eve')

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
        The spectral data (squared sound pressures in Pa^2) in an array with one value 
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
        the :attr:`sampling frequency<acoular.tprocess.SamplesGenerator.sample_freq>` 
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


class PowerSpectraImport( PowerSpectra ):
    """Provides a dummy class for using pre-calculated cross-spectral
    matrices. 

    This class does not calculate the cross-spectral matrix. Instead, 
    the user can inject one or multiple existing CSMs by setting the 
    :attr:`csm` attribute. This can be useful when algorithms shall be
    evaluated with existing CSM matrices.
    The frequency or frequencies contained by the CSM must be set via the 
    attr:`frequencies` attribute. The attr:`numchannels` attributes
    is determined on the basis of the CSM shape. 
    In contrast to the PowerSpectra object, the attributes 
    :attr:`sample_freq`, :attr:`time_data`, :attr:`source`,
    :attr:`block_size`, :attr:`calib`, :attr:`window`, 
    :attr:`overlap`, :attr:`cached`, and :attr:`num_blocks`
    have no functionality. 
    """

    #: The cross spectral matrix, 
    #: (number of frequencies, numchannels, numchannels) array of complex;
    csm = Property( 
        desc="cross spectral matrix")

    #: frequencies included in the cross-spectral matrix in ascending order.
    #: Compound trait that accepts arguments of type list, array, and float
    frequencies = Trait(None,(CArray,Float),
        desc="frequencies included in the cross-spectral matrix")

    #: Number of time data channels 
    numchannels = Property(depends_on=['digest'])

    time_data = Enum(None, 
        desc="PowerSpectraImport cannot consume time data")

    source = Enum(None, 
        desc="PowerSpectraImport cannot consume time data")

    # Sampling frequency of the signal, defaults to None
    sample_freq = Enum(None, 
        desc="sampling frequency")

    block_size = Enum(None, 
        desc="PowerSpectraImport does not operate on blocks of time data")

    calib = Enum(None,
        desc="PowerSpectraImport cannot calibrate the time data")

    window = Enum(None,
            desc="PowerSpectraImport does not perform windowing")

    overlap = Enum(None,
            desc="PowerSpectraImport does not consume time data")

    cached = Enum(False,
            desc="PowerSpectraImport has no caching capabilities")

    num_blocks = Enum(None,
            desc="PowerSpectraImport cannot determine the number of blocks")

    # Shadow trait, should not be set directly, for internal use.
    _ind_low = Int(0,
        desc="index of lowest frequency line")

    # Shadow trait, should not be set directly, for internal use.
    _ind_high = Trait(None,(Int,None),
        desc="index of highest frequency line")

    # internal identifier
    digest = Property( 
        depends_on = ['_csmsum', 
            ], 
        )

    #: Name of the cache file without extension, readonly.
    basename = Property( depends_on = 'digest', 
        desc="basename for cache file")

    # csm shadow trait, only for internal use.
    _csm = CArray()
        
    # CSM checksum to trigger digest calculation, only for internal use.
    _csmsum = Float() 

    def _get_basename( self ):
        return "csm_import_"+self.digest

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def _get_numchannels( self ):
        return self.csm.shape[1]

    def _get_csm ( self ):
        return self._csm

    def _set_csm (self, csm):
        if (len(csm.shape) != 3) or (csm.shape[1] != csm.shape[2]):
            raise ValueError(
                "The cross spectral matrix must have the following shape: (number of frequencies, numchannels, numchannels)!")
        self._csmsum = real(self._csm).sum() + (imag(self._csm)**2).sum() # to trigger new digest creation
        self._csm = csm

    @property_depends_on('digest')
    def _get_eva ( self ):
        """
        Eigenvalues of cross spectral matrix are either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        return self.calc_eva()

    @property_depends_on('digest')
    def _get_eve ( self ):
        """
        Eigenvectors of cross spectral matrix are either loaded from cache file or
        calculated and then additionally stored into cache.
        """
        return self.calc_eve()

    def fftfreq ( self ):
        """
        Return the Discrete Fourier Transform sample frequencies.
        
        Returns
        -------
        f : ndarray
            Array containing the frequencies.
        """
        if isinstance(self.frequencies,float): 
            return array([self.frequencies])
        elif isinstance(self.frequencies,ndarray):
            return self.frequencies
        elif self.frequencies is None:
            warn("No frequencies defined for PowerSpectraImport object!")
            return self.frequencies
        else:
            return self.frequencies