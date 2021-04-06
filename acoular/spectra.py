# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, E1103, C0103, R0901, R0902, R0903, R0904
#pylint: disable-msg=W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2020, Acoular Development Team.
#------------------------------------------------------------------------------
"""Estimation of power spectra and related tools

.. autosummary::
    :toctree: generated/

    PowerSpectra
    synthetic
"""
from warnings import warn

from numpy import array, ones, hanning, hamming, bartlett, blackman, \
dot, newaxis, zeros, empty, fft, linalg, \
searchsorted, isscalar, fill_diagonal, arange, zeros_like, sum
from traits.api import HasPrivateTraits, Int, Property, Instance, Trait, \
Range, Bool, cached_property, property_depends_on, Delegate, Float

from .fastFuncs import calcCSM
from .h5cache import H5cache
from .h5files import H5CacheFileBase
from .internal import digest
from .tprocess import SamplesGenerator
from .calib import Calib
from .configuration import config



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

    #: The :class:`~acoular.tprocess.SamplesGenerator` object that provides the data.
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

    #: FFT block size, one of: 128, 256, 512, 1024, 2048 ... 65536,
    #: defaults to 1024.
    block_size = Trait(1024, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        desc="number of samples per FFT block")

    # Shadow trait, should not be set directly, for internal use.
    _ind_low = Int(1,
        desc="index of lowest frequency line")

    # Shadow trait, should not be set directly, for internal use.
    _ind_high = Int(-1,
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
    _freqhc = Float(0)

    # Saves whether the user set indices or frequencies last, for internal use only,
    # not to be set directly, if True (default), indices are used for setting
    # the freq_range interval.
    _index_set_last = Bool(True)

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
    h5f = Instance( H5CacheFileBase, transient = True )
    
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

    def _set_freq_range( self, freq_range ):# by setting this the user sets _freqlc and _freqhc
        self._index_set_last = False
        self._freqlc = freq_range[0]
        self._freqhc = freq_range[1]

    @property_depends_on( 'time_data.sample_freq, block_size, _ind_low, _freqlc' )
    def _get_ind_low( self ):
        if self._index_set_last:
            return min(self._ind_low, self.block_size//2)
        else:
            return searchsorted(self.fftfreq()[:-1], self._freqlc)

    @property_depends_on( 'time_data.sample_freq, block_size, _ind_high, _freqhc' )
    def _get_ind_high( self ):
        if self._index_set_last:
            return min(self._ind_high, self.block_size//2)
        else:
            return searchsorted(self.fftfreq()[:-1], self._freqhc)

    def _set_ind_high(self, ind_high):# by setting this the user sets the lower index
        self._index_set_last = True
        self._ind_high = ind_high

    def _set_ind_low( self, ind_low):# by setting this the user sets the higher index
        self._index_set_last = True
        self._ind_low = ind_low

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

    def calc_csm( self ):
        """ csm calculation """
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

    def _get_filecache( self, traitname ):
        """
        function handles result caching of csm, eigenvectors and eigenvalues
        calculation depending on global/local caching behaviour.  
        """
        if traitname == 'csm':
            func = self.calc_csm
            numfreq = int(self.block_size/2 + 1)
            shape = (numfreq, self.time_data.numchannels, self.time_data.numchannels)
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
                (config.global_caching == 'individual' and self.cached == False)
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
                (config.global_caching == 'individual' and self.cached == False)
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
                (config.global_caching == 'individual' and self.cached == False)
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

