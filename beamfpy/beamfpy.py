# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
"""
beamfpy.py: classes for calculations in the frequency domain

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming

(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""

from numpy import array, ones, hanning, hamming, bartlett, blackman, invert, \
dot, newaxis, zeros, empty, fft, float32, float64, complex64, linalg, where, \
searchsorted, pi, multiply, sign, diag, arange, sqrt, exp, log10, int,\
reshape, hstack, vstack, eye, tril, size, clip
from sklearn.linear_model import LassoLars, LassoCV, LassoLarsCV, LassoLarsIC,\
 OrthogonalMatchingPursuit, SGDRegressor, LinearRegression, ElasticNet, \
 OrthogonalMatchingPursuitCV, Lasso
from sklearn.cross_validation import LeaveOneOut
from scipy.optimize import nnls
import tables
from traits.api import HasPrivateTraits, Float, Int, \
CArray, Property, Instance, Trait, Bool, Range, Delegate, Enum, \
cached_property, on_trait_change, property_depends_on
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons
from beamformer import faverage, gseidel, transfer,\
r_beam_psf, r_beam_psf1, r_beam_psf2, r_beam_psf3, r_beam_psf4, \
r_beamfull, r_beamfull_3d, r_beamfull_classic, r_beamfull_inverse, \
r_beamdiag, r_beamdiag_3d, r_beamdiag_classic, r_beamdiag_inverse, \
r_beamfull_os, r_beamfull_os_3d, r_beamfull_os_classic, r_beamfull_os_inverse, \
r_beamdiag_os, r_beamdiag_os_3d, r_beamdiag_os_classic, r_beamdiag_os_inverse

from h5cache import H5cache
from internal import digest
from grids import Grid, MicGeom, Environment
from timedomain import SamplesGenerator, Calib

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
        if not name in self.h5f.root:
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
            atom = tables.ComplexAtom(8)
            ac = self.h5f.createCArray(self.h5f.root, name, atom, csm_shape)
            ac[:] = csm
            return ac
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

class BeamformerBase( HasPrivateTraits ):
    """
    beamforming using the basic delay-and-sum algorithm
    """

    # PowerSpectra object that provides the cross spectral matrix
    freq_data = Trait(PowerSpectra, 
        desc="freq data object")

    # RectGrid object that provides the grid locations
    grid = Trait(Grid, 
        desc="beamforming grid")

    # MicGeom object that provides the microphone locations
    mpos = Trait(MicGeom, 
        desc="microphone geometry")
        
    # Environment object that provides speed of sound and grid-mic distances
    env = Trait(Environment(), Environment)

    # the speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")

    # flag, if true (default), the main diagonal is removed before beamforming
    r_diag = Bool(True, 
        desc="removal of diagonal")
    
    # type of steering vectors
    steer = Trait('true level', 'true location', 'classic', 'inverse', 
                  desc="type of steering vectors used")
                  
    # flag, if true (default), the result is cached in h5 files
    cached = Bool(True, 
        desc="cached flag")
                  
    # hdf5 cache file
    h5f = Instance(tables.File, transient = True )
    
    # the result, sound pressure squared in all grid locations
    # as (number of frequencies, nxsteps, nysteps) array of float
    result = Property(
        desc="beamforming result")
        
    # sound travel distances from microphone array center to grid points
    r0 = Property(
        desc="array center to grid distances")

    # sound travel distances from array microphones to grid points
    rm = Property(
        desc="array center to grid distances")
    
    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', \
            'r_diag', 'env.digest', 'steer'], 
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'freq_data.ind_low', 'freq_data.ind_high'], 
        )

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('r_diag', label='diagonal removed')], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
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

    @property_depends_on('digest')
    def _get_r0 ( self ):
        return self.env.r( self.c, self.grid.pos())

    @property_depends_on('digest')
    def _get_rm ( self ):
        return self.env.r( self.c, self.grid.pos(), self.mpos.mpos)

    @property_depends_on('ext_digest')
    def _get_result ( self ):
        """
        beamforming result is either loaded or calculated
        """
        _digest = ''
        while self.digest != _digest:
            _digest = self.digest
            name = self.__class__.__name__ + self.digest
            #print 1, name
            numchannels = self.freq_data.time_data.numchannels
            #print "nch", numchannels
            if  numchannels != self.mpos.num_mics or numchannels == 0:
                #return None
                raise ValueError("%i channels do not fit %i mics" % \
                    (numchannels, self.mpos.num_mics))
            numfreq = self.freq_data.block_size/2 + 1
            if self.cached:
                H5cache.get_cache( self, self.freq_data.basename)
                if not name in self.h5f.root:
                    group = self.h5f.createGroup(self.h5f.root, name)
                    shape = (numfreq, self.grid.size)
                    atom = tables.Float32Atom()
                    #filters = tables.Filters(complevel=5, complib='zlib')
                    ac = self.h5f.createCArray(group, 'result', atom, shape)
                    shape = (numfreq, )
                    atom = tables.BoolAtom()
                    fr = self.h5f.createCArray(group, 'freqs', atom, shape)
                else:
                    ac = self.h5f.getNode('/'+name, 'result')
                    fr = self.h5f.getNode('/'+name, 'freqs')
                if not fr[self.freq_data.ind_low:self.freq_data.ind_high].all():
                    self.calc(ac, fr)                  
                    self.h5f.flush()
            else:
                ac = zeros((numfreq, self.grid.size), dtype=float32)
                fr = zeros(numfreq, dtype=int)
                self.calc(ac,fr)
            #print 2, name
        return ac
        
    def get_beamfunc( self, os='' ):
        """
        returns the proper low-level beamforming routine
        """
        r_diag = {True: 'diag', False: 'full'}[self.r_diag]
        steer = {'true level': '', \
                'true location': '_3d', \
                'classic': '_classic', \
                'inverse': '_inverse'}[self.steer]
        return eval('r_beam'+r_diag+os+steer)

    def calc(self, ac, fr):
        """
        calculation of delay-and-sum beamforming result 
        for all missing frequencies
        """
        # prepare calculation
        kj = 2j*pi*self.freq_data.fftfreq()/self.c
        numchannels = self.freq_data.time_data.numchannels
        e = zeros((numchannels), 'D')
        r0 = self.r0
        rm = self.rm
        h = zeros((1, self.grid.size), 'd')
        # function
        beamfunc = self.get_beamfunc()
        if self.r_diag:
            adiv = 1.0/(numchannels*numchannels-numchannels)
            scalefunc = lambda h : adiv*multiply(h, (sign(h)+1-1e-35)/2)
        else:
            adiv = 1.0/(numchannels*numchannels)
            scalefunc = lambda h : adiv*h
        for i in self.freq_data.indices:
            if not fr[i]:
                csm = array(self.freq_data.csm[i][newaxis], dtype='complex128')
                kji = kj[i, newaxis]
                beamfunc(csm, e, h, r0, rm, kji)
                ac[i] = scalefunc(h)
                fr[i] = True
    
    def synthetic( self, freq, num=0):
        """
        returns synthesized frequency band values of beamforming result
        num = 0: single frequency line
        num = 1: octave band
        num = 3: third octave band
        etc.
        """
        res = self.result # trigger calculation
        f = self.freq_data.fftfreq()
        if len(f) == 0:
            return None#array([[1, ], ], 'd')
        try:
            if num == 0:
                # single frequency line
                h = self.result[searchsorted(f, freq)]
            else:
                f1 = searchsorted(f, freq*2.**(-0.5/num))
                f2 = searchsorted(f, freq*2.**(0.5/num))
                if f1 == f2:
                    h = self.result[f1]
                else:
                    h = sum(self.result[f1:f2], 0)
            return h.reshape(self.grid.shape)
        except IndexError:
            return None

    def integrate(self, sector):
        """
        integrates result map over the given sector
        where sector is a tuple with arguments for grid.indices
        e.g. array([xmin, ymin, xmax, ymax]) or array([x, y, radius])
        resp. array([rmin, phimin, rmax, phimax]), array([r, phi, radius]).
        returns spectrum
        """
#        ind = self.grid.indices(*sector)
#        gshape = self.grid.shape
#        r = self.result
#        rshape = r.shape
#        mapshape = (rshape[0], ) + gshape
#        h = r[:].reshape(mapshape)[ (s_[:], ) + ind ]
#        return h.reshape(h.shape[0], prod(h.shape[1:])).sum(axis=1)
        ind = self.grid.indices(*sector)
        gshape = self.grid.shape
        r = self.result
        h = zeros(r.shape[0])
        for i in range(r.shape[0]):
            h[i] = r[i].reshape(gshape)[ind].sum()
        return h
            
class BeamformerCapon( BeamformerBase ):
    """
    beamforming using the minimum variance or Capon algorithm
    """
    # flag for main diagonal removal is set to False
    r_diag = Enum(False, 
        desc="removal of diagonal")

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
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
        e = zeros((numchannels), 'D')
        h = zeros((1, self.grid.size), 'd')
        beamfunc = self.get_beamfunc()
        for i in self.freq_data.indices:
            if not fr[i]:
                csm = array(linalg.inv(array(self.freq_data.csm[i], \
                        dtype='complex128')), order='C')[newaxis]
                print csm.flags
                kji = kj[i, newaxis]
                beamfunc(csm, e, h, self.r0, self.rm, kji)
                ac[i] = 1.0/h
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
        depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', \
            'r_diag', 'env.digest', 'na', 'steer'], 
        )

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='component no', style='text')], 
            [Item('r_diag', label='diagonal removed')], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
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
        if na < 0:
            na = max(nm + na, 0)
        return min(nm - 1, na)

    def calc(self, ac, fr):
        """
        calculation of eigenvalue beamforming result 
        for all missing frequencies
        """
        # prepare calculation
        kj = 2j*pi*self.freq_data.fftfreq()/self.c
        na = int(self.na)
        numchannels = self.freq_data.time_data.numchannels
        e = zeros((numchannels), 'D')
        h = empty((1, self.grid.size), 'd')
        # function
        beamfunc = self.get_beamfunc('_os')
        if self.r_diag:
            adiv = 1.0/(numchannels*numchannels-numchannels)
            scalefunc = lambda h : adiv*multiply(h, (sign(h)+1-1e-35)/2)
        else:
            adiv = 1.0/(numchannels*numchannels)
            scalefunc = lambda h : adiv*h
        for i in self.freq_data.indices:        
            if not fr[i]:
                eva = array(self.freq_data.eva[i][newaxis], dtype='float64')
                eve = array(self.freq_data.eve[i][newaxis], dtype='complex128')
                kji = kj[i, newaxis]
                beamfunc(e, h, self.r0, self.rm, kji, eva, eve, na, na+1)
                ac[i] = scalefunc(h)
                fr[i] = True

class BeamformerMusic( BeamformerEig ):
    """
    beamforming using MUSIC algoritm
    """

    # flag for main diagonal removal is set to False
    r_diag = Enum(False, 
        desc="removal of diagonal")

    # assumed number of sources, should be set to a value not too small
    # defaults to 1
    n = Int(1, 
        desc="assumed number of sources")

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='no of sources', style='text')], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
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
        e = zeros((numchannels), 'D')
        h = empty((1, self.grid.size), 'd')
        beamfunc = self.get_beamfunc('_os')
        # function
        for i in self.freq_data.indices:        
            if not fr[i]:
                eva = array(self.freq_data.eva[i][newaxis], dtype='float64')
                eve = array(self.freq_data.eve[i][newaxis], dtype='complex128')
                kji = kj[i, newaxis]
                beamfunc(e, h, self.r0, self.rm, kji, eva, eve, 0, n)
                ac[i] = 4e-10*h.min()/h
                fr[i] = True

class PointSpreadFunction (HasPrivateTraits):
    """
    Array point spread function
    """
    # RectGrid object that provides the grid locations
    grid = Trait(Grid, 
        desc="beamforming grid")

    # indices of grid points to calculate the PSF for
    grid_indices = CArray( dtype=int, value=array([]), 
                     desc="indices of grid points for psf") #value=array([]), value=self.grid.pos(),
    
    # MicGeom object that provides the microphone locations
    mpos = Trait(MicGeom, 
        desc="microphone geometry")

    # Environment object that provides speed of sound and grid-mic distances
    env = Trait(Environment(), Environment)

    # the speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")

    # type of steering vectors
    steer = Trait('true level', 'true location', 'classic', 'inverse', 
                  'old_version',
                  desc="type of steering vectors used")

    # how to calculate and store the psf
    calcmode = Trait('single', 'block', 'full', 'readonly',
                     desc="mode of calculation / storage")
              
    # frequency 
    freq = Float(1.0, 
        desc="frequency")
        
    # sound travel distances from microphone array center to grid points
    r0 = Property(
        desc="array center to grid distances")
    
    # sound travel distances from array microphones to grid points
    rm = Property(
        desc="array to grid distances")
        
    # the actual point spread function
    psf = Property(
        desc="point spread function")

    # hdf5 cache file
    h5f = Instance(tables.File, transient = True)
    
    # internal identifier
    digest = Property( depends_on = ['mpos.digest', 'grid.digest', 'c', \
             'env.digest', 'steer'], cached = True)

    @cached_property
    def _get_digest( self ):
        return digest( self )

    @property_depends_on('digest')
    def _get_r0 ( self ):
        return self.env.r( self.c, self.grid.pos())
    
    @property_depends_on('digest')
    def _get_rm ( self ):
        return self.env.r( self.c, self.grid.pos(), self.mpos.mpos)

    def get_beam_psf( self ):
        """
        returns the proper low-level beamforming routine
        """
        steer = {'true level': '3', \
                'true location': '4', \
                'classic': '1', \
                'inverse': '2'}[self.steer]
        return eval('r_beam_psf'+steer)
    
    
    @property_depends_on('digest, freq')
    def _get_psf ( self ):
        """
        point spread function is either calculated or loaded from cache
        """
        gs = self.grid.size
        if not self.grid_indices.size:
            self.grid_indices = arange(gs)
        name = 'psf' + self.digest
        H5cache.get_cache( self, name)
        fr = ('Hz_%.2f' % self.freq).replace('.', '_')
        
        # get the cached data, or, if non-existing, create new structure
        if not fr in self.h5f.root:
            if self.calcmode == 'readonly':
                raise ValueError('Cannot calculate missing PSF (freq %s) in \'readonly\' mode.' % fr)
            
            group = self.h5f.createGroup(self.h5f.root, fr) 
            
            shape = (gs, gs)
            atom = tables.Float64Atom()
            ac = self.h5f.createCArray(group, 'result', atom, shape)
            
            shape = (gs,)
            atom = tables.BoolAtom()
            gp = self.h5f.createCArray(group, 'gridpts', atom, shape)
            
        else:
            ac = self.h5f.getNode('/'+fr, 'result')
            gp = self.h5f.getNode('/'+fr, 'gridpts')
        
        # are there grid points for which the PSF hasn't been calculated yet?
        if not gp[:][self.grid_indices].all():

            if self.calcmode == 'readonly':
                raise ValueError('Cannot calculate missing PSF (points) in \'readonly\' mode.')

            elif self.calcmode != 'full':
                # calc_ind has the form [True, True, False, True], except
                # when it has only 1 entry (value True/1 would be ambiguous)
                if self.grid_indices.size == 1:
                    calc_ind = [0]
                else:
                    calc_ind = invert(gp[:][self.grid_indices])
                
                # get indices which have the value True = not yet calculated
                g_ind_calc = self.grid_indices[calc_ind]
            
            
            r0 = self.r0
            rm = self.rm
            kj = 2j*pi*self.freq/self.c
            

            r_beam_psf = self.get_beam_psf()
            #{ 
            #    'true level'   : r_beam_psf3(hh, r0, r0[ind], rm, rm[ind], kj),
            #    'true location': r_beam_psf4(hh, r0[ind], rm, rm[ind], kj),
            #    'classic'      : r_beam_psf1(hh, r0[ind], rm, rm[ind], kj),
            #    'inverse'      : r_beam_psf2(hh, r0, r0[ind], rm, rm[ind], kj)
            #    }

            
            if self.calcmode == 'single':
            
                hh = ones((gs, 1), 'd')

              
                for ind in g_ind_calc:
                    # hh = hh / hh[ind] #psf4 & 3
                    # psf: ['h','rt0','rs0','rtm','rsm','kj']
                    """    
                    else:
                        e = zeros((self.mpos.num_mics), 'D')
                        e1 = e.copy()
                        r_beam_psf(e, e1, hh, self.r0, self.rm, kj)
                        h_out = hh[0] / diag(hh[0])
                    """
                    r_beam_psf(hh, r0, r0[[ind]], rm, rm[[ind]], kj)
                    
                    ac[:,ind] = hh[:,0] / hh[ind,0]
                    gp[ind] = True
                
            elif self.calcmode == 'full':
                hh = ones((gs, gs), 'd')
                r_beam_psf(hh, r0, r0, rm, rm, kj)
                
                gp[:] = True
                ac[:] = hh / diag(hh)

            else: # 'block'
                hh = ones((gs, g_ind_calc.size), 'd')
                r_beam_psf(hh, r0, r0[g_ind_calc], rm, rm[g_ind_calc], kj)
                hh /= diag(hh[g_ind_calc,:])[newaxis,:]
                
                indh = 0
                for ind in g_ind_calc:
                    gp[ind] = True
                    ac[:,ind] = hh[:,indh]
                    indh += 1

                
            self.h5f.flush()
        return ac[:][:,self.grid_indices]

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
    
    # type of steering vectors
    steer =  Delegate('beamformer')

    # number of iterations
    n_iter = Int(100, 
        desc="number of iterations")

    # how to calculate and store the psf
    calcmode = Trait('full', 'single', 'block', 'readonly',
                     desc="mode of psf calculation / storage")
    
    # internal identifier
    digest = Property( 
        depends_on = ['beamformer.digest', 'n_iter'], 
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'beamformer.ext_digest'], 
        )
    
    traits_view = View(
        [
            [Item('beamformer{}', style='custom')], 
            [Item('n_iter{Number of iterations}')], 
            [Item('steer{Type of steering vector}')], 
            [Item('calcmode{How to calculate PSF}')], 
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
        p = PointSpreadFunction(mpos=self.mpos, grid=self.grid, 
                                c=self.c, env=self.env, steer=self.steer,
                                calcmode=self.calcmode)
        for i in self.freq_data.indices:        
            if not fr[i]:
                p.freq = freqs[i]
                y = array(self.beamformer.result[i], dtype=float64)
                x = y.copy()
                psf = p.psf[:]
                gseidel(psf, y, x, self.n_iter, 1.0)
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

    # type of steering vectors
    steer =  Delegate('beamformer')

    # environment
    env =  Delegate('beamformer')
    
    # list of components to consider
    eva_list = CArray(
        desc="components")
        
    # helper: number of components to consider
    n = Int(1)

    # internal identifier
    digest = Property( 
        depends_on = ['beamformer.digest', 'eva_list'], 
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'beamformer.ext_digest'], 
        )
    
    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='number of components', style='text')], 
            [Item('r_diag', label='diagonal removed')], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
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
        """ sets the list of eigenvalues to consider """
        self.eva_list = arange(-1, -1-self.n, -1)

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
                ac[i, e.result[i].argmax()]+=e.freq_data.eva[i, n]/numchannels
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
    damp = Range(0.01, 1.0, 0.6, 
        desc="damping factor")

    # iteration stop criterion for automatic detection
    # iteration stops if power[i]>power[i-stopn]
    # defaults to 3
    stopn = Int(3, 
        desc="stop criterion index")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', \
        'env.digest', 'n', 'damp', 'stopn', 'steer'], )

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='no of iterations', style='text')], 
            [Item('r_diag', label='diagonal removed')], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
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
        e = zeros((numchannels), 'D')
        result = zeros((self.grid.size), 'f')
        fullbeamfunc = self.get_beamfunc()
        orthbeamfunc = self.get_beamfunc('_os')
        if self.r_diag:
            adiv = 1.0/(numchannels*numchannels-numchannels)
        else:
            adiv = 1.0/(numchannels*numchannels)
        if not self.n:
            J = numchannels*2
        else:
            J = self.n
        powers = zeros(J, 'd')
        h = zeros((1, self.grid.size), 'd')
        h1 = h.copy()
        # loop over frequencies
        for i in self.freq_data.indices:        
            if not fr[i]:
                kj = kjall[i, newaxis]
                csm = array(self.freq_data.csm[i][newaxis], \
                    dtype='complex128', copy=1)
                fullbeamfunc(csm, e, h, self.r0, self.rm, kj)
                h = h*adiv
                # CLEANSC Iteration
                result *= 0.0
                for j in range(J):
                    xi_max = h.argmax() #index of maximum
                    powers[j] = hmax = h[0, xi_max] #maximum
                    result[xi_max] += self.damp * hmax
                    if  j > self.stopn and hmax > powers[j-self.stopn]:
                        break
                    rm = self.rm[xi_max]
                    r0 = self.r0[xi_max]
                    if self.steer == 'true level':
                        rs = rm*r0*(1/(rm*rm)).sum(0)
                    elif self.steer == 'true location':
                        rs = rm*sqrt((1/(rm*rm)).sum(0)*numchannels)
                    elif self.steer == 'classic':
                        rs = 1.0*numchannels
                    elif self.steer == 'inverse':
                        rs = numchannels*r0/rm
                    wmax = numchannels*sqrt(adiv)*exp(-kj[0]*(r0-rm))/rs
                    hh = wmax.copy()
                    D1 = dot(csm[0]-diag(diag(csm[0])), wmax)/hmax
                    ww = wmax.conj()*wmax
                    for m in range(20):
                        H = hh.conj()*hh
                        hh = (D1+H*wmax)/sqrt(1+dot(ww, H))
                    hh = hh[:, newaxis]
                    csm1 = hmax*(hh*hh.conj().T)[newaxis, :, :]
                    orthbeamfunc(e, h1, self.r0, self.rm, kj, \
                        array((hmax, ))[newaxis, :], hh[newaxis, :], 0, 1)
                    h -= self.damp*h1*adiv
                    csm -= self.damp*csm1
#                print '%i iter of %i' % (j,J)
                ac[i] = result
                fr[i] = True

class BeamformerClean (BeamformerBase):
    """
    CLEAN Deconvolution
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

    # type of steering vectors
    steer =  Delegate('beamformer')

    # flag, if true (default), the main diagonal is removed before beamforming
    #r_diag =  Delegate('beamformer')
    
    # iteration damping factor
    # defaults to 0.6
    damp = Range(0.01, 1.0, 0.6, 
        desc="damping factor")
        
    # max number of iterations
    n_iter = Int(100, 
        desc="maximum number of iterations")

    # how to calculate and store the psf
    calcmode = Trait('block', 'full', 'single', 'readonly',
                     desc="mode of psf calculation / storage")
                     
    # internal identifier
    digest = Property( 
        depends_on = ['beamformer.digest', 'n_iter', 'damp'], 
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'beamformer.ext_digest'], 
        )
    
    traits_view = View(
        [
            [Item('beamformer{}', style='custom')], 
            [Item('')], 
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
        calculation of CLEAN result 
        for all missing frequencies
        """
        freqs = self.freq_data.fftfreq()
        gs = self.grid.size
        
        if self.calcmode == 'full':
            print 'Warning: calcmode = \'full\', slow CLEAN performance. Better use \'block\' or \'single\'.'
        p = PointSpreadFunction(mpos=self.mpos, grid=self.grid, 
                                c=self.c, env=self.env, steer=self.steer,
                                calcmode=self.calcmode)
        
        for i in self.freq_data.indices:        
            if not fr[i]:
                
                p.freq = freqs[i]
                dirty = array(self.beamformer.result[i], dtype=float64)
                clean = zeros(gs, 'd')
                
                i_iter = 0
                flag = True
                while flag:
                    # TODO: negative werte!!!
                    dirty_sum = abs(dirty).sum(0)
                    next_max = dirty.argmax(0)
                    p.grid_indices = array([next_max])
                    psf = p.psf.reshape(gs,)
                    new_amp = self.damp * dirty[next_max] #/ psf[next_max]
                    clean[next_max] += new_amp
                    dirty -= psf * new_amp
                    i_iter += 1
                    flag = (dirty_sum > abs(dirty).sum(0) \
                            and i_iter < self.n_iter \
                            and max(dirty) > 0)
                #print freqs[i],'Hz, Iterations:',i_iter
                
                ac[i] = clean            
                fr[i] = True

class BeamformerCMF ( BeamformerBase ):
    """
    Covariance Matrix Fitting (Yardibi2008)
    (not really a beamformer, but an inverse method)
    """
    # type of fit method
    method = Trait('LassoLars', 'LassoLarsBIC', \
        'OMPCV', 'NNLS', desc="fit method used")
        
    # weight factor
    # defaults to 0.0
    alpha = Range(0.0, 1.0, 0.0, 
        desc="Lasso weight factor")
    
    # maximum number of iterations
    # tradeoff between speed and precision
    # defaults to 500
    max_iter = Int(500, 
        desc="maximum number of iterations")

    # internal identifier
    digest = Property( 
        depends_on = ['mpos.digest', 'grid.digest', 'freq_data.digest', 'c', \
            'alpha', 'method', 'max_iter', 'env.digest', 'steer', 'r_diag'], 
        )

    traits_view = View(
        [
            [Item('mpos{}', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            [Item('method', label='fit method')], 
            [Item('max_iter', label='max iterations')], 
            [Item('alpha', label='Lasso weight factor')], 
            [Item('c', label='speed of sound')], 
            [Item('env{}', style='custom')], 
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
        calculation of delay-and-sum beamforming result 
        for all missing frequencies
        """
        def realify(M):
            return vstack([M.real,M.imag])

            
        # prepare calculation
        kj = 2j*pi*self.freq_data.fftfreq()/self.c
        nc = self.freq_data.time_data.numchannels
        r0 = self.r0
        rm = self.rm
        numpoints = rm.shape[0]

        hh = zeros((1, numpoints, nc), dtype='D')

            
        for i in self.freq_data.indices:
            if not fr[i]:
                # csm transposed b/c indices switched in faverage!
                csm = array(self.freq_data.csm[i], dtype='complex128',copy=1).T

                kji = kj[i, newaxis]
                transfer(hh, r0, rm, kji)
                h = hh[0].T
                
                # reduced Kronecker product (only where solution matrix != 0)
                Bc = ( h[:,:,newaxis] * \
                       h.conjugate().T[newaxis,:,:] )\
                         .transpose(2,0,1)
                Ac = Bc.reshape(nc*nc,numpoints)
                
                # get indices for upper triangular matrices (use tril b/c transposed)
                ind = reshape(tril(ones((nc,nc))), (nc*nc,)) > 0
                
                ind_im0 = (reshape(eye(nc),(nc*nc,)) == 0)[ind]
                if self.r_diag:
                    # omit main diagonal for noise reduction
                    ind_reim = hstack([ind_im0, ind_im0])
                else:
                    # take all real parts -- also main diagonal
                    ind_reim = hstack([ones(size(ind_im0),)>0,ind_im0])
                    ind_reim[0]=True # TODO: warum hier extra definiert??
#                    if sigma2:
#                        # identity matrix, needed when noise term sigma is used
#                        I  = eye(nc).reshape(nc*nc,1)                
#                        A = realify( hstack([Ac, I])[ind,:] )[ind_reim,:]
#                        # ... ac[i] = model.coef_[:-1]
#                    else:

                A = realify( Ac [ind,:] )[ind_reim,:]
                # use csm.T for column stacking reshape!
                R = realify( reshape(csm.T, (nc*nc,1))[ind,:] )[ind_reim,:]
#                print A.shape, R.shape
                # choose method
                if self.method == 'LassoLars':
                    model = LassoLars(alpha=self.alpha,max_iter=self.max_iter)
                elif self.method == 'LassoLarsBIC':
                    model = LassoLarsIC(criterion='bic',max_iter=self.max_iter)
                elif self.method == 'OMPCV':
                    model = OrthogonalMatchingPursuitCV()
#                model = ElasticNet(alpha=self.alpha, l1_ratio=0.7)
                # nnls is not in sklearn
                if self.method == 'NNLS':
                    ac[i] , x = nnls(A,R.flat)
                else:
                    model.fit(A,R[:,0])
                    ac[i] = model.coef_[:]
                fr[i] = True


def L_p ( x ):
    """
    calculates the sound pressure level from the sound pressure squared:

    L_p = 10 lg x/4e-10

    if x<0, return -350. dB
    """
    # new version to prevent division by zero warning for float32 arguments
    return 10*log10(clip(x/4e-10,1e-35,None))
#    return where(x>0, 10*log10(x/4e-10), -1000.)

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

def integrate(data, grid, sector):
    """
    integrates result map over the given sector
    where sector is a tuple with arguments for grid.indices
    e.g. array([xmin, ymin, xmax, ymax]) or array([x, y, radius])
    resp. array([rmin, phimin, rmax, phimax]), array([r, phi, radius]).
    returns spectrum
    """
    ind = grid.indices(*sector)
    gshape = grid.shape
    h = data.reshape(gshape)[ind].sum()
    return h
