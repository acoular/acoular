# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2017, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements beamformers in the frequency domain.

.. autosummary::
    :toctree: generated/
    
    SteeringVector

    BeamformerBase
    BeamformerFunctional
    BeamformerCapon
    BeamformerEig
    BeamformerMusic
    BeamformerClean
    BeamformerDamas
    BeamformerDamasPlus
    BeamformerOrth
    BeamformerCleansc
    BeamformerCMF
    BeamformerGIB

    PointSpreadFunction
    L_p
    integrate

"""

# imports from other packages
from __future__ import print_function, division

from numpy import array, ones, full, hanning, hamming, bartlett, blackman, \
invert, dot, newaxis, zeros, empty, fft, float32, float64, complex64, linalg, \
where, searchsorted, pi, multiply, sign, diag, arange, sqrt, exp, log10, int,\
reshape, hstack, vstack, eye, tril, size, clip, tile, round, delete, \
absolute, argsort, sort, sum, hsplit, fill_diagonal, zeros_like, isclose, \
vdot, flatnonzero, einsum, ndarray, isscalar

from sklearn.linear_model import LassoLars, LassoLarsCV, LassoLarsIC,\
OrthogonalMatchingPursuit, ElasticNet, OrthogonalMatchingPursuitCV, Lasso

from scipy.optimize import nnls, linprog
from scipy.linalg import inv, eigh, eigvals, fractional_matrix_power
from warnings import warn

import tables
from traits.api import HasPrivateTraits, Float, Int, ListInt, ListFloat, \
CArray, Property, Instance, Trait, Bool, Range, Delegate, Enum, Any, \
cached_property, on_trait_change, property_depends_on
from traits.trait_errors import TraitError

from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons

from .fastFuncs import beamformerFreq, calcTransfer, calcPointSpreadFunction, \
damasSolverGaussSeidel

from .h5cache import H5cache
from .internal import digest
from .grids import Grid
from .microphones import MicGeom
from .environments import Environment
from .spectra import PowerSpectra, _precision


class SteeringVector( HasPrivateTraits ):
    """ 
    Basic class for implementing steering vectors with monopole source transfer models
    """
    
    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    grid = Trait(Grid, 
        desc="beamforming grid")
    
    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Trait(MicGeom, 
        desc="microphone geometry")
        
    #: Type of steering vectors, see also :ref:`Sarradj, 2012<Sarradj2012>`. Defaults to 'true level'.
    steer_type = Trait('true level', 'true location', 'classic', 'inverse',
                  desc="type of steering vectors used")
    
    #: :class:`~acoular.environments.Environment` or derived object, 
    #: which provides information about the sound propagation in the medium.
    #: Defaults to standard :class:`~acoular.environments.Environment` object.
    env = Instance(Environment(), Environment)
    
    # TODO: add caching capability for transfer function
    # Flag, if "True" (not default), the transfer function is 
    # cached in h5 files and does not have to be recomputed during subsequent 
    # program runs. 
    # Be aware that setting this to "True" may result in high memory usage.
    #cached = Bool(False, 
    #              desc="cache flag for transfer function")    
    
    
    # Sound travel distances from microphone array center to grid 
    # points or reference position (readonly). Feature may change.
    r0 = Property(desc="array center to grid distances")

    # Sound travel distances from array microphones to grid 
    # points (readonly). Feature may change.
    rm = Property(desc="all array mics to grid distances")
    
    # mirror trait for ref
    _ref = Any(array([0.,0.,0.]),
               desc="reference position or distance")
    
    #: Reference position or distance at which to evaluate the sound pressure 
    #: of a grid point. 
    #: If set to a scalar, this is used as reference distance to the grid points.
    #: If set to a vector, this is interpreted as x,y,z coordinates of the reference position.
    #: Defaults to [0.,0.,0.].
    ref = Property(desc="reference position or distance")
    
    def _set_ref (self, ref):
        if isscalar(ref):
            try:
                self._ref = absolute(float(ref))
            except:
                raise TraitError(args=self,
                                 name='ref', 
                                 info='Float or CArray(3,)',
                                 value=ref) 
        elif len(ref) == 3:
            self._ref = array(ref, dtype=float)
        else:
            raise TraitError(args=self,
                             name='ref', 
                             info='Float or CArray(3,)',
                             value=ref)
      
    def _get_ref (self):
        return self._ref
    
    
    # internal identifier
    digest = Property( 
        depends_on = ['steer_type', 'env.digest', 'grid.digest', 'mics.digest', '_ref'])
    
    # internal identifier, use for inverse methods, excluding steering vector type
    inv_digest = Property( 
        depends_on = ['env.digest', 'grid.digest', 'mics.digest', '_ref'])
        
    @property_depends_on('grid.digest, env.digest, _ref')
    def _get_r0 ( self ):
        if isscalar(self.ref) and self.ref > 0:
            return full((self.grid.size,), self.ref)
        else:
            return self.env._r(self.grid.pos())

    @property_depends_on('grid.digest, mics.digest, env.digest')
    def _get_rm ( self ):
        return self.env._r(self.grid.pos(), self.mics.mpos)
 
    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @cached_property
    def _get_inv_digest( self ):
        return digest( self )
    
    def transfer(self, f, ind=None):
        """
        Calculates the transfer matrix for one frequency. 
        
        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the transfer function of the gridpoints addressed by 
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed
        
        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the transfer matrix for the given frequency
        """
        #if self.cached:
        #    warn('Caching of transfer function is not yet supported!', Warning)
        #    self.cached = False
        
        if ind is None:
            trans = calcTransfer(self.r0, self.rm, array(2*pi*f/self.env.c))
        elif not isinstance(ind,ndarray):
            trans = calcTransfer(self.r0[ind], self.rm[ind, :][newaxis], array(2*pi*f/self.env.c))#[0, :]
        else:
            trans = calcTransfer(self.r0[ind], self.rm[ind, :], array(2*pi*f/self.env.c))
        return trans
    
    def steer_vector(self, f, ind=None):
        """
        Calculates the steering vectors based on the transfer function
        See also :ref:`Sarradj, 2012<Sarradj2012>`.
        
        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the steering vectors of the gridpoints addressed by 
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed
        
        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the steering vectors for the given frequency
        """
        func = {'classic' : lambda x: x / absolute(x) / x.shape[-1],
                'inverse' : lambda x: 1. / x.conj() / x.shape[-1],
                'true level' : lambda x: x / einsum('ij,ij->i',x,x.conj())[:,newaxis],
                'true location' : lambda x: x / sqrt(einsum('ij,ij->i',x,x.conj()) * x.shape[-1])[:,newaxis]
                }[self.steer_type]
        return func(self.transfer(f, ind))
    
    
class BeamformerBase( HasPrivateTraits ):
    """
    Beamforming using the basic delay-and-sum algorithm in the frequency domain.
    """
    
    
    # Instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes
    # that contains information about the steering vector. This is a private trait.
    # Do not set this directly, use `steer` trait instead.
    _steer_obj = Instance(SteeringVector(), SteeringVector)   
    
    #: :class:`~acoular.fbeamform.SteeringVector` or derived object. 
    #: Defaults to :class:`~acoular.fbeamform.SteeringVector` object.
    steer = Property(desc="steering vector object")  
    
    def _get_steer(self):
        return self._steer_obj
    
    def _set_steer(self, steer):
        if isinstance(steer, SteeringVector):
            self._steer_obj = steer
        elif steer in ('true level', 'true location', 'classic', 'inverse'):
            # Type of steering vectors, see also :ref:`Sarradj, 2012<Sarradj2012>`.
            warn("Deprecated use of 'steer' trait. "
                 "Please use object of class 'SteeringVector' in the future.", 
                 Warning, stacklevel = 2)
            self._steer_obj = SteeringVector(steer_type = steer)
        else:
            raise(TraitError(args=self,
                             name='steer', 
                             info='SteeringVector',
                             value=steer))

    # --- List of backwards compatibility traits and their setters/getters -----------
    
    # :class:`~acoular.environments.Environment` or derived object. 
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    env = Property()
    
    def _get_env(self):
        return self._steer_obj.env    
    
    def _set_env(self, env):
        warn("Deprecated use of 'env' trait. ", Warning, stacklevel = 2)
        self._steer_obj.env = env
    
    # The speed of sound.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    c = Property()
    
    def _get_c(self):
        return self._steer_obj.env.c
    
    def _set_c(self, c):
        warn("Deprecated use of 'c' trait. ", Warning, stacklevel = 2)
        self._steer_obj.env.c = c
   
    # :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    grid = Property()

    def _get_grid(self):
        return self._steer_obj.grid
    
    def _set_grid(self, grid):
        warn("Deprecated use of 'grid' trait. ", Warning, stacklevel = 2)
        self._steer_obj.grid = grid
    
    # :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait
    mpos = Property()
    
    def _get_mpos(self):
        return self._steer_obj.mics
    
    def _set_mpos(self, mpos):
        warn("Deprecated use of 'mpos' trait. ", Warning, stacklevel = 2)
        self._steer_obj.mics = mpos
    
    
    # Sound travel distances from microphone array center to grid points (r0)
    # and all array mics to grid points (rm). Readonly.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait
    r0 = Property()
    def _get_r0(self):
        return self._steer_obj.r0
    
    rm = Property()
    def _get_rm(self):
        return self._steer_obj.rm
    
    # --- End of backwards compatibility traits --------------------------------------

    #: :class:`~acoular.spectra.PowerSpectra` object that provides the 
    #: cross spectral matrix and eigenvalues
    freq_data = Trait(PowerSpectra, 
                      desc="freq data object")

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, 
                  desc="removal of diagonal")
    
    #: If r_diag==True: if r_diag_norm==0.0, the standard  
    #: normalization = num_mics/(num_mics-1) is used. 
    #: If r_diag_norm !=0.0, the user input is used instead.  
    #: If r_diag==False, the normalization is 1.0 either way. 
    r_diag_norm = Float(0.0, 
                        desc="If diagonal of the csm is removed, some signal energy is lost." 
                        "This is handled via this normalization factor." 
                        "Internally, the default is: num_mics / (num_mics - 1).") 
    
    #: Floating point precision of property result. Corresponding to numpy dtypes. Default = 64 Bit.
    precision = Trait('float64', 'float32',
            desc="precision (32/64 Bit) of result, corresponding to numpy dtypes")
    
    #: Boolean flag, if 'True' (default), the result is cached in h5 files.
    cached = Bool(True, 
        desc="cached flag")
                  
    # hdf5 cache file
    h5f = Instance(tables.File, transient = True )
    
    #: The beamforming result as squared sound pressure values 
    #: at all grid point locations (readonly).
    #: Returns a (number of frequencies, number of gridpoints) array of floats.
    result = Property(
        desc="beamforming result")
    
    # internal identifier
    digest = Property( 
        depends_on = ['freq_data.digest', 'r_diag', 'r_diag_norm', 'precision', '_steer_obj.digest'])

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'freq_data.ind_low', 'freq_data.ind_high'], 
        )

    traits_view = View(
        [
            [Item('r_diag', label='Diagonal removed')], 
            [Item('steer', label='Steering vector')], 
#            [Item('env{}', style='custom')], 
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
        This is the :attr:`result` getter routine.
        The beamforming result is either loaded or calculated.
        """
        _digest = ''
        while self.digest != _digest:
            _digest = self.digest
            name = self.__class__.__name__ + self.digest
            numchannels = self.freq_data.numchannels
            if  numchannels != self.steer.mics.num_mics or numchannels == 0:
                raise ValueError("%i channels do not fit %i mics" % (numchannels, self.steer.mics.num_mics))
            numfreq = self.freq_data.fftfreq().shape[0]# block_size/2 + 1steer_obj
            precisionTuple = _precision(self.precision)
            if self.cached:
                H5cache.get_cache( self, self.freq_data.basename)
                if not name in self.h5f.root:
                    group = self.h5f.create_group(self.h5f.root, name)
                    shape = (numfreq, self.steer.grid.size)
                    atom = precisionTuple[3]()
                    filters = tables.Filters(complevel=5, complib='blosc')
                    ac = self.h5f.create_carray(group, 'result', atom, shape, filters=filters)
                    shape = (numfreq, )
                    atom = tables.BoolAtom()
                    fr = self.h5f.create_carray(group, 'freqs', atom, shape, filters=filters)
                else:
                    ac = self.h5f.get_node('/'+name, 'result')
                    fr = self.h5f.get_node('/'+name, 'freqs')
                if not fr[self.freq_data.ind_low:self.freq_data.ind_high].all():
                    self.calc(ac, fr)                  
                    self.h5f.flush()
            else:
                ac = zeros((numfreq, self.steer.grid.size), dtype=self.precision)
                fr = zeros(numfreq, dtype='int64')
                self.calc(ac,fr)
        return ac
        
    def sig_loss_norm(self):
        """ 
        If the diagonal of the CSM is removed one has to handle the loss 
        of signal energy --> Done via a normalization factor.
        """
        if not self.r_diag:  # Full CSM --> no normalization needed 
            normFactor = 1.0 
        elif self.r_diag_norm == 0.0:  # Removed diag: standard normalization factor 
            nMics = float(self.freq_data.numchannels) 
            normFactor = nMics / (nMics - 1) 
        elif self.r_diag_norm != 0.0:  # Removed diag: user defined normalization factor 
            normFactor = self.r_diag_norm 
        return normFactor


    def _beamformer_params(self):
        """
        Manages the parameters for calling of the core beamformer functionality.
        This is a workaround to allow faster calculation and may change in the
        future.
        
        Returns
        -------
            - String containing the steering vector type
            - Function for frequency-dependent steering vector calculation
                
        """
        if type(self.steer) == SteeringVector: # for simple steering vector, use faster method
            param_type = self.steer.steer_type
            def param_steer_func(f): return (self.steer.r0, self.steer.rm, 2*pi*f/self.steer.env.c )
        else:
            param_type = 'custom'
            param_steer_func = self.steer.steer_vector
        return param_type, param_steer_func

    def calc(self, ac, fr):
        """
        Calculates the delay-and-sum beamforming result for the frequencies 
        defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`result` or calling
        its :meth:`synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        f = self.freq_data.fftfreq()#[inds]
        param_steer_type, steer_vector = self._beamformer_params()
        for i in self.freq_data.indices:
            if not fr[i]:
                csm = array(self.freq_data.csm[i], dtype='complex128')
                beamformerOutput = beamformerFreq(param_steer_type, 
                                                  self.r_diag, 
                                                  self.sig_loss_norm(), 
                                                  steer_vector(f[i]), 
                                                  csm)[0]
                if self.r_diag:  # set (unphysical) negative output values to 0
                    indNegSign = sign(beamformerOutput) < 0
                    beamformerOutput[indNegSign] = 0.0
                ac[i] = beamformerOutput
                fr[i] = True
    
    def synthetic( self, f, num=0):
        """
        Evaluates the beamforming result for an arbitrary frequency band.
        
        Parameters
        ----------
        f: float
            Band center frequency. 
        num : integer
            Controls the width of the frequency bands considered; defaults to
            0 (single frequency line).
            
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
            The synthesized frequency band values of the beamforming result at 
            each grid point .
            Note that the frequency resolution and therefore the bandwidth 
            represented by a single frequency line depends on 
            the :attr:`sampling frequency<acoular.sources.SamplesGenerator.sample_freq>` and 
            used :attr:`FFT block size<acoular.spectra.PowerSpectra.block_size>`.
        """
        res = self.result # trigger calculation
        freq = self.freq_data.fftfreq()
        if len(freq) == 0:
            return None
        
        indices = self.freq_data.indices
        
        if num == 0:
            # single frequency line
            ind = searchsorted(freq, f)
            if ind >= len(freq):
                warn('Queried frequency (%g Hz) not in resolved '
                              'frequency range. Returning zeros.' % f, 
                              Warning, stacklevel = 2)
                h = zeros_like(res[0])
            else:
                if freq[ind] != f:
                    warn('Queried frequency (%g Hz) not in set of '
                         'discrete FFT sample frequencies. '
                         'Using frequency %g Hz instead.' % (f,freq[ind]), 
                         Warning, stacklevel = 2)
                if not (ind in indices):
                    warn('Beamforming result may not have been calculated '
                         'for queried frequency. Check '
                         'freq_data.ind_low and freq_data.ind_high!',
                          Warning, stacklevel = 2)
                h = res[ind]
        else:
            # fractional octave band
            f1 = f*2.**(-0.5/num)
            f2 = f*2.**(+0.5/num)
            ind1 = searchsorted(freq, f1)
            ind2 = searchsorted(freq, f2)
            if ind1 == ind2:
                warn('Queried frequency band (%g to %g Hz) does not '
                     'include any discrete FFT sample frequencies. '
                     'Returning zeros.' % (f1,f2), 
                     Warning, stacklevel = 2)
                h = zeros_like(res[0])
            else:
                h = sum(res[ind1:ind2], 0)
                if not ((ind1 in indices) and (ind2 in indices)):
                    warn('Beamforming result may not have been calculated '
                         'for all queried frequencies. Check '
                         'freq_data.ind_low and freq_data.ind_high!',
                          Warning, stacklevel = 2)
        return h.reshape(self.steer.grid.shape)


    def integrate(self, sector):
        """
        Integrates result map over a given sector.
        
        Parameters
        ----------
        sector: array of floats
              Tuple with arguments for the 'indices' method 
              of a :class:`~acoular.grids.Grid`-derived class 
              (e.g. :meth:`RectGrid.indices<acoular.grids.RectGrid.indices>` 
              or :meth:`RectGrid3D.indices<acoular.grids.RectGrid3D.indices>`).
              Possible sectors would be *array([xmin, ymin, xmax, ymax])* 
              or *array([x, y, radius])*.
              
        Returns
        -------
        array of floats
            The spectrum (all calculated frequency bands) for the integrated sector.
        """
        #resp. array([rmin, phimin, rmax, phimax]), array([r, phi, radius]).
        
#        ind = self.grid.indices(*sector)
#        gshape = self.grid.shape
#        r = self.result
#        rshape = r.shape
#        mapshape = (rshape[0], ) + gshape
#        h = r[:].reshape(mapshape)[ (s_[:], ) + ind ]
#        return h.reshape(h.shape[0], prod(h.shape[1:])).sum(axis=1)
        ind = self.steer.grid.indices(*sector)
        gshape = self.steer.grid.shape
        r = self.result
        h = zeros(r.shape[0])
        for i in range(r.shape[0]):
            h[i] = r[i].reshape(gshape)[ind].sum()
        return h

class BeamformerFunctional( BeamformerBase ):
    """
    Functional beamforming after :ref:`Dougherty, 2014<Dougherty2014>`.
    """

    #: Functional exponent, defaults to 1 (= Classic Beamforming).
    gamma = Float(1, 
        desc="functional exponent")

    # internal identifier
    digest = Property(depends_on = ['freq_data.digest', '_steer_obj.digest', 'r_diag', 'gamma'])
    
    #: Functional Beamforming is only well defined for full CSM
    r_diag = Enum(False, 
                  desc="False, as Functional Beamformer is only well defined for the full CSM")

    traits_view = View(
        [
#            [Item('mics{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('gamma', label='Exponent', style='simple')], 
#            [Item('env{}', style='custom')], 
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
        Calculates the Functional Beamformer result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        f = self.freq_data.fftfreq()
        normFactor = self.sig_loss_norm()
        param_steer_type, steer_vector = self._beamformer_params()
        for i in self.freq_data.indices:
            if not fr[i]:
                if self.r_diag:
                    # This case is not used at the moment (see Trait r_diag)  
                    # It would need some testing as structural changes were not tested...
#==============================================================================
#                     One cannot use spectral decomposition when diagonal of csm is removed,
#                     as the resulting modified eigenvectors are not orthogonal to each other anymore.
#                     Therefor potentiating cannot be applied only to the eigenvalues.
#                     --> To avoid this the root of the csm (removed diag) is calculated directly.
#                     WATCH OUT: This doesn't really produce good results.
#==============================================================================
                    csm = self.freq_data.csm[i]
                    fill_diagonal(csm, 0)
                    csmRoot = fractional_matrix_power(csm, 1.0 / self.gamma)
                    beamformerOutput, steerNorm = beamformerFreq(param_steer_type, 
                                                                 self.r_diag, 
                                                                 1.0, 
                                                                 steer_vector(f[i]), 
                                                                 csmRoot)
                    beamformerOutput /= steerNorm  # take normalized steering vec
                    
                    # set (unphysical) negative output values to 0
                    indNegSign = sign(beamformerOutput) < 0
                    beamformerOutput[indNegSign] = 0.0
                else:
                    eva = array(self.freq_data.eva[i], dtype='float64') ** (1.0 / self.gamma)
                    eve = array(self.freq_data.eve[i], dtype='complex128')
                    beamformerOutput, steerNorm = beamformerFreq(param_steer_type, 
                                                                 self.r_diag, 
                                                                 1.0, 
                                                                 steer_vector(f[i]), 
                                                                 (eva, eve))
                    beamformerOutput /= steerNorm  # take normalized steering vec
                ac[i] = (beamformerOutput ** self.gamma) * steerNorm * normFactor  # the normalization must be done outside the beamformer
                fr[i] = True
            
class BeamformerCapon( BeamformerBase ):
    """
    Beamforming using the Capon (Mininimum Variance) algorithm, 
    see :ref:`Capon, 1969<Capon1969>`.
    """
    # Boolean flag, if 'True', the main diagonal is removed before beamforming;
    # for Capon beamforming r_diag is set to 'False'.
    r_diag = Enum(False, 
        desc="removal of diagonal")

    traits_view = View(
        [
#            [Item('mics{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
#            [Item('env{}', style='custom')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )

    def calc(self, ac, fr):
        """
        Calculates the Capon result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """        
        f = self.freq_data.fftfreq()
        nMics = self.freq_data.numchannels
        normFactor = self.sig_loss_norm() * nMics**2
        param_steer_type, steer_vector = self._beamformer_params()
        for i in self.freq_data.indices:
            if not fr[i]:
                csm = array(linalg.inv(array(self.freq_data.csm[i], dtype='complex128')), order='C')
                beamformerOutput = beamformerFreq(param_steer_type, 
                                                  self.r_diag, 
                                                  normFactor, 
                                                  steer_vector(f[i]), 
                                                  csm)[0]
                ac[i] = 1.0 / beamformerOutput
                fr[i] = True

class BeamformerEig( BeamformerBase ):
    """
    Beamforming using eigenvalue and eigenvector techniques,
    see :ref:`Sarradj et al., 2005<Sarradj2005>`.
    """
    #: Number of component to calculate: 
    #: 0 (smallest) ... :attr:`~acoular.sources.SamplesGenerator.numchannels`-1;
    #: defaults to -1, i.e. numchannels-1
    n = Int(-1, 
        desc="No. of eigenvalue")

    # Actual component to calculate, internal, readonly.
    na = Property(
        desc="No. of eigenvalue")

    # internal identifier
    digest = Property( 
        depends_on = ['freq_data.digest', '_steer_obj.digest', 'r_diag', 'n'])

    traits_view = View(
        [
#            [Item('mics{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='Component No.', style='simple')], 
            [Item('r_diag', label='Diagonal removed')], 
#            [Item('env{}', style='custom')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )
    
    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @property_depends_on('steer.mics, n')
    def _get_na( self ):
        na = self.n
        nm = self.steer.mics.num_mics
        if na < 0:
            na = max(nm + na, 0)
        return min(nm - 1, na)

    def calc(self, ac, fr):
        """
        Calculates the result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        f = self.freq_data.fftfreq()
        na = int(self.na)  # eigenvalue taken into account
        normFactor = self.sig_loss_norm()
        param_steer_type, steer_vector = self._beamformer_params()
        for i in self.freq_data.indices:
            if not fr[i]:
                eva = array(self.freq_data.eva[i], dtype='float64')
                eve = array(self.freq_data.eve[i], dtype='complex128')
                beamformerOutput = beamformerFreq(param_steer_type, 
                                                  self.r_diag, 
                                                  normFactor, 
                                                  steer_vector(f[i]), 
                                                  (eva[na:na+1], eve[:, na:na+1]))[0]
                if self.r_diag:  # set (unphysical) negative output values to 0
                    indNegSign = sign(beamformerOutput) < 0
                    beamformerOutput[indNegSign] = 0
                ac[i] = beamformerOutput
                fr[i] = True

class BeamformerMusic( BeamformerEig ):
    """
    Beamforming using the MUSIC algorithm, see :ref:`Schmidt, 1986<Schmidt1986>`.
    """

    # Boolean flag, if 'True', the main diagonal is removed before beamforming;
    # for MUSIC beamforming r_diag is set to 'False'.
    r_diag = Enum(False, 
        desc="removal of diagonal")

    # assumed number of sources, should be set to a value not too small
    # defaults to 1
    n = Int(1, 
        desc="assumed number of sources")

    traits_view = View(
        [
#            [Item('mics{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='No. of sources', style='simple')], 
#            [Item('env{}', style='custom')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )

    def calc(self, ac, fr):
        """
        Calculates the MUSIC result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        f = self.freq_data.fftfreq()
        nMics = self.freq_data.numchannels
        n = int(self.steer.mics.num_mics-self.na)
        normFactor = self.sig_loss_norm() * nMics**2
        param_steer_type, steer_vector = self._beamformer_params()
        for i in self.freq_data.indices:
            if not fr[i]:
                eva = array(self.freq_data.eva[i], dtype='float64')
                eve = array(self.freq_data.eve[i], dtype='complex128')
                beamformerOutput = beamformerFreq(param_steer_type, 
                                                  self.r_diag, 
                                                  normFactor, 
                                                  steer_vector(f[i]), 
                                                  (eva[:n], eve[:, :n]))[0]
                ac[i] = 4e-10*beamformerOutput.min() / beamformerOutput
                fr[i] = True

class PointSpreadFunction (HasPrivateTraits):
    """
    The point spread function.
    
    This class provides tools to calculate the PSF depending on the used 
    microphone geometry, focus grid, flow environment, etc.
    The PSF is needed by several deconvolution algorithms to correct
    the aberrations when using simple delay-and-sum beamforming.
    """
    
        # Instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes
    # that contains information about the steering vector. This is a private trait.
    # Do not set this directly, use `steer` trait instead.
    _steer_obj = Instance(SteeringVector(), SteeringVector)   
    
    #: :class:`~acoular.fbeamform.SteeringVector` or derived object. 
    #: Defaults to :class:`~acoular.fbeamform.SteeringVector` object.
    steer = Property(desc="steering vector object")  
    
    def _get_steer(self):
        return self._steer_obj
    
    def _set_steer(self, steer):
        if isinstance(steer, SteeringVector):
            self._steer_obj = steer
        elif steer in ('true level', 'true location', 'classic', 'inverse'):
            # Type of steering vectors, see also :ref:`Sarradj, 2012<Sarradj2012>`.
            warn("Deprecated use of 'steer' trait. "
                 "Please use object of class 'SteeringVector' in the future.", 
                 Warning, stacklevel = 2)
            self._steer_obj = SteeringVector(steer_type = steer)
        else:
            raise(TraitError(args=self,
                             name='steer', 
                             info='SteeringVector',
                             value=steer))

    # --- List of backwards compatibility traits and their setters/getters -----------
    
    # :class:`~acoular.environments.Environment` or derived object. 
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    env = Property()
    
    def _get_env(self):
        return self._steer_obj.env    
    
    def _set_env(self, env):
        warn("Deprecated use of 'env' trait. ", Warning, stacklevel = 2)
        self._steer_obj.env = env
    
    # The speed of sound.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    c = Property()
    
    def _get_c(self):
        return self._steer_obj.env.c
    
    def _set_c(self, c):
        warn("Deprecated use of 'c' trait. ", Warning, stacklevel = 2)
        self._steer_obj.env.c = c
   
    # :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait.
    grid = Property()

    def _get_grid(self):
        return self._steer_obj.grid
    
    def _set_grid(self, grid):
        warn("Deprecated use of 'grid' trait. ", Warning, stacklevel = 2)
        self._steer_obj.grid = grid
    
    # :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait
    mpos = Property()
    
    def _get_mpos(self):
        return self._steer_obj.mics
    
    def _set_mpos(self, mpos):
        warn("Deprecated use of 'mpos' trait. ", Warning, stacklevel = 2)
        self._steer_obj.mics = mpos
    
    
    # Sound travel distances from microphone array center to grid points (r0)
    # and all array mics to grid points (rm). Readonly.
    # Deprecated! Only kept for backwards compatibility. 
    # Now governed by :attr:`steer` trait
    r0 = Property()
    def _get_r0(self):
        return self._steer_obj.r0
    
    rm = Property()
    def _get_rm(self):
        return self._steer_obj.rm
    
    # --- End of backwards compatibility traits --------------------------------------
    
    
    #: Indices of grid points to calculate the PSF for.
    grid_indices = CArray( dtype=int, value=array([]), 
                     desc="indices of grid points for psf") #value=array([]), value=self.grid.pos(),
    
    #: Flag that defines how to calculate and store the point spread function
    #: defaults to 'single'.
    #:
    #: * 'full': Calculate the full PSF (for all grid points) in one go (should be used if the PSF at all grid points is needed, as with :class:`DAMAS<BeamformerDamas>`)
    #: * 'single': Calculate the PSF for the grid points defined by :attr:`grid_indices`, one by one (useful if not all PSFs are needed, as with :class:`CLEAN<BeamformerClean>`)
    #: * 'block': Calculate the PSF for the grid points defined by :attr:`grid_indices`, in one go (useful if not all PSFs are needed, as with :class:`CLEAN<BeamformerClean>`)
    #: * 'readonly': Do not attempt to calculate the PSF since it should already be cached (useful if multiple processes have to access the cache file)
    calcmode = Trait('single', 'block', 'full', 'readonly',
                     desc="mode of calculation / storage")
              
    #: Floating point precision of property psf. Corresponding to numpy dtypes. Default = 64 Bit.
    precision = Trait('float64', 'float32',
            desc="precision (32/64 Bit) of result, corresponding to numpy dtypes")
    
    #: The actual point spread function.
    psf = Property(desc="point spread function")
    
    #: Frequency to evaluate the PSF for; defaults to 1.0. 
    freq = Float(1.0, desc="frequency")

    # hdf5 cache file
    h5f = Instance(tables.File, transient = True)
    
    # internal identifier
    digest = Property( depends_on = ['_steer_obj.digest', 'precision'], cached = True)

    @cached_property
    def _get_digest( self ):
        return digest( self )
    

    def _psfCall(self, ind):
        """
        Manages the calling of the core psf functionality.
        
        Parameters
        ----------
        ind : list of int
            Indices of gridpoints which are assumed to be sources.
            Normalization factor for the beamforming result (e.g. removal of diag is compensated with this.)

        Returns
        -------
        The psf [1, nGridPoints, len(ind)]
        """
        if type(self.steer) == SteeringVector: # for simple steering vector, use faster method
            result = calcPointSpreadFunction(self.steer.steer_type, 
                                             self.steer.r0, 
                                             self.steer.rm, 
                                             2*pi*self.freq[newaxis]/self.env.c, 
                                             ind, self.precision)[0]
        else: # for arbitrary steering sectors, use general calculation
            # there is a version of this in fastFuncs, may be used later after runtime testing and debugging
            product = dot(self.steer.steer_vector(self.freq).conj(), self.steer.transfer(self.freq,ind).T)
            result = (product * product.conj()).real
        return result

    def _get_psf ( self ):
        """
        This is the :attr:`psf` getter routine.
        The point spread function is either loaded or calculated.
        """
        steer = self.steer
        gs = steer.grid.size
        if not self.grid_indices.size:
            self.grid_indices = arange(gs)
        name = 'psf' + self.digest
        H5cache.get_cache( self, name)
        fr = ('Hz_%.2f' % self.freq).replace('.', '_')
        precisionTuple = _precision(self.precision)
        
        # check wether self.freq is part of SteeringVector.f
        #freqInSteerObjFreq = isclose(array(self._steer_obj.f), self.freq)
        #if freqInSteerObjFreq.any():
        #    freqInd = flatnonzero(freqInSteerObjFreq)[0]
        #else:
        #    warn('PointSpreadFunction.freq (%s Hz) was appended to PointSpreadFunction._steer_obj.f, '\
        #         'as it was not an element of the original list!' % self.freq, Warning, stacklevel = 2)
        #    self._steer_obj.f.append(self.freq)
        #    freqInd = int(-1)
        
        # get the cached data, or, if non-existing, create new structure
        if not fr in self.h5f.root:
            if self.calcmode == 'readonly':
                raise ValueError('Cannot calculate missing PSF (freq %s) in \'readonly\' mode.' % fr)
            
            group = self.h5f.create_group(self.h5f.root, fr) 
            shape = (gs, gs)
            atom = precisionTuple[3]()
            filters = tables.Filters(complevel=5, complib='blosc')
            ac = self.h5f.create_carray(group, 'result', atom, shape, filters=filters)
            shape = (gs,)
            atom = tables.BoolAtom()
            gp = self.h5f.create_carray(group, 'gridpts', atom, shape, filters=filters)
            
        else:
            ac = self.h5f.get_node('/'+fr, 'result')
            gp = self.h5f.get_node('/'+fr, 'gridpts')
        
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
            
            if self.calcmode == 'single': # calculate selected psfs one-by-one
                for ind in g_ind_calc:
                    ac[:,ind] = self._psfCall([ind])[:,0]
                    gp[ind] = True
            elif self.calcmode == 'full': # calculate all psfs in one go
                gp[:] = True
                ac[:] = self._psfCall(arange(gs))
            else: # 'block' # calculate selected psfs in one go
                hh = self._psfCall(g_ind_calc)
                indh = 0
                for ind in g_ind_calc:
                    gp[ind] = True
                    ac[:,ind] = hh[:,indh]
                    indh += 1
            self.h5f.flush()
        return ac[:][:,self.grid_indices]

class BeamformerDamas (BeamformerBase):
    """
    DAMAS deconvolution, see :ref:`Brooks and Humphreys, 2006<BrooksHumphreys2006>`.
    Needs a-priori delay-and-sum beamforming (:class:`BeamformerBase`).
    """

    #: :class:`BeamformerBase` object that provides data for deconvolution.
    beamformer = Trait(BeamformerBase)

    #: :class:`~acoular.spectra.PowerSpectra` object that provides the cross spectral matrix; 
    #: is set automatically.
    freq_data = Delegate('beamformer')

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming; 
    #: is set automatically.
    r_diag =  Delegate('beamformer')
    
    #: instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes,
    #: that contains information about the steering vector. Is set automatically.
    steer = Delegate('beamformer')
    
    #: Floating point precision of result, is set automatically.
    precision = Delegate('beamformer')
    
    #: The floating-number-precision of the PSFs. Default is 64 bit.
    psf_precision = Trait('float64', 'float32', 
                          desc="precision of PSF")

    #: Number of iterations, defaults to 100.
    n_iter = Int(100, 
        desc="number of iterations")
    
    #: Damping factor in modified gauss-seidel
    damp = Float(1.0,
                          desc="damping factor in modified gauss-seidel-DAMAS-approach")

    #: Flag that defines how to calculate and store the point spread function, 
    #: defaults to 'full'. See :attr:`PointSpreadFunction.calcmode` for details.
    calcmode = Trait('full', 'single', 'block', 'readonly',
                     desc="mode of psf calculation / storage")
    
    # internal identifier
    digest = Property( 
        depends_on = ['beamformer.digest', 'n_iter', 'damp', 'psf_precision'], 
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'beamformer.ext_digest'], 
        )
    
    traits_view = View(
        [
            [Item('beamformer{}', style='custom')], 
            [Item('n_iter{Number of iterations}')], 
#            [Item('steer{Type of steering vector}')], 
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
        Calculates the DAMAS result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        A Gauss-Seidel algorithm implemented in C is used for computing the result.
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        f = self.freq_data.fftfreq()
        p = PointSpreadFunction(steer=self.steer, calcmode=self.calcmode, precision=self.psf_precision)
        for i in self.freq_data.indices:
            if not fr[i]:
                y = array(self.beamformer.result[i])
                x = y.copy()
                p.freq = f[i]
                psf = p.psf[:]
                damasSolverGaussSeidel(psf, y, self.n_iter, self.damp, x)
                ac[i] = x
                fr[i] = True

class BeamformerDamasPlus (BeamformerDamas):
    """
    DAMAS deconvolution, see :ref:`Brooks and Humphreys, 2006<BrooksHumphreys2006>`,
    for solving the system of equations, instead of the original Gauss-Seidel 
    iterations, this class employs the NNLS or linear programming solvers from 
    scipy.optimize or one  of several optimization algorithms from the scikit-learn module.
    Needs a-priori delay-and-sum beamforming (:class:`BeamformerBase`).
    """
    
    #: Type of fit method to be used ('LassoLars', 
    #: 'OMPCV', 'LP', or 'NNLS', defaults to 'NNLS').
    #: These methods are implemented in 
    #: the `scikit-learn <http://scikit-learn.org/stable/user_guide.html>`_ 
    #: module or within scipy.optimize respectively.
    method = Trait('NNLS','LP','LassoLars', 'OMPCV',  
                   desc="method used for solving deconvolution problem")
    
    #: Weight factor for LassoLars method,
    #: defaults to 0.0.
    # (Values in the order of 10^⁻9 should produce good results.)
    alpha = Range(0.0, 1.0, 0.0,
                  desc="Lasso weight factor")
    
    #: Maximum number of iterations,
    #: tradeoff between speed and precision;
    #: defaults to 500
    max_iter = Int(500,
                   desc="maximum number of iterations")
    
    #: Unit multiplier for evaluating, e.g., nPa instead of Pa. 
    #: Values are converted back before returning. 
    #: Temporary conversion may be necessary to not reach machine epsilon
    #: within fitting method algorithms. Defaults to 1e9.
    unit_mult = Float(1e9,
                      desc = "unit multiplier")
    
    # internal identifier
    digest = Property( 
        depends_on = ['beamformer.digest','alpha', 'method', 
                      'max_iter', 'unit_mult'], 
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'beamformer.ext_digest'], 
        )
    
    traits_view = View(
        [
            [Item('beamformer{}', style='custom')], 
            [Item('method{Solver}')],
            [Item('max_iter{Max. number of iterations}')], 
            [Item('alpha', label='Lasso weight factor')], 
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
        Calculates the DAMAS result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        f = self.freq_data.fftfreq()
        p = PointSpreadFunction(steer=self.steer, calcmode=self.calcmode, precision=self.psf_precision)
        unit = self.unit_mult
        for i in self.freq_data.indices:
            if not fr[i]:
                y = self.beamformer.result[i] * unit
                p.freq = f[i]
                psf = p.psf[:]

                if self.method == 'NNLS':
                    resopt = nnls(psf,y)[0]
                elif self.method == 'LP': # linear programming (Dougherty)
                    if self.r_diag:
                        warn('Linear programming solver may fail when CSM main '
                              'diagonal is removed for delay-and-sum beamforming.', 
                              Warning, stacklevel = 5)
                    cT = -1*psf.sum(1) # turn the minimization into a maximization
                    resopt = linprog(c=cT, A_ub=psf, b_ub=y).x # defaults to simplex method and non-negative x
                elif self.method == 'LassoLars':
                    model = LassoLars(alpha = self.alpha * unit, 
                                      max_iter = self.max_iter)
                else: # self.method == 'OMPCV':
                    model = OrthogonalMatchingPursuitCV()
                
                
                if self.method in ('NNLS','LP'):
                    ac[i] = resopt / unit
                else: # sklearn models
                    model.fit(psf,y)
                    ac[i] = model.coef_[:] / unit
                
                fr[i] = True

class BeamformerOrth (BeamformerBase):
    """
    Orthogonal beamforming, see :ref:`Sarradj, 2010<Sarradj2010>`.
    Needs a-priori beamforming with eigenvalue decomposition (:class:`BeamformerEig`).
    """

    #: :class:`BeamformerEig` object that provides data for deconvolution.
    beamformer = Trait(BeamformerEig)

    #: :class:`~acoular.spectra.PowerSpectra` object that provides the cross spectral matrix 
    #: and eigenvalues, is set automatically.    
    freq_data = Delegate('beamformer')

    #: Flag, if 'True' (default), the main diagonal is removed before beamforming, 
    #: is set automatically.
    r_diag =  Delegate('beamformer')
    
    #: instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes,
    #: that contains information about the steering vector. Is set automatically.
    steer = Delegate('beamformer')

    #: List of components to consider, use this to directly set the eigenvalues
    #: used in the beamformer. Alternatively, set :attr:`n`.
    eva_list = CArray(
        desc="components")
        
    #: Number of components to consider, defaults to 1. If set, 
    #: :attr:`eva_list` will contain
    #: the indices of the n largest eigenvalues. Setting :attr:`eva_list` 
    #: afterwards will override this value.
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
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='Number of components', style='simple')], 
            [Item('r_diag', label='Diagonal removed')], 
#            [Item('env{}', style='custom')], 
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
        Calculates the Orthogonal Beamforming result for the frequencies 
        defined by :attr:`freq_data`.
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        # prepare calculation
        ii = []
        for i in self.freq_data.indices:        
            if not fr[i]:
                ii.append(i)
        numchannels = self.freq_data.numchannels
        e = self.beamformer
        for n in self.eva_list:
            e.n = n
            for i in ii:
                ac[i, e.result[i].argmax()]+=e.freq_data.eva[i, n]/numchannels
        for i in ii:
            fr[i] = True
    
class BeamformerCleansc( BeamformerBase ):
    """
    CLEAN-SC deconvolution, see :ref:`Sijtsma, 2007<Sijtsma2007>`.
    Classic delay-and-sum beamforming is already included.
    """

    #: no of CLEAN-SC iterations
    #: defaults to 0, i.e. automatic (max 2*numchannels)
    n = Int(0, 
        desc="no of iterations")

    #: iteration damping factor
    #: defaults to 0.6
    damp = Range(0.01, 1.0, 0.6, 
        desc="damping factor")

    #: iteration stop criterion for automatic detection
    #: iteration stops if power[i]>power[i-stopn]
    #: defaults to 3
    stopn = Int(3, 
        desc="stop criterion index")

    # internal identifier
    digest = Property( 
        depends_on = ['freq_data.digest', '_steer_obj.digest', 'r_diag', 'n', 'damp', 'stopn'])

    traits_view = View(
        [
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='No. of iterations', style='simple')], 
            [Item('r_diag', label='Diagonal removed')], 
#            [Item('env{}', style='custom')], 
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
        Calculates the CLEAN-SC result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """

        # prepare calculation
        normFactor = self.sig_loss_norm()
        numchannels = self.freq_data.numchannels
        f = self.freq_data.fftfreq()
        result = zeros((self.steer.grid.size), 'f') 
        normFac = self.sig_loss_norm()
        if not self.n:
            J = numchannels*2
        else:
            J = self.n
        powers = zeros(J, 'd')
        
        param_steer_type, steer_vector = self._beamformer_params()
        for i in self.freq_data.indices:
            if not fr[i]:
                csm = array(self.freq_data.csm[i], dtype='complex128', copy=1)
                #h = self.steer._beamformerCall(f[i], self.r_diag, normFactor, (csm,))[0]
                h = beamformerFreq(param_steer_type, 
                                   self.r_diag, 
                                   normFactor, 
                                   steer_vector(f[i]), 
                                   csm)[0]
                # CLEANSC Iteration
                result *= 0.0
                for j in range(J):
                    xi_max = h.argmax() #index of maximum
                    powers[j] = hmax = h[xi_max] #maximum
                    result[xi_max] += self.damp * hmax
                    if  j > self.stopn and hmax > powers[j-self.stopn]:
                        break
                    wmax = self.steer.steer_vector(f[i],xi_max) * sqrt(normFac)
                    wmax = wmax[0].conj()  # as old code worked with conjugated csm..should be updated
                    hh = wmax.copy()
                    D1 = dot(csm.T - diag(diag(csm)), wmax)/hmax
                    ww = wmax.conj()*wmax
                    for m in range(20):
                        H = hh.conj()*hh
                        hh = (D1+H*wmax)/sqrt(1+dot(ww, H))
                    hh = hh[:, newaxis]
                    csm1 = hmax*(hh*hh.conj().T)
                    
                    #h1 = self.steer._beamformerCall(f[i], self.r_diag, normFactor, (array((hmax, ))[newaxis, :], hh[newaxis, :].conjugate()))[0]
                    h1 = beamformerFreq(param_steer_type, 
                                        self.r_diag, 
                                        normFactor, 
                                        steer_vector(f[i]), 
                                        (array((hmax, )), hh.conj()))[0]
                    h -= self.damp * h1
                    csm -= self.damp * csm1.T#transpose(0,2,1)
                ac[i] = result
                fr[i] = True

class BeamformerClean (BeamformerBase):
    """
    CLEAN deconvolution, see :ref:`Hoegbom, 1974<Hoegbom1974>`.
    Needs a-priori delay-and-sum beamforming (:class:`BeamformerBase`).
    """

    # BeamformerBase object that provides data for deconvolution
    beamformer = Trait(BeamformerBase)

    # PowerSpectra object that provides the cross spectral matrix
    freq_data = Delegate('beamformer')

    # flag, if true (default), the main diagonal is removed before beamforming
    #r_diag =  Delegate('beamformer')
    
    #: instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes,
    #: that contains information about the steering vector. Is set automatically.
    steer = Delegate('beamformer')
    
    #: Floating point precision of result, is set automatically.
    precision = Delegate('beamformer')
    
    #: The floating-number-precision of the PSFs. Default is 64 bit.
    psf_precision = Trait('float64', 'float32', 
                     desc="precision of PSF.")
    
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
        depends_on = ['beamformer.digest', 'n_iter', 'damp', 'psf_precision'], 
        )

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'beamformer.ext_digest'], 
        )
    
    traits_view = View(
        [
            [Item('beamformer{}', style='custom')], 
            [Item('n_iter{Number of iterations}')], 
#            [Item('steer{Type of steering vector}')], 
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
        Calculates the CLEAN result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        f = self.freq_data.fftfreq()
        gs = self.steer.grid.size
        
        if self.calcmode == 'full':
            warn("calcmode = 'full', possibly slow CLEAN performance. "
                 "Better use 'block' or 'single'.", Warning, stacklevel = 2)
        p = PointSpreadFunction(steer=self.steer, calcmode=self.calcmode, precision=self.psf_precision)
        for i in self.freq_data.indices:
            if not fr[i]:
                p.freq = f[i]
                dirty = self.beamformer.result[i]
                clean = zeros(gs, dtype=dirty.dtype)
                
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
                
                ac[i] = clean            
                fr[i] = True

class BeamformerCMF ( BeamformerBase ):
    """
    Covariance Matrix Fitting, see :ref:`Yardibi et al., 2008<Yardibi2008>`.
    This is not really a beamformer, but an inverse method.
    """

    #: Type of fit method to be used ('LassoLars', 'LassoLarsBIC', 
    #: 'OMPCV' or 'NNLS', defaults to 'LassoLars').
    #: These methods are implemented in 
    #: the `scikit-learn <http://scikit-learn.org/stable/user_guide.html>`_ 
    #: module.
    method = Trait('LassoLars', 'LassoLarsBIC',  \
        'OMPCV', 'NNLS', desc="fit method used")
        
    #: Weight factor for LassoLars method,
    #: defaults to 0.0.
    #: (Use values in the order of 10^⁻9 for good results.)
    alpha = Range(0.0, 1.0, 0.0, 
        desc="Lasso weight factor")
    
    #: Maximum number of iterations,
    #: tradeoff between speed and precision;
    #: defaults to 500
    max_iter = Int(500, 
        desc="maximum number of iterations")

    
    #: Unit multiplier for evaluating, e.g., nPa instead of Pa. 
    #: Values are converted back before returning. 
    #: Temporary conversion may be necessary to not reach machine epsilon
    #: within fitting method algorithms. Defaults to 1e9.
    unit_mult = Float(1e9,
                      desc = "unit multiplier")

    # internal identifier
    digest = Property( 
        depends_on = ['freq_data.digest', 'alpha', 'method', 'max_iter', 'unit_mult', 'r_diag', 'steer.inv_digest'], 
        )

    traits_view = View(
        [
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('method', label='Fit method')], 
            [Item('max_iter', label='No. of iterations')], 
            [Item('alpha', label='Lasso weight factor')], 
            [Item('c', label='Speed of sound')], 
#            [Item('env{}', style='custom')], 
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
        Calculates the CMF result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters
        """
        
        # function to repack complex matrices to deal with them in real number space
        def realify(M):
            return vstack([M.real,M.imag])

            
        # prepare calculation
        i = self.freq_data.indices
        f = self.freq_data.fftfreq()
        nc = self.freq_data.numchannels
        numpoints = self.steer.grid.size
        unit = self.unit_mult

        for i in self.freq_data.indices:        
            if not fr[i]:
                csm = array(self.freq_data.csm[i], dtype='complex128',copy=1)

                h = self.steer.transfer(f[i]).T
                
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
                R = realify( reshape(csm.T, (nc*nc,1))[ind,:] )[ind_reim,:] * unit
                # choose method
                if self.method == 'LassoLars':
                    model = LassoLars(alpha = self.alpha * unit,
                                      max_iter = self.max_iter)
                elif self.method == 'LassoLarsBIC':
                    model = LassoLarsIC(criterion = 'bic',
                                        max_iter = self.max_iter)
                elif self.method == 'OMPCV':
                    model = OrthogonalMatchingPursuitCV()

                # nnls is not in sklearn
                if self.method == 'NNLS':
                    ac[i] , x = nnls(A,R.flat)
                    ac[i] /= unit
                else:
                    model.fit(A,R[:,0])
                    ac[i] = model.coef_[:] / unit
                fr[i] = True

class BeamformerGIB(BeamformerEig):  #BeamformerEig #BeamformerBase
    """
    Beamforming GIB methods with different normalizations,
    """
    
    #: Unit multiplier for evaluating, e.g., nPa instead of Pa. 
    #: Values are converted back before returning. 
    #: Temporary conversion may be necessary to not reach machine epsilon
    #: within fitting method algorithms. Defaults to 1e9.
    unit_mult = Float(1e9,
                      desc = "unit multiplier")

    #: Maximum number of iterations,
    #: tradeoff between speed and precision;
    #: defaults to 10
    max_iter = Int(10, 
                   desc="maximum number of iterations")

    #: Type of fit method to be used ('Suzuki', 'LassoLars', 'LassoLarsCV', 'LassoLarsBIC', 
    #: 'OMPCV' or 'NNLS', defaults to 'Suzuki').
    #: These methods are implemented in 
    #: the `scikit-learn <http://scikit-learn.org/stable/user_guide.html>`_ 
    #: module.
    method = Trait('Suzuki', 'InverseIRLS', 'LassoLars', 'LassoLarsBIC','LassoLarsCV',  \
        'OMPCV', 'NNLS', desc="fit method used")

    #: Weight factor for LassoLars method,
    #: defaults to 0.0.
    alpha = Range(0.0, 1.0, 0.0, 
        desc="Lasso weight factor")
    # (use values in the order of 10^⁻9 for good results) 
    
    #: Norm to consider for the regularization in InverseIRLS and Suzuki methods 
    #: defaults to L-1 Norm
    pnorm= Float(1,desc="Norm for regularization")

    #: Beta - Fraction of sources maintained after each iteration
    #: defaults to 0.9 
    beta =  Float(0.9,desc="fraction of sources maintained")
    
    #: eps - Regularization parameter for Suzuki algorithm
    #: defaults to 0.05. 
    eps_perc =  Float(0.05,desc="regularization parameter")

    # This feature is not fully supported may be changed in the next release 
    # First eigenvalue to consider. Defaults to 0.
    m = Int(0,
                      desc = "First eigenvalue to consider")
    
    
    # internal identifier++++++++++++++++++++++++++++++++++++++++++++++++++
    digest = Property( 
        depends_on = ['steer.inv_digest', 'freq_data.digest', \
            'alpha', 'method', 'max_iter', 'unit_mult', 'eps_perc',\
            'pnorm', 'beta','n', 'm'], 
        )

    traits_view = View(
        [
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('method', label='Fit method')], 
            [Item('max_iter', label='No. of iterations')], 
            [Item('alpha', label='Lasso weight factor')], 
            [Item('c', label='Speed of sound')], 
#            [Item('env{}', style='custom')], 
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
        nm = self.steer.mics.num_mics
        if na < 0:
            na = max(nm + na, 0)
        return min(nm - 1, na)

    def calc(self, ac, fr):
        
        """
        Calculates the result for the frequencies defined by :attr:`freq_data`
        
        This is an internal helper function that is automatically called when 
        accessing the beamformer's :attr:`~BeamformerBase.result` or calling
        its :meth:`~BeamformerBase.synthetic` method.        
        
        Parameters
        ----------
        ac : array of floats
            This array of dimension ([number of frequencies]x[number of gridpoints])
            is used as call-by-reference parameter and contains the calculated
            value after calling this method. 
        fr : array of booleans
            The entries of this [number of frequencies]-sized array are either 
            'True' (if the result for this frequency has already been calculated)
            or 'False' (for the frequencies where the result has yet to be calculated).
            After the calculation at a certain frequency the value will be set
            to 'True'
        
        Returns
        -------
        This method only returns values through the *ac* and *fr* parameters

        """        
        # prepare calculation
        f = self.freq_data.fftfreq()
        n = int(self.n)  
        m = int(self.m)                             #number of eigenvalues
        numchannels = self.freq_data.numchannels   #number of channels
        numpoints = self.steer.grid.size
        hh = zeros((1, numpoints, numchannels), dtype='D')
        
        #Generate a cross spectral matrix, and perform the eigenvalue decomposition
        for i in self.freq_data.indices:        
            if not fr[i]:
                #for monopole and source strenght Q needs to define density
                #calculate a transfer matrix A 
                hh = self.steer.transfer(f[i])
                A=hh.T                 
                #eigenvalues and vectors               
                csm = array(self.freq_data.csm[i], dtype='complex128',copy=1)
                eva,eve=eigh(csm)
                eva = eva[::-1]
                eve = eve[:, ::-1] 
                eva[eva < max(eva)/1e12] = 0 #set small values zo 0, lowers numerical errors in simulated data
                #init sources    
                qi=zeros([n+m,numpoints], dtype='complex128')
                #Select the number of coherent modes to be processed referring to the eigenvalue distribution.
                #for s in arange(n):  
                for s in list(range(m,n+m)): 
                    #Generate the corresponding eigenmodes
                    emode=array(sqrt(eva[s])*eve[:,s], dtype='complex128')
                    # choose method for computation
                    if self.method == 'Suzuki':
                        leftpoints=numpoints
                        locpoints=arange(numpoints)         
                        weights=diag(ones(numpoints))             
                        epsilon=arange(self.max_iter)              
                        for it in arange(self.max_iter): 
                            if numchannels<=leftpoints:
                                AWA= dot(dot(A[:,locpoints],weights),A[:,locpoints].conj().T)
                                epsilon[it] = max(absolute(eigvals(AWA)))*self.eps_perc
                                qi[s,locpoints]=dot(dot(dot(weights,A[:,locpoints].conj().T),inv(AWA+eye(numchannels)*epsilon[it])),emode)
                            elif numchannels>leftpoints:
                                AA=dot(A[:,locpoints].conj().T,A[:,locpoints])
                                epsilon[it] = max(absolute(eigvals(AA)))*self.eps_perc
                                qi[s,locpoints]=dot(dot(inv(AA+inv(weights)*epsilon[it]),A[:,locpoints].conj().T),emode)                                                       
                            if self.beta < 1 and it > 1:   
                                #Reorder from the greatest to smallest magnitude to define a reduced-point source distribution , and reform a reduced transfer matrix 
                                leftpoints=int(round(numpoints*self.beta**(it+1)))                                                                                          
                                idx = argsort(abs(qi[s,locpoints]))[::-1]   
                                #print(it, leftpoints, locpoints, idx )
                                locpoints= delete(locpoints,[idx[leftpoints::]])             
                                qix=zeros([n+m,leftpoints], dtype='complex128')                      
                                qix[s,:]=qi[s,locpoints]
                                #calc weights for next iteration 
                                weights=diag(absolute(qix[s,:])**(2-self.pnorm))    
                            else:                          
                                weights=diag((absolute(qi[s,:])**(2-self.pnorm)))    
                         
                    elif self.method == 'InverseIRLS':                         
                        weights=eye(numpoints)
                        locpoints=arange(numpoints)
                        for it in arange(self.max_iter): 
                            if numchannels<=numpoints: 
                                wtwi=inv(dot(weights.T,weights))  
                                aH=A.conj().T                       
                                qi[s,:]=dot(dot(wtwi,aH),dot(inv(dot(A,dot(wtwi,aH))),emode))                            
                                weights=diag(absolute(qi[s,:])**((2-self.pnorm)/2))
                                weights=weights/sum(absolute(weights))                                 
                            elif numchannels>numpoints:
                                wtw=dot(weights.T,weights)
                                qi[s,:]= dot(dot(inv(dot(dot(A.conj.T,wtw),A)),dot( A.conj().T,wtw)) ,emode)
                                weights=diag(absolute(qi[s,:])**((2-self.pnorm)/2))
                                weights=weights/sum(absolute(weights))                  
                    else:
                        locpoints=arange(numpoints) 
                        unit = self.unit_mult
                        AB = vstack([hstack([A.real,-A.imag]),hstack([A.imag,A.real])])
                        R  = hstack([emode.real.T,emode.imag.T]) * unit
                        if self.method == 'LassoLars':
                            model = LassoLars(alpha=self.alpha * unit,max_iter=self.max_iter)
                        elif self.method == 'LassoLarsBIC':
                            model = LassoLarsIC(criterion='bic',max_iter=self.max_iter)
                        elif self.method == 'OMPCV':
                            model = OrthogonalMatchingPursuitCV()
                        elif self.method == 'LassoLarsCV':
                            model = LassoLarsCV()                        
                        if self.method == 'NNLS':
                            x , zz = nnls(AB,R)
                            qi_real,qi_imag = hsplit(x/unit, 2) 
                        else:
                            model.fit(AB,R)
                            qi_real,qi_imag = hsplit(model.coef_[:]/unit, 2)
                        #print(s,qi.size)    
                        qi[s,locpoints] = qi_real+qi_imag*1j
                #Generate source maps of all selected eigenmodes, and superpose source intensity for each source type.
                ac[i] = zeros([1,numpoints])
                ac[i,locpoints] = sum(absolute(qi[:,locpoints])**2,axis=0)
                fr[i] = True    

def L_p ( x ):
    """
    Calculates the sound pressure level from the squared sound pressure.
    
    :math:`L_p = 10 \lg ( x / 4\cdot 10^{-10})`
    
    Parameters
    ----------
    x: array of floats
        The squared sound pressure values
        
    Returns
    -------
    array of floats
        The corresponding sound pressure levels in dB. 
        If `x<0`, -350.0 dB is returned.
    """
    # new version to prevent division by zero warning for float32 arguments
    return 10*log10(clip(x/4e-10,1e-35,None))
#    return where(x>0, 10*log10(x/4e-10), -1000.)

def integrate(data, grid, sector):
    """
    Integrates a sound pressure map over a given sector.
    
    This function can be applied on beamforming results to
    quantitatively analyze the sound pressure in a given sector.
    If used with :meth:`Beamformer.result()<acoular.fbeamform.BeamformerBase.result>`, 
    the output is identical to the result of the intrinsic 
    :meth:`Beamformer.integrate<acoular.fbeamform.BeamformerBase.integrate>` method.
    It can, however, also be used with the 
    :meth:`Beamformer.synthetic<acoular.fbeamform.BeamformerBase.synthetic>`
    output.
    
    Parameters
    ----------
    data: array of floats
        Contains the calculated sound pressures in Pa.        
        If data has the same number of entries than the number of grid points
        only one value is returned.
        In case of a 2-D array with the second dimension identical 
        to the number of grid points an array containing as many entries as
        the first dimension is returned.
    grid: Grid object 
        Object of a :class:`~acoular.grids.Grid`-derived class 
        that provides the grid locations.        
    sector: array of floats
        Tuple with arguments for the `indices` method 
        of a :class:`~acoular.grids.Grid`-derived class 
        (e.g. :meth:`RectGrid.indices<acoular.grids.RectGrid.indices>` 
        or :meth:`RectGrid3D.indices<acoular.grids.RectGrid3D.indices>`).
        Possible sectors would be `array([xmin, ymin, xmax, ymax])`
        or `array([x, y, radius])`.
          
    Returns
    -------
    array of floats
        The spectrum (all calculated frequency bands) for the integrated sector.
    """
    
    ind = grid.indices(*sector)
    gshape = grid.shape
    gsize = grid.size
    if size(data) == gsize: # one value per grid point
        h = data.reshape(gshape)[ind].sum()
    elif data.ndim == 2 and data.shape[1] == gsize:
        h = zeros(data.shape[0])
        for i in range(data.shape[0]):
            h[i] = data[i].reshape(gshape)[ind].sum()
    return h
