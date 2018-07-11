# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2017, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements beamformers in the frequency domain.

.. autosummary::
    :toctree: generated/

    SteeringVector
    SteeringVectorInduct
    
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

from numpy import array, ones, hanning, hamming, bartlett, blackman, invert, \
dot, newaxis, zeros, empty, fft, float32, float64, complex64, linalg, where, \
searchsorted, pi, multiply, sign, diag, arange, sqrt, exp, log10, int,\
reshape, hstack, vstack, eye, tril, size, clip, tile, round, delete, \
absolute, argsort, sort, sum, hsplit, fill_diagonal, zeros_like, isclose, \
vdot, flatnonzero, unique, int32, in1d, mean, inf

from sklearn.linear_model import LassoLars, LassoLarsCV, LassoLarsIC,\
OrthogonalMatchingPursuit, ElasticNet, OrthogonalMatchingPursuitCV, Lasso

from scipy.optimize import nnls, linprog
from scipy.linalg import inv, eigh, eigvals, fractional_matrix_power
from scipy.special import jn
from warnings import warn

import tables
from traits.api import HasPrivateTraits, Float, Int, ListInt, ListFloat, \
CArray, Property, Instance, Trait, Bool, Range, Delegate, Enum, \
cached_property, on_trait_change, property_depends_on

from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons

from .fastFuncs import beamformerFreq, calcTransfer, calcPointSpreadFunction, \
damasSolverGaussSeidel, greens_func_Induct

from .h5cache import H5cache
from .internal import digest
from .grids import Grid
from .microphones import MicGeom
from .environments import Environment, InductUniformFlow, cartToCyl
from .spectra import PowerSpectra, _precision


class SteeringVector( HasPrivateTraits ):
    """ 
    Basic class for implementing steering vectors without flow in free field
    environment.
    """
    #: List of frequency bins
    f = ListFloat(desc="frequency bins")  # should be ListFloat: With Carray, traits 
    # always recalculates everything (especially transferfunctions and psf),
    # even though f got overwritten with its original List (which happens often with the current implementation)
    
    #: The speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")
    
    #: List of free field wave numbers, corresponding to :attr:`SteeringVector.f`
    #: and :attr:`SteeringVector.c`; read only
    k = Property(depends_on=['f', 'c'])

    #: Type of steering vectors, see also :ref:`Sarradj, 2012<Sarradj2012>`.
    steer_type = Trait('true level', 'true location', 'classic', 'inverse',
                  desc="type of steering vectors used")
    
    #: :class:`~acoular.environments.Environment` or derived object, 
    #: which provides information about the sound propagation in the medium.
    env = Instance(Environment(), Environment)
    
    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    grid = Trait(Grid, 
        desc="beamforming grid")
    
    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mpos = Trait(MicGeom, 
        desc="microphone geometry")
    
    #: Sound travel distances from microphone array center to grid 
    #: points (readonly).
    r0 = Property(desc="array center to grid distances")

    #: Sound travel distances from array microphones to grid 
    #: points (readonly).
    rm = Property(desc="all array mics to grid distances")
    
    #: Transfer matrix (vector of transfer-vectors) of dimension [nFreq, nGrid, nMics]; read only
    transfer = Property(depends_on=['f', 'c', 'env.digest', 'grid.digest', 'mpos.digest'], 
                        desc = 'transfer matrix')
    
    #: Steering vector of dimension [nFreq, nGrid, nMics], corresponding to transfer; read-only
    steer_vector = Property(depends_on='digest', desc = 'steering vector')
    
    # internal identifier
    digest = Property( 
        depends_on = ['f', 'c', 'steer_type', 'env.digest', 'grid.digest', 'mpos.digest'])
    
    @cached_property
    def _get_k(self):
        return 2 * pi * array(self.f) / self.c
    
    @property_depends_on('c, grid.digest, env.digest')
    def _get_r0 ( self ):
        return self.env.r( self.c, self.grid.pos())

    @property_depends_on('c, grid.digest, mpos.digest, env.digest')
    def _get_rm ( self ):
        return self.env.r( self.c, self.grid.pos(), self.mpos.mpos)
 
    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @cached_property
    def _get_transfer(self):
        trans = calcTransfer(self.r0, self.rm, array(self.k))
        return trans
    
    @cached_property
    def _get_steer_vector(self):
        steer = zeros_like(self.transfer, 'complex128')
        for indFreq in range(steer.shape[0]):
            steer[indFreq, :, :] = self.calc_steer_vector(self.transfer[indFreq, :, :])
        return steer
    
    def _transfer_specific_gridpoint(self, ind):
        """
        Calculates the transfer function of ONE grid point. Is used in clean-sc
        when not the whole transfer matrix is needed.
        
        Parameters
        ----------
        ind : int
            Identifier for wich gridpoint the transfer funtion is calculated
        
        Returns
        -------
        trans : complex128[nFreqs, nMics]
        """
        trans = calcTransfer(self.r0[ind], self.rm[ind, :][newaxis], array(self.k))[:, 0, :]
        return trans
    
    def calc_steer_vector(self, a):
        """
        Calculates the steering vector from a given transfer function a.
        See also :ref:`Sarradj, 2012<Sarradj2012>`.
        
        Parameters
        ----------
        a : complex128[nGridpoints, nMics]
            Transfer functions
        
        Returns
        -------
        h : complex128[nGridpoints, nMics]
            Steering vectors
        """
        nGrid, nMics = a.shape
        func = {'classic' : lambda x: x / abs(x) / nMics,
                'inverse' : lambda x: 1. / x.conj() / nMics,
                'true level' : lambda x: x / vdot(x, x),
                'true location' : lambda x: x / sqrt(vdot(x, x) * nMics)}[self.steer_type]
        steerOutput = array([func(a[cntGrid, :]) for cntGrid in range(nGrid)])
        return steerOutput
    
    def _beamformerCall(self, indFreq, rDiag, normFac, tupleCSM):
        """
        Manages the calling of the core beamformer functionality.
        
        Parameters
        ----------
        indFreq : int
            The index for which item of this classes property k the beamformer should be performed.
        rDiag : bool
            Should the diagonal of the csm should be considered in beamformer
            (rDiag==False) or not (rDiag==True).
        normFac : float
            Normalization factor for the beamforming result (e.g. removal of diag is compensated with this.)
        tupleCSM : either (csm,) or (eigValues, eigVectors)
            See information header of beamformerFreq in fastFuncs.py for information on those inputs.

        Returns
        -------
        The results of beamformerFreq, see fastFuncs.py 
        """
        tupleSteer = (self.r0, self.rm, self.k[indFreq][newaxis])
        result = beamformerFreq(self.steer_type, rDiag, normFac, tupleSteer, tupleCSM)
        return result
    
    def _psfCall(self, freqInd, sourceInd, precision):
        """
        Manages the calling of the core psf functionality.
        
        Parameters
        ----------
        freqInd : int
            Index for which entry of Trait f the psf should be calculated.
        sourceInd : list of int
            Indices of gridpoints which are assumed to be sources.
            Normalization factor for the beamforming result (e.g. removal of diag is compensated with this.)
        precision : string ('float32' or 'float64')
            Precision of psf.

        Returns
        -------
        The psf [1, nGridPoints, len(sourceInd)]
        """
        result = calcPointSpreadFunction(self.steer_type, self.r0, self.rm, self.k[freqInd][newaxis], sourceInd, precision)
        return result


class SteeringVectorInduct( SteeringVector ):
    """
    Class for implementing a steering vector for sound 
    propagation in a circular duct (without hub).
    """
    #: :class:`~acoular.environments.InductUniformFlow`, which provides 
    #: information about the sound propagation in the medium.
    env = Instance(InductUniformFlow)
    
    #: Maximum attenuation level (dB) per radius of duct. All modes with 
    #: attenuation levels above this value are not taken into account.
    max_attenuation = Float(0.0, 
                            desc="Maximum level of attenuation of modes."
                            "Modes with values above this Float are not taken into account.")

    #: List of azimuthal mode orders which are observed (one Int per entry of Trait f).
    #: E.g. if considered_azi_modes=[-5,10] then for the first entry of f only 
    #: the (m,n) modes with m=-5 are taken into account for the transfer vectors,
    #: respectively m=10 for the second entry of f. The default is on empty List
    #: in which case for all entries of k all (m,n) modes of :attr:`modal_properties`
    #: are taken into account for the transfer vectors.
    considered_azi_modes = ListInt()
    
    #: When calculating the transfer function: Should the propagation from gridpoint to mics be
    #: normalized with the propagation from gridpoint to specific point (e.g. array center)? 
    #: Default is True (see :attr:`transfer_norm_ref_point`).
    transfer_normalized = Bool(True)
    
    #: Only relevant if :attr:`transfer_normalized` is True. Specifies the normalization point
    #: for the transfer function (see :attr:`transfer_normalized`). Default is [inf, inf, inf]
    #: which normalizes with the propagation to the array center.
    transfer_norm_ref_point = CArray(dtype=float64, shape=(3, ), value=(inf, inf, inf))
    
    #: grid positions in cylindrical coordinates; read only
    grid_cyl = Property(depends_on='grid.digest')
    
    #: grid positions in cylindrical coordinates; read only
    mpos_cyl = Property(depends_on='mpos.digest')
    
    #: List with table of modal properties of each (m,n)-mode (in rows). Column entries are:
    #: [azim. order, rad. order, sigma, alpha.real, alpha.imag, k+, k-, norm-Factor].
    #: The List contains nFreq tables. See Phd-thesis of Ulf Tapken (TU Berlin) for 
    #: information on sigma, alpha, k+- and the normalization factor (=F_mn in the thesis); Read-only
    modal_properties = Property(depends_on=['f', 'c', 'max_attenuation', 'env.digest'])
    
    #: Transfer matrix (vector of transfer-vectors) of dimension [nFreq, nGrid, nMics]; read only
    transfer = Property(depends_on=['f', 'c', 'max_attenuation', 'env.digest', 'considered_azi_modes', \
                                    'grid.digest', 'mpos.digest', 'transfer_normalized', 'transfer_norm_ref_point'], 
                        desc = 'transfer matrix')
    
    # internal identifier
    digest = Property(depends_on = ['f', 'c', 'steer_type', 'env.digest', 'grid.digest', 'mpos.digest', 'considered_azi_modes', \
                                    'max_attenuation', 'transfer_normalized', 'transfer_norm_ref_point'])
 
    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @cached_property
    def _get_grid_cyl(self):
        return cartToCyl(self.grid.pos())
    
    @cached_property
    def _get_mpos_cyl(self):
        return cartToCyl(self.mpos.mpos)
    
    @cached_property
    def _get_modal_properties(self):
        if len(self.k) == 0:
            raise Exception('Property f of SteeringVectorInduct must be set with Floats, but is [] instead!')
        result = self.env.modal_properties(self.k, self.max_attenuation, self.c)
        return result
    
    @cached_property
    def _get_transfer(self):
        nFreq, nMics, nGrid = len(self.k), self.mpos_cyl.shape[1], self.grid_cyl.shape[1]
        trans = zeros((nFreq, nGrid, nMics), dtype='complex128')

        for cntFreq in range(nFreq):
            modeProps = self.modal_properties[cntFreq]
            uniqueAziModes = unique(int32(modeProps[:, 0]))
    
            # Check which modegroups with same aziMode should be taken into account
            if self.considered_azi_modes == []:  # default case
                usedAziModes = uniqueAziModes  # create one transfer matrix for all (m,n) modes
            else:
                considerHelp= in1d(self.considered_azi_modes[cntFreq], uniqueAziModes)
                if not considerHelp.all():
                    raise Exception('Specified azi-mode of considered_azi_modes[%s] must be '
                                    'within possible azimuthal mode range of [%s, %s], but %s was given.' 
                                    % (cntFreq, uniqueAziModes[0], uniqueAziModes[-1], self.considered_azi_modes[cntFreq]))
                usedAziModes = self.considered_azi_modes[cntFreq]  # create one transfer matrix for all (m,n) modes with fixed m
            
            # locate all active (m,n)-modes
            usedModes = in1d(int32(modeProps[:, 0]), usedAziModes)
            usedModesInd = flatnonzero(usedModes)
            nActiveModes = usedModesInd.shape[0]
            
            # calc all needed bessel functions
            besselMic = zeros((nActiveModes, nMics), dtype='float64')
            besselGrid = zeros((nGrid, nActiveModes), dtype='float64')
            for cntUsed in range(nActiveModes):
                besselMic[cntUsed, :] = jn(modeProps[usedModesInd[cntUsed], 0], modeProps[usedModesInd[cntUsed], 2] * self.mpos_cyl[1, :] / self.env.R)
                besselGrid[:, cntUsed] = jn(modeProps[usedModesInd[cntUsed], 0], modeProps[usedModesInd[cntUsed], 2] * self.grid_cyl[1, :] / self.env.R)

            # calc greens function from all grids to all mics
            greens_func_Induct(modeProps[usedModes, 0], modeProps[usedModes, 3], modeProps[usedModes, 4], 
                      modeProps[usedModes, 5], modeProps[usedModes, 6], modeProps[usedModes, 7], 
                      self.mpos_cyl[0, :], self.mpos_cyl[2, :], 
                      self.grid_cyl[0, :], self.grid_cyl[2, :], 
                      besselMic, besselGrid, trans[cntFreq, :, :])
            
            if self.transfer_normalized:
                if (self.transfer_norm_ref_point == array([inf, inf, inf])).all():  # take center of array
                    pointOfRef = mean(self.mpos.mpos, axis=-1)
                else:  # take user defined point
                    pointOfRef = self.transfer_norm_ref_point
                pointOfRefCyl = cartToCyl(pointOfRef)
                if pointOfRefCyl[1] < 1e-5:  # all modes except (0,0) have no influence here
                    usedModesRef = (modeProps[:, 0:2] == [0,0]).sum(axis=1) == 2
                else:
                    usedModesRef = ones((modeProps.shape[0]), dtype='bool')
                transHelpArrayCenter = zeros((nGrid, 1), dtype='complex128')
                greens_func_Induct(modeProps[usedModesRef, 0], modeProps[usedModesRef, 3], modeProps[usedModesRef, 4],
                          modeProps[usedModesRef, 5], modeProps[usedModesRef, 6], modeProps[usedModesRef, 7], 
                          [pointOfRefCyl[0]], [pointOfRef[2]], 
                          self.grid_cyl[0, :], self.grid_cyl[2, :], 
                          array([1])[newaxis], ones((nGrid, 1)), transHelpArrayCenter)
                trans[cntFreq, :, :] /= transHelpArrayCenter
        return trans
    
    def _transfer_specific_gridpoint(self, ind):
        """
        Calculates the transfer function of ONE grid point. Is used in clean-sc
        when not the whole transfer matrix is needed.
        
        Parameters
        ----------
        ind : int
            Identifier for wich gridpoint the transfer funtion is calculated
        
        Returns
        -------
        trans : complex128[nFreqs, nMics]
        """
        # basically needed in clean-sc (freefield steering vectors), when not all transfer functions must be calculated.
        # In here this is only implemented for class compatibility with SteeringVector (basically senseless as the full transfer
        # vectors are calculated for any beamformer anyway).
        trans = self.transfer[:, ind, :]
        return trans
    
    def _beamformerCall(self, indFreq, rDiag, normFac, tupleCSM):
        """
        Manages the calling of the core beamformer functionality.
        
        Parameters
        ----------
        indFreq : int
            The index for which item of this classes property k the beamformer should be performed.
        rDiag : bool
            Should the diagonal of the csm should be considered in beamformer
            (rDiag==False) or not (rDiag==True).
        normFac : float
            Normalization factor for the beamforming result (e.g. removal of diag is compensated with this.)
        tupleCSM : either (csm,) or (eigValues, eigVectors)
            See information header of beamformerFreq in fastFuncs.py for information on those inputs.

        Returns
        -------
        The results of beamformerFreq, see fastFuncs.py 
        """
        steer = self.steer_vector[indFreq, :, :][newaxis]
        result = beamformerFreq('custom', rDiag, normFac, (steer,), tupleCSM)
        return result
    
    def _psfCall(self, freqInd, sourceInd, precision):
        """
        Manages the calling of the core psf functionality.
        
        Parameters
        ----------
        freqInd : int
            Index for which entry of Trait f the psf should be calculated.
        sourceInd : list of int
            Indices of gridpoints which are assumed to be sources.
            Normalization factor for the beamforming result (e.g. removal of diag is compensated with this.)
        precision : string ('float32' or 'float64')
            Precision of psf.

        Returns
        -------
        The psf [1, nGridPoints, len(sourceInd)]
        """
        # Perform steer^H * transfer (see information header in fastFuncs.calcPointSpreadFunction)
        prelim = dot(self.steer_vector[freqInd, :, :].conj(), self.transfer[freqInd, sourceInd, :].T)
        psf = (prelim * prelim.conj()).real.astype(precision)
        return psf[newaxis]


class BeamformerBase( HasPrivateTraits ):
    """
    Beamforming using the basic delay-and-sum algorithm in the frequency domain.
    """
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.c` for information.
    c = Property(desc="speed of sound")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.steer_type` for information.
    steer = Property(desc="type of steering vectors used")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.env` for information.
    env = Property()
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.grid` for information.
    grid = Property(desc="beamforming grid")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.mpos` for information.
    mpos = Property(desc="microphone geometry")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.r0` for information.
    r0 = Property(desc="array center to grid distances")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.rm` for information.
    rm = Property(desc="all array mics to grid distances")
    
    #: instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes,
    #: that contains information about the steering vector.
    steer_obj = Instance(SteeringVector(), SteeringVector)  # creates standard steering vector in constructor of BeamformerBase

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
        depends_on = ['freq_data.digest', 'r_diag', 'r_diag_norm', 'precision', 'steer_obj.digest'])

    # internal identifier
    ext_digest = Property( 
        depends_on = ['digest', 'freq_data.ind_low', 'freq_data.ind_high'], 
        )

    traits_view = View(
        [
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('r_diag', label='Diagonal removed')], 
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
    
    @cached_property
    def _get_ext_digest( self ):
        return digest( self, 'ext_digest' )
    
    def _get_c(self):
        return self.steer_obj.c
    
    def _get_steer(self):
        return self.steer_obj.steer_type
    
    def _get_env(self):
        return self.steer_obj.env
    
    def _get_grid(self):
        return self.steer_obj.grid
    
    def _get_mpos(self):
        return self.steer_obj.mpos
    
    def _get_r0(self):
        return self.steer_obj.r0
    
    def _get_rm(self):
        return self.steer_obj.rm
    
    def _set_c(self, c):
        self.steer_obj.c = c
    
    def _set_steer(self, steer):
        self.steer_obj.steer_type = steer
    
    def _set_env(self, env):
        self.steer_obj.env = env
    
    def _set_grid(self, grid):
        self.steer_obj.grid = grid
    
    def _set_mpos(self, mpos):
        self.steer_obj.mpos = mpos

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
            if  numchannels != self.steer_obj.mpos.num_mics or numchannels == 0:
                raise ValueError("%i channels do not fit %i mics" % (numchannels, self.steer_obj.mpos.num_mics))
            numfreq = self.freq_data.fftfreq().shape[0]# block_size/2 + 1
            precisionTuple = _precision(self.precision)
            if self.cached:
                H5cache.get_cache( self, self.freq_data.basename)
                if not name in self.h5f.root:
                    group = self.h5f.create_group(self.h5f.root, name)
                    shape = (numfreq, self.steer_obj.grid.size)
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
                ac = zeros((numfreq, self.steer_obj.grid.size), dtype=self.precision)
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        normFactor = self.sig_loss_norm()
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                csm = array(self.freq_data.csm[i[cntFreq]][newaxis], dtype='complex128')
                beamformerOutput = self.steer_obj._beamformerCall(cntFreq, self.r_diag, normFactor, (csm,))[0]
                if self.r_diag:  # set (unphysical) negative output values to 0
                    indNegSign = sign(beamformerOutput) < 0
                    beamformerOutput[indNegSign] = 0.0
                ac[i[cntFreq]] = beamformerOutput
                fr[i[cntFreq]] = True
    
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
        return h.reshape(self.steer_obj.grid.shape)


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
        ind = self.steer_obj.grid.indices(*sector)
        gshape = self.steer_obj.grid.shape
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
    digest = Property(depends_on = ['freq_data.digest', 'steer_obj.digest', 'r_diag', 'gamma'])
    
    #: Functional Beamforming is only well defined for full CSM
    r_diag = Enum(False, 
                  desc="False, as Functional Beamformer is only well defined for the full CSM")

    traits_view = View(
        [
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('gamma', label='Exponent', style='simple')], 
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        normFactor = self.sig_loss_norm()
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
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
                    csm = self.freq_data.csm[i[cntFreq]]
                    fill_diagonal(csm, 0)
                    csmRoot = fractional_matrix_power(csm, 1.0 / self.gamma)
                    beamformerOutput, steerNorm = self.steer_obj._beamformerCall(cntFreq, self.r_diag, 1.0, (csmRoot[newaxis],))
                    beamformerOutput /= steerNorm  # take normalized steering vec
                    
                    # set (unphysical) negative output values to 0
                    indNegSign = sign(beamformerOutput) < 0
                    beamformerOutput[indNegSign] = 0.0
                else:
                    eva = array(self.freq_data.eva[i[cntFreq]][newaxis], dtype='float64') ** (1.0 / self.gamma)
                    eve = array(self.freq_data.eve[i[cntFreq]][newaxis], dtype='complex128')
                    beamformerOutput, steerNorm = self.steer_obj._beamformerCall(cntFreq, self.r_diag, 1.0, (eva, eve))  # takes all EigVal into account
                    beamformerOutput /= steerNorm  # take normalized steering vec
                ac[i[cntFreq]] = (beamformerOutput ** self.gamma) * steerNorm * normFactor  # the normalization must be done outside the beamformer
                fr[i[cntFreq]] = True
            
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
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('c', label='Speed of sound')], 
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        nMics = self.freq_data.numchannels
        normFactor = self.sig_loss_norm() * nMics**2
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                csm = array(linalg.inv(array(self.freq_data.csm[i[cntFreq]], dtype='complex128')), order='C')[newaxis]
                beamformerOutput = self.steer_obj._beamformerCall(cntFreq, self.r_diag, normFactor, (csm,))[0]
                ac[i[cntFreq]] = 1.0 / beamformerOutput
                fr[i[cntFreq]] = True

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
        depends_on = ['freq_data.digest', 'steer_obj.digest', 'r_diag', 'n'])

    traits_view = View(
        [
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='Component No.', style='simple')], 
            [Item('r_diag', label='Diagonal removed')], 
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
    
    @property_depends_on('steer_obj.mpos, n')
    def _get_na( self ):
        na = self.n
        nm = self.steer_obj.mpos.num_mics
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        na = int(self.na)  # eigenvalue taken into account
        normFactor = self.sig_loss_norm()
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                eva = array(self.freq_data.eva[i[cntFreq]][newaxis], dtype='float64')
                eve = array(self.freq_data.eve[i[cntFreq]][newaxis], dtype='complex128')
                beamformerOutput = self.steer_obj._beamformerCall(cntFreq, self.r_diag, normFactor, (eva[:, na:na+1], eve[:, :, na:na+1]))[0]
                if self.r_diag:  # set (unphysical) negative output values to 0
                    indNegSign = sign(beamformerOutput) < 0
                    beamformerOutput[indNegSign] = 0
                ac[i[cntFreq]] = beamformerOutput
                fr[i[cntFreq]] = True

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
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='No. of sources', style='simple')], 
            [Item('c', label='Speed of sound')], 
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        nMics = self.freq_data.numchannels
        n = int(self.steer_obj.mpos.num_mics-self.na)
        normFactor = self.sig_loss_norm() * nMics**2
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                eva = array(self.freq_data.eva[i[cntFreq]][newaxis], dtype='float64')
                eve = array(self.freq_data.eve[i[cntFreq]][newaxis], dtype='complex128')
                beamformerOutput = self.steer_obj._beamformerCall(cntFreq, self.r_diag, normFactor, (eva[:, :n], eve[:, :, :n]))[0]
                ac[i[cntFreq]] = 4e-10*beamformerOutput.min() / beamformerOutput
                fr[i[cntFreq]] = True

class PointSpreadFunction (HasPrivateTraits):
    """
    The point spread function.
    
    This class provides tools to calculate the PSF depending on the used 
    microphone geometry, focus grid, flow environment, etc.
    The PSF is needed by several deconvolution algorithms to correct
    the aberrations when using simple delay-and-sum beamforming.
    """
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.grid` for information.
    grid = Property(desc="beamforming grid")
    
    #: Indices of grid points to calculate the PSF for.
    grid_indices = CArray( dtype=int, value=array([]), 
                     desc="indices of grid points for psf") #value=array([]), value=self.grid.pos(),
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.mpos` for information.
    mpos = Property(desc="microphone geometry")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.env` for information.
    env = Property()
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.c` for information.
    c = Property(desc="speed of sound")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.steer_type` for information.
    steer = Property(desc="type of steering vectors used")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.r0` for information.
    r0 = Property(desc="array center to grid distances")
    
    #: Dummy property for Backward compatibility. See :attr:`~acoular.fbeamform.SteeringVector.rm` for information.
    rm = Property(desc="all array mics to grid distances")
    
    #: instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes,
    #: that contains information about the steering vector.
    steer_obj = Trait(SteeringVector)

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
    digest = Property( depends_on = ['steer_obj.digest', 'precision'], cached = True)

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    def _get_c(self):
        return self.steer_obj.c
    
    def _get_steer(self):
        return self.steer_obj.steer_type
    
    def _get_env(self):
        return self.steer_obj.env
    
    def _get_grid(self):
        return self.steer_obj.grid
    
    def _get_mpos(self):
        return self.steer_obj.mpos
    
    def _get_r0(self):
        return self.steer_obj.r0
    
    def _get_rm(self):
        return self.steer_obj.rm
    
    def _set_c(self, c):
        self.steer_obj.c = c
    
    def _set_steer(self, steer):
        self.steer_obj.steer_type = steer
    
    def _set_env(self, env):
        self.steer_obj.env = env
    
    def _set_grid(self, grid):
        self.steer_obj.grid = grid
    
    def _set_mpos(self, mpos):
        self.steer_obj.mpos = mpos

    def _get_psf ( self ):
        """
        This is the :attr:`psf` getter routine.
        The point spread function is either loaded or calculated.
        """
        gs = self.steer_obj.grid.size
        if not self.grid_indices.size:
            self.grid_indices = arange(gs)
        name = 'psf' + self.digest
        H5cache.get_cache( self, name)
        fr = ('Hz_%.2f' % self.freq).replace('.', '_')
        precisionTuple = _precision(self.precision)
        
        # check wether self.freq is part of SteeringVector.f
        freqInSteerObjFreq = isclose(array(self.steer_obj.f), self.freq)
        if freqInSteerObjFreq.any():
            freqInd = flatnonzero(freqInSteerObjFreq)[0]
        else:
            warn('PointSpreadFunction.freq (%s Hz) was appended to PointSpreadFunction.steer_obj.f, '\
                 'as it was not an element of the original list!' % self.freq, Warning, stacklevel = 2)
            self.steer_obj.f.append(self.freq)
            freqInd = int(-1)
        
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
            if self.calcmode == 'single':
                for ind in g_ind_calc:
                    ac[:,ind] = self.steer_obj._psfCall(freqInd, [ind], self.precision)[0,:,0]
                    gp[ind] = True
            elif self.calcmode == 'full':
                gp[:] = True
                ac[:] = self.steer_obj._psfCall(freqInd, arange(self.steer_obj.grid.size), self.precision)[0,:,:]
            else: # 'block'
                hh = self.steer_obj._psfCall(freqInd, g_ind_calc, self.precision)
                indh = 0
                for ind in g_ind_calc:
                    gp[ind] = True
                    ac[:,ind] = hh[0,:,indh]
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
    steer_obj = Delegate('beamformer')
    
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        p = PointSpreadFunction(steer_obj=self.steer_obj, calcmode=self.calcmode, precision=self.psf_precision)
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                y = array(self.beamformer.result[i[cntFreq]])
                x = y.copy()
                p.freq = self.steer_obj.f[cntFreq]
                psf = p.psf[:]
                damasSolverGaussSeidel(psf, y, self.n_iter, self.damp, x)
                ac[i[cntFreq]] = x
                fr[i[cntFreq]] = True

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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        p = PointSpreadFunction(steer_obj=self.steer_obj, calcmode=self.calcmode, precision=self.psf_precision)
        unit = self.unit_mult
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                y = self.beamformer.result[i[cntFreq]] * unit
                p.freq = self.steer_obj.f[cntFreq]
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
                    ac[i[cntFreq]] = resopt / unit
                else: # sklearn models
                    model.fit(psf,y)
                    ac[i[cntFreq]] = model.coef_[:] / unit
                
                fr[i[cntFreq]] = True

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
    steer_obj = Delegate('beamformer')

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
        depends_on = ['freq_data.digest', 'steer_obj.digest', 'r_diag', 'n', 'damp', 'stopn'])

    traits_view = View(
        [
#            [Item('mpos{}', style='custom')], 
#            [Item('grid', style='custom'), '-<>'], 
            [Item('n', label='No. of iterations', style='simple')], 
            [Item('r_diag', label='Diagonal removed')], 
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        result = zeros((self.steer_obj.grid.size), 'f') 
        normFac = self.sig_loss_norm()
        if not self.n:
            J = numchannels*2
        else:
            J = self.n
        powers = zeros(J, 'd')
        
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                csm = array(self.freq_data.csm[i[cntFreq]][newaxis], dtype='complex128', copy=1)
                h = self.steer_obj._beamformerCall(cntFreq, self.r_diag, normFactor, (csm,))[0]

                # CLEANSC Iteration
                result *= 0.0
                for j in range(J):
                    xi_max = h.argmax() #index of maximum
                    powers[j] = hmax = h[0, xi_max] #maximum
                    result[xi_max] += self.damp * hmax
                    if  j > self.stopn and hmax > powers[j-self.stopn]:
                        break
                    trans = self.steer_obj._transfer_specific_gridpoint(xi_max)[cntFreq, :][newaxis]
                    wmax = self.steer_obj.calc_steer_vector(trans)[0] * sqrt(normFac)
                    wmax = wmax.conj()  # as old code worked with conjugated csm..should be updated
                    hh = wmax.copy()
                    D1 = dot(csm[0].T - diag(diag(csm[0])), wmax)/hmax
                    ww = wmax.conj()*wmax
                    for m in range(20):
                        H = hh.conj()*hh
                        hh = (D1+H*wmax)/sqrt(1+dot(ww, H))
                    hh = hh[:, newaxis]
                    csm1 = hmax*(hh*hh.conj().T)[newaxis, :, :]
                    h1 = self.steer_obj._beamformerCall(cntFreq, self.r_diag, normFactor, (array((hmax, ))[newaxis, :], hh[newaxis, :].conjugate()))[0]
                    h -= self.damp * h1
                    csm -= self.damp * csm1.transpose(0,2,1)
                ac[i[cntFreq]] = result
                fr[i[cntFreq]] = True

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
    steer_obj = Delegate('beamformer')
    
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        gs = self.steer_obj.grid.size
        
        if self.calcmode == 'full':
            print('Warning: calcmode = \'full\', slow CLEAN performance. Better use \'block\' or \'single\'.')
        p = PointSpreadFunction(steer_obj=self.steer_obj, calcmode=self.calcmode, precision=self.psf_precision)
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                p.freq = self.steer_obj.f[cntFreq]
                dirty = self.beamformer.result[i[cntFreq]]
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
                
                ac[i[cntFreq]] = clean            
                fr[i[cntFreq]] = True

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
        depends_on = ['freq_data.digest', 'c', 'alpha', 'method', 'max_iter', 'unit_mult', 'r_diag', 'steer_obj.digest'], 
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
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        nc = self.freq_data.numchannels
        numpoints = self.steer_obj.grid.size
        unit = self.unit_mult
        hh = zeros((1, numpoints, nc), dtype='D')

        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                csm = array(self.freq_data.csm[i[cntFreq]], dtype='complex128',copy=1)

                hh = self.steer_obj.transfer
                h = hh[cntFreq].T
                
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
                    ac[i[cntFreq]] , x = nnls(A,R.flat)
                    ac[i[cntFreq]] /= unit
                else:
                    model.fit(A,R[:,0])
                    ac[i[cntFreq]] = model.coef_[:] / unit
                fr[i[cntFreq]] = True

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
        depends_on = ['steer_obj.digest', 'freq_data.digest', \
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
        nm = self.steer_obj.mpos.num_mics
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
        i = self.freq_data.indices
        self.steer_obj.f = (self.freq_data.fftfreq()[i]).tolist()
        n = int(self.n)  
        m = int(self.m)                             #number of eigenvalues
        numchannels = self.freq_data.numchannels   #number of channels
        numpoints = self.steer_obj.grid.size
        hh = zeros((1, numpoints, numchannels), dtype='D')
        
        #Generate a cross spectral matrix, and perform the eigenvalue decomposition
        for cntFreq in range(len(i)):
            if not fr[i[cntFreq]]:
                #for monopole and source strenght Q needs to define density
                #calculate a transfer matrix A 
                hh = self.steer_obj.transfer
                A=hh[cntFreq].T                 
                #eigenvalues and vectors               
                csm = array(self.freq_data.csm[i[cntFreq]], dtype='complex128',copy=1)
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
                ac[i[cntFreq]] = zeros([1,numpoints])
                ac[i[cntFreq],locpoints] = sum(absolute(qi[:,locpoints])**2,axis=0)
                fr[i[cntFreq]] = True    

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
