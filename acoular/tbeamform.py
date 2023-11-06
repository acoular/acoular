# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements beamformers in the time domain.

.. autosummary::
    :toctree: generated/

    BeamformerTime
    BeamformerTimeTraj
    BeamformerTimeSq
    BeamformerTimeSqTraj
    BeamformerCleant
    BeamformerCleantTraj
    BeamformerCleantSq
    BeamformerCleantSqTraj
    IntegratorSectorTime
"""

# imports from other packages
from __future__ import print_function, division
from numpy import array, newaxis, empty, sqrt, arange, r_, zeros, \
histogram, unique, dot, where, s_ , sum, isscalar, full, ceil, argmax,\
interp,concatenate, float32, float64, int32, int64
from numpy.linalg import norm
from traits.api import Float, CArray, Property, Trait, Bool, Delegate, \
cached_property, List, Instance, Range, Int, Enum
from traits.trait_errors import TraitError
from warnings import warn

# acoular imports
from .internal import digest
from .grids import RectGrid
from .trajectory import Trajectory
from .tprocess import TimeInOut
from .fbeamform import SteeringVector, L_p
from .tfastfuncs import _delayandsum4, _delayandsum5, \
    _steer_I, _steer_II, _steer_III, _steer_IV, _delays, _modf


def const_power_weight( bf ):
    """
    Internal helper function for :class:`BeamformerTime`
    
    Provides microphone weighting 
    to make the power per unit area of the
    microphone array geometry constant.
    
    Parameters
    ----------
    bf: :class:`BeamformerTime` object
        
          
    Returns
    -------
    array of floats
        The weight factors.
    """

    r = bf.steer.env._r(zeros((3, 1)), bf.steer.mics.mpos) # distances to center
    # round the relative distances to one decimal place
    r = (r/r.max()).round(decimals=1)
    ru, ind = unique(r, return_inverse=True)
    ru = (ru[1:]+ru[:-1])/2
    count, bins = histogram(r, r_[0, ru, 1.5*r.max()-0.5*ru[-1]])
    bins *= bins
    weights = sqrt((bins[1:]-bins[:-1])/count)
    weights /= weights.mean()
    return weights[ind]

# possible choices for spatial weights
possible_weights = {'none':None, 
                    'power':const_power_weight}


class BeamformerTime( TimeInOut ):
    """
    Provides a basic time domain beamformer with time signal output
    for a spatially fixed grid.
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
        if type(steer) == SteeringVector:
            # This condition may be replaced at a later time by: isinstance(steer, SteeringVector): -- (derived classes allowed)
            self._steer_obj = steer
        elif steer in ('true level', 'true location', 'classic', 'inverse'):
            # Type of steering vectors, see also :ref:`Sarradj, 2012<Sarradj2012>`.
            warn("Deprecated use of 'steer' trait. "
                 "Please use object of class 'SteeringVector' in the future.", 
                 Warning, stacklevel = 2)
            self._steer_obj.steer_type = steer
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

    #: Number of channels in output (=number of grid points).
    numchannels = Delegate('grid', 'size')

    #: Spatial weighting function.
    weights = Trait('none', possible_weights, 
        desc="spatial weighting function")
    # (from timedomain.possible_weights)

    # buffer with microphone time signals used for processing. Internal use 
    buffer = CArray(desc="buffer containing microphone signals")
    
    # index indicating position of current processing sample. Internal use.
    bufferIndex = Int(desc="index indicating position in buffer")

    # internal identifier
    digest = Property( 
        depends_on = ['_steer_obj.digest', 'source.digest', 'weights', '__class__'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    def _get_weights(self):
        if self.weights_:
            w = self.weights_(self)[newaxis]
        else:
            w = 1.0
        return w

    def _fill_buffer(self,num):
        """ generator that fills the signal buffer """
        weights = self._get_weights()
        for block in self.source.result(num):
            block *= weights
            ns = block.shape[0]
            bufferSize = self.buffer.shape[0]
            self.buffer[0:(bufferSize-ns)] = self.buffer[-(bufferSize-ns):] 
            self.buffer[-ns:,:] = block
            self.bufferIndex -= ns
            yield
         
    def result( self, num=2048 ):
        """
        Python generator that yields the *squared* deconvolved beamformer 
        output with optional removal of autocorrelation block-wise.
        
        Parameters
        ----------
        num : integer, defaults to 2048
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape  \
        (num, :attr:`~BeamformerTime.numchannels`). 
            :attr:`~BeamformerTime.numchannels` is usually very \
            large (number of grid points).
            The last block may be shorter than num. \
            The output starts for signals that were emitted 
            from the grid at `t=0`.
        """
        # initialize values
        fdtype = float64
        idtype = int64
        numMics = self.steer.mics.num_mics
        n_index = arange(0,num+1)[:,newaxis]
        c = self.steer.env.c/self.source.sample_freq
        amp = empty((1,self.grid.size,numMics),dtype=fdtype)
        #delays = empty((1,self.grid.size,numMics),dtype=fdtype)
        d_index = empty((1,self.grid.size,numMics),dtype=idtype)
        d_interp2 = empty((1,self.grid.size,numMics),dtype=fdtype)
        _steer_III(self.rm[newaxis,:,:],self.r0[newaxis,:],amp)
        _delays(self.rm[newaxis,:,:], c, d_interp2, d_index)
        amp.shape = amp.shape[1:]
        #delays.shape = delays.shape[1:]
        d_index.shape = d_index.shape[1:]
        d_interp2.shape = d_interp2.shape[1:]
        maxdelay = int((self.rm/c).max())+2 + num # +2 because of interpolation
        initialNumberOfBlocks = int(ceil(maxdelay/num))
        bufferSize=initialNumberOfBlocks*num
        self.buffer = zeros((bufferSize,numMics), dtype=float)
        self.bufferIndex = bufferSize # indexing current time sample in buffer 
        fill_buffer_generator = self._fill_buffer(num)
        for _ in range(initialNumberOfBlocks):
            next(fill_buffer_generator)

        # start processing
        flag = True
        while flag:
            samplesleft = self.buffer.shape[0]-self.bufferIndex
            if samplesleft-maxdelay <= 0:
                num += samplesleft-maxdelay
                maxdelay += samplesleft-maxdelay
                n_index = arange(0,num+1)[:,newaxis]
                flag=False
            # init step
            p_res = array(
                self.buffer[self.bufferIndex:self.bufferIndex+maxdelay,:])
            Phi, autopow = self.delay_and_sum(num,p_res,d_interp2,d_index,amp)
            if 'Cleant' not in self.__class__.__name__:
                if 'Sq' not in self.__class__.__name__:
                    yield Phi[:num]
                elif self.r_diag:                 
                    yield (Phi[:num]**2 - autopow[:num]).clip(min=0)
                else:
                    yield Phi[:num]**2
            else:
                Gamma = zeros(Phi.shape)
                Gamma_autopow = zeros(Phi.shape)
                J = 0
                # deconvolution 
                while (J < self.n_iter):
                    # print(f"start clean iteration {J+1} of max {self.n_iter}")
                    if self.r_diag:
                        powPhi = (Phi[:num]**2-autopow).sum(0).clip(min=0)
                    else:
                        powPhi = (Phi[:num]**2).sum(0)
                    imax = argmax(powPhi)
                    t_float = d_interp2[imax] + d_index[imax] + n_index
                    t_ind = t_float.astype(int64)
                    for m in range(numMics): 
                        p_res[t_ind[:num+1,m],m] -= self.damp*interp(t_ind[:num+1,m],
                                                                t_float[:num,m],
                                                                    Phi[:num,imax]*self.r0[imax]/self.rm[imax,m],
                                                                    )
                    nextPhi, nextAutopow = self.delay_and_sum(num,p_res,d_interp2,d_index,amp)
                    if self.r_diag:
                        pownextPhi = (nextPhi[:num]**2-nextAutopow).sum(0).clip(min=0)
                    else:
                        pownextPhi = (nextPhi[:num]**2).sum(0)
                    # print(f"total signal power: {powPhi.sum()}")
                    if pownextPhi.sum() < powPhi.sum(): # stopping criterion
                        Gamma[:num,imax] += self.damp*Phi[:num,imax]
                        Gamma_autopow[:num,imax] = autopow[:num,imax].copy()
                        Phi=nextPhi
                        autopow=nextAutopow
                        # print(f"clean max: {L_p((Gamma**2).sum(0)/num).max()} dB")
                        J += 1
                    else:
                        break
                if 'Sq' not in self.__class__.__name__:
                    yield Gamma[:num]
                elif self.r_diag:                 
                    yield Gamma[:num]**2 - (self.damp**2)*Gamma_autopow[:num]
                else:
                    yield Gamma[:num]**2
            self.bufferIndex += num
            try:
                next(fill_buffer_generator)
            except: 
                pass

    def delay_and_sum(self,num,p_res,d_interp2,d_index,amp): 
        ''' standard delay-and-sum method ''' 
        result = empty((num, self.grid.size), dtype=float) # output array
        autopow = empty((num, self.grid.size), dtype=float) # output array
        _delayandsum4(p_res, d_index, d_interp2, amp, result, autopow)
        return result, autopow          
            

class BeamformerTimeSq( BeamformerTime ):
    """
    Provides a time domain beamformer with time-dependend
    power signal output and possible autopower removal
    for a spatially fixed grid.
    """
    
    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, 
        desc="removal of diagonal")

    # internal identifier
    digest = Property( 
        depends_on = ['_steer_obj.digest', 'source.digest', 'r_diag', \
                      'weights', '__class__'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        

class BeamformerTimeTraj( BeamformerTime ):
    """
    Provides a basic time domain beamformer with time signal output
    for a grid moving along a trajectory.
    """


    #: :class:`~acoular.trajectory.Trajectory` or derived object.
    #: Start time is assumed to be the same as for the samples.
    trajectory = Trait(Trajectory, 
        desc="trajectory of the grid center")

    #: Reference vector, perpendicular to the y-axis of moving grid.
    rvec = CArray( dtype=float, shape=(3, ), value=array((0, 0, 0)), 
        desc="reference vector")
    
    #: Considering of convective amplification in beamforming formula.
    conv_amp = Bool(False, 
        desc="determines if convective amplification of source is considered")

    #: Floating point and integer precision
    precision = Trait(64, [32,64], 
        desc="numeric precision")

    # internal identifier
    digest = Property( 
        depends_on = ['_steer_obj.digest', 'source.digest', 'weights', 'precision',\
                      'rvec','conv_amp','trajectory.digest', '__class__'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    def get_moving_gpos(self):
        """
        Python generator that yields the moving grid coordinates samplewise 
        """
        def cross(a, b):
            """ cross product for fast computation
                because numpy.cross is ultra slow in this case
            """
            return array([a[1]*b[2] - a[2]*b[1],
                        a[2]*b[0] - a[0]*b[2],
                        a[0]*b[1] - a[1]*b[0]])

        start_t = 0.0
        gpos = self.grid.pos()
        trajg = self.trajectory.traj( start_t, delta_t=1/self.source.sample_freq)
        trajg1 = self.trajectory.traj( start_t, delta_t=1/self.source.sample_freq, 
                                  der=1)
        rflag = (self.rvec == 0).all() #flag translation vs. rotation
        if rflag:
            for g in trajg:
                # grid is only translated, not rotated
                tpos = gpos + array(g)[:, newaxis]
                yield tpos
        else:
            for (g,g1) in zip(trajg,trajg1):
                # grid is both translated and rotated
                loc = array(g) #translation array([0., 0.4, 1.])
                dx = array(g1) #direction vector (new x-axis)
                dy = cross(self.rvec, dx) # new y-axis
                dz = cross(dx, dy) # new z-axis
                RM = array((dx, dy, dz)).T # rotation matrix
                RM /= sqrt((RM*RM).sum(0)) # column normalized
                tpos = dot(RM, gpos)+loc[:, newaxis] # rotation+translation
#                print(loc[:])
                yield tpos

    def get_macostheta(self,g1,tpos,rm):
        vvec = array(g1) # velocity vector
        ma = norm(vvec)/self.steer.env.c # machnumber
        fdv = (vvec/sqrt((vvec*vvec).sum()))[:, newaxis] # unit vecor velocity
        mpos = self.steer.mics.mpos[:, newaxis, :]
        rmv = tpos[:, :, newaxis]-mpos
        return (ma*sum(rmv.reshape((3, -1))*fdv, 0)\
                                  /rm.reshape(-1)).reshape(rm.shape)

    def get_r0( self, tpos ):
        if isscalar(self.steer.ref) and self.steer.ref > 0:
            return self.steer.ref#full((self.steer.grid.size,), self.steer.ref)
        else:
            return self.env._r(tpos)

    def increase_buffer( self, num ): 
        ar = zeros((num,self.steer.mics.num_mics), dtype=self.buffer.dtype)
        self.buffer = concatenate((ar,self.buffer), axis=0)
        self.bufferIndex += num

    def result( self, num=2048 ):
        """
        Python generator that yields the deconvolved output block-wise. 
        
        Parameters
        ----------
        num : integer, defaults to 2048
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape  \
        (num, :attr:`~BeamformerTime.numchannels`). 
            :attr:`~BeamformerTime.numchannels` is usually very \
            large (number of grid points).
            The last block may be shorter than num. \
            The output starts for signals that were emitted 
            from the grid at `t=0`.
        """
        # initialize values
        if self.precision==64:
            fdtype = float64
            idtype = int64
        else:
            fdtype = float32
            idtype = int32
        w = self._get_weights()
        c = self.steer.env.c/self.source.sample_freq
        numMics = self.steer.mics.num_mics
        mpos = self.steer.mics.mpos.astype(fdtype)
        m_index = arange(numMics, dtype=idtype)
        n_index = arange(num,dtype=idtype)[:,newaxis]
        blockrm = empty((num,self.grid.size,numMics),dtype=fdtype)
        blockrmconv = empty((num,self.grid.size,numMics),dtype=fdtype)
        amp = empty((num,self.grid.size,numMics),dtype=fdtype)
        #delays = empty((num,self.grid.size,numMics),dtype=fdtype)
        d_index = empty((num,self.grid.size,numMics),dtype=idtype)
        d_interp2 = empty((num,self.grid.size,numMics),dtype=fdtype)
        blockr0 = empty((num,self.grid.size),dtype=fdtype)
        self.buffer = zeros((2*num,numMics), dtype=fdtype)
        self.bufferIndex = self.buffer.shape[0] 
        movgpos = self.get_moving_gpos() # create moving grid pos generator
        movgspeed = self.trajectory.traj(0.0, delta_t=1/self.source.sample_freq, 
              der=1)
        fill_buffer_generator = self._fill_buffer(num)
        for i in range(2): 
            next(fill_buffer_generator)
        
        # preliminary implementation of different steering vectors
        steer_func = {'classic' : _steer_I,
                'inverse' : _steer_II,
                'true level' : _steer_III,
                'true location' : _steer_IV
                }[self.steer.steer_type]

        # start processing
        flag = True
        dflag = True # data is available 
        while flag:
            for i in range(num):
                tpos = next(movgpos).astype(fdtype)
                rm = self.steer.env._r( tpos, mpos )#.astype(fdtype) 
                blockr0[i,:] = self.get_r0(tpos)
                blockrm[i,:,:] = rm
                if self.conv_amp:
                    ht = next(movgspeed)
                    blockrmconv[i,:,:] = rm*(1-self.get_macostheta(ht,tpos,rm))**2
            if self.conv_amp:
                steer_func(blockrmconv, blockr0, amp)
            else:
                steer_func(blockrm, blockr0, amp)
            _delays(blockrm, c, d_interp2, d_index)
            #_modf(delays, d_interp2, d_index)
            maxdelay = (d_index.max((1,2)) + arange(0,num)).max()+2 # + because of interpolation
            # increase buffer size because of greater delays
            while maxdelay > self.buffer.shape[0] and dflag:
                self.increase_buffer(num)
                try:
                    next(fill_buffer_generator)
                except:
                    dflag = False
            samplesleft = self.buffer.shape[0]-self.bufferIndex
            # last block may be shorter
            if samplesleft-maxdelay <= 0:
                num = sum((d_index.max((1,2))+1+arange(0,num)) < samplesleft)
                n_index = arange(num,dtype=idtype)[:,newaxis]
                flag=False
            # init step
            p_res = self.buffer[self.bufferIndex:self.bufferIndex+maxdelay,:].copy()
            Phi, autopow = self.delay_and_sum(num,p_res,d_interp2,d_index,amp)
            if 'Cleant' not in self.__class__.__name__:
                if 'Sq' not in self.__class__.__name__:
                    yield Phi[:num]
                elif self.r_diag:                 
                    yield (Phi[:num]**2 - autopow[:num]).clip(min=0)
                else:
                    yield Phi[:num]**2
            else:
                # choose correct distance
                if self.conv_amp: 
                    blockrm1 = blockrmconv
                else:
                    blockrm1 = blockrm
                Gamma = zeros(Phi.shape,dtype=fdtype)
                Gamma_autopow = zeros(Phi.shape,dtype=fdtype)
                J = 0
                t_ind = arange(p_res.shape[0],dtype=idtype)
                # deconvolution 
                while (J < self.n_iter):
                    # print(f"start clean iteration {J+1} of max {self.n_iter}")
                    if self.r_diag:
                        powPhi = (Phi[:num]*Phi[:num]-autopow).sum(0).clip(min=0)
                    else:
                        powPhi = (Phi[:num]*Phi[:num]).sum(0)
                    # find index of max power focus point
                    imax = argmax(powPhi)
                    # find backward delays
                    t_float = (d_interp2[:num,imax,m_index] + \
                               d_index[:num,imax,m_index] + \
                               n_index).astype(fdtype)
                    # determine max/min delays in sample units
                    # + 2 because we do not want to extrapolate behind the last sample
                    ind_max = t_float.max(0).astype(idtype)+2 
                    ind_min = t_float.min(0).astype(idtype)
                    # store time history at max power focus point
                    h = Phi[:num,imax]*blockr0[:num,imax]
                    for m in range(numMics):
                        # subtract interpolated time history from microphone signals
                        p_res[ind_min[m]:ind_max[m],m] -= self.damp*interp(
                            t_ind[ind_min[m]:ind_max[m]], 
                            t_float[:num,m],
                            h/blockrm1[:num,imax,m],
                                )
                    nextPhi, nextAutopow = self.delay_and_sum(num,p_res,d_interp2,d_index,amp)
                    if self.r_diag:
                        pownextPhi = (nextPhi[:num]*nextPhi[:num]-nextAutopow).sum(0).clip(min=0)
                    else:
                        pownextPhi = (nextPhi[:num]*nextPhi[:num]).sum(0)                 
                    # print(f"total signal power: {powPhi.sum()}")
                    if pownextPhi.sum() < powPhi.sum(): # stopping criterion
                        Gamma[:num,imax] += self.damp*Phi[:num,imax]
                        Gamma_autopow[:num,imax] = autopow[:num,imax].copy()
                        Phi=nextPhi
                        autopow=nextAutopow
                        # print(f"clean max: {L_p((Gamma**2).sum(0)/num).max()} dB")
                        J += 1
                    else:
                        break
                if 'Sq' not in self.__class__.__name__:
                    yield Gamma[:num]
                elif self.r_diag: 
                    yield Gamma[:num]**2 - (self.damp**2)*Gamma_autopow[:num]
                else:
                    yield Gamma[:num]**2
            self.bufferIndex += num
            try:
                next(fill_buffer_generator)
            except: 
                dflag = False
                pass

    def delay_and_sum(self,num,p_res,d_interp2,d_index,amp): 
        ''' standard delay-and-sum method ''' 
        if self.precision==64:
            fdtype = float64
        else:
            fdtype = float32
        result = empty((num, self.grid.size), dtype=fdtype) # output array
        autopow = empty((num, self.grid.size), dtype=fdtype) # output array
        _delayandsum5(p_res, d_index, d_interp2, amp, result, autopow)
        return result, autopow          
  
        
class BeamformerTimeSqTraj( BeamformerTimeSq, BeamformerTimeTraj ):
    """
    Provides a time domain beamformer with time-dependent
    power signal output and possible autopower removal
    for a grid moving along a trajectory.
    """
    
    # internal identifier
    digest = Property( 
        depends_on = ['_steer_obj.digest', 'source.digest', 'r_diag', 'weights', 'precision',\
                      'rvec','conv_amp','trajectory.digest', '__class__'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
                               

class BeamformerCleant( BeamformerTime ):
    """
    CLEANT deconvolution method, see :ref:`Cousson et al., 2019<Cousson2019>`.
    
    An implementation of the CLEAN method in time domain. This class can only 
    be used for static sources.
    """

    #: Boolean flag, always False
    r_diag = Enum(False, 
                  desc="False, as we do not remove autopower in this beamformer")

    #: iteration damping factor also referred as loop gain in Cousson et al. 
    #: defaults to 0.6
    damp = Range(0.01, 1.0, 0.6, 
        desc="damping factor (loop gain)")

    #: max number of iterations
    n_iter = Int(100, 
        desc="maximum number of iterations")
     
    # internal identifier
    digest = Property( 
        depends_on = ['_steer_obj.digest', 'source.digest', 'weights',  \
                      '__class__','damp','n_iter'],
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)


class BeamformerCleantSq( BeamformerCleant ):
    """
    CLEANT deconvolution method, see :ref:`Cousson et al., 2019<Cousson2019>`
    with optional removal of autocorrelation.
    
    An implementation of the CLEAN method in time domain. This class can only 
    be used for static sources.
    """

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, 
        desc="removal of diagonal")

    # internal identifier
    digest = Property( 
        depends_on = ['_steer_obj.digest', 'source.digest', 'weights',  \
                      '__class__','damp','n_iter','r_diag'],
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    

class BeamformerCleantTraj( BeamformerCleant, BeamformerTimeTraj ):
    """
    CLEANT deconvolution method, see :ref:`Cousson et al., 2019<Cousson2019>`.
    
    An implementation of the CLEAN method in time domain for moving sources
    with known trajectory. 
    """
    #: Floating point and integer precision
    precision = Trait(32, [32,64], 
        desc="numeric precision")

    # internal identifier
    digest = Property( 
        depends_on = ['_steer_obj.digest', 'source.digest', 'weights', 'precision', \
                      '__class__','damp','n_iter', 'rvec','conv_amp',
                      'trajectory.digest'],
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    

class BeamformerCleantSqTraj( BeamformerCleantTraj, BeamformerTimeSq ):
    """
    CLEANT deconvolution method, see :ref:`Cousson et al., 2019<Cousson2019>`
    with optional removal of autocorrelation.
    
    An implementation of the CLEAN method in time domain for moving sources
    with known trajectory. 
    """

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, 
        desc="removal of diagonal")

    # internal identifier
    digest = Property( 
        depends_on = ['_steer_obj.digest', 'source.digest', 'weights', 'precision', \
                      '__class__','damp','n_iter', 'rvec','conv_amp',
                      'trajectory.digest','r_diag'],
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    

class IntegratorSectorTime( TimeInOut ):
    """
    Provides an Integrator in the time domain.
    """

    #: :class:`~acoular.grids.RectGrid` object that provides the grid locations.
    grid = Trait(RectGrid, 
        desc="beamforming grid")
        
    #: List of sectors in grid
    sectors = List()

    #: Clipping, in Dezibel relative to maximum (negative values)
    clip = Float(-350.0)

    #: Number of channels in output (= number of sectors).
    numchannels = Property( depends_on = ['sectors', ])

    # internal identifier
    digest = Property( 
        depends_on = ['sectors', 'clip', 'grid.digest', 'source.digest', \
        '__class__'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    @cached_property
    def _get_numchannels ( self ):
        return len(self.sectors)

    def result( self, num=1 ):
        """
        Python generator that yields the source output integrated over the given 
        sectors, block-wise.
        
        Parameters
        ----------
        num : integer, defaults to 1
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`). 
        :attr:`numchannels` is the number of sectors.
        The last block may be shorter than num.
        """
        inds = [self.grid.indices(*sector) for sector in self.sectors]
        gshape = self.grid.shape
        o = empty((num, self.numchannels), dtype=float) # output array
        for r in self.source.result(num):
            ns = r.shape[0]
            mapshape = (ns,) + gshape
            rmax = r.max()
            rmin = rmax * 10**(self.clip/10.0)
            r = where(r>rmin, r, 0.0)
            i = 0
            for ind in inds:
                h = r[:].reshape(mapshape)[ (s_[:],) + ind ]
                o[:ns, i] = h.reshape(h.shape[0], -1).sum(axis=1)
                i += 1
            yield o[:ns]