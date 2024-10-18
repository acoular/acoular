# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
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
from warnings import warn

from numpy import (
    arange,
    argmax,
    array,
    ceil,
    dot,
    empty,
    float32,
    float64,
    histogram,
    int32,
    int64,
    interp,
    isscalar,
    newaxis,
    r_,
    s_,
    sqrt,
    sum,
    unique,
    where,
    zeros,
)
from scipy.linalg import norm
from traits.api import Bool, CArray, Delegate, Enum, Float, Instance, Int, List, Property, Range, Trait, cached_property
from traits.trait_errors import TraitError

from .base import SamplesGenerator, TimeOut
from .fbeamform import SteeringVector
from .grids import RectGrid

# acoular imports
from .internal import digest
from .tfastfuncs import _delayandsum4, _delayandsum5, _delays
from .tools.utils import SamplesBuffer
from .trajectory import Trajectory


def const_power_weight(bf):
    """Internal helper function for :class:`BeamformerTime`.

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
    r = bf.steer.env._r(zeros((3, 1)), bf.steer.mics.mpos)  # distances to center
    # round the relative distances to one decimal place
    r = (r / r.max()).round(decimals=1)
    ru, ind = unique(r, return_inverse=True)
    ru = (ru[1:] + ru[:-1]) / 2
    count, bins = histogram(r, r_[0, ru, 1.5 * r.max() - 0.5 * ru[-1]])
    bins *= bins
    weights = sqrt((bins[1:] - bins[:-1]) / count)
    weights /= weights.mean()
    return weights[ind]


# possible choices for spatial weights
possible_weights = {'none': None, 'power': const_power_weight}


class BeamformerTime(TimeOut):
    """Provides a basic time domain beamformer with time signal output
    for a spatially fixed grid.
    """

    #: Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes
    # that contains information about the steering vector. This is a private trait.
    # Do not set this directly, use `steer` trait instead.
    _steer_obj = Instance(SteeringVector(), SteeringVector)

    #: :class:`~acoular.fbeamform.SteeringVector` or derived object.
    #: Defaults to :class:`~acoular.fbeamform.SteeringVector` object.
    steer = Property(desc='steering vector object')

    def _get_steer(self):
        return self._steer_obj

    def _set_steer(self, steer):
        if type(steer) == SteeringVector:
            # This condition may be replaced at a later time by: isinstance(steer, SteeringVector): -- (derived classes allowed)
            self._steer_obj = steer
        elif steer in ('true level', 'true location', 'classic', 'inverse'):
            # Type of steering vectors, see also :ref:`Sarradj, 2012<Sarradj2012>`.
            msg = (
                "Deprecated use of 'steer' trait. Please use the 'steer' with an object of class "
                "'SteeringVector'. Using a string to specify the steer type will be removed in "
                'version 25.01.'
            )
            warn(
                msg,
                DeprecationWarning,
                stacklevel=2,
            )
            self._steer_obj.steer_type = steer
        else:
            raise (TraitError(args=self, name='steer', info='SteeringVector', value=steer))

    # --- List of backwards compatibility traits and their setters/getters -----------

    # :class:`~acoular.environments.Environment` or derived object.
    # Deprecated! Only kept for backwards compatibility.
    # Now governed by :attr:`steer` trait.
    env = Property()

    def _get_env(self):
        return self._steer_obj.env

    def _set_env(self, env):
        msg = (
            "Deprecated use of 'env' trait. Please use the 'steer' trait with an object of class  "
            "'SteeringVector'. The 'env' trait will be removed in version 25.01."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        self._steer_obj.env = env

    # The speed of sound.
    # Deprecated! Only kept for backwards compatibility.
    # Now governed by :attr:`steer` trait.
    c = Property()

    def _get_c(self):
        return self._steer_obj.env.c

    def _set_c(self, c):
        msg = (
            "Deprecated use of 'c' trait. Please use the 'steer' trait with an object of class  "
            "'SteeringVector'. The 'c' trait will be removed in version 25.01."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        self._steer_obj.env.c = c

    # :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    # Deprecated! Only kept for backwards compatibility.
    # Now governed by :attr:`steer` trait.
    grid = Property()

    def _get_grid(self):
        return self._steer_obj.grid

    def _set_grid(self, grid):
        msg = (
            "Deprecated use of 'grid' trait. Please use the 'steer' trait with an object of class  "
            "'SteeringVector'. The 'grid' trait will be removed in version 25.01."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        self._steer_obj.grid = grid

    # :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    # Deprecated! Only kept for backwards compatibility.
    # Now governed by :attr:`steer` trait
    mpos = Property()

    def _get_mpos(self):
        return self._steer_obj.mics

    def _set_mpos(self, mpos):
        msg = (
            "Deprecated use of 'mpos' trait. Please use the 'steer' trait with an object of class  "
            "'SteeringVector' which has an attribute 'mics'. The 'mpos' trait will be removed in version 25.01."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        self._steer_obj.mics = mpos

    # Sound travel distances from microphone array center to grid points (r0)
    # and all array mics to grid points (rm). Readonly.
    # Deprecated! Only kept for backwards compatibility.
    # Now governed by :attr:`steer` trait
    r0 = Property()

    def _get_r0(self):
        msg = (
            "Deprecated use of 'r0' trait. Please use the 'steer' trait with an object of class  "
            "'SteeringVector'. The 'r0' trait will be removed in version 25.01."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._steer_obj.r0

    rm = Property()

    def _get_rm(self):
        msg = (
            "Deprecated use of 'rm' trait. Please use the 'steer' trait with an object of class  "
            "'SteeringVector'. The 'rm' trait will be removed in version 25.01."
        )
        warn(msg, DeprecationWarning, stacklevel=2)
        return self._steer_obj.rm

    # --- End of backwards compatibility traits --------------------------------------

    #: Number of channels in output (=number of grid points).
    numchannels = Delegate('grid', 'size')

    #: Spatial weighting function.
    weights = Trait('none', possible_weights, desc='spatial weighting function')
    # (from timedomain.possible_weights)

    # internal identifier
    digest = Property(
        depends_on=['_steer_obj.digest', 'source.digest', 'weights', '__class__'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_weights(self):
        return self.weights_(self)[newaxis] if self.weights_ else 1.0

    def result(self, num=2048):
        """Python generator that yields the time-domain beamformer output.

        The output time signal starts for source signals that were emitted from
        the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, :attr:`~BeamformerTime.numchannels`).
                :attr:`~BeamformerTime.numchannels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than num.
        """
        # initialize values
        steer_func = self.steer._steer_funcs_time[self.steer.steer_type]
        fdtype = float64
        idtype = int64
        numMics = self.steer.mics.num_mics
        n_index = arange(0, num + 1)[:, newaxis]
        c = self.steer.env.c / self.source.sample_freq
        amp = empty((1, self.grid.size, numMics), dtype=fdtype)
        # delays = empty((1,self.grid.size,numMics),dtype=fdtype)
        d_index = empty((1, self.grid.size, numMics), dtype=idtype)
        d_interp2 = empty((1, self.grid.size, numMics), dtype=fdtype)
        steer_func(self.steer.rm[newaxis, :, :], self.steer.r0[newaxis, :], amp)
        _delays(self.steer.rm[newaxis, :, :], c, d_interp2, d_index)
        amp.shape = amp.shape[1:]
        # delays.shape = delays.shape[1:]
        d_index.shape = d_index.shape[1:]
        d_interp2.shape = d_interp2.shape[1:]
        max_sample_delay = int((self.steer.rm / c).max()) + 2
        weights = self._get_weights()

        buffer = SamplesBuffer(
            source=self.source,
            length=int(ceil((num + max_sample_delay) / num)) * num,
            result_num=num + max_sample_delay,
            shift_index_by='num',
            dtype=fdtype,
        )
        for p_res in buffer.result(num):
            p_res *= weights
            if p_res.shape[0] < buffer.result_num:  # last block shorter
                num = p_res.shape[0] - max_sample_delay
                n_index = arange(0, num + 1)[:, newaxis]
            # init step
            Phi, autopow = self._delay_and_sum(num, p_res, d_interp2, d_index, amp)
            if 'Cleant' not in self.__class__.__name__:
                if 'Sq' not in self.__class__.__name__:
                    yield Phi[:num]
                elif self.r_diag:
                    yield (Phi[:num] ** 2 - autopow[:num]).clip(min=0)
                else:
                    yield Phi[:num] ** 2
            else:
                p_res_copy = p_res.copy()
                Gamma = zeros(Phi.shape)
                Gamma_autopow = zeros(Phi.shape)
                J = 0
                # deconvolution
                while self.n_iter > J:
                    # print(f"start clean iteration {J+1} of max {self.n_iter}")
                    powPhi = (Phi[:num] ** 2 - autopow).sum(0).clip(min=0) if self.r_diag else (Phi[:num] ** 2).sum(0)
                    imax = argmax(powPhi)
                    t_float = d_interp2[imax] + d_index[imax] + n_index
                    t_ind = t_float.astype(int64)
                    for m in range(numMics):
                        p_res_copy[t_ind[: num + 1, m], m] -= self.damp * interp(
                            t_ind[: num + 1, m],
                            t_float[:num, m],
                            Phi[:num, imax] * self.steer.r0[imax] / self.steer.rm[imax, m],
                        )
                    nextPhi, nextAutopow = self._delay_and_sum(num, p_res_copy, d_interp2, d_index, amp)
                    if self.r_diag:
                        pownextPhi = (nextPhi[:num] ** 2 - nextAutopow).sum(0).clip(min=0)
                    else:
                        pownextPhi = (nextPhi[:num] ** 2).sum(0)
                    # print(f"total signal power: {powPhi.sum()}")
                    if pownextPhi.sum() < powPhi.sum():  # stopping criterion
                        Gamma[:num, imax] += self.damp * Phi[:num, imax]
                        Gamma_autopow[:num, imax] = autopow[:num, imax].copy()
                        Phi = nextPhi
                        autopow = nextAutopow
                        # print(f"clean max: {L_p((Gamma**2).sum(0)/num).max()} dB")
                        J += 1
                    else:
                        break
                if 'Sq' not in self.__class__.__name__:
                    yield Gamma[:num]
                elif self.r_diag:
                    yield Gamma[:num] ** 2 - (self.damp**2) * Gamma_autopow[:num]
                else:
                    yield Gamma[:num] ** 2

    def _delay_and_sum(self, num, p_res, d_interp2, d_index, amp):
        """Standard delay-and-sum method."""
        result = empty((num, self.grid.size), dtype=float)  # output array
        autopow = empty((num, self.grid.size), dtype=float)  # output array
        _delayandsum4(p_res, d_index, d_interp2, amp, result, autopow)
        return result, autopow


class BeamformerTimeSq(BeamformerTime):
    """Provides a time domain beamformer with time-dependend
    power signal output and possible autopower removal
    for a spatially fixed grid.
    """

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, desc='removal of diagonal')

    # internal identifier
    digest = Property(
        depends_on=['_steer_obj.digest', 'source.digest', 'r_diag', 'weights', '__class__'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """Python generator that yields the **squared** time-domain beamformer output.

        The squared output time signal starts for source signals that were emitted from
        the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, :attr:`~BeamformerTime.numchannels`).
                :attr:`~BeamformerTime.numchannels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than num.
        """
        return super().result(num)


class BeamformerTimeTraj(BeamformerTime):
    """Provides a basic time domain beamformer with time signal output
    for a grid moving along a trajectory.
    """

    #: :class:`~acoular.trajectory.Trajectory` or derived object.
    #: Start time is assumed to be the same as for the samples.
    trajectory = Trait(Trajectory, desc='trajectory of the grid center')

    #: Reference vector, perpendicular to the y-axis of moving grid.
    rvec = CArray(dtype=float, shape=(3,), value=array((0, 0, 0)), desc='reference vector')

    #: Considering of convective amplification in beamforming formula.
    conv_amp = Bool(False, desc='determines if convective amplification of source is considered')

    #: Floating point and integer precision
    precision = Trait(64, [32, 64], desc='numeric precision')

    # internal identifier
    digest = Property(
        depends_on=[
            '_steer_obj.digest',
            'source.digest',
            'weights',
            'precision',
            'rvec',
            'conv_amp',
            'trajectory.digest',
            '__class__',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_moving_gpos(self):
        """Python generator that yields the moving grid coordinates samplewise."""

        def cross(a, b):
            """Cross product for fast computation
            because numpy.cross is ultra slow in this case.
            """
            return array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

        start_t = 0.0
        gpos = self.grid.gpos
        trajg = self.trajectory.traj(start_t, delta_t=1 / self.source.sample_freq)
        trajg1 = self.trajectory.traj(start_t, delta_t=1 / self.source.sample_freq, der=1)
        rflag = (self.rvec == 0).all()  # flag translation vs. rotation
        if rflag:
            for g in trajg:
                # grid is only translated, not rotated
                tpos = gpos + array(g)[:, newaxis]
                yield tpos
        else:
            for g, g1 in zip(trajg, trajg1):
                # grid is both translated and rotated
                loc = array(g)  # translation array([0., 0.4, 1.])
                dx = array(g1)  # direction vector (new x-axis)
                dy = cross(self.rvec, dx)  # new y-axis
                dz = cross(dx, dy)  # new z-axis
                RM = array((dx, dy, dz)).T  # rotation matrix
                RM /= sqrt((RM * RM).sum(0))  # column normalized
                tpos = dot(RM, gpos) + loc[:, newaxis]  # rotation+translation
                #                print(loc[:])
                yield tpos

    def _get_macostheta(self, g1, tpos, rm):
        vvec = array(g1)  # velocity vector
        ma = norm(vvec) / self.steer.env.c  # machnumber
        fdv = (vvec / sqrt((vvec * vvec).sum()))[:, newaxis]  # unit vecor velocity
        mpos = self.steer.mics.mpos[:, newaxis, :]
        rmv = tpos[:, :, newaxis] - mpos
        return (ma * sum(rmv.reshape((3, -1)) * fdv, 0) / rm.reshape(-1)).reshape(rm.shape)

    def get_r0(self, tpos):
        if isscalar(self.steer.ref) and self.steer.ref > 0:
            return self.steer.ref  # full((self.steer.grid.size,), self.steer.ref)
        return self.env._r(tpos)

    def result(self, num=2048):
        """Python generator that yields the time-domain beamformer output.

        The output time signal starts for source signals that were emitted from
        the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, :attr:`~BeamformerTime.numchannels`).
                :attr:`~BeamformerTime.numchannels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than num.
        """
        # initialize values
        if self.precision == 64:
            fdtype = float64
            idtype = int64
        else:
            fdtype = float32
            idtype = int32
        w = self._get_weights()
        c = self.steer.env.c / self.source.sample_freq
        numMics = self.steer.mics.num_mics
        mpos = self.steer.mics.mpos.astype(fdtype)
        m_index = arange(numMics, dtype=idtype)
        n_index = arange(num, dtype=idtype)[:, newaxis]
        blockrm = empty((num, self.grid.size, numMics), dtype=fdtype)
        blockrmconv = empty((num, self.grid.size, numMics), dtype=fdtype)
        amp = empty((num, self.grid.size, numMics), dtype=fdtype)
        # delays = empty((num,self.grid.size,numMics),dtype=fdtype)
        d_index = empty((num, self.grid.size, numMics), dtype=idtype)
        d_interp2 = empty((num, self.grid.size, numMics), dtype=fdtype)
        blockr0 = empty((num, self.grid.size), dtype=fdtype)
        movgpos = self._get_moving_gpos()  # create moving grid pos generator
        movgspeed = self.trajectory.traj(0.0, delta_t=1 / self.source.sample_freq, der=1)
        weights = self._get_weights()

        # preliminary implementation of different steering vectors
        steer_func = self.steer._steer_funcs_time[self.steer.steer_type]

        # start processing
        flag = True
        buffer = SamplesBuffer(source=self.source, length=num * 2, shift_index_by='num', dtype=fdtype)
        buffered_result = buffer.result(num)
        while flag:
            for i in range(num):
                tpos = next(movgpos).astype(fdtype)
                rm = self.steer.env._r(tpos, mpos)  # .astype(fdtype)
                blockr0[i, :] = self.get_r0(tpos)
                blockrm[i, :, :] = rm
                if self.conv_amp:
                    ht = next(movgspeed)
                    blockrmconv[i, :, :] = rm * (1 - self._get_macostheta(ht, tpos, rm)) ** 2
            if self.conv_amp:
                steer_func(blockrmconv, blockr0, amp)
            else:
                steer_func(blockrm, blockr0, amp)
            _delays(blockrm, c, d_interp2, d_index)
            max_sample_delay = (d_index.max((1, 2)) + 2).max()  # + because of interpolation
            buffer.result_num = num + max_sample_delay

            try:
                time_block = next(buffered_result)
                time_block *= weights
            except StopIteration:
                break
            if time_block.shape[0] < buffer.result_num:  # last block shorter
                num = sum((d_index.max((1, 2)) + 1 + arange(0, num)) < time_block.shape[0])
                n_index = arange(num, dtype=idtype)[:, newaxis]
                flag = False
            # init step
            p_res = time_block.copy()
            Phi, autopow = self._delay_and_sum(num, p_res, d_interp2, d_index, amp)
            if 'Cleant' not in self.__class__.__name__:
                if 'Sq' not in self.__class__.__name__:
                    yield Phi[:num]
                elif self.r_diag:
                    yield (Phi[:num] ** 2 - autopow[:num]).clip(min=0)
                else:
                    yield Phi[:num] ** 2
            else:
                # choose correct distance
                blockrm1 = blockrmconv if self.conv_amp else blockrm
                Gamma = zeros(Phi.shape, dtype=fdtype)
                Gamma_autopow = zeros(Phi.shape, dtype=fdtype)
                J = 0
                t_ind = arange(p_res.shape[0], dtype=idtype)
                # deconvolution
                while self.n_iter > J:
                    # print(f"start clean iteration {J+1} of max {self.n_iter}")
                    if self.r_diag:
                        powPhi = (Phi[:num] * Phi[:num] - autopow).sum(0).clip(min=0)
                    else:
                        powPhi = (Phi[:num] * Phi[:num]).sum(0)
                    # find index of max power focus point
                    imax = argmax(powPhi)
                    # find backward delays
                    t_float = (d_interp2[:num, imax, m_index] + d_index[:num, imax, m_index] + n_index).astype(fdtype)
                    # determine max/min delays in sample units
                    # + 2 because we do not want to extrapolate behind the last sample
                    ind_max = t_float.max(0).astype(idtype) + 2
                    ind_min = t_float.min(0).astype(idtype)
                    # store time history at max power focus point
                    h = Phi[:num, imax] * blockr0[:num, imax]
                    for m in range(numMics):
                        # subtract interpolated time history from microphone signals
                        p_res[ind_min[m] : ind_max[m], m] -= self.damp * interp(
                            t_ind[ind_min[m] : ind_max[m]],
                            t_float[:num, m],
                            h / blockrm1[:num, imax, m],
                        )
                    nextPhi, nextAutopow = self._delay_and_sum(num, p_res, d_interp2, d_index, amp)
                    if self.r_diag:
                        pownextPhi = (nextPhi[:num] * nextPhi[:num] - nextAutopow).sum(0).clip(min=0)
                    else:
                        pownextPhi = (nextPhi[:num] * nextPhi[:num]).sum(0)
                    # print(f"total signal power: {powPhi.sum()}")
                    if pownextPhi.sum() < powPhi.sum():  # stopping criterion
                        Gamma[:num, imax] += self.damp * Phi[:num, imax]
                        Gamma_autopow[:num, imax] = autopow[:num, imax].copy()
                        Phi = nextPhi
                        autopow = nextAutopow
                        # print(f"clean max: {L_p((Gamma**2).sum(0)/num).max()} dB")
                        J += 1
                    else:
                        break
                if 'Sq' not in self.__class__.__name__:
                    yield Gamma[:num]
                elif self.r_diag:
                    yield Gamma[:num] ** 2 - (self.damp**2) * Gamma_autopow[:num]
                else:
                    yield Gamma[:num] ** 2

    def _delay_and_sum(self, num, p_res, d_interp2, d_index, amp):
        """Standard delay-and-sum method."""
        fdtype = float64 if self.precision == 64 else float32
        result = empty((num, self.grid.size), dtype=fdtype)  # output array
        autopow = empty((num, self.grid.size), dtype=fdtype)  # output array
        _delayandsum5(p_res, d_index, d_interp2, amp, result, autopow)
        return result, autopow


class BeamformerTimeSqTraj(BeamformerTimeSq, BeamformerTimeTraj):
    """Provides a time domain beamformer with time-dependent
    power signal output and possible autopower removal
    for a grid moving along a trajectory.
    """

    # internal identifier
    digest = Property(
        depends_on=[
            '_steer_obj.digest',
            'source.digest',
            'r_diag',
            'weights',
            'precision',
            'rvec',
            'conv_amp',
            'trajectory.digest',
            '__class__',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """Python generator that yields the **squared** time-domain beamformer output.

        The squared output time signal starts for source signals that were emitted from
        the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, :attr:`~BeamformerTime.numchannels`).
                :attr:`~BeamformerTime.numchannels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than num.
        """
        return super().result(num)


class BeamformerCleant(BeamformerTime):
    """CLEANT deconvolution method.

    An implementation of the CLEAN method in time domain. This class can only
    be used for static sources. See :cite:`Cousson2019` for details.
    """

    #: Boolean flag, always False
    r_diag = Enum(False, desc='False, as we do not remove autopower in this beamformer')

    #: iteration damping factor also referred as loop gain in Cousson et al.
    #: defaults to 0.6
    damp = Range(0.01, 1.0, 0.6, desc='damping factor (loop gain)')

    #: max number of iterations
    n_iter = Int(100, desc='maximum number of iterations')

    # internal identifier
    digest = Property(
        depends_on=['_steer_obj.digest', 'source.digest', 'weights', '__class__', 'damp', 'n_iter'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """Python generator that yields the deconvolved time-domain beamformer output.

        The output starts for signals that were emitted from the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, :attr:`~BeamformerTime.numchannels`).
                :attr:`~BeamformerTime.numchannels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than num.
        """
        return super().result(num)


class BeamformerCleantSq(BeamformerCleant):
    """CLEANT deconvolution method with optional removal of autocorrelation.

    An implementation of the CLEAN method in time domain. This class can only
    be used for static sources. See :cite:`Cousson2019` for details on the method
    and :cite:`Kujawski2020` for details on the autocorrelation removal.
    """

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, desc='removal of diagonal')

    # internal identifier
    digest = Property(
        depends_on=['_steer_obj.digest', 'source.digest', 'weights', '__class__', 'damp', 'n_iter', 'r_diag'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """Python generator that yields the *squared* deconvolved time-domain beamformer output.

        The output starts for signals that were emitted from the :class:`~acoular.grids.Grid` at `t=0`.
        Per default, block-wise removal of autocorrelation is performed, which can be turned of by setting
        :attr:`r_diag` to `False`.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, :attr:`~BeamformerTime.numchannels`).
                :attr:`~BeamformerTime.numchannels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than num.
        """
        return super().result(num)


class BeamformerCleantTraj(BeamformerCleant, BeamformerTimeTraj):
    """CLEANT deconvolution method.

    An implementation of the CLEAN method in time domain for moving sources
    with known trajectory. See :cite:`Cousson2019` for details.
    """

    #: Floating point and integer precision
    precision = Trait(32, [32, 64], desc='numeric precision')

    # internal identifier
    digest = Property(
        depends_on=[
            '_steer_obj.digest',
            'source.digest',
            'weights',
            'precision',
            '__class__',
            'damp',
            'n_iter',
            'rvec',
            'conv_amp',
            'trajectory.digest',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """Python generator that yields the deconvolved time-domain beamformer output.

        The output starts for signals that were emitted from the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, :attr:`~BeamformerTime.numchannels`).
                :attr:`~BeamformerTime.numchannels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than num.
        """
        return super().result(num)


class BeamformerCleantSqTraj(BeamformerCleantTraj, BeamformerTimeSq):
    """CLEANT deconvolution method with optional removal of autocorrelation.

    An implementation of the CLEAN method in time domain for moving sources
    with known trajectory. See :cite:`Cousson2019` for details on the method and
    :cite:`Kujawski2020` for details on the autocorrelation removal.
    """

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, desc='removal of diagonal')

    # internal identifier
    digest = Property(
        depends_on=[
            '_steer_obj.digest',
            'source.digest',
            'weights',
            'precision',
            '__class__',
            'damp',
            'n_iter',
            'rvec',
            'conv_amp',
            'trajectory.digest',
            'r_diag',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """Python generator that yields the *squared* deconvolved time-domain beamformer output.

        The output starts for signals that were emitted from the :class:`~acoular.grids.Grid` at `t=0`.
        Per default, block-wise removal of autocorrelation is performed, which can be turned of by setting
        :attr:`r_diag` to `False`.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, :attr:`~BeamformerTime.numchannels`).
                :attr:`~BeamformerTime.numchannels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than num.
        """
        return super().result(num)


class IntegratorSectorTime(TimeOut):
    """Provides an Integrator in the time domain."""

    #: Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    #: :class:`~acoular.grids.RectGrid` object that provides the grid locations.
    grid = Trait(RectGrid, desc='beamforming grid')

    #: List of sectors in grid
    sectors = List()

    #: Clipping, in Dezibel relative to maximum (negative values)
    clip = Float(-350.0)

    #: Number of channels in output (= number of sectors).
    numchannels = Property(depends_on=['sectors'])

    # internal identifier
    digest = Property(
        depends_on=['sectors', 'clip', 'grid.digest', 'source.digest', '__class__'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_numchannels(self):
        return len(self.sectors)

    def result(self, num=1):
        """Python generator that yields the source output integrated over the given
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
        o = empty((num, self.numchannels), dtype=float)  # output array
        for r in self.source.result(num):
            ns = r.shape[0]
            mapshape = (ns,) + gshape
            rmax = r.max()
            rmin = rmax * 10 ** (self.clip / 10.0)
            r = where(r > rmin, r, 0.0)
            for i, ind in enumerate(inds):
                h = r[:].reshape(mapshape)[(s_[:],) + ind]
                o[:ns, i] = h.reshape(h.shape[0], -1).sum(axis=1)
                i += 1
            yield o[:ns]
