# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implements beamformers in the time domain.

.. inheritance-diagram::
                acoular.tbeamform
    :top-classes:
                acoular.base.TimeOut
    :parts: 1

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

import numpy as np
import scipy.linalg as spla
from traits.api import Bool, CArray, Enum, Float, Instance, Int, List, Map, Property, Range, cached_property

from .base import SamplesGenerator, TimeOut
from .fbeamform import SteeringVector
from .grids import RectGrid

# acoular imports
from .internal import digest
from .process import SamplesBuffer
from .tfastfuncs import _delayandsum4, _delayandsum5, _delays
from .trajectory import Trajectory


def const_power_weight(bf):
    """
    Internal helper function for :class:`BeamformerTime`.

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
    r = bf.steer.env._r(np.zeros((3, 1)), bf.steer.mics.pos)  # distances to center
    # round the relative distances to one decimal place
    r = (r / r.max()).round(decimals=1)
    ru, ind = np.unique(r, return_inverse=True)
    ru = (ru[1:] + ru[:-1]) / 2
    count, bins = np.histogram(r, np.r_[0, ru, 1.5 * r.max() - 0.5 * ru[-1]])
    bins *= bins
    weights = np.sqrt((bins[1:] - bins[:-1]) / count)
    weights /= weights.mean()
    return weights[ind]


# possible choices for spatial weights
possible_weights = {'none': None, 'power': const_power_weight}


class BeamformerTime(TimeOut):
    """
    Provides a basic time domain beamformer with time signal output.

    for a spatially fixed grid.
    """

    #: Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    # Instance of :class:`~acoular.fbeamform.SteeringVector` or its derived classes
    # that contains information about the steering vector. This is a private trait.
    # Do not set this directly, use `steer` trait instead.
    steer = Instance(SteeringVector, args=())

    #: Number of channels in output (=number of grid points).
    num_channels = Property()

    #: Spatial weighting function.
    weights = Map(possible_weights, default_value='none', desc='spatial weighting function')
    # (from timedomain.possible_weights)

    # internal identifier
    digest = Property(
        depends_on=['steer.digest', 'source.digest', 'weights'],
    )

    def _get_num_channels(self):
        return self.steer.grid.size

    def _get_digest(self):
        return digest(self)

    def _get_weights(self):
        return self.weights_(self)[np.newaxis] if self.weights_ else 1.0

    def result(self, num=2048):
        """
        Python generator that yields the time-domain beamformer output.

        The output time signal starts for source signals that were emitted from
        the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : :class:`int`
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`~BeamformerTime.num_channels`).
                :attr:`~BeamformerTime.num_channels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than ``num``.
        """
        # initialize values
        steer_func = self.steer._steer_funcs_time[self.steer.steer_type]
        fdtype = np.float64
        idtype = np.int64
        num_mics = self.steer.mics.num_mics
        n_index = np.arange(0, num + 1)[:, np.newaxis]
        c = self.steer.env.c / self.source.sample_freq
        amp = np.empty((1, self.steer.grid.size, num_mics), dtype=fdtype)
        # delays = np.empty((1,self.steer.grid.size,num_mics),dtype=fdtype)
        d_index = np.empty((1, self.steer.grid.size, num_mics), dtype=idtype)
        d_interp2 = np.empty((1, self.steer.grid.size, num_mics), dtype=fdtype)
        steer_func(self.steer.rm[np.newaxis, :, :], self.steer.r0[np.newaxis, :], amp)
        _delays(self.steer.rm[np.newaxis, :, :], c, d_interp2, d_index)
        amp.shape = amp.shape[1:]
        # delays.shape = delays.shape[1:]
        d_index.shape = d_index.shape[1:]
        d_interp2.shape = d_interp2.shape[1:]
        max_sample_delay = int((self.steer.rm / c).max()) + 2
        weights = self._get_weights()

        buffer = SamplesBuffer(
            source=self.source,
            length=int(np.ceil((num + max_sample_delay) / num)) * num,
            result_num=num + max_sample_delay,
            shift_index_by='num',
            dtype=fdtype,
        )
        for p_res in buffer.result(num):
            p_res *= weights
            if p_res.shape[0] < buffer.result_num:  # last block shorter
                num = p_res.shape[0] - max_sample_delay
                # exit loop if there is not enough data left to be processed
                if num <= 0:
                    break
                n_index = np.arange(0, num + 1)[:, np.newaxis]
            # init step
            phi, autopow = self._delay_and_sum(num, p_res, d_interp2, d_index, amp)
            if 'Cleant' not in self.__class__.__name__:
                if 'Sq' not in self.__class__.__name__:
                    yield phi[:num]
                elif self.r_diag:
                    yield (phi[:num] ** 2 - autopow[:num]).clip(min=0)
                else:
                    yield phi[:num] ** 2
            else:
                p_res_copy = p_res.copy()
                gamma = np.zeros(phi.shape)
                gamma_autopow = np.zeros(phi.shape)
                j = 0
                # deconvolution
                while self.n_iter > j:
                    # print(f"start clean iteration {j+1} of max {self.n_iter}")
                    pow_phi = (phi[:num] ** 2 - autopow).sum(0).clip(min=0) if self.r_diag else (phi[:num] ** 2).sum(0)
                    imax = np.argmax(pow_phi)
                    t_float = d_interp2[imax] + d_index[imax] + n_index
                    t_ind = t_float.astype(np.int64)
                    for m in range(num_mics):
                        p_res_copy[t_ind[: num + 1, m], m] -= self.damp * np.interp(
                            t_ind[: num + 1, m],
                            t_float[:num, m],
                            phi[:num, imax] * self.steer.r0[imax] / self.steer.rm[imax, m],
                        )
                    next_phi, next_autopow = self._delay_and_sum(num, p_res_copy, d_interp2, d_index, amp)
                    if self.r_diag:
                        pow_next_phi = (next_phi[:num] ** 2 - next_autopow).sum(0).clip(min=0)
                    else:
                        pow_next_phi = (next_phi[:num] ** 2).sum(0)
                    # print(f"total signal power: {pow_phi.sum()}")
                    if pow_next_phi.sum() < pow_phi.sum():  # stopping criterion
                        gamma[:num, imax] += self.damp * phi[:num, imax]
                        gamma_autopow[:num, imax] = autopow[:num, imax].copy()
                        phi = next_phi
                        autopow = next_autopow
                        # print(f"clean max: {L_p((gamma**2).sum(0)/num).max()} dB")
                        j += 1
                    else:
                        break
                if 'Sq' not in self.__class__.__name__:
                    yield gamma[:num]
                elif self.r_diag:
                    yield gamma[:num] ** 2 - (self.damp**2) * gamma_autopow[:num]
                else:
                    yield gamma[:num] ** 2

    def _delay_and_sum(self, num, p_res, d_interp2, d_index, amp):
        """Standard delay-and-sum method."""
        result = np.empty((num, self.steer.grid.size), dtype=float)  # output array
        autopow = np.empty((num, self.steer.grid.size), dtype=float)  # output array
        _delayandsum4(p_res, d_index, d_interp2, amp, result, autopow)
        return result, autopow


class BeamformerTimeSq(BeamformerTime):
    """
    Time domain beamformer with squared output and optional autopower removal.

    Provides a time domain beamformer with time-dependend power signal output and possible autopower
    removal for a spatially fixed grid.
    """

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, desc='removal of diagonal')

    # internal identifier
    digest = Property(
        depends_on=['steer.digest', 'source.digest', 'r_diag', 'weights'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """
        Python generator that yields the **squared** time-domain beamformer output.

        The squared output time signal starts for source signals that were emitted from
        the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : :class:`int`
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`~BeamformerTime.num_channels`).
                :attr:`~BeamformerTime.num_channels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than ``num``.
        """
        return super().result(num)


class BeamformerTimeTraj(BeamformerTime):
    """
    Provides a basic time domain beamformer with time signal output.

    for a grid moving along a trajectory.
    """

    #: :class:`~acoular.trajectory.Trajectory` or derived object.
    #: Start time is assumed to be the same as for the samples.
    trajectory = Instance(Trajectory, desc='trajectory of the grid center')

    #: Reference vector, perpendicular to the y-axis of moving grid.
    rvec = CArray(dtype=float, shape=(3,), value=np.array((0, 0, 0)), desc='reference vector')

    #: Considering of convective amplification in beamforming formula.
    conv_amp = Bool(False, desc='determines if convective amplification of source is considered')

    #: Floating point and integer precision
    precision = Enum(64, 32, desc='numeric precision')

    # internal identifier
    digest = Property(
        depends_on=[
            'steer.digest',
            'source.digest',
            'weights',
            'precision',
            'rvec',
            'conv_amp',
            'trajectory.digest',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _get_moving_gpos(self):
        """Python generator that yields the moving grid coordinates samplewise."""

        def cross(a, b):
            """
            Cross product for fast computation.

            because numpy.cross is ultra slow in this case.
            """
            return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])

        start_t = 0.0
        gpos = self.steer.grid.pos
        trajg = self.trajectory.traj(start_t, delta_t=1 / self.source.sample_freq)
        trajg1 = self.trajectory.traj(start_t, delta_t=1 / self.source.sample_freq, der=1)
        rflag = (self.rvec == 0).all()  # flag translation vs. rotation
        if rflag:
            for g in trajg:
                # grid is only translated, not rotated
                tpos = gpos + np.array(g)[:, np.newaxis]
                yield tpos
        else:
            for g, g1 in zip(trajg, trajg1):
                # grid is both translated and rotated
                loc = np.array(g)  # translation array([0., 0.4, 1.])
                dx = np.array(g1)  # direction vector (new x-axis)
                dy = cross(self.rvec, dx)  # new y-axis
                dz = cross(dx, dy)  # new z-axis
                rm = np.array((dx, dy, dz)).T  # rotation matrix
                rm /= np.sqrt((rm * rm).sum(0))  # column normalized
                tpos = np.dot(rm, gpos) + loc[:, np.newaxis]  # rotation+translation
                #                print(loc[:])
                yield tpos

    def _get_macostheta(self, g1, tpos, rm):
        vvec = np.array(g1)  # velocity vector
        ma = spla.norm(vvec) / self.steer.env.c  # machnumber
        fdv = (vvec / np.sqrt((vvec * vvec).sum()))[:, np.newaxis]  # unit vecor velocity
        mpos = self.steer.mics.pos[:, np.newaxis, :]
        rmv = tpos[:, :, np.newaxis] - mpos
        return (ma * np.sum(rmv.reshape((3, -1)) * fdv, 0) / rm.reshape(-1)).reshape(rm.shape)

    def get_r0(self, tpos):
        """
        Get reference distance for grid positions.

        Parameters
        ----------
        tpos : :class:`numpy.ndarray`
            Grid positions.

        Returns
        -------
        :class:`float` or :class:`numpy.ndarray`
            Reference distance(s).
        """
        if np.isscalar(self.steer.ref) and self.steer.ref > 0:
            return self.steer.ref  # full((self.steer.grid.size,), self.steer.ref)
        return self.steer.env._r(tpos)

    def result(self, num=2048):
        """
        Python generator that yields the time-domain beamformer output.

        The output time signal starts for source signals that were emitted from
        the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : :class:`int`
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`~BeamformerTime.num_channels`).
                :attr:`~BeamformerTime.num_channels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than ``num``.
        """
        # initialize values
        if self.precision == 64:
            fdtype = np.float64
            idtype = np.int64
        else:
            fdtype = np.float32
            idtype = np.int32
        c = self.steer.env.c / self.source.sample_freq
        num_mics = self.steer.mics.num_mics
        mpos = self.steer.mics.pos.astype(fdtype)
        m_index = np.arange(num_mics, dtype=idtype)
        n_index = np.arange(num, dtype=idtype)[:, np.newaxis]
        blockrm = np.empty((num, self.steer.grid.size, num_mics), dtype=fdtype)
        blockrmconv = np.empty((num, self.steer.grid.size, num_mics), dtype=fdtype)
        amp = np.empty((num, self.steer.grid.size, num_mics), dtype=fdtype)
        # delays = np.empty((num,self.steer.grid.size,num_mics),dtype=fdtype)
        d_index = np.empty((num, self.steer.grid.size, num_mics), dtype=idtype)
        d_interp2 = np.empty((num, self.steer.grid.size, num_mics), dtype=fdtype)
        blockr0 = np.empty((num, self.steer.grid.size), dtype=fdtype)
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
                num = np.sum((d_index.max((1, 2)) + 1 + np.arange(0, num)) < time_block.shape[0])
                n_index = np.arange(num, dtype=idtype)[:, np.newaxis]
                flag = False
            # init step
            p_res = time_block.copy()
            phi, autopow = self._delay_and_sum(num, p_res, d_interp2, d_index, amp)
            if 'Cleant' not in self.__class__.__name__:
                if 'Sq' not in self.__class__.__name__:
                    yield phi[:num]
                elif self.r_diag:
                    yield (phi[:num] ** 2 - autopow[:num]).clip(min=0)
                else:
                    yield phi[:num] ** 2
            else:
                # choose correct distance
                blockrm1 = blockrmconv if self.conv_amp else blockrm
                gamma = np.zeros(phi.shape, dtype=fdtype)
                gamma_autopow = np.zeros(phi.shape, dtype=fdtype)
                j = 0
                t_ind = np.arange(p_res.shape[0], dtype=idtype)
                # deconvolution
                while self.n_iter > j:
                    # print(f"start clean iteration {j+1} of max {self.n_iter}")
                    if self.r_diag:
                        pow_phi = (phi[:num] * phi[:num] - autopow).sum(0).clip(min=0)
                    else:
                        pow_phi = (phi[:num] * phi[:num]).sum(0)
                    # find index of max power focus point
                    imax = np.argmax(pow_phi)
                    # find backward delays
                    t_float = (d_interp2[:num, imax, m_index] + d_index[:num, imax, m_index] + n_index).astype(fdtype)
                    # determine max/min delays in sample units
                    # + 2 because we do not want to extrapolate behind the last sample
                    ind_max = t_float.max(0).astype(idtype) + 2
                    ind_min = t_float.min(0).astype(idtype)
                    # store time history at max power focus point
                    h = phi[:num, imax] * blockr0[:num, imax]
                    for m in range(num_mics):
                        # subtract interpolated time history from microphone signals
                        p_res[ind_min[m] : ind_max[m], m] -= self.damp * np.interp(
                            t_ind[ind_min[m] : ind_max[m]],
                            t_float[:num, m],
                            h / blockrm1[:num, imax, m],
                        )
                    next_phi, next_autopow = self._delay_and_sum(num, p_res, d_interp2, d_index, amp)
                    if self.r_diag:
                        pow_next_phi = (next_phi[:num] * next_phi[:num] - next_autopow).sum(0).clip(min=0)
                    else:
                        pow_next_phi = (next_phi[:num] * next_phi[:num]).sum(0)
                    # print(f"total signal power: {pow_phi.sum()}")
                    if pow_next_phi.sum() < pow_phi.sum():  # stopping criterion
                        gamma[:num, imax] += self.damp * phi[:num, imax]
                        gamma_autopow[:num, imax] = autopow[:num, imax].copy()
                        phi = next_phi
                        autopow = next_autopow
                        # print(f"clean max: {L_p((gamma**2).sum(0)/num).max()} dB")
                        j += 1
                    else:
                        break
                if 'Sq' not in self.__class__.__name__:
                    yield gamma[:num]
                elif self.r_diag:
                    yield gamma[:num] ** 2 - (self.damp**2) * gamma_autopow[:num]
                else:
                    yield gamma[:num] ** 2

    def _delay_and_sum(self, num, p_res, d_interp2, d_index, amp):
        """Standard delay-and-sum method."""
        fdtype = np.float64 if self.precision == 64 else np.float32
        result = np.empty((num, self.steer.grid.size), dtype=fdtype)  # output array
        autopow = np.empty((num, self.steer.grid.size), dtype=fdtype)  # output array
        _delayandsum5(p_res, d_index, d_interp2, amp, result, autopow)
        return result, autopow


class BeamformerTimeSqTraj(BeamformerTimeSq, BeamformerTimeTraj):
    """
    Time domain beamformer with squared output for a grid moving along a trajectory.

    Provides a time domain beamformer with time-dependent power signal output and possible autopower
    removal for a grid moving along a trajectory.
    """

    # internal identifier
    digest = Property(
        depends_on=[
            'steer.digest',
            'source.digest',
            'r_diag',
            'weights',
            'precision',
            'rvec',
            'conv_amp',
            'trajectory.digest',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """
        Python generator that yields the **squared** time-domain beamformer output.

        The squared output time signal starts for source signals that were emitted from
        the :class:`~acoular.grids.Grid` at `t=0`.

        Parameters
        ----------
        num : :class:`int`
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`~BeamformerTime.num_channels`).
                :attr:`~BeamformerTime.num_channels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than ``num``.
        """
        return super().result(num)


class BeamformerCleant(BeamformerTime):
    """
    CLEANT deconvolution method.

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
        depends_on=['steer.digest', 'source.digest', 'weights', 'damp', 'n_iter'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """
        Python generator that yields the deconvolved time-domain beamformer output.

        The output starts for signals that were emitted from the :class:`~acoular.grids.Grid` at
        `t=0`.

        Parameters
        ----------
        num : :class:`int`
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`~BeamformerTime.num_channels`).
                :attr:`~BeamformerTime.num_channels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than ``num``.
        """
        return super().result(num)


class BeamformerCleantSq(BeamformerCleant):
    """
    CLEANT deconvolution method with optional removal of autocorrelation.

    An implementation of the CLEAN method in time domain. This class can only
    be used for static sources. See :cite:`Cousson2019` for details on the method
    and :cite:`Kujawski2020` for details on the autocorrelation removal.
    """

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, desc='removal of diagonal')

    # internal identifier
    digest = Property(
        depends_on=['steer.digest', 'source.digest', 'weights', 'damp', 'n_iter', 'r_diag'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=2048):
        """
        Python generator that yields the *squared* deconvolved time-domain beamformer output.

        The output starts for signals that were emitted from the :class:`~acoular.grids.Grid` at
        `t=0`. Per default, block-wise removal of autocorrelation is performed, which can be turned
        of by setting :attr:`r_diag` to `False`.

        Parameters
        ----------
        num : :class:`int`
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`~BeamformerTime.num_channels`).
                :attr:`~BeamformerTime.num_channels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than ``num``.
        """
        return super().result(num)


class BeamformerCleantTraj(BeamformerCleant, BeamformerTimeTraj):
    """
    CLEANT deconvolution method.

    An implementation of the CLEAN method in time domain for moving sources
    with known trajectory. See :cite:`Cousson2019` for details.
    """

    #: Floating point and integer precision
    precision = Enum(32, 64, desc='numeric precision')

    # internal identifier
    digest = Property(
        depends_on=[
            'steer.digest',
            'source.digest',
            'weights',
            'precision',
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
        """
        Python generator that yields the deconvolved time-domain beamformer output.

        The output starts for signals that were emitted from the :class:`~acoular.grids.Grid` at
        `t=0`.

        Parameters
        ----------
        num : :class:`int`
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`~BeamformerTime.num_channels`).
                :attr:`~BeamformerTime.num_channels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than ``num``.
        """
        return super().result(num)


class BeamformerCleantSqTraj(BeamformerCleantTraj, BeamformerTimeSq):
    """
    CLEANT deconvolution method with optional removal of autocorrelation.

    An implementation of the CLEAN method in time domain for moving sources
    with known trajectory. See :cite:`Cousson2019` for details on the method and
    :cite:`Kujawski2020` for details on the autocorrelation removal.
    """

    #: Boolean flag, if 'True' (default), the main diagonal is removed before beamforming.
    r_diag = Bool(True, desc='removal of diagonal')

    # internal identifier
    digest = Property(
        depends_on=[
            'steer.digest',
            'source.digest',
            'weights',
            'precision',
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
        """
        Python generator that yields the *squared* deconvolved time-domain beamformer output.

        The output starts for signals that were emitted from the :class:`~acoular.grids.Grid` at
        `t=0`. Per default, block-wise removal of autocorrelation is performed, which can be turned
        of by setting :attr:`r_diag` to `False`.

        Parameters
        ----------
        num : :class:`int`
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block). Defaults to 2048.

        Yields
        ------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`~BeamformerTime.num_channels`).
                :attr:`~BeamformerTime.num_channels` is usually very \
                large (number of grid points).
                The last block returned by the generator may be shorter than ``num``.
        """
        return super().result(num)


class IntegratorSectorTime(TimeOut):
    """Provides an Integrator in the time domain."""

    #: Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    #: :class:`~acoular.grids.RectGrid` object that provides the grid locations.
    grid = Instance(RectGrid, desc='beamforming grid')

    #: List of sectors in grid
    sectors = List()

    #: Clipping, in Decibel relative to maximum (negative values)
    clip = Float(-350.0)

    #: Number of channels in output (= number of sectors).
    num_channels = Property(depends_on=['sectors'])

    # internal identifier
    digest = Property(
        depends_on=['sectors', 'clip', 'grid.digest', 'source.digest'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_num_channels(self):
        return len(self.sectors)

    def result(self, num=1):
        """
        Python generator that yields the source output integrated over specified grid sectors.

        Parameters
        ----------
        num : :class:`int`
            Size of the blocks to be yielded (number of samples per block). Default is ``1``.

        Returns
        -------
        :class:`numpy.ndarray`
            Samples in blocks of shape (``num``, :attr:`num_channels`).
        :attr:`num_channels` is the number of sectors.
        The last block may be shorter than num.
        """
        inds = [self.grid.indices(*sector) for sector in self.sectors]
        gshape = self.grid.shape
        o = np.empty((num, self.num_channels), dtype=float)  # output array
        for r in self.source.result(num):
            ns = r.shape[0]
            mapshape = (ns,) + gshape
            rmax = r.max()
            rmin = rmax * 10 ** (self.clip / 10.0)
            r = np.where(r > rmin, r, 0.0)
            for i, ind in enumerate(inds):
                h = r[:].reshape(mapshape)[(np.s_[:],) + ind]
                o[:ns, i] = h.reshape(h.shape[0], -1).sum(axis=1)
                i += 1
            yield o[:ns]
