# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Contains NUMBA accelerated functions for time-domain beamformers."""

import numba as nb
import numpy as np


@nb.njit(
    [
        (
            nb.float64[:, ::1],
            nb.int64[:, ::1],
            nb.float64[:, ::1],
            nb.float64[:, ::1],
            nb.float64[:, ::1],
            nb.float64[:, ::1],
        ),
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def _delayandsum4(data, offsets, ifactor2, steeramp, out, autopower):
    """Performs one time step of delay and sum with output and additional autopower removal.

    Parameters
    ----------
    data : float64[nSamples, nMics]
        The time history for all channels.
    offsets : int64[gridSize, nMics]
        Indices for each grid point and each channel.
    ifactor2: float64[gridSize, nMics]
        Second interpolation factor, the first one is computed internally.
    steeramp: float64[gridSize, nMics]
        Amplitude factor from steering vector.

    Returns
    -------
    None : as the inputs out and autopower get overwritten.

    """
    gridsize, num_channels = offsets.shape
    num = out.shape[0]
    zero_constant = data.dtype.type(0.0)
    for n in nb.prange(num):
        for gi in nb.prange(gridsize):
            out[n, gi] = zero_constant
            autopower[n, gi] = zero_constant
            for mi in range(num_channels):
                ind = (gi, mi)
                r = (
                    data[offsets[ind] + n, mi] * (1.0 - ifactor2[ind]) + data[offsets[ind] + n + 1, mi] * ifactor2[ind]
                ) * steeramp[ind]
                out[n, gi] += r
                autopower[n, gi] += r * r


@nb.njit(
    [
        (
            nb.float32[:, ::1],
            nb.int32[:, :, ::1],
            nb.float32[:, :, ::1],
            nb.float32[:, :, ::1],
            nb.float32[:, ::1],
            nb.float32[:, ::1],
        ),
        (
            nb.float64[:, ::1],
            nb.int64[:, :, ::1],
            nb.float64[:, :, ::1],
            nb.float64[:, :, ::1],
            nb.float64[:, ::1],
            nb.float64[:, ::1],
        ),
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def _delayandsum5(data, offsets, ifactor2, steeramp, out, autopower):
    """Performs one time step of delay and sum with output and additional autopower removal.

    Parameters
    ----------
    data : float64[nSamples, nMics]
        The time history for all channels.
    offsets : int64[nBlockSamples, gridSize, nMics]
        Indices for each grid point and each channel.
    ifactor2: float64[nBlockSamples,gridSize, nMics]
        Second interpolation factor, the first one is computed internally.
    steeramp: float64[nBlockSamples,gridSize, nMics]
        Amplitude factor from steering vector.

    Returns
    -------
    None : as the inputs out and autopower get overwritten.

    """
    num, gridsize, num_channels = offsets.shape
    num = out.shape[0]
    # ZERO = data.dtype.type(0.)
    one_constant = data.dtype.type(1.0)
    for n in nb.prange(num):
        for gi in nb.prange(gridsize):
            out[n, gi] = 0
            autopower[n, gi] = 0
            for mi in range(num_channels):
                ind = offsets[n, gi, mi] + n
                r = (
                    data[ind, mi] * (one_constant - ifactor2[n, gi, mi]) + data[ind + 1, mi] * ifactor2[n, gi, mi]
                ) * steeramp[
                    n,
                    gi,
                    mi,
                ]
                out[n, gi] += r
                autopower[n, gi] += r * r


@nb.njit(
    [
        (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def _steer_I(rm, r0, amp):  # noqa: ARG001, N802
    num, gridsize, num_channels = rm.shape
    amp[0, 0, 0] = 1.0 / num_channels  # to get the same type for rm2 as for rm
    nr = amp[0, 0, 0]
    for n in nb.prange(num):
        for gi in nb.prange(gridsize):
            for mi in nb.prange(num_channels):
                amp[n, gi, mi] = nr


@nb.njit(
    [
        (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def _steer_II(rm, r0, amp):  # noqa: N802
    num, gridsize, num_channels = rm.shape
    amp[0, 0, 0] = 1.0 / num_channels  # to get the same type for rm2 as for rm
    nr = amp[0, 0, 0]
    for n in nb.prange(num):
        for gi in nb.prange(gridsize):
            rm2 = np.divide(nr, r0[n, gi])
            for mi in nb.prange(num_channels):
                amp[n, gi, mi] = rm[n, gi, mi] * rm2


@nb.njit(
    [
        (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def _steer_III(rm, r0, amp):  # noqa: N802
    num, gridsize, num_channels = rm.shape
    rm20 = rm[0, 0, 0] - rm[0, 0, 0]  # to get the same type for rm2 as for rm
    rm1 = rm[0, 0, 0] / rm[0, 0, 0]
    for n in nb.prange(num):
        for gi in nb.prange(gridsize):
            rm2 = rm20
            for mi in nb.prange(num_channels):
                rm2 += np.divide(rm1, np.square(rm[n, gi, mi]))
            rm2 *= r0[n, gi]
            for mi in nb.prange(num_channels):
                amp[n, gi, mi] = np.divide(rm1, rm[n, gi, mi] * rm2)


@nb.njit(
    [
        (nb.float32[:, :, :], nb.float32[:, :], nb.float32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :], nb.float64[:, :, :]),
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def _steer_IV(rm, r0, amp):  # noqa: ARG001, N802
    num, gridsize, num_channels = rm.shape
    amp[0, 0, 0] = np.sqrt(1.0 / num_channels)  # to get the same type for rm2 as for rm
    nr = amp[0, 0, 0]
    rm1 = rm[0, 0, 0] / rm[0, 0, 0]
    rm20 = rm[0, 0, 0] - rm[0, 0, 0]  # to get the same type for rm2 as for rm
    for n in nb.prange(num):
        for gi in nb.prange(gridsize):
            rm2 = rm20
            for mi in nb.prange(num_channels):
                rm2 += np.divide(rm1, np.square(rm[n, gi, mi]))
            rm2 = np.sqrt(rm2)
            for mi in nb.prange(num_channels):
                amp[n, gi, mi] = np.divide(nr, rm[n, gi, mi] * rm2)


@nb.njit(
    [
        (nb.float32[:, :, ::1], nb.float32, nb.float32[:, :, ::1], nb.int32[:, :, ::1]),
        (nb.float64[:, :, ::1], nb.float64, nb.float64[:, :, ::1], nb.int64[:, :, ::1]),
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def _delays(rm, c, interp2, index):
    num, gridsize, num_channels = rm.shape
    invc = 1 / c
    intt = index.dtype.type
    for n in nb.prange(num):
        for gi in nb.prange(gridsize):
            for mi in nb.prange(num_channels):
                delays = invc * rm[n, gi, mi]
                index[n, gi, mi] = intt(delays)
                interp2[n, gi, mi] = delays - nb.int64(delays)


@nb.njit(
    [
        (nb.float32[:, :, :], nb.float32[:, :, :], nb.int32[:, :, :]),
        (nb.float64[:, :, :], nb.float64[:, :, :], nb.int64[:, :, :]),
    ],
    cache=True,
    parallel=True,
    fastmath=True,
)
def _modf(delays, interp2, index):
    num, gridsize, num_channels = delays.shape
    for n in nb.prange(num):
        for gi in nb.prange(gridsize):
            for mi in nb.prange(num_channels):
                index[n, gi, mi] = int(delays[n, gi, mi])
                interp2[n, gi, mi] = delays[n, gi, mi] - index[n, gi, mi]


if __name__ == '__main__':
    foo = _delays
    print(foo.parallel_diagnostics(level=4))
