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


@nb.njit(
    [(nb.float64[:, :], nb.float64[:], nb.float64, nb.float64[:, :])],
    cache=True,
    parallel=True,
)
def iir_time_varying_kernel(data, cos_theta, pole_radius, zi):
    """Apply time-varying 2nd-order IIR notch filter to multi-channel data.

    Processes all samples and channels in compiled code.  The filter
    coefficients are recomputed per sample from ``cos_theta[n]``.
    Channels are processed in parallel via :func:`numba.prange`.

    Parameters
    ----------
    data : numpy.ndarray, shape (N, K)
        Input signal, *N* samples * *K* channels.
    cos_theta : numpy.ndarray, shape (N,)
        Precomputed ``cos(2 pi * freq[n] / sample_freq)`` per sample.
    pole_radius : float
        Pole radius (0 < r < 1).
    zi : numpy.ndarray, shape (K, 2)
        Filter state per channel, **modified in-place**.

    Returns
    -------
    numpy.ndarray, shape (N, K)
        Filtered output signal.
    """
    N, K = data.shape
    r = pole_radius
    r2 = r * r
    output = np.empty((N, K), dtype=np.float64)

    for k in nb.prange(K):
        z0 = zi[k, 0]
        z1 = zi[k, 1]

        for n in range(N):
            ct = cos_theta[n]
            b1 = -2.0 * ct
            a1 = -2.0 * r * ct

            x = data[n, k]

            # Transposed Direct-Form II difference equations (b0 = b2 = 1, a2 = r^2):
            #   y[n]   = x[n] + z0
            #   z0_new = b1*x[n] - a1*y[n] + z1
            #   z1_new = x[n] - r^2*y[n]
            # See: https://ccrma.stanford.edu/~jos/filters/Transposed_Direct_Forms.html
            y = x + z0
            z0 = b1 * x - a1 * y + z1
            z1 = x - r2 * y

            output[n, k] = y

        zi[k, 0] = z0
        zi[k, 1] = z1

    return output


@nb.njit(
    [
        (
            nb.float64[:, :],  # data
            nb.float64,  # pole_radius
            nb.float64,  # sample_freq
            nb.float64,  # step_size
            nb.int64,  # smooth_window
            nb.float64[:, :],  # zi
            nb.float64[:],  # current_freq
            nb.float64[:, :],  # beta_state
            nb.float64[:, :],  # freq_history
            nb.float64[:],  # freq_history_sum
            nb.float64,  # gradient_leak
        ),
    ],
    cache=True,
    parallel=True,
)
def iir_lms_kernel(
    data,
    pole_radius,
    sample_freq,
    step_size,
    smooth_window,
    zi,
    current_freq,
    beta_state,
    freq_history,
    freq_history_sum,
    gradient_leak,
):
    r"""Apply LMS-adaptive notch filter to multi-channel data.

    All state arrays are modified in-place.  Uses a circular buffer
    with a running sum for O(1) moving-average smoothing.
    Channels are processed in parallel via :func:`numba.prange`.

    The gradient dy/dtheta is computed with a leaky recursive formulation
    (Nehorai 1985) for stable frequency tracking:

    .. math::

        g[n] = 2\\sin(\\theta)(x[n-1] - r \\cdot y[n-1])
               + \\lambda (2r\\cos(\\theta) g[n-1] - r^2 g[n-2])

    where lambda (*gradient_leak*) controls the trade-off between gradient
    accuracy (lambda -> 1) and numerical stability (lambda -> 0).

    Parameters
    ----------
    data : numpy.ndarray, shape (N, K)
        Input signal.
    pole_radius : float
        Pole radius (0 < r < 1).
    sample_freq : float
        Sampling frequency in Hz.
    step_size : float
        LMS step size.
    smooth_window : int
        Moving-average window length.
    zi : numpy.ndarray, shape (K, 2)
        Filter state per channel, modified in-place.
    current_freq : numpy.ndarray, shape (K,)
        Current frequency estimate per channel, modified in-place.
    beta_state : numpy.ndarray, shape (K, 5)
        Gradient state per channel, modified in-place.
        Layout: ``[x(n-1), y(n-1), power_est, g(n-1), g(n-2)]``.
    freq_history : numpy.ndarray, shape (K, smooth_window)
        Circular buffer of recent frequencies, modified in-place.
    freq_history_sum : numpy.ndarray, shape (K,)
        Running sum of freq_history per channel, modified in-place.
    gradient_leak : float
        Leak factor lambda for recursive gradient.

    Returns
    -------
    output : numpy.ndarray, shape (N, K)
        Filtered output signal.
    learned_trajectory : numpy.ndarray, shape (N,)
        Per-sample learned frequency (from channel 0).
    learned_all : numpy.ndarray, shape (K, N)
        Per-sample learned frequency for every channel.
    """
    N, K = data.shape
    r = pole_radius
    r2 = r * r
    two_pi = 2.0 * np.pi
    nyquist = sample_freq / 2.0 - 1.0
    sw = smooth_window
    lk = gradient_leak

    output = np.empty((N, K), dtype=np.float64)
    learned_all = np.empty((K, N), dtype=np.float64)

    for k in nb.prange(K):
        z0 = zi[k, 0]
        z1 = zi[k, 1]
        freq_k = current_freq[k]
        bst0 = beta_state[k, 0]
        bst1 = beta_state[k, 1]
        power_est = beta_state[k, 2]
        if power_est <= 0.0:
            power_est = max(data[0, k] * data[0, k], 1.0)
        g_prev = beta_state[k, 3]
        g_prev2 = beta_state[k, 4]
        fh_sum = freq_history_sum[k]

        for n in range(N):
            learned_all[k, n] = freq_k

            theta = two_pi * freq_k / sample_freq
            ct = np.cos(theta)
            st = np.sin(theta)

            b1 = -2.0 * ct
            a1 = -2.0 * r * ct

            x = data[n, k]

            # Direct-form II transposed
            y = x + z0
            z0 = b1 * x - a1 * y + z1
            z1 = x - r2 * y

            output[n, k] = y

            # Leaky recursive gradient dy[n]/dtheta (Nehorai 1985, eq. 14-15;
            # DOI: 10.1109/TASSP.1985.1164643):
            #   g[n] = 2*sin(theta)*(x[n-1] - r*y[n-1])
            #          + lambda*(2r*cos(theta)*g[n-1] - r^2*g[n-2])
            # lambda = gradient_leak controls recursion depth (0 = instantaneous,
            # 1 = full IIR gradient). bst0 = x[n-1], bst1 = y[n-1].
            g_n = 2.0 * st * (bst0 - r * bst1) + lk * (2.0 * r * ct * g_prev - r2 * g_prev2)
            g_prev2 = g_prev
            g_prev = g_n

            bst1 = y
            bst0 = x

            # NLMS: exponentially weighted estimate of input power sigma^2_x.
            # Normalised step mu/sigma^2_x decouples convergence speed from signal
            # amplitude (Tan & Jiang 2009, DOI: 10.1109/MSP.2009.934189).
            power_est = 0.99 * power_est + 0.01 * (x * x)
            normalized_step = step_size / (power_est + 1e-8)

            # Gradient-descent update on theta: theta[n+1] = theta[n] - mu_norm * e[n] * g[n]
            # where e[n] = y[n] is the IIR output (residual at notch frequency).
            theta_new = theta - normalized_step * y * g_n
            theta_new = theta_new % two_pi

            freq_new = (theta_new / two_pi) * sample_freq
            if freq_new < 1.0:
                freq_new = 1.0
            elif freq_new > nyquist:
                freq_new = nyquist

            # Circular-buffer moving average
            buf_idx = n % sw
            old_val = freq_history[k, buf_idx]
            freq_history[k, buf_idx] = freq_new
            fh_sum += freq_new - old_val

            freq_k = fh_sum / sw

        # Write back per-channel state
        zi[k, 0] = z0
        zi[k, 1] = z1
        current_freq[k] = freq_k
        beta_state[k, 0] = bst0
        beta_state[k, 1] = bst1
        beta_state[k, 2] = power_est
        beta_state[k, 3] = g_prev
        beta_state[k, 4] = g_prev2
        freq_history_sum[k] = fh_sum

    learned_trajectory = learned_all[0].copy()

    return output, learned_trajectory, learned_all


@nb.njit(
    [
        (
            nb.float64[:, :],  # data
            nb.int64,  # num_sources
            nb.int64,  # num_harmonics
            nb.float64,  # pole_radius
            nb.float64,  # sample_freq
            nb.float64[:],  # step_sizes
            nb.int64,  # smooth_window
            nb.float64[:, :, :, :],  # zi
            nb.float64[:],  # current_freq
            nb.float64[:, :],  # source_state
            nb.float64[:, :],  # freq_history
            nb.float64[:],  # freq_history_sum
            nb.float64[:, :, :, :],  # harm_grad_state
            nb.float64[:, :, :, :, :],  # grad_prop_zi
            nb.float64,  # gradient_leak
        ),
    ],
    cache=True,
)
def iir_harmonic_cascade_lms_kernel(
    data,
    num_sources,
    num_harmonics,
    pole_radius,
    sample_freq,
    step_sizes,
    smooth_window,
    zi,
    current_freq,
    source_state,
    freq_history,
    freq_history_sum,
    harm_grad_state,
    grad_prop_zi,
    gradient_leak,
):
    """Joint-optimisation harmonic cascade LMS kernel.

    Implements the thesis-aligned MIMO cascade (Harvey 2019, eq. 3.56 /
    3.59 / 3.60) where:

    - One theta_s per source (harmonic *m* uses ``m * theta_s``).
    - Shared theta_s across all *K* channels.
    - Recursive gradient beta_{m,s} chains through all *M* harmonics.
    - Per-source LMS update averages gradient contributions across *K*
      channels.
    - Joint optimisation: final cascade output is the error for **all**
      sources, and each source's gradient is propagated through all
      downstream filters to compute ``d(final_output)/d theta_s``.

    Parameters
    ----------
    data : numpy.ndarray, shape (N, K)
        Input signal.
    num_sources : int
        Number of tonal sources (*S*).
    num_harmonics : int
        Number of harmonics per source (*M*).
    pole_radius : float
        Pole radius (0 < r < 1).
    sample_freq : float
        Sampling frequency in Hz.
    step_sizes : numpy.ndarray, shape (S,)
        Per-source LMS step size.
    smooth_window : int
        Moving-average window length.
    zi : numpy.ndarray, shape (S, M, K, 2)
        Per-harmonic, per-channel IIR state, modified in-place.
    current_freq : numpy.ndarray, shape (S,)
        Shared fundamental frequency per source, modified in-place.
    source_state : numpy.ndarray, shape (S, 1)
        ``[power_est]`` per source, modified in-place.
    freq_history : numpy.ndarray, shape (S, smooth_window)
        Circular buffer for frequency smoothing, modified in-place.
    freq_history_sum : numpy.ndarray, shape (S,)
        Running sum, modified in-place.
    harm_grad_state : numpy.ndarray, shape (S, M, K, 4)
        Gradient state per harmonic/channel, modified in-place.
    grad_prop_zi : numpy.ndarray, shape (S, S, M, K, 2)
        IIR state for propagating source *s*'s gradient through
        downstream source *s_ds*'s harmonic *m* filter.
    gradient_leak : float
        Leak factor for recursive gradient.

    Returns
    -------
    output : numpy.ndarray, shape (N, K)
        Filtered signal (after full S x M cascade).
    learned : numpy.ndarray, shape (S, N)
        Per-sample fundamental frequency per source.
    """
    N, K = data.shape
    S = num_sources
    M = num_harmonics
    r = pole_radius
    r2 = r * r
    two_pi = 2.0 * np.pi
    nyquist = sample_freq / 2.0 - 1.0
    sw = smooth_window
    lk = gradient_leak
    # Gradient normaliser: 1/Sum_{m=1}^{M} m^2 = 6 / [M(M+1)(2M+1)]
    # (from the closed-form sum-of-squares formula Sum m^2 = M(M+1)(2M+1)/6).
    # Keeps the effective LMS step size constant regardless of M, preventing
    # divergence when many harmonics are tracked simultaneously.
    grad_norm = 6.0 / (M * (M + 1) * (2 * M + 1)) if M > 1 else 1.0

    output = np.empty((N, K), dtype=np.float64)
    learned = np.empty((S, N), dtype=np.float64)

    error_buf = np.empty((S, K), dtype=np.float64)
    grad_buf = np.empty((S, K), dtype=np.float64)
    source_grad_raw = np.empty(S, dtype=np.float64)

    # Initialise source_state power estimates on first call
    for s in range(S):
        if source_state[s, 0] <= 0.0:
            p = 0.0
            for k in range(K):
                p += data[0, k] * data[0, k]
            source_state[s, 0] = max(p / K, 1.0)

    for n in range(N):
        for s in range(S):
            learned[s, n] = current_freq[s]

        # Forward pass: filter all K channels through S*M cascade
        raw_power_sum = 0.0
        for k in range(K):
            raw_power_sum += data[n, k] * data[n, k]

        for k in range(K):
            val = data[n, k]

            for s in range(S):
                theta_s = two_pi * current_freq[s] / sample_freq

                beta_curr = 0.0
                saved_beta_prev_n1 = 0.0
                saved_beta_prev_n2 = 0.0

                for m_idx in range(M):
                    m = m_idx + 1
                    m_theta = m * theta_s
                    ct = np.cos(m_theta)
                    st = np.sin(m_theta)

                    # IIR notch filter at m*theta_s
                    z0 = zi[s, m_idx, k, 0]
                    z1 = zi[s, m_idx, k, 1]

                    b1 = -2.0 * ct
                    a1 = -2.0 * r * ct

                    y = val + z0
                    z0_new = b1 * val - a1 * y + z1
                    z1_new = val - r2 * y

                    zi[s, m_idx, k, 0] = z0_new
                    zi[s, m_idx, k, 1] = z1_new

                    # Gradient beta_{m,k,s}(n)
                    beta_n1 = harm_grad_state[s, m_idx, k, 0]
                    beta_n2 = harm_grad_state[s, m_idx, k, 1]
                    y_in_prev = harm_grad_state[s, m_idx, k, 2]
                    y_out_prev = harm_grad_state[s, m_idx, k, 3]

                    next_saved_n1 = beta_n1
                    next_saved_n2 = beta_n2

                    beta_m_n = (
                        beta_curr
                        + lk * (-2.0 * ct * saved_beta_prev_n1 + saved_beta_prev_n2)
                        + 2.0 * m * st * y_in_prev
                        + lk * (2.0 * r * ct * beta_n1 - r2 * beta_n2)
                        - 2.0 * m * r * st * y_out_prev
                    )

                    harm_grad_state[s, m_idx, k, 1] = beta_n1
                    harm_grad_state[s, m_idx, k, 0] = beta_m_n
                    harm_grad_state[s, m_idx, k, 2] = val
                    harm_grad_state[s, m_idx, k, 3] = y

                    saved_beta_prev_n1 = next_saved_n1
                    saved_beta_prev_n2 = next_saved_n2

                    beta_curr = beta_m_n
                    val = y

                source_grad_raw[s] = beta_curr

            output[n, k] = val

            # Joint-optimisation chain-rule propagation (Harvey 2019, eqs. 3.59-3.60):
            # The final cascade output is the error for *all* sources. To compute
            # d(final_output)/d theta_s, the local gradient beta_{M,k,s}[n] must be
            # propagated through every downstream source's S*M notch filters using
            # the same IIR structure (grad_prop_zi carries the filter state).
            # For the last source (s == S-1) no downstream filters exist.
            for s in range(S):
                if s == S - 1:
                    grad_buf[s, k] = source_grad_raw[s]
                else:
                    prop = source_grad_raw[s]
                    for s_ds in range(s + 1, S):
                        theta_ds = two_pi * current_freq[s_ds] / sample_freq
                        for m_idx in range(M):
                            m_theta = (m_idx + 1) * theta_ds
                            ct_ds = np.cos(m_theta)

                            gz0 = grad_prop_zi[s, s_ds, m_idx, k, 0]
                            gz1 = grad_prop_zi[s, s_ds, m_idx, k, 1]

                            b1_ds = -2.0 * ct_ds
                            a1_ds = -2.0 * r * ct_ds

                            yg = prop + gz0
                            gz0_new = b1_ds * prop - a1_ds * yg + gz1
                            gz1_new = prop - r2 * yg

                            grad_prop_zi[s, s_ds, m_idx, k, 0] = gz0_new
                            grad_prop_zi[s, s_ds, m_idx, k, 1] = gz1_new
                            prop = yg

                    grad_buf[s, k] = prop

            for s in range(S):
                error_buf[s, k] = val

        # Cross-channel averaged LMS update per source (Harvey 2019, section 3.3.3):
        # theta_s is shared across all K channels, so the gradient is averaged
        # over channels before the update - equivalent to a single-source
        # update on the channel-mean signal, reducing variance.
        for s in range(S):
            update_sum = 0.0
            for k in range(K):
                update_sum += error_buf[s, k] * grad_buf[s, k]
            update_avg = update_sum / K

            power_est = source_state[s, 0]
            power_est = 0.99 * power_est + 0.01 * (raw_power_sum / K)
            source_state[s, 0] = power_est

            normalized_step = step_sizes[s] / (power_est + 1e-8)

            theta_s = two_pi * current_freq[s] / sample_freq
            theta_new = theta_s - normalized_step * grad_norm * update_avg
            theta_new = theta_new % two_pi

            freq_new = (theta_new / two_pi) * sample_freq
            if freq_new < 1.0:
                freq_new = 1.0
            elif freq_new > nyquist:
                freq_new = nyquist

            buf_idx = n % sw
            old_val = freq_history[s, buf_idx]
            freq_history[s, buf_idx] = freq_new
            freq_history_sum[s] += freq_new - old_val

            current_freq[s] = freq_history_sum[s] / sw

    return output, learned


if __name__ == '__main__':
    foo = _delays
    print(foo.parallel_diagnostics(level=4))
