"""Tests for :class:`~acoular.sources.MovingPointSourceDipole`.

All tests use a source that moves on a straight line with constant velocity, for which the emission
time can be computed in closed form, independent of the Newton-Raphson iteration in
:meth:`~acoular.sources.MovingPointSourceDipole.get_emission_time`.
"""

import acoular as ac

import numpy as np
import pytest

FS = 12800.0  # sampling frequency
NUM = 2000  # number of samples
C = 343.0  # speed of sound
V = np.array([-50.0, 0.0, 0.0])  # source velocity, the source moves in -x direction
P0 = np.array([4.0, 0.0, 3.0])  # source position at t = 0; it passes over the microphones at t = 0.08 s
UP = 16  # default up-sampling factor of the source
MICS = np.array([[0.0, 0.0, 0.0], [1.5, 2.0, 0.0], [-2.0, -1.0, 0.0]])  # one position per row


def get_trajectory(shift=(0.0, 0.0, 0.0)):
    """Straight line with constant velocity, translated by ``shift``.

    The spline through collinear, equidistant points is linear, so the source position is exactly
    ``P0 + V * t + shift``.
    """
    return ac.Trajectory(points={t: tuple(P0 + V * t + shift) for t in (-0.1, 0.0, 0.1, 0.2)})


def get_source(mic_pos, cls=ac.MovingPointSourceDipole, shift=(0.0, 0.0, 0.0), **kwargs):
    """Source emitting white noise, received at ``mic_pos`` (one microphone position per row).

    ``shift`` translates the whole scene: the microphones and the trajectory are moved by the same
    vector. Distances, emission times and radial Mach numbers depend only on the positions of the
    source and the microphones relative to each other and must therefore not change, only the
    source positions are moved by ``shift``.

    ``cls`` is the source class, e.g. :class:`~acoular.sources.MovingPointSource` instead of the
    dipole. ``kwargs`` are passed to it (e.g. ``direction`` or ``conv_amp``).
    """
    return cls(
        signal=ac.WNoiseGenerator(sample_freq=FS, num_samples=NUM, seed=1),
        mics=ac.MicGeom(pos_total=(np.array(mic_pos, dtype=float) + shift).T),
        trajectory=get_trajectory(shift),
        env=ac.Environment(c=C),
        **kwargs,
    )


def get_output(src, num=256):
    """All output blocks of ``src`` (copied, the block buffer may be reused, see #654)."""
    return np.vstack([block.copy() for block in src.result(num)])


def exact_emission(mic_pos):
    """Emission times, source positions and radial Mach numbers of the straight line trajectory.

    The sound received at microphone ``m`` at the time ``t = n / FS`` was emitted at the time
    ``te = t - tau`` from the source position ``p(te) = P0 + V te``. The propagation time ``tau``
    satisfies ``|m - p(te)| = c tau``.

    Derivation of ``tau``: let ``r0 = m - p(t)`` be the vector from the source position at the
    receiving time to the microphone. Since ``p(te) = p(t) - V tau``, the vector from the source
    position at the emission time to the microphone is ``m - p(te) = r0 + V tau``. Squaring
    ``|r0 + V tau| = c tau`` gives ``r0.r0 + 2 tau V.r0 + tau**2 V.V = c**2 tau**2``, i.e. the
    quadratic equation

        a tau**2 - 2 b tau - r0.r0 = 0,   with   a = c**2 - V.V   and   b = V.r0,

    with the roots ``tau = (b +- sqrt(b**2 + a r0.r0)) / a``. For a subsonic source (``a > 0``),
    ``sqrt(b**2 + a r0.r0) >= |b|``, so the root with ``-`` is not positive (emission in the
    future) and the root with ``+`` is the only physical solution. For ``V = 0`` it reduces to
    ``tau = |r0| / c``.

    Returns the emission times (shape (NUM, num_mics)), the source positions (shape
    (3, NUM, num_mics)), the unit vectors from the source to the microphones (same shape) and the
    radial Mach numbers (shape (NUM, num_mics)).
    """
    t = np.arange(NUM) / FS  # receiving times, shape (NUM,)
    m = np.array(mic_pos, dtype=float).T[:, np.newaxis, :]  # microphone positions, shape (3, 1, M)
    # vectors from the source position at the receiving time, p(t) = P0 + V t, to the microphones,
    # shape (3, NUM, M)
    r0 = m - (P0[:, np.newaxis, np.newaxis] + V[:, np.newaxis, np.newaxis] * t[np.newaxis, :, np.newaxis])
    a = C**2 - V @ V  # c**2 - |V|**2, positive for a subsonic source
    b = np.einsum('i,inm->nm', V, r0)  # V.r0 for each sample n and microphone m, shape (NUM, M)
    tau = (b + np.sqrt(b * b + a * (r0 * r0).sum(0))) / a  # propagation time t - te (positive root)
    te = t[:, np.newaxis] - tau
    xs = P0[:, np.newaxis, np.newaxis] + V[:, np.newaxis, np.newaxis] * te[np.newaxis]
    unit = (m - xs) / np.sqrt(((m - xs) ** 2).sum(0))  # unit vectors from the source to the microphones
    mach = np.einsum('i,inm->nm', V, unit) / C  # radial Mach number, positive for an approaching source
    return te, xs, unit, mach


@pytest.mark.parametrize('direction', [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])
def test_conv_amp_factor(direction):
    """``conv_amp=True`` multiplies the output by ``(1 - Mr)**-2`` per microphone and sample.

    The same source is computed with and without ``conv_amp``, and the ratio of the two outputs is
    compared with ``(1 - Mr)**-2``. The radial Mach number ``Mr`` is computed independently from
    the exact emission time, not with
    :meth:`~acoular.sources.MovingPointSourceDipole.get_emission_time`. The test therefore fails
    if ``conv_amp`` has no effect and also if it is applied with a wrong ``Mr``. The factor
    amplifies the approaching source (``Mr > 0``) and attenuates the receding source, as for
    :class:`~acoular.sources.MovingPointSource`.

    For the axis (1, 0, 0), ``cos(theta)`` passes through zero when the source passes the
    microphones. There, the output is small and its relative rounding error is larger (see the
    comment on ``atol``).
    """
    out_off = get_output(get_source(MICS, direction=direction, conv_amp=False))  # shape (NUM, M)
    out_on = get_output(get_source(MICS, direction=direction, conv_amp=True))
    mach = exact_emission(MICS)[3]  # exact radial Mach number, shape (NUM, M)
    gain = (1 - mach) ** -2  # expected convective amplification, 0.81 ... 1.33 in this scene

    # rtol: accuracy of the factor. atol: the two-pole difference 1/rm1 - 1/rm2 (pole distance
    # ~5e-7 m at a distance of ~5 m) loses about 7 digits, so its relative rounding error is about
    # 2e-9 / |cos(theta)|. atol covers the samples close to cos(theta) = 0 (axis (1, 0, 0)).
    np.testing.assert_allclose(out_on, out_off * gain, rtol=1e-6, atol=1e-8 * np.abs(out_off).max())

    # Direction of the effect: louder when approaching, quieter when receding. Excluded are the
    # samples close to zero (their ratio is not meaningful) and |Mr| <= 0.01 (factor close to 1).
    valid = np.abs(out_off) > 1e-3 * np.abs(out_off).max()
    ratio = out_on[valid] / out_off[valid]
    assert (ratio[mach[valid] > 0.01] > 1).all()
    assert (ratio[mach[valid] < -0.01] < 1).all()
    # Both cases occur in the scene, so the two checks above are not empty (all() of an empty
    # array is True).
    assert (mach > 0.01).any()
    assert (mach < -0.01).any()


@pytest.mark.parametrize(('time', 'sign'), [(0.0, 1.0), (0.15, -1.0)], ids=['approaching', 'receding'])
def test_emission_time_mach_sign_and_translation(time, sign):
    """``get_emission_time`` returns the radial Mach number of :class:`MovingPointSource`.

    ``Mr`` is positive for an approaching and negative for a receding source and, like the
    emission time and the distance, does not change if the whole scene is translated.
    """
    shift = np.array([100.0, -50.0, 20.0])
    t = np.full(len(MICS), time)
    te, rm, mach, xs = get_source(MICS).get_emission_time(t.copy(), 0)
    te_s, rm_s, mach_s, xs_s = get_source(MICS, shift=shift).get_emission_time(t.copy(), 0)

    n = round(time * FS)
    te_exact, _, _, mach_exact = exact_emission(MICS)
    assert (sign * mach > 0).all()
    np.testing.assert_allclose(te, te_exact[n], rtol=0, atol=0.1 / UP / FS)
    np.testing.assert_allclose(mach, mach_exact[n], rtol=1e-6)
    np.testing.assert_allclose(te_s, te, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rm_s, rm, rtol=1e-9)
    np.testing.assert_allclose(mach_s, mach, rtol=1e-9)
    np.testing.assert_allclose(xs_s, xs + shift[:, np.newaxis], rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    ('direction', 'expected'),
    [
        (
            (0.0, 0.0, 1.0),
            [
                -0.14497008777013354,
                0.021685052557430577,
                -0.07727106787715758,
                -0.1231321548206848,
                -0.009526378704628096,
                -0.11874714118572606,
                0.033334711654074305,
                0.18330908843212204,
                -0.15707827843043917,
            ],
        ),
        (
            (1.0, 0.0, 0.0),
            [
                -0.2333379322089594,
                0.0268901013030552,
                -0.0676967797033321,
                -0.06417051817121426,
                -0.00170854164583097,
                0.01744788265497663,
                -0.015313098101692723,
                -0.1396885740630108,
                0.1660915244972694,
            ],
        ),
    ],
)
def test_single_microphone_at_origin_unchanged(direction, expected):
    """For a single microphone at the origin and ``conv_amp=False``, the output is unchanged.

    ``expected`` holds the output of Acoular 26.08 for the same source at the 9 samples
    ``indices = [0, 250, 500, ..., 1750, 1999]`` (every 250th sample and the last one), i.e.
    ``expected[k]`` is the output at sample ``indices[k]``. The samples cover the approach of the
    source, the pass-by (received between the samples 1000 and 1250) and the recession, and for the
    axis (1, 0, 0) the change of sign of the directivity.

    For a microphone at the origin, the fix of the radial Mach number does not change the output:
    the unit vector from the origin to the source used before is the negative unit vector from the
    source to the microphone, so the Newton-Raphson iteration is the same. With ``conv_amp=False``,
    the convective amplification is not applied.
    """
    out = get_output(get_source([MICS[0]], direction=direction))  # shape (NUM, 1)
    indices = [*range(0, NUM, 250), NUM - 1]  # samples 0, 250, ..., 1750 and the last sample 1999
    np.testing.assert_allclose(out[indices, 0], expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize(('trait', 'value'), [('conv_amp', True)])
def test_digest_depends_on(trait, value):
    """The digest changes with the traits that change the output.

    Acoular uses the digest as the key of cached results. If a trait that changes the output is
    missing in ``depends_on`` of the digest, two sources that differ only in this trait get the
    same digest, and a cache may return the result of the other source. Each case sets one trait
    to a value different from its default and checks that the digest changes.
    """
    src = get_source(MICS)  # default values, e.g. conv_amp=False
    digest = src.digest
    setattr(src, trait, value)  # set the trait by its name, e.g. src.conv_amp = True
    assert src.digest != digest
