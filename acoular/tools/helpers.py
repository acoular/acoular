# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements some helper functions that are useful for Acoular.

.. autosummary::
    :toctree: generated/

    synthetic
    return_result
    barspectrum
    bardata
    c_air
"""

from warnings import warn

from numpy import (
    array,
    concatenate,
    isscalar,
    newaxis,
    searchsorted,
    sum,  # noqa A004
    where,
    zeros_like,
)
from numpy.ma import masked_where

from acoular.tools.utils import mole_fraction_of_water_vapor


def synthetic(data, freqs, f, num=3):
    """Returns synthesized frequency band values of spectral data.

    If used with :meth:`Beamformer.result()<acoular.fbeamform.BeamformerBase.result>`
    and only one frequency band, the output is identical to the result of the intrinsic
    :meth:`Beamformer.synthetic<acoular.fbeamform.BeamformerBase.synthetic>` method.
    It can, however, also be used with the
    :meth:`Beamformer.integrate<acoular.fbeamform.BeamformerBase.integrate>`
    output and more frequency bands.

    Parameters
    ----------
    data : array of floats
        The spectral data (squared sound pressures in Pa^2) in an array with one value
        per frequency line.
        The number of entries must be identical to the number of
        grid points.
    freqs : array of floats
        The frequencies that correspond to the input *data* (as yielded by
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
        the :attr:`sampling frequency<acoular.base.SamplesGenerator.sample_freq>`
        and used :attr:`FFT block size<acoular.spectra.PowerSpectra.block_size>`.

    """
    if isscalar(f):
        f = (f,)
    if num == 0:
        # single frequency lines
        res = []
        for i in f:
            ind = searchsorted(freqs, i)
            if ind >= len(freqs):
                warn(
                    f'Queried frequency ({i:g} Hz) not in resolved frequency range. Returning zeros.',
                    Warning,
                    stacklevel=2,
                )
                h = zeros_like(data[0])
            else:
                if freqs[ind] != i:
                    warn(
                        f'Queried frequency ({i:g} Hz) not in set of '
                        'discrete FFT sample frequencies. '
                        f'Using frequency {freqs[ind]:g} Hz instead.',
                        Warning,
                        stacklevel=2,
                    )
                h = data[ind]
            res += [h]
    else:
        # fractional octave bands
        res = []
        for i in f:
            f1 = i * 2.0 ** (-0.5 / num)
            f2 = i * 2.0 ** (+0.5 / num)
            ind1 = searchsorted(freqs, f1)
            ind2 = searchsorted(freqs, f2)
            if ind1 == ind2:
                warn(
                    f'Queried frequency band ({f1:g} to {f2:g} Hz) does not '
                    'include any discrete FFT sample frequencies. '
                    'Returning zeros.',
                    Warning,
                    stacklevel=2,
                )
                h = zeros_like(data[0])
            else:
                h = sum(data[ind1:ind2], 0)
            res += [h]
    return array(res)


def return_result(source, nmax=-1, num=128):
    """Collects the output from a
    :meth:`SamplesGenerator.result()<acoular.base.SamplesGenerator.result>`
    generator and returns an assembled array with all the data.

    Parameters
    ----------
    source: SamplesGenerator or derived object.
        This is the  :class:`SamplesGenerator<acoular.base.SamplesGenerator>` data source.
    nmax: integer
        With this parameter, a maximum number of output samples can be set
        (first dimension of array). If set to -1 (default), samples are
        collected as long as the generator yields them.
    num : integer
        This parameter defines the size of the blocks that are fetched.
        Defaults to 128.

    Returns
    -------
    array of floats (number of samples, source.num_channels)
        Array that holds all the data.

    """
    resulter = (_.copy() for _ in source.result(num))

    if nmax > 0:
        nblocks = (nmax - 1) // num + 1
        return concatenate([res for _, res in zip(range(nblocks), resulter)])[:nmax]
    return concatenate(list(resulter))


def barspectrum(data, fftfreqs, num=3, bar=True, xoffset=0.0):
    """Returns synthesized frequency band values of spectral data to be plotted
    as bar graph with the matlpotlib plot command.


    Parameters
    ----------
    data : array of floats
        The spectral data (sound pressures in Pa) in an array with one value
        per frequency line.
    fftfreqs : array of floats
        Discrete frequencies from FFT.
    num : integer
        Controls the width of the frequency bands considered; defaults to
        3 (third-octave band).
    bar : bool
        If True, returns bar-like curve. If False, normal plot (direct
        line between data points) is returned.
    xoffset : float
        If bar is True, offset of the perpendicular line (helpful if
        plotting several curves above each other).

        ===  =====================
        num  frequency band width
        ===  =====================
        1    octave band
        3    third-octave band
        ===  =====================

    Returns
    -------
    (flulist, plist, fc)
    flulist : array of floats
        Lower/upper band frequencies in plottable format.
    plist : array of floats
        Corresponding synthesized frequency band values in plottable format.
    fc : array of floats
        Evaluated band center frequencies.

    """
    if num not in [1, 3]:
        print('Only octave and third-octave bands supported at this moment.')
        return (0, 0, 0)

    # preferred center freqs after din en iso 266 for third-octave bands
    fcbase = array([31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250])
    # DIN band center frequencies from 31.5 Hz to 25 kHz
    fc = concatenate((fcbase, fcbase * 10.0, fcbase[:] * 100.0))[:: (3 // num)]

    # exponent for band width calculation
    ep = 1.0 / (2.0 * num)

    # lowest and highest possible center frequencies
    # for chosen band and sampling frequency
    f_low = fftfreqs[1] * 2**ep
    f_high = fftfreqs[-1] * 2**-ep
    # get possible index range
    i_low = 0 if fc[0] >= f_low else where(fc < f_low)[0][-1]

    i_high = fc.shape[0] if fc[-1] <= f_high else where(fc > f_high)[0][0]

    # synthesize sound pressure values
    p = array([synthetic(data, fftfreqs, list(fc[i_low:i_high]), num)])

    if bar:
        # upper and lower band borders
        flu = concatenate(
            (
                fc[i_low : i_low + 1] * 2**-ep,
                (fc[i_low : i_high - 1] * 2**ep + fc[i_low + 1 : i_high] * 2**-ep) / 2.0,
                fc[i_high - 1 : i_high] * 2**ep,
            ),
        )
        # band borders as coordinates for bar plotting
        flulist = 2 ** (2 * xoffset * ep) * (array([1, 1])[:, newaxis] * flu[newaxis, :]).T.reshape(-1)[1:-1]
        # sound pressures as list for bar plotting
        plist = (array([1, 1])[:, newaxis] * p[newaxis, :]).T.reshape(-1)
    else:
        flulist = fc[i_low:i_high]
        plist = p[0, :]
    # print(flulist.shape, plist.shape)
    return (flulist, plist, fc[i_low:i_high])


def bardata(data, fc, num=3, bar=True, xoffset=0.0, masked=-360):
    """Returns data to be plotted
    as bar graph with the matlpotlib plot command.


    Parameters
    ----------
    data : array of floats
        The spectral data
    fc : array of floats
        Band center frequencies
    bar : bool
        If True, returns bar-like curve. If False, normal plot (direct
        line between data points) is returned.
    xoffset : float
        If bar is True, offset of the perpendicular line (helpful if
        plotting several curves above each other).

        ===  =====================
        num  frequency band width
        ===  =====================
        1    octave band
        3    third-octave band
        ===  =====================

    Returns
    -------
    (flulist, plist)
    flulist : array of floats
        Lower/upper band frequencies in plottable format.
    plist : array of floats
        Corresponding values in plottable format.

    """
    ep = 1.0 / (2.0 * num)

    if bar:
        # upper and lower band borders
        flu = concatenate((fc[:1] * 2**-ep, (fc[:-1] * 2**ep + fc[1:] * 2**-ep) / 2.0, fc[-1:] * 2**ep))
        # band borders as coordinates for bar plotting
        flulist = 2 ** (xoffset * 1.0 / num) * (array([1, 1])[:, newaxis] * flu[newaxis, :]).T.reshape(-1)[1:-1]
        # sound pressures as list for bar plotting
        plist = (array([1, 1])[:, newaxis] * data[newaxis, :]).T.reshape(-1)
    else:
        flulist = fc
        plist = data
    # print(flulist.shape, plist.shape)
    if masked > -360:
        plist = masked_where(plist <= masked, plist)
    return (flulist, plist)


def c_air(t, h, p=101325, co2=0.04):
    r"""
    Calculates the speed of sound in air according to Eq.(15) in :cite:`Cramer1993`.

    This function calculates the speed of sound in air based on temperature, pressure, relative
    humidity, and CO\ :sub:`2` concentration. To calculate the mole fraction of water vapor in
    the air, :meth:`~acoular.tools.utils.mole_fraction_of_water_vapor` uses the more recent work
    of :cite:`Davis1992` to obtain the saturation vapor pressure.

    The function is only valid over the temperature range from 0°C to 30°C (273.15 K to 303.15 K),
    for the pressure range 60 to 110 kPa, a water vapor mole fraction up to 0.06, and CO2
    concentrations up to 1%.

    Parameters
    ----------
    t : float
        Temperature in (°C).
    h : float
        Humidity in percent (0 to 100).
    p : float
        Atmospheric pressure in Pa (default is the standard pressure 101325 Pa).
    co2 : float
        Carbon dioxide concentration in percent (default is 0.04%).

    Returns
    -------
    float
        Speed of sound in air in m/s.

    Raises
    ------
    ValueError
        If the temperature is out of range (0°C to 30°C), the pressure is out of range
        (60 kPa to 110 kPa), the water vapor mole fraction is out of range (up to 0.06),
        or the CO\ :sub:`2` concentration is out of range (up to 1%).

    Notes
    -----
    The speed of sound in air is calculated using the following equation:

    .. math::

        \begin{aligned}
        c(t, p, x_w, c_{CO_2}) = & a_0 + a_1 t + a_2 t^2 + \left(a_3 + a_4 t + a_5 t^2\right) x_w \\
        & + \left(a_6 + a_7 t + a_8 t^2\right) p + \left(a_9 + a_{10} t + a_{11} t^2\right) x_c \\
        & + a_{12} x_w^2 + a_{13} p^2 + a_{14} x_c^2 + a_{15} x_w p x_c
        \end{aligned}

    where:
        - :math:`t` is the temperature in c.
        - :math:`x_w` is the water vapor mole fraction.
        - :math:`x_c` is the carbon dioxide mole fraction (:math:`x_c = c_{CO_2} / 100`).
        - :math:`p` is the atmospheric pressure in Pa.

    Examples
    --------
    Code for reproducing Fig.1 from :cite:`Cramer1993`

    .. plot:: plots/c_air.py
    """
    if t < 0 or t > 30:
        msg = 'Temperature out of range (0°C to 30°C)'
        raise ValueError(msg)
    if p < 60000 or p > 110000:
        msg = 'Pressure out of range (60 kPa to 110 kPa)'
        raise ValueError(msg)

    # Calculate water vapor mole fraction
    x_w = mole_fraction_of_water_vapor(h / 100, t + 273.15, p)
    if x_w > 0.06:
        msg = 'Water vapor mole fraction out of range (up to 0.06)'
        raise ValueError(msg)

    if co2 > 1.0:
        msg = 'CO2 concentration out of range (up to 1%)'
        raise ValueError(msg)

    # Convert CO2 concentration from percent to mole fraction
    x_c = co2 / 100

    # Coefficients from Eq.(15)
    a0 = 331.5024
    a1 = 0.603055
    a2 = -0.000528
    a3 = 51.471935
    a4 = 0.1495874
    a5 = -0.000782
    a6 = -1.82 * 10**-7
    a7 = 3.73 * 10**-8
    a8 = -2.93 * 10**-10
    a9 = -85.20931
    a10 = -0.228525
    a11 = 5.91 * 10**-5
    a12 = -2.835149
    a13 = -2.15 * 10**-13
    a14 = 29.179762
    a15 = 0.000486
    return (
        a0
        + a1 * t
        + a2 * t**2
        + (a3 + a4 * t + a5 * t**2) * x_w
        + (a6 + a7 * t + a8 * t**2) * p
        + (a9 + a10 * t + a11 * t**2) * x_c
        + a12 * x_w**2
        + a13 * p**2
        + a14 * x_c**2
        + a15 * x_w * p * x_c
    )
