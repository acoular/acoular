"""Utility classes intended for internal use in Acoular.

.. autosummary::
    :toctree: generated/

    get_file_basename
    find_basename
    mole_fraction_of_water_vapor
"""

from pathlib import Path

import numpy as np


def get_file_basename(file, alternative_basename='void'):
    """Return the basename of the file.

    Parameters
    ----------
    file : str
        File path.

    Returns
    -------
    str
        Basename of the file.
    """
    basename = Path(file).stem
    return basename if basename else alternative_basename


def find_basename(source, alternative_basename='void'):
    """Return the basename of the original source.

    Traverses the source chain of the object and returns the basename of the original source.
    If the source object does not have a basename, uses the alternative basename.

    Parameters
    ----------
    source : instance
        :class:`~acoular.base.Generator` derived object
    alternative_basename : str
        Alternative basename to use if the source object does not have a basename.


    Returns
    -------
    str
        Basename of the original source.
    """
    while source:
        basename = getattr(source, 'basename', None)
        if basename is not None:
            return basename
        source = getattr(source, 'source', None)
    return alternative_basename


def mole_fraction_of_water_vapor(h, t, p=101325):
    r"""Mole fraction of water vapor in the air for real gases.

    Calculates the mole fraction of water vapor in air from the relative humidity,
    based on the equations provided in the appendix of :cite:`Cramer1993` and
    the enhancement factors from :cite:`Davis1992`.

    Parameters
    ----------
    h : float
        Relative humidity as a fraction [0,1].
    t : float
        Thermodynamic temperature in K.
    p : float
        Atmospheric pressure in Pa (default is the standard pressure 101325 Pa).

    Returns
    -------
    float
        Mole fraction of water vapor.

    Notes
    -----
    The mole fraction is calculated as:

    .. math::
        x_w = h \cdot f \cdot \frac{p_{sv}}{p},

    where:
      - :math:`h` is the relative humidity as a fraction [0,1].
      - :math:`f` is the enhancement factor:

        .. math::
            f = 1.00062 + 3.14 \times 10^{-8} \cdot p + 5.6 \times 10^{-7} \cdot t^2.

      - :math:`p_{sv}` is the saturation vapor pressure of water vapor in air:

        .. math::
            p_{sv} = \exp(A \cdot t^2 + B \cdot t + C + \frac{D}{t}),

        with the updated coefficients from :cite:`Davis1992`:

        .. math::
            A = 1.2378847 \times 10^{-5}, \\
            B = -1.9121316 \times 10^{-2}, \\
            C = 33.93711047, \\
            D = -6.3431645 \times 10^3.

    """
    f = 1.00062 + 3.14 * 10 ** (-8) * p + 5.6 * 10 ** (-7) * t**2  # enhancement factor
    # Saturation vapor pressure using updated coefficients from Davis (1992)
    A = 1.2378847 * 10 ** (-5)  # noqa: N806
    B = -1.9121316 * 10 ** (-2)  # noqa: N806
    C = 33.93711047  # noqa: N806
    D = -6.3431645 * 10**3  # noqa: N806
    p_sv = np.exp(A * t**2 + B * t + C + D / t)  # p_sv in Pa
    return h * f * p_sv / p
