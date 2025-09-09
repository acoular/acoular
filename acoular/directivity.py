# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""
Implements methods required for directivity shared by source
"""

import numpy as np
from traits.api import (
    CArray,
    Property,
    Str,
)

# acoular imports
from .microphones import MicGeom

def get_radiation_angles(direction, mpos, sourceposition):
    r"""
    Calculate the azimuthal and elevation angles between the microphones and the source.

    The function computes the azimuth (``azi``) and elevation (``ele``) angles between each
    microphone position and the source position, taking into account the orientation of the
    spherical harmonics provided by the parameter ``direction``.

    Parameters
    ----------
    direction : :class:`numpy.ndarray` of shape ``(3,)``
        Unit vector representing the spherical harmonic orientation. It should be a 3-element array
        corresponding to the ``x``, ``y``, and ``z`` components of the direction.
    mpos : :class:`numpy.ndarray` of shape ``(3, N)``
        Microphone positions in a 3D Cartesian coordinate system. The array should have 3 rows (the
        ``x``, ``y`` and ``z`` coordinates) and ``N`` columns (one for each microphone).
    sourceposition : :class:`numpy.ndarray` of shape ``(3,)``
        Position of the source in a 3D Cartesian coordinate system. It should be a 3-element array
        corresponding to the ``x``, ``y``, and ``z`` coordinates of the source.

    Returns
    -------
    azi : :class:`numpy.ndarray` of shape ``(N,)``
        Azimuth angles in radians between the microphones and the source. The range of the values is
        :math:`[0, 2\pi)`.
    ele : :class:`numpy.ndarray` of shape ``(N,)``
        Elevation angles in radians between the microphones and the source. The range of the values
        is :math:`[0, \pi]`.

    See Also
    --------
    :func:`numpy.linalg.norm` :
        Computes the norm of a vector.
    :func:`numpy.arctan2` :
        Computes the arctangent of two variables, preserving quadrant information.

    Notes
    -----
    - The function accounts for a coordinate system transformation where the ``z``-axis in Acoular
      corresponds to the ``y``-axis in spherical coordinates, and the ``y``-axis in Acoular
      corresponds to the ``z``-axis in spherical coordinates.
    - The elevation angle (``ele``) is adjusted to the range :math:`[0, \pi]` by adding
      :math:`\pi/2` after the initial calculation.

    Examples
    --------
    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> direction = [1, 0, 0]
    >>> mpos = np.array([[1, 2], [0, 0], [0, 1]])  # Two microphones
    >>> sourceposition = [0, 0, 0]
    >>> azi, ele = ac.sources.get_radiation_angles(direction, mpos, sourceposition)
    >>> azi
    array([0.       , 5.8195377])
    >>> ele
    array([4.71238898, 4.71238898])
    """
    # direction of the Spherical Harmonics
    direc = np.array(direction, dtype=float)
    direc = direc / spla.norm(direc)
    # distances
    source_to_mic_vecs = mpos - np.array(sourceposition).reshape((3, 1))
    source_to_mic_vecs[2] *= -1  # invert z-axis (acoular)    #-1
    # z-axis (acoular) -> y-axis (spherical)
    # y-axis (acoular) -> z-axis (spherical)
    # theta
    ele = np.arctan2(np.sqrt(source_to_mic_vecs[0] ** 2 + source_to_mic_vecs[2] ** 2), source_to_mic_vecs[1])
    ele += np.arctan2(np.sqrt(direc[0] ** 2 + direc[2] ** 2), direc[1])
    ele += np.pi * 0.5  # convert from [-pi/2, pi/2] to [0,pi] range
    # phi
    azi = np.arctan2(source_to_mic_vecs[2], source_to_mic_vecs[0])
    azi += np.arctan2(direc[2], direc[0])
    azi = np.mod(azi, 2 * np.pi)
    return azi, ele

class MicGeomDirectional(MicGeom):
    """
    Extension of MicGeom where directivity can be specified for each microphone
    """
    # @TODO none of the XML parsing mechanics has been implemented
    # @TODO decide how to store/represent each directivity

    # @TODO don't store these as strings
    #: Array containing directivity for each microphone, including invalid ones
    directivities_total = CArray(dtype=Str, desc='directivity for each microphone')

    # @TODO don't store these as strings
    #: Array containing directivity for each microphone, excluding those in invalid_channels (read-only)
    directivities = Property(depends_on=['pos_total', 'invalid_channels'], desc='directivity for each microphone')

    #: Vectors defining the local orientation of the microphones relative to global space
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientations_total = CArray(dtype=CArray(3,3), desc='orientations for each microphone')

    orientations = Property(depends_on=['orientations_total', 'invalid_channels'], desc='orientation for each microphone')
