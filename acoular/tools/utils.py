"""Utility classes intended for internal use in Acoular.

.. autosummary::
    :toctree: generated/

    mole_fraction_of_water_vapor
"""

from pathlib import Path


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
