# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implements support for array microphone arrangements.

.. autosummary::
    :toctree: generated/

    MicGeom
"""

# imports from other packages
import xml.dom.minidom
from pathlib import Path

from numpy import array, average
from scipy.spatial.distance import cdist
from traits.api import (
    CArray,
    File,
    HasStrictTraits,
    ListInt,
    Property,
    cached_property,
    on_trait_change,
)

# acoular imports
from .deprecation import deprecated_alias
from .internal import digest


@deprecated_alias({'mpos_tot': 'pos_total', 'mpos': 'pos', 'from_file': 'file'}, read_only=['mpos'])
class MicGeom(HasStrictTraits):
    """
    Provide the geometric arrangement of microphones in an array.

    This class allows you to define, import, and manage the spatial positions of
    microphones in a microphone array. The positions can be read from an XML file or set
    programmatically. Invalid microphones can be excluded by specifying their indices.

    Notes
    -----
    - If :attr:`file` is updated, the :meth:`import_mpos` method is automatically called to update
      :attr:`pos_total`.
    - Small numerical values in the computed :attr:``center`` are set to zero for numerical
      stability.
    """

    #: Path to the XML file containing microphone positions. The XML file should have elements with
    #: the tag ``pos`` and attributes ``Name``, ``x``, ``y``, and ``z``.
    file = File(filter=['*.xml'], exists=True, desc='name of the xml file to import')

    #: Array containing the ``x, y, z`` positions of all microphones, including invalid ones, shape
    #: ``(3,`` :attr:`num_mics` ``)``. This is set automatically when :attr:`file` changes or
    #: explicitly by assigning an array of floats.
    pos_total = CArray(dtype=float, shape=(3, None), desc='x, y, z position of all microphones')

    #: Array containing the ``x, y, z`` positions of valid microphones (i.e., excluding those in
    #: :attr:`invalid_channels`), shape ``(3,`` :attr:`num_mics` ``)``. (read-only)
    pos = Property(depends_on=['pos_total', 'invalid_channels'], desc='x, y, z position of used microphones')

    #: List of indices indicating microphones to be excluded from calculations and results.
    #: Default is ``[]``.
    invalid_channels = ListInt(desc='list of invalid channels')

    #: Number of valid microphones in the array. (read-only)
    num_mics = Property(depends_on=['pos'], desc='number of microphones in the geometry')

    #: The geometric center of the array, calculated as the arithmetic mean of the positions of all
    #: valid microphones. (read-only)
    center = Property(depends_on=['pos'], desc='array center')

    #: The maximum distance between any two valid microphones in the array. (read-only)
    aperture = Property(depends_on=['pos'], desc='array aperture')

    #: A unique identifier for the geometry, based on its properties. (read-only)
    digest = Property(depends_on=['pos'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_pos(self):
        if len(self.invalid_channels) == 0:
            return self.pos_total
        allr = [i for i in range(self.pos_total.shape[-1]) if i not in self.invalid_channels]
        return self.pos_total[:, array(allr)]

    @cached_property
    def _get_num_mics(self):
        return self.pos.shape[-1]

    @cached_property
    def _get_center(self):
        if self.pos.any():
            center = average(self.pos, axis=1)
            # set very small values to zero
            center[abs(center) < 1e-16] = 0.0
            return center
        return None

    @cached_property
    def _get_aperture(self):
        if self.pos.any():
            return cdist(self.pos.T, self.pos.T).max()
        return None

    @on_trait_change('file')
    def import_mpos(self):
        """
        Import the microphone positions from an XML file.

        This method parses the XML file specified in :attr:`file` and extracts the ``x``, ``y``, and
        ``z`` positions of microphones. The data is stored in :attr:`pos_total` attribute as an
        array of shape ``(3,`` :attr:`num_mics` ``)``.

        This method is called when :attr:`file` changes.

        Raises
        ------
        xml.parsers.expat.ExpatError
            If the XML file is malformed or cannot be parsed.
        ValueError
            If the attributes ``x``, ``y``, or ``z`` in any ``<pos>`` element are missing or cannot
            be converted to a float.

        Examples
        --------
        The microphone geometry changes by changing the :attr:`file` attribute.

        >>> from acoular import MicGeom  # doctest: +SKIP
        >>> mg = MicGeom(file='/path/to/geom1.xml')  # doctest: +SKIP
        >>> mg.center  # doctest: +SKIP
        array([-0.25,  0.  ,  0.25]) # doctest: +SKIP
        >>> mg.file = '/path/to/geom2.xml'  # doctest: +SKIP
        >>> mg.center  # doctest: +SKIP
        array([0.        , 0.33333333, 0.66666667]) # doctest: +SKIP
        """
        doc = xml.dom.minidom.parse(self.file)
        names = []
        xyz = []
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            xyz.append([float(el.getAttribute(a)) for a in 'xyz'])
        self.pos_total = array(xyz, 'd').swapaxes(0, 1)

    def export_mpos(self, filename):
        """
        Export the microphone positions to an XML file.

        This method generates an XML file containing the positions of all valid microphones in the
        array. Each microphone is represented by a ``<pos>`` element with ``Name``, ``x``, ``y``,
        and ``z`` attributes. The generated XML is formatted to match the structure required for
        importing into the :class:`MicGeom` class.

        Parameters
        ----------
        filename : str
            The name of the file to which the microphone positions will be written. The file
            extension must be ``.xml``.

        Raises
        ------
        OSError
            If the file cannot be written due to permissions issues or invalid file paths.

        Notes
        -----
        - The file will be saved in UTF-8 encoding.
        - The ``Name`` attribute for each microphone is set as ``"Point {i+1}"``, where ``i`` is the
          index of the microphone.
        - This method only exports the positions of the valid microphones (those not listed in
          :attr:`invalid_channels`).
        """
        filepath = Path(filename)
        basename = filepath.stem
        with filepath.open('w', encoding='utf-8') as f:
            f.write(f'<?xml version="1.1" encoding="utf-8"?><MicArray name="{basename}">\n')
            for i in range(self.pos.shape[-1]):
                f.write(
                    f'  <pos Name="Point {i+1}" x="{self.pos[0, i]}" y="{self.pos[1, i]}" z="{self.pos[2, i]}"/>\n',
                )
            f.write('</MicArray>')
