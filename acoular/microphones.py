# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements support for array microphone arrangements.

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
    List,
    Property,
    cached_property,
    on_trait_change,
)

# acoular imports
from .deprecation import deprecated_alias
from .internal import digest


@deprecated_alias({'mpos_tot': 'pos_total', 'mpos': 'pos', 'from_file': 'file'}, read_only=['mpos'])
class MicGeom(HasStrictTraits):
    """Provides the geometric arrangement of microphones in the array.

    The geometric arrangement of microphones is read in from an
    xml-source with element tag names `pos` and attributes Name, `x`, `y` and `z`.
    Can also be used with programmatically generated arrangements.
    """

    #: Name of the .xml-file from which to read the data.
    file = File(filter=['*.xml'], exists=True, desc='name of the xml file to import')

    #: Positions as (3, :attr:`num_mics`) array of floats, may include also invalid
    #: microphones (if any). Set either automatically on change of the
    #: :attr:`file` argument or explicitly by assigning an array of floats.
    pos_total = CArray(dtype=float, shape=(3, None), desc='x, y, z position of all microphones')

    #: Positions as (3, :attr:`num_mics`) array of floats, without invalid
    #: microphones; readonly.
    pos = Property(depends_on=['pos_total', 'invalid_channels'], desc='x, y, z position of used microphones')

    #: List that gives the indices of channels that should not be considered.
    #: Defaults to a blank list.
    invalid_channels = List(int, desc='list of invalid channels')

    #: Number of used microphones in the array; readonly.
    num_mics = Property(depends_on=['pos'], desc='number of microphones in the geometry')

    #: Center of the array (arithmetic mean of all used array positions); readonly.
    center = Property(depends_on=['pos'], desc='array center')

    #: Aperture of the array (greatest extent between two microphones); readonly.
    aperture = Property(depends_on=['pos'], desc='array aperture')

    # internal identifier
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
        """Import the microphone positions from .xml file.
        Called when :attr:`file` changes.
        """
        doc = xml.dom.minidom.parse(self.file)
        names = []
        xyz = []
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            xyz.append([float(el.getAttribute(a)) for a in 'xyz'])
        self.pos_total = array(xyz, 'd').swapaxes(0, 1)

    def export_mpos(self, filename):
        """Export the microphone positions to .xml file.

        Parameters
        ----------
        filename : str
            Name of the file to which the microphone positions are written.
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
