# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements support for array microphone arrangements.

.. autosummary::
    :toctree: generated/

    MicGeom

"""

# imports from other packages
import errno
from os import path, strerror

from numpy import array, average
from scipy.spatial.distance import cdist
from traits.api import Bool, CArray, File, HasPrivateTraits, ListInt, Property, cached_property, on_trait_change

from .internal import digest


class MicGeom(HasPrivateTraits):
    """Provides the geometric arrangement of microphones in the array.

    The geometric arrangement of microphones is read in from an
    xml-source with element tag names `pos` and attributes Name, `x`, `y` and `z`.
    Can also be used with programmatically generated arrangements.
    """

    #: Name of the .xml-file from wich to read the data.
    from_file = File(filter=['*.xml'], desc='name of the xml file to import')

    #: Validate mic geom from file
    validate_file = Bool(True, desc='Validate mic geom from file')

    #: Basename of the .xml-file, without the extension; is set automatically / readonly.
    basename = Property(depends_on='from_file', desc='basename of xml file')

    #: List that gives the indices of channels that should not be considered.
    #: Defaults to a blank list.
    invalid_channels = ListInt(desc='list of invalid channels')

    #: Number of microphones in the array; readonly.
    num_mics = Property(depends_on=['mpos'], desc='number of microphones in the geometry')

    #: Center of the array (arithmetic mean of all used array positions); readonly.
    center = Property(depends_on=['mpos'], desc='array center')

    #: Aperture of the array (greatest extent between two microphones); readonly.
    aperture = Property(depends_on=['mpos'], desc='array aperture')

    #: Positions as (3, :attr:`num_mics`) array of floats, may include also invalid
    #: microphones (if any). Set either automatically on change of the
    #: :attr:`from_file` argument or explicitely by assigning an array of floats.
    mpos_tot = CArray(dtype=float, desc='x, y, z position of all microphones')

    #: Positions as (3, :attr:`num_mics`) array of floats, without invalid
    #: microphones; readonly.
    mpos = Property(depends_on=['mpos_tot', 'invalid_channels'], desc='x, y, z position of microphones')

    # internal identifier
    digest = Property(depends_on=['mpos'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        return path.splitext(path.basename(self.from_file))[0]

    @cached_property
    def _get_mpos(self):
        if self.validate_file:
            if len(self.invalid_channels) == 0:
                return self.mpos_tot
            allr = [i for i in range(self.mpos_tot.shape[-1]) if i not in self.invalid_channels]
            return self.mpos_tot[:, array(allr)]
        raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), self.from_file)

    @cached_property
    def _get_num_mics(self):
        return self.mpos.shape[-1]

    @cached_property
    def _get_center(self):
        if self.mpos.any():
            center = average(self.mpos, axis=1)
            # set very small values to zero
            center[abs(center) < 1e-16] = 0.0
            return center
        return None

    @cached_property
    def _get_aperture(self):
        if self.mpos.any():
            return cdist(self.mpos.T, self.mpos.T).max()
        return None

    @on_trait_change('basename')
    def import_mpos(self):
        """Import the microphone positions from .xml file.
        Called when :attr:`basename` changes.
        """
        if not path.isfile(self.from_file):
            # no file there
            self.mpos_tot = array([], 'd')
            # raise error: File not found on _get functions
            self.validate_file = False

        import xml.dom.minidom

        doc = xml.dom.minidom.parse(self.from_file)
        names = []
        xyz = []
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            xyz.append([float(el.getAttribute(a)) for a in 'xyz'])
        self.mpos_tot = array(xyz, 'd').swapaxes(0, 1)
        self.validate_file = True

    def export_mpos(self, filename):
        """Export the microphone positions to .xml file.

        Parameters
        ----------
        filename : str
            Name of the file to which the microphone positions are written.
        """
        basename = path.splitext(path.basename(filename))[0]
        with open(filename, 'w') as f:
            f.write(f'<?xml version="1.1" encoding="utf-8"?><MicArray name="{basename}">\n')
            for i in range(self.mpos.shape[-1]):
                f.write(
                    f'  <pos Name="Point {i+1}" x="{self.mpos[0, i]}" y="{self.mpos[1, i]}" z="{self.mpos[2, i]}"/>\n',
                )
            f.write('</MicArray>')
