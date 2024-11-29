# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements calibration of multichannel time signals.

.. autosummary::
    :toctree: generated/

    Calib
"""

# imports from other packages
import xml.dom.minidom

from numpy import array
from traits.api import CArray, CLong, File, HasPrivateTraits, Property, cached_property, on_trait_change

# acoular imports
from .deprecation import deprecated_alias
from .internal import digest


@deprecated_alias({'from_file': 'file'})
class Calib(HasPrivateTraits):
    """Container for calibration data in `*.xml` format.

    This class serves as interface to load calibration data for the used
    microphone array. The calibration factors are stored as [Pa/unit].
    """

    #: Name of the .xml file to be imported.
    file = File(filter=['*.xml'], exists=True, desc='name of the xml file to import')

    #: Number of microphones in the calibration data,
    #: is set automatically / read from file.
    num_mics = CLong(0, desc='number of microphones in the geometry')

    #: Array of calibration factors,
    #: is set automatically / read from file.
    data = CArray(desc='calibration data')

    # Internal identifier
    digest = Property(depends_on=['data'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @on_trait_change('file')
    def import_data(self):
        """Loads the calibration data from `*.xml` file ."""
        doc = xml.dom.minidom.parse(self.file)
        names = []
        data = []
        for element in doc.getElementsByTagName('pos'):
            names.append(element.getAttribute('Name'))
            data.append(float(element.getAttribute('factor')))
        self.data = array(data, 'd')
        self.num_mics = self.data.shape[0]
