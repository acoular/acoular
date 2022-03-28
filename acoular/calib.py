# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""
Implements calibration of multichannel time signals.

.. autosummary::
    :toctree: generated/

    Calib
"""

# imports from other packages
from numpy import array
from traits.api import HasPrivateTraits, CLong, File, CArray, Property, \
cached_property, on_trait_change
from os import path

# acoular imports
from .internal import digest

class Calib( HasPrivateTraits ):
    """
    Container for calibration data in `*.xml` format
    
    This class serves as interface to load calibration data for the used
    microphone array.
    """

    #: Name of the .xml file to be imported.
    from_file = File(filter=['*.xml'], 
        desc="name of the xml file to import")

    #: Basename of the .xml-file. Readonly / is set automatically.
    basename = Property( depends_on = 'from_file', 
        desc="basename of xml file")
    
    #: Number of microphones in the calibration data, 
    #: is set automatically / read from file.
    num_mics = CLong( 0, 
        desc="number of microphones in the geometry")

    #: Array of calibration factors, 
    #: is set automatically / read from file.
    data = CArray(
        desc="calibration data")

    # Internal identifier
    digest = Property( depends_on = ['basename', ] )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_basename( self ):
        if not path.isfile(self.from_file):
            return ''
        return path.splitext(path.basename(self.from_file))[0]
    
    @on_trait_change('basename')
    def import_data( self ):
        """ 
        Loads the calibration data from `*.xml` file .
        """
        if not path.isfile(self.from_file):
            # empty calibration
            if self.basename=='':
                self.data = None 
                self.num_mics = 0
            # no file there
            else:
                self.data = array([1.0, ], 'd')
                self.num_mics = 1
            return
        import xml.dom.minidom
        doc = xml.dom.minidom.parse(self.from_file)
        names = []
        data = []
        for element in doc.getElementsByTagName('pos'):
            names.append(element.getAttribute('Name'))
            data.append(float(element.getAttribute('factor')))
        self.data = array(data, 'd')
        self.num_mics = self.data.shape[0]

