# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
"""
beamfpy.py: classes for calculations in the time domain

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming
 
(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""

# imports from other packages
from numpy import array
from traits.api import HasPrivateTraits, CLong, File, CArray, Property, \
cached_property, on_trait_change
from traitsui.api import View
from traitsui.menu import OKCancelButtons
from os import path

# beamfpy imports
from .internal import digest

class Calib( HasPrivateTraits ):
    """
    container for calibration data that is loaded from
    an .xml-file
    """

    # name of the .xml file
    from_file = File(filter=['*.xml'], 
        desc="name of the xml file to import")

    # basename of the .xml-file
    basename = Property( depends_on = 'from_file', 
        desc="basename of xml file")
    
    # number of microphones in the calibration data 
    num_mics = CLong( 0, 
        desc="number of microphones in the geometry")

    # array of calibration factors
    data = CArray(
        desc="calibration data")

    # internal identifier
    digest = Property( depends_on = ['basename', ] )

    traits_view = View(
        ['from_file{File name}', 
            ['num_mics~{Number of microphones}', 
                '|[Properties]'
            ]
        ], 
        title='Calibration data', 
        buttons = OKCancelButtons
                    )

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
        "loads the calibration data from .xml file"
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

