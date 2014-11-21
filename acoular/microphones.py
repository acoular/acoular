# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements support for array microphone arrangements

.. autosummary::
    :toctree: generated/

    MicGeom

"""

# imports from other packages
from numpy import array
from traits.api import HasPrivateTraits, Property, File, \
CArray, List, cached_property, on_trait_change
from traitsui.api import View
from traitsui.menu import OKCancelButtons
from os import path

from .internal import digest


class MicGeom( HasPrivateTraits ):
    """
    Provides the geometric arrangement of microphones in the mic. array.
    
    The geometric arrangement of microphones is read in from an 
    xml-source with element tag names 'pos' and attributes Name, x, y and z. 
    Can also be used with programmatically generated arrangements.
    """

    #: Name of the .xml-file from wich to read the data.
    from_file = File(filter=['*.xml'],
        desc="name of the xml file to import")

    #: Basename of the .xml-file, without the extension, is set automatically / readonly.
    basename = Property( depends_on = 'from_file',
        desc="basename of xml file")

    #: List that gives the indices of channels that should not be considered.
    #: Defaults to a blank list.
    invalid_channels = List(
        desc="list of invalid channels")

    #: Number of microphones in the array, readonly.
    num_mics = Property( depends_on = ['mpos', ],
        desc="number of microphones in the geometry")

    #: Positions as (3, :attr:`num_mics`) array of floats, may include also invalid
    #: microphones (if any). Set either automatically on change of the
    #: :attr:`from_file` argument or explicitely by assigning an array of floats.
    mpos_tot = CArray(
        desc="x, y, z position of all microphones")

    #: Positions as (3, num_mics) array of floats, without invalid
    #: microphones, readonly.
    mpos = Property( depends_on = ['mpos_tot', 'invalid_channels'],
        desc="x, y, z position of microphones")

    # internal identifier
    digest = Property( depends_on = ['mpos', ])

    traits_view = View(
        ['from_file',
        'num_mics~',
        '|[Microphone geometry]'
        ],
#        title='Microphone geometry',
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.from_file))[0]

    @cached_property
    def _get_mpos( self ):
        if len(self.invalid_channels)==0:
            return self.mpos_tot
        allr = range(self.mpos_tot.shape[-1])
        for channel in self.invalid_channels:
            if channel in allr:
                allr.remove(channel)
        return self.mpos_tot[:, array(allr)]

    @cached_property
    def _get_num_mics( self ):
        return self.mpos.shape[-1]

    @on_trait_change('basename')
    def import_mpos( self ):
        """
        Import the microphone positions from .xml file.
        Called when :attr:`basename` changes.
        """
        if not path.isfile(self.from_file):
            # no file there
            self.mpos_tot = array([], 'd')
            self.num_mics = 0
            return
        import xml.dom.minidom
        doc = xml.dom.minidom.parse(self.from_file)
        names = []
        xyz = []
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            xyz.append(map(lambda a : float(el.getAttribute(a)), 'xyz'))
        self.mpos_tot = array(xyz, 'd').swapaxes(0, 1)
#        self.num_mics = self.mpos.shape[1]

