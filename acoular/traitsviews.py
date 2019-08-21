# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2019, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements support for array microphone arrangements

.. autosummary::
    :toctree: generated/

    MicGeom

"""

# imports from other packages
from traitsui.api import View
from traitsui.menu import OKCancelButtons

from .microphones import MicGeom
from .spectra import PowerSpectra
from .calib import Calib
    
MicGeom.class_trait_view('traits_view',
                         View(
                                 ['from_file',
                                  'num_mics~',
                                  '|[Microphone geometry]'
                                  ],
                                  buttons = OKCancelButtons
                                  )
                         )  
                         
PowerSpectra.class_trait_view('traits_view',
                              View(
                                      ['time_data@{}', 
                                       'calib@{}', 
                                       ['block_size', 
                                        'window', 
                                        'overlap', 
                                        ['ind_low{Low Index}', 
                                         'ind_high{High Index}', 
                                         '-[Frequency range indices]'], 
                                         ['num_blocks~{Number of blocks}', 
                                          'freq_range~{Frequency range}', 
                                          '-'], 
                                          '[FFT-parameters]'
                                          ], 
                                          ], 
                                          buttons = OKCancelButtons
                                          )
                             )

Calib.class_trait_view('traits_view',
                       View(
                               ['from_file{File name}', 
                                ['num_mics~{Number of microphones}', 
                                 '|[Properties]'
                                 ]
                                ], 
                                title='Calibration data', 
                                buttons = OKCancelButtons
                                )
                         )
                         