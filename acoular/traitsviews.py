# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2019, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements support separate traits_view definitions for all relevant
classes to lift the traitsui requirement for the Acoular package
"""

# imports from other packages
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons

from .microphones import MicGeom
    
MicGeom.class_trait_view('traits_view',
                         View(
                                 ['from_file',
                                  'num_mics~',
                                  '|[Microphone geometry]'
                                  ],
                                  buttons = OKCancelButtons
                                  )
                         )  
                         
from .spectra import PowerSpectra

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

from .calib import Calib

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

from .trajectory import Trajectory

Trajectory.class_trait_view('traits_view',
                            View(
                                    [Item('points', style='custom')
                                    ], 
                                    title='Grid center trajectory', 
                                    buttons = OKCancelButtons
                                    )
                            )

from .grids import RectGrid, RectGrid3D

RectGrid.class_trait_view('traits_view',
                          View(
                                  [['x_min', 'y_min', '|'],
                                   ['x_max', 'y_max', 'z', 
                                    'increment', 'size~{Grid size}', '|'],
                                    '-[Map extension]'
                                    ]
                                   )
                          )

# increment3D omitted in view for easier handling, can be added later
RectGrid3D.class_trait_view('traits_view',
                            View(
                                    [
                                            ['x_min', 'y_min', 'z_min', '|'],
                                            ['x_max', 'y_max', 'z_max', 
                                             'increment', 'size~{Grid size}', 
                                             '|'],
                                             '-[Map extension]'
                                             ]
                                            )
                            )



                         