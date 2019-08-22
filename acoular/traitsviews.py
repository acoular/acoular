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
                ['x_max', 'y_max', 'z_max', 'increment', 
                 'size~{Grid size}', '|'],
                 '-[Map extension]'
        ]
        )
                            )

from .tbeamform import BeamformerTime, BeamformerTimeSq, BeamformerTimeSqTraj,\
BeamformerTimeTraj, IntegratorSectorTime
    
BeamformerTime.class_trait_view('traits_view',
                                View(
        [
            [Item('steer{}', style='custom')], 
            [Item('source{}', style='custom'), '-<>'], 
            [Item('weights{}', style='simple')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )
                                )

BeamformerTimeSq.class_trait_view('traits_view',
                                  View(
        [
            [Item('steer{}', style='custom')], 
            [Item('source{}', style='custom'), '-<>'], 
            [Item('r_diag', label='diagonal removed')], 
            [Item('weights{}', style='simple')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )
                                )

BeamformerTimeTraj.class_trait_view('traits_view',
                                    View(
        [
            [Item('steer{}', style='custom')], 
            [Item('source{}', style='custom'), '-<>'], 
            [Item('trajectory{}', style='custom')],
            [Item('weights{}', style='simple')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )
                                )

BeamformerTimeSqTraj.class_trait_view('traits_view',
                                      View(
        [
            [Item('steer{}', style='custom')], 
            [Item('source{}', style='custom'), '-<>'], 
            [Item('trajectory{}', style='custom')],
            [Item('r_diag', label='diagonal removed')], 
            [Item('weights{}', style='simple')], 
            '|'
        ], 
        title='Beamformer options', 
        buttons = OKCancelButtons
        )
                                )

IntegratorSectorTime.class_trait_view('traits_view',
                                      View(
        [
            [Item('sectors', style='custom')], 
            [Item('grid', style='custom'), '-<>'], 
            '|'
        ], 
        title='Integrator', 
        buttons = OKCancelButtons
        )
                                )

from .environments import UniformFlowEnvironment, GeneralFlowEnvironment,\
SlotJet, OpenJet

UniformFlowEnvironment.class_trait_view('traits_view',
                                        View(
            [
                ['ma{Flow Mach number}', 'fdv{Flow vector}'], 
                '|[Uniform Flow]'
            ]
        )
                                        )

GeneralFlowEnvironment.class_trait_view('traits_view',
                                        View(
            [
                ['ff{Flow field}', 'N{Max. number of rays}', 'Om{Max. solid angle }'], 
                '|[General Flow]'
            ]
        )
                                         )

SlotJet.class_trait_view('traits_view',
                         View(
            [
                ['v0{Exit velocity}', 'origin{Jet origin}',
                 'flow', 'plane',
                'B{Slot width}'], 
                '|[Slot jet]'
            ]
        )
                         )

OpenJet.class_trait_view('traits_view',
                         View(
            [
                ['v0{Exit velocity}', 'origin{Jet origin}', 
                'D{Nozzle diameter}'], 
                '|[Open jet]'
            ]
        )
                         )

from .tprocess import TimeInOut, TimeAverage, FiltFiltOctave, WriteWAV, WriteH5

TimeInOut.class_trait_view('traits_view',
                           View(
        Item('source', style='custom')
                    )
                           )

TimeAverage.class_trait_view('traits_view',
                             View(
        [Item('source', style='custom'), 
         'naverage{Samples to average}', 
            ['sample_freq~{Output sampling frequency}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Linear average', 
        buttons = OKCancelButtons
                    )
                             )

FiltFiltOctave.class_trait_view('traits_view',
                                View(
        [Item('source', style='custom'), 
         'band{Center frequency}', 
         'fraction{Bandwidth}', 
            ['sample_freq~{Output sampling frequency}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Linear average', 
        buttons = OKCancelButtons
                    )
                                )

WriteWAV.class_trait_view('traits_view',
                           View(
        [Item('source', style='custom'), 
            ['basename~{File name}', 
            '|[Properties]'], 
            '|'
        ], 
        title='Write wav file', 
        buttons = OKCancelButtons
                    )
                          )

WriteH5.class_trait_view('traits_view',
                         View(
        [Item('source', style='custom'), 
            ['name{File name}', 
            '|[Properties]'], 
            '|'
        ], 
        title='write .h5', 
        buttons = OKCancelButtons
                    )
                         )
                        