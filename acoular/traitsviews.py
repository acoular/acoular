# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements support separate traits_view definitions for all relevant
classes to lift the traitsui requirement for the Acoular package.
"""

# imports from other packages
from traitsui.api import Item, View
from traitsui.menu import OKCancelButtons

from .base import TimeOut
from .calib import Calib
from .environments import GeneralFlowEnvironment, OpenJet, RotatingFlow, SlotJet, UniformFlowEnvironment
from .fbeamform import (
    BeamformerBase,
    BeamformerCapon,
    BeamformerClean,
    BeamformerCleansc,
    BeamformerCMF,
    BeamformerDamas,
    BeamformerDamasPlus,
    BeamformerEig,
    BeamformerFunctional,
    BeamformerGIB,
    BeamformerMusic,
    BeamformerOrth,
)
from .grids import RectGrid, RectGrid3D
from .microphones import MicGeom
from .process import Average
from .sources import MaskedTimeSamples, SourceMixer, TimeSamples
from .spectra import PowerSpectra
from .tbeamform import BeamformerTime, BeamformerTimeSq, BeamformerTimeSqTraj, BeamformerTimeTraj, IntegratorSectorTime
from .tprocess import FiltFiltOctave, WriteH5, WriteWAV
from .trajectory import Trajectory

MicGeom.class_trait_view(
    'traits_view',
    View(
        [
            'file',
            'num_mics~',
            '|[Microphone geometry]',
        ],
        buttons=OKCancelButtons,
    ),
)

PowerSpectra.class_trait_view(
    'traits_view',
    View(
        [
            'time_data@{}',
            'calib@{}',
            [
                'block_size',
                'window',
                'overlap',
                ['ind_low{Low Index}', 'ind_high{High Index}', '-[Frequency range indices]'],
                ['num_blocks~{Number of blocks}', 'freq_range~{Frequency range}', '-'],
                '[FFT-parameters]',
            ],
        ],
        buttons=OKCancelButtons,
    ),
)

Calib.class_trait_view(
    'traits_view',
    View(
        [
            'file{File name}',
            [
                'num_mics~{Number of microphones}',
                '|[Properties]',
            ],
        ],
        title='Calibration data',
        buttons=OKCancelButtons,
    ),
)

Trajectory.class_trait_view(
    'traits_view',
    View(
        [
            Item('points', style='custom'),
        ],
        title='Grid center trajectory',
        buttons=OKCancelButtons,
    ),
)

RectGrid.class_trait_view(
    'traits_view',
    View(
        [
            ['x_min', 'y_min', '|'],
            ['x_max', 'y_max', 'z', 'increment', 'size~{Grid size}', '|'],
            '-[Map extension]',
        ],
    ),
)

# increment3D omitted in view for easier handling, can be added later
RectGrid3D.class_trait_view(
    'traits_view',
    View(
        [
            ['x_min', 'y_min', 'z_min', '|'],
            ['x_max', 'y_max', 'z_max', 'increment', 'size~{Grid size}', '|'],
            '-[Map extension]',
        ],
    ),
)

BeamformerTime.class_trait_view(
    'traits_view',
    View(
        [
            [Item('steer{}', style='custom')],
            [Item('source{}', style='custom'), '-<>'],
            [Item('weights{}', style='simple')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerTimeSq.class_trait_view(
    'traits_view',
    View(
        [
            [Item('steer{}', style='custom')],
            [Item('source{}', style='custom'), '-<>'],
            [Item('r_diag', label='diagonal removed')],
            [Item('weights{}', style='simple')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerTimeTraj.class_trait_view(
    'traits_view',
    View(
        [
            [Item('steer{}', style='custom')],
            [Item('source{}', style='custom'), '-<>'],
            [Item('trajectory{}', style='custom')],
            [Item('weights{}', style='simple')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerTimeSqTraj.class_trait_view(
    'traits_view',
    View(
        [
            [Item('steer{}', style='custom')],
            [Item('source{}', style='custom'), '-<>'],
            [Item('trajectory{}', style='custom')],
            [Item('r_diag', label='diagonal removed')],
            [Item('weights{}', style='simple')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

IntegratorSectorTime.class_trait_view(
    'traits_view',
    View(
        [
            [Item('sectors', style='custom')],
            [Item('grid', style='custom'), '-<>'],
            '|',
        ],
        title='Integrator',
        buttons=OKCancelButtons,
    ),
)

UniformFlowEnvironment.class_trait_view(
    'traits_view',
    View(
        [
            ['ma{Flow Mach number}', 'fdv{Flow vector}'],
            '|[Uniform Flow]',
        ],
    ),
)

GeneralFlowEnvironment.class_trait_view(
    'traits_view',
    View(
        [
            ['ff{Flow field}', 'N{Max. number of rays}', 'Om{Max. solid angle }'],
            '|[General Flow]',
        ],
    ),
)

SlotJet.class_trait_view(
    'traits_view',
    View(
        [
            ['v0{Exit velocity}', 'origin{Jet origin}', 'flow', 'plane', 'B{Slot width}'],
            '|[Slot jet]',
        ],
    ),
)

OpenJet.class_trait_view(
    'traits_view',
    View(
        [
            ['v0{Exit velocity}', 'origin{Jet origin}', 'D{Nozzle diameter}'],
            '|[Open jet]',
        ],
    ),
)

RotatingFlow.class_trait_view(
    'traits_view',
    View(
        [
            ['v0{flow velocity}', 'origin{Jet origin}', 'rpm{ revolutions }'],
            '|[RotatingFlow]',
        ],
    ),
)

TimeOut.class_trait_view(
    'traits_view',
    View(
        Item('source', style='custom'),
    ),
)

Average.class_trait_view(
    'traits_view',
    View(
        [
            Item('source', style='custom'),
            'naverage{Samples to average}',
            ['sample_freq~{Output sampling frequency}', '|[Properties]'],
            '|',
        ],
        title='Linear average',
        buttons=OKCancelButtons,
    ),
)

FiltFiltOctave.class_trait_view(
    'traits_view',
    View(
        [
            Item('source', style='custom'),
            'band{Center frequency}',
            'fraction{Bandwidth}',
            ['sample_freq~{Output sampling frequency}', '|[Properties]'],
            '|',
        ],
        title='Linear average',
        buttons=OKCancelButtons,
    ),
)

WriteWAV.class_trait_view(
    'traits_view',
    View(
        [
            Item('source', style='custom'),
            ['basename~{File name}', '|[Properties]'],
            '|',
        ],
        title='Write wav file',
        buttons=OKCancelButtons,
    ),
)

WriteH5.class_trait_view(
    'traits_view',
    View(
        [
            Item('source', style='custom'),
            ['name{File name}', '|[Properties]'],
            '|',
        ],
        title='write .h5',
        buttons=OKCancelButtons,
    ),
)

TimeSamples.class_trait_view(
    'traits_view',
    View(
        [
            'name{File name}',
            [
                'sample_freq~{Sampling frequency}',
                'num_channels~{Number of channels}',
                'num_samples~{Number of samples}',
                '|[Properties]',
            ],
            '|',
        ],
        title='Time data',
        buttons=OKCancelButtons,
    ),
)

MaskedTimeSamples.class_trait_view(
    'traits_view',
    View(
        [
            'name{File name}',
            ['start{From sample}', Item('stop', label='to', style='text'), '-'],
            'invalid_channels{Invalid channels}',
            [
                'sample_freq~{Sampling frequency}',
                'num_channels~{Number of channels}',
                'num_samples~{Number of samples}',
                '|[Properties]',
            ],
            '|',
        ],
        title='Time data',
        buttons=OKCancelButtons,
    ),
)

SourceMixer.class_trait_view(
    'traits_view',
    View(
        Item('sources', style='custom'),
    ),
)


BeamformerBase.class_trait_view(
    'traits_view',
    View(
        [
            [Item('r_diag', label='Diagonal removed')],
            [Item('steer', label='Steering vector')],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerFunctional.class_trait_view(
    'traits_view',
    View(
        [
            #            [Item('mics{}', style='custom')],
            #            [Item('grid', style='custom'), '-<>'],
            [Item('gamma', label='Exponent', style='simple')],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerCapon.class_trait_view(
    'traits_view',
    View(
        [
            #            [Item('mics{}', style='custom')],
            #            [Item('grid', style='custom'), '-<>'],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerEig.class_trait_view(
    'traits_view',
    View(
        [
            #            [Item('mics{}', style='custom')],
            #            [Item('grid', style='custom'), '-<>'],
            [Item('n', label='Component No.', style='simple')],
            [Item('r_diag', label='Diagonal removed')],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerMusic.class_trait_view(
    'traits_view',
    View(
        [
            #            [Item('mics{}', style='custom')],
            #            [Item('grid', style='custom'), '-<>'],
            [Item('n', label='No. of sources', style='simple')],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerDamas.class_trait_view(
    'traits_view',
    View(
        [
            [Item('beamformer{}', style='custom')],
            [Item('n_iter{Number of iterations}')],
            #            [Item('steer{Type of steering vector}')],
            [Item('calcmode{How to calculate PSF}')],
            '|',
        ],
        title='Beamformer denconvolution options',
        buttons=OKCancelButtons,
    ),
)

BeamformerDamasPlus.class_trait_view(
    'traits_view',
    View(
        [
            [Item('beamformer{}', style='custom')],
            [Item('method{Solver}')],
            [Item('n_iter{Max. number of iterations}')],
            [Item('alpha', label='Lasso weight factor')],
            [Item('calcmode{How to calculate PSF}')],
            '|',
        ],
        title='Beamformer denconvolution options',
        buttons=OKCancelButtons,
    ),
)

BeamformerOrth.class_trait_view(
    'traits_view',
    View(
        [
            #            [Item('mpos{}', style='custom')],
            #            [Item('grid', style='custom'), '-<>'],
            [Item('n', label='Number of components', style='simple')],
            [Item('r_diag', label='Diagonal removed')],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerCleansc.class_trait_view(
    'traits_view',
    View(
        [
            #            [Item('mpos{}', style='custom')],
            #            [Item('grid', style='custom'), '-<>'],
            [Item('n_iter', label='No. of iterations', style='simple')],
            [Item('r_diag', label='Diagonal removed')],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerClean.class_trait_view(
    'traits_view',
    View(
        [
            [Item('beamformer{}', style='custom')],
            [Item('n_iter{Number of iterations}')],
            #            [Item('steer{Type of steering vector}')],
            [Item('calcmode{How to calculate PSF}')],
            '|',
        ],
        title='Beamformer denconvolution options',
        buttons=OKCancelButtons,
    ),
)

BeamformerCMF.class_trait_view(
    'traits_view',
    View(
        [
            #            [Item('mpos{}', style='custom')],
            #            [Item('grid', style='custom'), '-<>'],
            [Item('method', label='Fit method')],
            [Item('n_iter', label='(Max.) no. of iterations')],
            [Item('alpha', label='Lasso weight factor')],
            [Item('c', label='Speed of sound')],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)

BeamformerGIB.class_trait_view(
    'traits_view',
    View(
        [
            #            [Item('mpos{}', style='custom')],
            #            [Item('grid', style='custom'), '-<>'],
            [Item('method', label='Fit method')],
            [Item('n_iter', label='(Max.) no. of iterations')],
            [Item('alpha', label='Lasso weight factor')],
            [Item('c', label='Speed of sound')],
            #            [Item('env{}', style='custom')],
            '|',
        ],
        title='Beamformer options',
        buttons=OKCancelButtons,
    ),
)
