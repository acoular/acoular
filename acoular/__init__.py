# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""The Acoular library: several classes for the implementation of acoustic beamforming."""

import os

from .configuration import config
from .version import __author__, __date__, __version__

if config.have_sounddevice:
    from .sdinput import SoundDeviceSamplesGenerator

from . import demo, tools
from .calib import Calib
from .environments import (
    Environment,
    FlowField,
    GeneralFlowEnvironment,
    OpenJet,
    RotatingFlow,
    SlotJet,
    UniformFlowEnvironment,
    cartToCyl,
    cylToCart,
)
from .fbeamform import (
    BeamformerAdaptiveGrid,
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
    BeamformerGridlessOrth,
    BeamformerMusic,
    BeamformerOrth,
    BeamformerSODIX,
    L_p,
    PointSpreadFunction,
    SteeringVector,
    integrate,
)
from .grids import (
    CircSector,
    ConvexSector,
    Grid,
    ImportGrid,
    LineGrid,
    MergeGrid,
    MultiSector,
    PolySector,
    RectGrid,
    RectGrid3D,
    RectSector,
    RectSector3D,
    Sector,
)
from .microphones import MicGeom
from .signals import (
    FiltWNoiseGenerator,
    GenericSignalGenerator,
    PNoiseGenerator,
    SignalGenerator,
    SineGenerator,
    WNoiseGenerator,
)
from .sources import (
    LineSource,
    MaskedTimeSamples,
    MovingLineSource,
    MovingPointSource,
    MovingPointSourceDipole,
    PointSource,
    PointSourceConvolve,
    PointSourceDipole,
    SourceMixer,
    SphericalHarmonicSource,
    TimeSamples,
    UncorrelatedNoiseSource,
)
from .spectra import BaseSpectra, FFTSpectra, PowerSpectra, PowerSpectraImport, synthetic
from .spectra import PowerSpectra as EigSpectra
from .tbeamform import (
    BeamformerCleant,
    BeamformerCleantSq,
    BeamformerCleantSqTraj,
    BeamformerCleantTraj,
    BeamformerTime,
    BeamformerTimeSq,
    BeamformerTimeSqTraj,
    BeamformerTimeTraj,
    IntegratorSectorTime,
)
from .tprocess import (
    AngleTracker,
    ChannelMixer,
    Filter,
    FilterBank,
    FiltFiltOctave,
    FiltFreqWeight,
    FiltOctave,
    MaskedTimeInOut,
    Mixer,
    OctaveFilterBank,
    SamplesGenerator,
    SampleSplitter,
    SpatialInterpolator,
    SpatialInterpolatorConstantRotation,
    SpatialInterpolatorRotation,
    TimeAverage,
    TimeCache,
    TimeConvolve,
    TimeCumAverage,
    TimeExpAverage,
    TimeInOut,
    TimePower,
    TimeReverse,
    Trigger,
    WriteH5,
    WriteWAV,
)
from .trajectory import Trajectory
