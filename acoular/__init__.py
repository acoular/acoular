# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""The Acoular library: several classes for the implementation of acoustic beamforming."""

import os  # noqa: I001

# config must be imported before any submodules containing numpy, see #322.
from .configuration import config

from . import demo, tools, aiaa
from .base import (
    Generator,
    InOut,
    SamplesGenerator,
    SpectraGenerator,
    SpectraOut,
    TimeOut,
)
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
from .fprocess import IRFFT, RFFT, AutoPowerSpectra, CrossPowerSpectra
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
from .process import Average, Cache, SampleSplitter
from .sdinput import SoundDeviceSamplesGenerator
from .signals import (
    FiltWNoiseGenerator,
    GenericSignalGenerator,
    NoiseGenerator,
    PeriodicSignalGenerator,
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
from .tools.helpers import synthetic
from .spectra import BaseSpectra, PowerSpectra, PowerSpectraImport
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
    MaskedTimeOut,
    Mixer,
    OctaveFilterBank,
    SpatialInterpolator,
    SpatialInterpolatorConstantRotation,
    SpatialInterpolatorRotation,
    TimeConvolve,
    TimeCumAverage,
    TimeExpAverage,
    TimePower,
    TimeReverse,
    Trigger,
    WriteH5,
    WriteWAV,
)
from .trajectory import Trajectory
from .version import __author__, __date__, __version__

# Register the Acoular matplotlib style
def _register_plot_style():
    """Register the Acoular matplotlib style with matplotlib's style library."""
    try:
        from pathlib import Path

        from matplotlib import rc_params_from_file, style

        style_path = Path(__file__).parent / 'plots.mplstyle'
        if style_path.exists():
            style_dict = rc_params_from_file(str(style_path), use_default_template=False)
            style.library['acoular.plots'] = style_dict
    except ImportError:
        # matplotlib is not installed, skip style registration
        pass

_register_plot_style()
del _register_plot_style  # Clean up namespace
