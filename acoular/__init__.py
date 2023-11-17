# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------

"""
The Acoular library: several classes for the implemetation of 
acoustic beamforming
"""

from .version import __author__, __date__, __version__

import os 

from .configuration import config

from .fileimport import time_data_import, csv_import, td_import, \
bk_mat_import, datx_import
try:
    from .nidaqimport import nidaq_import
except:
    pass

try:
    from .sdinput import SoundDeviceSamplesGenerator
except:
    pass

from .tbeamform import IntegratorSectorTime, \
BeamformerTime, BeamformerTimeSq, BeamformerTimeTraj, BeamformerTimeSqTraj,\
BeamformerCleant, BeamformerCleantSq, BeamformerCleantTraj, BeamformerCleantSqTraj
from .tprocess import SamplesGenerator, TimeInOut, MaskedTimeInOut, ChannelMixer, \
Mixer, TimeAverage, TimeReverse, TimePower, FiltFiltOctave, FiltOctave, TimeCache, \
WriteWAV, WriteH5, SpatialInterpolator, SpatialInterpolatorRotation, Trigger, \
SampleSplitter, AngleTracker, SpatialInterpolatorConstantRotation, Filter, \
TimeExpAverage, FiltFreqWeight, TimeCumAverage, FilterBank, OctaveFilterBank, TimeConvolve
from .calib import Calib
from .trajectory import Trajectory
from .grids import Grid, RectGrid, RectGrid3D, Sector, RectSector, RectSector3D, CircSector,\
    PolySector, MultiSector, MergeGrid, LineGrid, ImportGrid, ConvexSector
from .environments import cartToCyl, cylToCart, Environment, UniformFlowEnvironment, RotatingFlow, \
FlowField, OpenJet, SlotJet, GeneralFlowEnvironment
from .microphones import MicGeom
from .spectra import BaseSpectra, FFTSpectra, PowerSpectra, PowerSpectraImport, PowerSpectra as EigSpectra, synthetic

from .fbeamform import BeamformerBase, BeamformerCapon, BeamformerEig, \
BeamformerMusic, BeamformerDamas, BeamformerDamasPlus, BeamformerOrth, BeamformerCleansc, \
BeamformerCMF,BeamformerSODIX, BeamformerClean, BeamformerFunctional, BeamformerGIB, L_p, integrate, \
PointSpreadFunction, SteeringVector, BeamformerAdaptiveGrid, BeamformerGridlessOrth

from .sources import PointSource, MovingPointSource, \
TimeSamples, MaskedTimeSamples, PointSourceDipole, UncorrelatedNoiseSource, \
SourceMixer, SphericalHarmonicSource, LineSource, MovingPointSourceDipole, \
MovingLineSource, PointSourceConvolve
from .signals import SineGenerator, WNoiseGenerator, SignalGenerator,\
PNoiseGenerator, GenericSignalGenerator, FiltWNoiseGenerator

from . import tools

from . import demo
