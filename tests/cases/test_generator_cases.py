# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implement test cases for all :class:`acoular.base.Generator` derived classes."""

import warnings

import acoular as ac
import numpy as np
import pytest
from pytest_cases import case, get_case_id, parametrize, parametrize_with_cases

from tests.cases.test_grid_cases import Sectors
from tests.utils import get_subclasses

SKIP_DEFAULT = [
    ac.Generator,
    ac.InOut,
    ac.SamplesGenerator,
    ac.SpectraGenerator,
    ac.SpectraOut,
    ac.TimeInOut,
    ac.TimeOut,
    ac.SourceMixer,
    ac.TimeConvolve,
    ac.UncorrelatedNoiseSource,
    ac.LineSource,
    ac.PointSource,
    ac.PointSourceDipole,
    ac.SphericalHarmonicSource,
    ac.Average,
    ac.TimeSamples,
    ac.MaskedTimeSamples,
    ac.BeamformerTime,
    ac.BeamformerTimeSq,
    ac.BeamformerCleant,
    ac.BeamformerCleantSq,
    ac.BeamformerTimeTraj,
    ac.BeamformerTimeSqTraj,
    ac.BeamformerCleantTraj,
    ac.BeamformerCleantSqTraj,
    ac.FFTSpectra,
    ac.IntegratorSectorTime,
    ac.MovingPointSource,
    ac.MovingLineSource,
    ac.MovingPointSourceDipole,
    ac.FilterBank,
    ac.OctaveFilterBank,
    ac.RFFT,
    ac.SampleSplitter,
    ac.SoundDeviceSamplesGenerator,
    ac.PointSourceConvolve,
    ac.SpatialInterpolator,
    ac.SpatialInterpolatorConstantRotation,
    ac.SpatialInterpolatorRotation,
    ac.WriteH5,
    ac.WriteWAV,
    ac.Calib,
    # deprecated:
    ac.TimeAverage,
    ac.MaskedTimeInOut,
    ac.TimeCache,
]

DEFAULT = [cls for cls in get_subclasses(ac.Generator) if cls not in SKIP_DEFAULT]


def case_single_microphone():
    return ac.MicGeom(pos_total=np.array([[0, 0, 0]]).T)


def case_two_microphones():
    return ac.MicGeom(pos_total=np.array([[0, 0, 0], [0, 1, 0]]).T)


class Generators:
    """Test cases for all :class:`acoular.base.Generator` derived classes.

    New generators should be added here. If no dedicated test case is added for a
    :class:`acoular.base.Generator` derived class, the class is still included in the test suite
    through the use of the `case_default` case. If a dedicated test case was added for a generator,
    it should be added to the `SKIP_DEFAULT` list, which excludes the class from `case_default`.

    """

    @parametrize('acoular_cls', DEFAULT)
    def case_default(self, acoular_cls, time_data_source):
        source = time_data_source
        if acoular_cls.__name__ in SKIP_DEFAULT:
            pytest.skip()
        traits = acoular_cls.class_traits()
        if traits.get('source'):
            if 'SpectraGenerator' in traits['source'].info() and not issubclass(acoular_cls, ac.RFFT):
                source = ac.RFFT(source=source, block_size=8)
            return acoular_cls(source=source)
        return acoular_cls()

    def case_TimeSamples(self, small_source_case):
        return ac.TimeSamples(file=small_source_case.time.file)

    def case_MaskedTimeSamples(self, small_source_case):
        return ac.MaskedTimeSamples(file=small_source_case.time.file, start=0, stop=50)

    @parametrize_with_cases('mic_setup', cases=[case_single_microphone, case_two_microphones], ids=['1ch', '2ch'])
    def case_UncorrectedNoiseSource(self, mic_setup):
        return ac.UncorrelatedNoiseSource(
            signal=ac.WNoiseGenerator(sample_freq=1000, num_samples=50, seed=1), mics=mic_setup, sample_freq=1000
        )

    def case_SourceMixer(self, time_data_source):
        return ac.SourceMixer(sources=[time_data_source])

    @parametrize('weight', ['F', 'S', 'I'])
    def case_TimeExpAverage(self, time_data_source, weight):
        return ac.TimeExpAverage(source=time_data_source, weight=weight)

    def case_Average(self, time_data_source):
        return ac.Average(source=time_data_source, num_per_average=2)

    @case(id='sources')
    @parametrize_with_cases('mic_setup', cases=[case_single_microphone, case_two_microphones], ids=['1ch', '2ch'])
    @parametrize(
        'acoular_cls',
        [ac.PointSource, ac.PointSourceDipole, ac.LineSource, ac.SphericalHarmonicSource, ac.PointSourceConvolve],
    )
    def case_point_sources(self, acoular_cls, mic_setup):
        src = acoular_cls(signal=ac.WNoiseGenerator(sample_freq=1000, num_samples=50, seed=1), mics=mic_setup)
        if issubclass(acoular_cls, ac.PointSourceConvolve):
            kernel = np.zeros(5)
            kernel[1] = 1
            src.kernel = kernel
        if issubclass(acoular_cls, ac.LineSource):
            src.source_strength = [1.0]
        if issubclass(acoular_cls, ac.SphericalHarmonicSource):
            src.alpha = np.ones(1)
        return src

    @parametrize('conv_amp', [True, False])
    @parametrize('acoular_cls', [ac.MovingPointSource, ac.MovingLineSource, ac.MovingPointSourceDipole])
    def case_point_sources_traj(self, acoular_cls, conv_amp, moving_source_case):
        src = acoular_cls(
            signal=moving_source_case.signal,
            mics=moving_source_case.mics,
            trajectory=moving_source_case.traj,
            conv_amp=conv_amp,
        )
        if issubclass(acoular_cls, ac.LineSource):
            src.source_strength = [1.0]
        return src

    @parametrize('bf', [ac.BeamformerTime, ac.BeamformerTimeSq, ac.BeamformerCleant, ac.BeamformerCleantSq])
    @parametrize('steer_type', ['true location', 'inverse', 'classic', 'true level'])
    def case_beamformer(self, bf, steer_type, moving_source_case):
        moving_source_case.steer_fixed.steer_type = steer_type
        bf = bf(source=moving_source_case.source, steer=moving_source_case.steer_fixed)
        if issubclass(bf.__class__, ac.BeamformerCleant):
            bf.n_iter = 2
        return bf

    @parametrize(
        'bf', [ac.BeamformerTimeTraj, ac.BeamformerTimeSqTraj, ac.BeamformerCleantTraj, ac.BeamformerCleantSqTraj]
    )
    @parametrize('steer_type', ['true location', 'inverse', 'classic', 'true level'])
    @parametrize('conv_amp', [True, False])
    def case_beamformer_traj(self, bf, steer_type, conv_amp, moving_source_case):
        moving_source_case.steer_moving.steer_type = steer_type
        bf = bf(
            source=moving_source_case.source,
            steer=moving_source_case.steer_moving,
            trajectory=moving_source_case.traj,
            conv_amp=conv_amp,
        )
        if issubclass(bf.__class__, ac.BeamformerCleant):
            bf.n_iter = 2
        return bf

    @parametrize_with_cases('sector', cases=Sectors, filter=lambda cf: 'numpy' in get_case_id(cf))
    def case_IntegratorSectorTime(self, regression_source_case, sector):
        bf = ac.BeamformerTimeSq(source=regression_source_case.calib, steer=regression_source_case.steer)
        return ac.IntegratorSectorTime(source=bf, grid=regression_source_case.grid, sectors=[sector])

    def case_OctaveFilterBank(self, moving_source_case):
        return ac.OctaveFilterBank(source=moving_source_case.source, hband=30)

    def case_RFFT(self, time_data_source):
        return ac.RFFT(source=time_data_source, block_size=8)

    def case_SampleSplitter(self, time_data_source):
        ss = ac.SampleSplitter(source=time_data_source)
        tps = [ac.TimePower(source=ss) for _ in range(2)]
        ss.register_object(*tps)
        return ac.SourceMixer(sources=tps)

    @parametrize('method', ['linear', 'spline', 'rbf-multiquadric', 'rbf-cubic', 'IDW', 'custom', 'sinc'])
    def case_SpatialInterpolator(self, moving_source_case, method):
        return ac.SpatialInterpolator(mics=moving_source_case.mics, source=moving_source_case.source, method=method)

    def case_WriteH5(self, time_data_source, tmp_path):
        return ac.WriteH5(source=time_data_source, file=tmp_path / 'test.h5')

    def case_WriteWAV(self, time_data_source, tmp_path):
        return ac.WriteWAV(source=time_data_source, file=tmp_path / 'test.wav')

    def case_TimeSamplesAIAABenchmark(self, aiaa_bechmark_time_data_file):
        return ac.aiaa.TimeSamplesAIAABenchmark(file=aiaa_bechmark_time_data_file)

    def case_TriggerAIAABenchmark(self, aiaa_bechmark_trigger_file):
        return ac.aiaa.TriggerAIAABenchmark(file=aiaa_bechmark_trigger_file)

    @parametrize('domain', ['time', 'frequency'])
    def case_Calib(self, time_data_source, domain):
        if domain == 'time':
            data = time_data_source
            shape = data.num_channels
        else:
            data = ac.RFFT(source=time_data_source, block_size=8)
            shape = data.num_channels * data.num_freqs
        cal = np.abs(np.zeros(shape))
        return ac.Calib(source=data, data=cal)

    @parametrize('acoular_cls', [ac.TimeAverage, ac.MaskedTimeInOut, ac.TimeCache])
    def case_deprecated(self, time_data_source, acoular_cls):
        args = {'num_per_average': 2} if acoular_cls is ac.TimeAverage else {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            with pytest.deprecated_call():
                obj = acoular_cls(source=time_data_source, **args)
        return obj
