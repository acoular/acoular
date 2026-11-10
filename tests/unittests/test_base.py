# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for the common Acoular object base class."""

from abc import abstractmethod

import acoular as ac
from acoular.internal import digest

from tests.utils import get_all_classes

import numpy as np
import pytest
from traits.api import Instance, Int, Property, TraitError, property_depends_on


class DigestObject(ac.AcoularBase):
    value = Int(1)

    digest = Property(fget=ac.AcoularBase._get_digest, depends_on=['value'])


class ExtendedDigestObject(DigestObject):
    extra = Int(2)

    digest = Property(fget=ac.AcoularBase._get_digest, depends_on=['value', 'extra'])


class NestedDigestObject(ac.AcoularBase):
    source = Instance(DigestObject)

    digest = Property(fget=ac.AcoularBase._get_digest, depends_on=['source.digest'])


@pytest.fixture(params=[ac.Generator, ac.SamplesGenerator, ac.SpectraGenerator])
def generator_cls(request):
    class ConcreteGenerator(request.param):
        def result(self, num):
            yield np.zeros((num, self.num_channels))

    return ConcreteGenerator


def test_acoular_base_rejects_unknown_traits():
    obj = ac.AcoularBase()
    with pytest.raises(TraitError):
        obj.unknown = 1


def test_acoular_base_supports_abstract_methods():
    class AbstractObject(ac.AcoularBase):
        @abstractmethod
        def result(self):
            pass

    with pytest.raises(TypeError, match='abstract'):
        AbstractObject()


def test_shared_digest_getter_caches_and_invalidates(mocker):
    getter = mocker.spy(ac.base, 'digest')
    obj = DigestObject()
    before = obj.digest
    assert obj.digest == before
    assert getter.call_count == 1
    assert before == digest(obj)

    obj.value = 2
    assert obj.digest != before
    assert getter.call_count == 2
    assert obj.digest == digest(obj)


def test_digest_is_read_only():
    with pytest.raises(TraitError):
        DigestObject().digest = 'replacement'


@pytest.mark.parametrize('name', ['value', 'extra'])
def test_redeclared_digest_uses_shared_getter(name):
    obj = ExtendedDigestObject()
    before = obj.digest
    setattr(obj, name, 3)
    assert obj.digest != before
    assert obj.digest == digest(obj)


def test_nested_digest_invalidates_on_source_change_and_replacement():
    source = DigestObject()
    obj = NestedDigestObject(source=source)
    before = obj.digest

    source.value = 2
    assert obj.digest != before
    before = obj.digest

    obj.source = DigestObject(value=3)
    assert obj.digest != before
    assert obj.digest == digest(obj)


def test_inherited_digest_includes_subclass_identity():
    class InheritedDigestObject(DigestObject):
        pass

    obj = DigestObject()
    inherited = InheritedDigestObject()
    assert inherited.digest == digest(inherited)
    assert inherited.digest != obj.digest
    assert DigestObject(value=obj.value).digest == obj.digest


@pytest.mark.parametrize(
    'block_cls',
    [
        cls
        for cls in get_all_classes(hastraits_only=True)
        if issubclass(cls, ac.TimeOut) and cls.class_traits()['digest'].cached
    ],
)
def test_timeout_classes_use_shared_cached_getter(block_cls):
    assert issubclass(block_cls, ac.AcoularBase)
    assert block_cls.class_traits()['digest'].property_fields[0] is ac.AcoularBase._get_digest
    assert '_get_digest' not in block_cls.__dict__


def test_environment_digest_caches_and_invalidates(mocker):
    getter = mocker.spy(ac.base, 'digest')
    obj = ac.Environment()
    before = obj.digest
    assert obj.digest == before
    assert before == digest(obj)
    assert getter.call_count == 1

    obj.c = 340.0
    assert obj.digest != before
    assert obj.digest == digest(obj)
    assert getter.call_count == 2

    obj.roi = np.ones((3, 2))
    assert obj.digest == digest(obj)
    assert getter.call_count == 2


def test_environment_digest_invalidates_steering():
    env = ac.Environment()
    steer = ac.SteeringVector(env=env)
    before = steer.digest
    env.c = 340.0
    assert steer.digest != before
    assert steer.digest == digest(steer)

    before = steer.digest
    steer.env = ac.Environment(c=330.0)
    assert steer.digest != before
    assert steer.digest == digest(steer)


def test_environment_subclass_can_override_digest_getter():
    class CustomEnvironment(ac.Environment):
        def _get_digest(self):
            return f'custom_{self.c}'

    obj = CustomEnvironment()
    assert obj.digest == 'custom_343.0'
    obj.c = 340.0
    assert obj.digest == 'custom_340.0'


def test_generator_digest_caches_and_tracks_dependencies(generator_cls, mocker):
    getter = mocker.spy(ac.base, 'digest')
    obj = generator_cls(sample_freq=48000, num_samples=16, num_channels=2)
    before = obj.digest
    assert obj.digest == before
    assert getter.call_count == 1
    assert before == digest(obj)
    downstream = ac.Cache(source=obj)

    for name in obj.trait('digest').depends_on:
        before = obj.digest
        downstream_before = downstream.digest
        setattr(obj, name, getattr(obj, name) + 1)
        assert obj.digest != before
        assert obj.digest == digest(obj)
        assert downstream.digest != downstream_before
        assert downstream.digest == digest(downstream)


def test_generator_digest_tracks_computed_properties(generator_cls):
    class ComputedGenerator(generator_cls):
        length = Int(16)
        num_samples = Property(depends_on=['length'])

        @property_depends_on(['length'])
        def _get_num_samples(self):
            return self.length

    obj = ComputedGenerator()
    before = obj.digest
    obj.length = 32
    assert obj.digest != before
    assert obj.digest == digest(obj)


def test_generator_subclass_can_override_digest_getter(generator_cls):
    class CustomGenerator(generator_cls):
        def _get_digest(self):
            return f'custom_{self.num_samples}'

    obj = CustomGenerator(num_samples=16)
    assert obj.digest == 'custom_16'
    obj.num_samples = 32
    assert obj.digest == 'custom_32'


def test_beamformer_time_digest_caches_and_tracks_pipeline_changes(mocker):
    getter = mocker.spy(ac.base, 'digest')
    obj = ac.BeamformerTime(
        source=ac.TimeSamples(data=np.ones((16, 2)), sample_freq=48000),
        steer=ac.SteeringVector(grid=ac.RectGrid(), mics=ac.MicGeom(pos_total=np.zeros((3, 2))), env=ac.Environment()),
    )
    before = obj.digest
    assert obj.digest == before
    assert sum(call.args[0] is obj for call in getter.call_args_list) == 1
    assert before == digest(obj)
    downstream = ac.TimePower(source=obj)

    changes = [
        ('source.sample_freq', 44100),
        ('steer.env.c', 340.0),
        ('steer.grid.increment', 0.2),
        ('steer.mics.pos_total', np.ones((3, 2))),
        ('steer.mics.invalid_channels', [0]),
        ('steer.ref', 1.0),
        ('steer.steer_type', 'classic'),
        ('weights', 'power'),
        ('source', ac.TimeSamples(data=np.ones((16, 2)), sample_freq=32000)),
        (
            'steer',
            ac.SteeringVector(grid=ac.RectGrid(), mics=ac.MicGeom(pos_total=np.zeros((3, 2))), env=ac.Environment()),
        ),
    ]
    for path, value in changes:
        before = obj.digest
        downstream_before = downstream.digest
        target = obj
        *parents, name = path.split('.')
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, name, value)
        assert obj.digest != before, path
        assert obj.digest == digest(obj), path
        assert downstream.digest != downstream_before, path
        assert downstream.digest == digest(downstream), path

    before = obj.digest
    downstream_before = downstream.digest
    obj.steer.mics.invalid_channels.append(0)
    assert obj.digest != before
    assert obj.digest == digest(obj)
    assert downstream.digest != downstream_before


def test_beamformer_time_subclass_can_override_digest_getter():
    class CustomBeamformer(ac.BeamformerTime):
        def _get_digest(self):
            return f'custom_{self.weights}'

    obj = CustomBeamformer()
    assert obj.digest == 'custom_none'
    obj.weights = 'power'
    assert obj.digest == 'custom_power'


def test_timeout_digest_tracks_pipeline_changes():
    source = ac.TimeSamples(data=np.ones((16, 2)), sample_freq=48000)
    power = ac.TimePower(source=source)
    block = ac.ChannelMixer(source=power, weights=[1.0, 1.0])
    before = block.digest
    assert before == digest(block)

    source.sample_freq = 44100
    assert block.digest != before
    before = block.digest

    block.weights = [0.5, 1.0]
    assert block.digest != before
    before = block.digest

    power.source = ac.TimeSamples(data=np.ones((16, 2)), sample_freq=32000)
    assert block.digest != before
    assert block.digest == digest(block)


def test_timeout_inherited_digest_includes_class_identity():
    source = ac.TimeSamples(data=np.ones((16, 2)))
    assert ac.TimePower(source=source).digest != ac.TimeCumAverage(source=source).digest


def test_timeout_subclass_can_override_getter():
    class CustomTimePower(ac.TimePower):
        def _get_digest(self):
            return 'custom'

    assert CustomTimePower().digest == 'custom'


@pytest.mark.parametrize(
    'obj_cls', [cls for cls in get_all_classes(hastraits_only=True) if 'digest' in cls.class_traits()]
)
def test_all_digest_classes_derive_from_shared_base(obj_cls):
    assert issubclass(obj_cls, ac.AcoularBase)


@pytest.mark.parametrize(
    'obj_cls',
    [
        cls
        for cls in get_all_classes(hastraits_only=True)
        if 'digest' in cls.class_traits() and cls.class_traits()['digest'].cached
    ],
)
def test_all_cached_digest_classes_use_shared_getter(obj_cls):
    assert obj_cls.class_traits()['digest'].property_fields[0] is ac.AcoularBase._get_digest
    assert '_get_digest' not in obj_cls.__dict__


def test_base_spectra_is_concrete_with_shared_digest():
    obj = ac.BaseSpectra()
    assert obj.digest == digest(obj)
    before = obj.digest
    obj.block_size = 128
    assert obj.digest != before


def test_noise_subclass_does_not_need_a_digest_getter():
    class CustomNoise(ac.NoiseGenerator):
        def signal(self):
            return np.zeros(self.num_samples)

    obj = CustomNoise(num_samples=4)
    np.testing.assert_array_equal(obj.signal(), np.zeros(4))
    assert obj.digest == digest(obj)
    before = obj.digest
    obj.seed = 1
    assert obj.digest != before


def test_flow_subclass_does_not_need_a_digest_getter():
    class ConstantFlow(ac.FlowField):
        def v(self, _xx):
            return np.zeros(3), np.zeros((3, 3))

    obj = ConstantFlow()
    assert obj.digest == digest(obj)


@pytest.mark.parametrize(
    'obj_cls', [ac.FlowField, ac.Grid, ac.SignalGenerator, ac.PeriodicSignalGenerator, ac.NoiseGenerator]
)
def test_domain_interfaces_remain_abstract(obj_cls):
    with pytest.raises(TypeError, match='abstract'):
        obj_cls()


def test_grid_subclass_does_not_need_a_digest_getter():
    class SinglePointGrid(ac.Grid):
        def _get_size(self):
            return 1

        def _get_shape(self):
            return (1,)

        def _get_pos(self):
            return np.zeros((3, 1))

    obj = SinglePointGrid()
    assert obj.size == 1
    assert obj.shape == (1,)
    np.testing.assert_array_equal(obj.pos, np.zeros((3, 1)))
    assert obj.digest == digest(obj)


def test_grid_position_getter_remains_abstract():
    assert '_get_pos' in ac.Grid.__abstractmethods__


def test_existing_adaptive_grid_instantiation_is_preserved():
    obj = ac.BeamformerAdaptiveGrid()
    assert obj.digest == digest(obj)
