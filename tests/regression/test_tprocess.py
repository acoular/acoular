# get all classes in module acoular.tprocess

import importlib
import inspect
import pkgutil

import acoular as ac
import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

SKIP = [
    'TimeConvolve', 'Generator', 'InOut',
    'SamplesGenerator',  'SpectraGenerator',
    'SpectraOut', 'TimeInOut', 'TimeOut',
    'SourceMixer']

def get_all_classes():
    classes = []
    package = importlib.import_module('acoular')
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_info.name:  # ensure class is defined in the current module
                classes.append(obj)
    return classes

def get_generators():
    classes = get_all_classes()
    classes_with_source_samples_generator = []
    for cls in classes:
        if issubclass(cls, ac.Generator):# and cls.class_traits().get('source'):
            classes_with_source_samples_generator.append(cls)
    return classes_with_source_samples_generator

def get_generators_time():
    classes = get_generators()
    return [cls for cls in classes if issubclass(cls, ac.SamplesGenerator)]

def get_generators_freq():
    classes = get_generators()
    return [cls for cls in classes if issubclass(cls, ac.SpectraGenerator)]

@pytest.fixture
def case_single_microphone():
    return ac.MicGeom(mpos_tot=np.array([[0, 0, 0]])[:, np.newaxis])

@pytest.fixture
def case_two_microphones():
    return ac.MicGeom(mpos_tot=np.array([[0, 0, 0], [0, 1, 0]]).T)

@pytest.fixture
def case_WNoiseGenerator():
    return ac.WNoiseGenerator(sample_freq=1000, numsamples=50, seed=1)

@pytest.fixture
def case_SineGenerator():
    return ac.SineGenerator(sample_freq=1000, numsamples=50, freq=100)

# # Create a strategy for random SineGenerator objects
# @st.composite
# def sine_generator_strategy():
#     return st.builds(
#         ac.SineGenerator,
#         sample_freq=st.floats(min_value=100.0, max_value=5000.0),
#         numsamples=st.integers(min_value=1, max_value=100000),
#         freq=st.integers(min_value=1, max_value=20000)
#     )


class Generators:
    """
    should include:
    - multi-channel (source)
    - single-channel (source)
    - different block sizes
    """

    @parametrize("acoular_cls", get_generators_time())
    def case_default(self, acoular_cls, source_case_short):
        if (acoular_cls.__name__ in SKIP):
            pytest.skip()
        source = source_case_short.source
        traits = acoular_cls.class_traits()
        if traits.get('source'):
            if "SpectraGenerator" in traits['source'].info() and not issubclass(acoular_cls, ac.RFFT):
                source = ac.RFFT(source=source)
            return acoular_cls(source=source)
        # if issubclass(acoular_cls, ac.PointSource):
        #     signal = ac.WNoiseGenerator(sample_freq=1000, numsamples=1000, seed=1)
        #     return acoular_cls(mics=source_case_short.mics, signal=signal)
        return acoular_cls()

    def case_SourceMixer(self, source_case_short):
        return ac.SourceMixer(sources=[source_case_short.source])

    @parametrize('weight', ["F", "S", "I"])
    def case_TimeExpAverage(self, source_case_short, weight):
        return ac.TimeExpAverage(source=source_case_short.source, weight=weight)

    @parametrize('signal', [case_WNoiseGenerator, case_SineGenerator])
    @parametrize('mics', [case_single_microphone, case_two_microphones])
    def case_point_sources(self, signal, mics):
        return ac.PointSource(signal=signal, mics=mics)

    # @parametrize("acoular_cls", get_generators_freq())
    # def case_result_freq(self, source_case_short):
    #     return ac.RFFT(source=source_case_short.source)


@parametrize("num", [1, 32])
@parametrize_with_cases("acoular_cls_instance", cases=Generators)
# @given(num=st.integers(min_value=1, max_value=1000))
# @settings(max_examples=4, suppress_health_check=HealthCheck) # suppress health check (fixtures are not reset!)
def test_result_generator(acoular_cls_instance, num): # don't use default value if @parametrize is used
    result = next(acoular_cls_instance.result(num=num))
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == min(acoular_cls_instance.numsamples, num)
    assert result.shape[1] == acoular_cls_instance.numchannels
    # full_result = ac.tools.return_result(acoular_cls_instance, num=num)
#    assert np.concatenate([result, full_result]).shape[0] == acoular_cls_instance.numsamples

@parametrize_with_cases("acoular_cls_instance", cases=Generators)
def test_regression(snapshot, acoular_cls_instance):
    gen = acoular_cls_instance.result(num=1)
    result = next(gen)  # Trigger calculation
    snapshot.check(result, rtol=5e-5, atol=5e-8)  # uses numpy.testing.assert_allclose

# @parametrize_with_cases("acoular_cls_instance", cases=Generators)
# def test_result_generator_freq(acoular_cls_instance):
#     # Test generator function with each class
#     gen = acoular_cls_instance.result(num=1)
#     data = next(gen)  # Trigger calculation
#     assert isinstance(data, np.ndarray)
#     assert data.shape[0] == 1
#     assert data.shape[1] == acoular_cls_instance.numchannels*acoular_cls_instance.numfreqs


@parametrize_with_cases("bf", cases=Beamformer)
def test_beamformer(bf):
    assert True
