# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for result generators."""

import acoular as ac

from tests.cases.test_generator_cases import Generators
from tests.utils import get_result

import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases


# @given(num=st.integers(min_value=1, max_value=1000))
# suppress health check (fixtures are not reset!)
# @settings(max_examples=4, suppress_health_check=HealthCheck)
@parametrize('num', [1, 32], ids=['num=1', 'num=32'])
@parametrize_with_cases('obj', cases=Generators)
def test_result_generator(obj, num):  # don't use default value if @parametrize is used
    """
    Test for result generators: checks block sizes and total sample count.

    Parameters
    ----------
    obj : instance of :class:`~acoular.base.Generator`
        Generator instance to be tested (cases from Generators)
    num : :class:`int`
        Number of samples to return by the generator
    """
    xfails = {
        ac.BeamformerTime: 'Issue #525',
        ac.IntegratorSectorTime: 'Issue #525',  # test case relies on BeamformerTime as source
    }
    if reason := xfails.get(type(obj)):
        pytest.xfail(reason)

    blocks = get_result(obj, num=num, f=list)
    assert len(blocks) > 0, 'Generator did not yield any blocks.'

    obj_copy = obj
    while isinstance(obj_copy, ac.InOut):
        obj_copy = obj_copy.source
    if isinstance(obj_copy, ac.SamplesGenerator):
        expected_cols = obj.num_channels
    elif isinstance(obj_copy, ac.SpectraGenerator):
        expected_cols = obj.num_channels * obj.num_freqs
    else:
        msg = 'Unknown generator type for shape check.'
        raise AssertionError(msg)

    for block in blocks[:-1]:
        assert isinstance(block, np.ndarray)
        assert block.shape[0] == num, f'Non-final block has wrong size: {block.shape[0]} != {num}'
        assert block.shape[1] == expected_cols

    last_block = blocks[-1]
    assert isinstance(last_block, np.ndarray)
    assert last_block.shape[0] > 0, 'Last block is empty.'
    assert last_block.shape[1] == expected_cols

    result = np.concatenate(blocks, axis=0)
    assert result.shape[0] == obj.num_samples, f'Total samples {result.shape[0]} != {obj.num_samples}'


def _time_samples_sources(generator, seen=None):
    """Yield TimeSamples objects in a generator's source graph."""
    if seen is None:
        seen = set()
    if id(generator) in seen:
        return
    seen.add(id(generator))

    if isinstance(generator, ac.TimeSamples):
        yield generator
    source = getattr(generator, 'source', None)
    if isinstance(source, ac.Generator):
        yield from _time_samples_sources(source, seen)
    for source in getattr(generator, 'sources', []):
        if isinstance(source, ac.Generator):
            yield from _time_samples_sources(source, seen)


def _retained_blocks(generator):
    first = next(generator)
    expected = first.copy()
    try:
        second = next(generator)
    except StopIteration:
        return None
    return first, expected, second


@parametrize('num', [1])
@parametrize_with_cases('obj', cases=Generators)
def test_result_block_stability(obj, num):
    """A retained block must not change when its generator advances."""
    xfails = {
        ac.IRFFT: 'Issue #654',
        ac.IntegratorSectorTime: 'Issue #654',
        ac.MaskedTimeOut: 'Issue #654',
        ac.MovingPointSourceDipole: 'Issue #654',
        ac.OctaveFilterBank: 'Issue #654',
        ac.PointSource: 'Issue #654',
        ac.PointSourceDipole: 'Issue #654',
        ac.RFFT: 'Issue #654',
        ac.TimeReverse: 'Issue #654',
    }
    if reason := xfails.get(type(obj)):
        pytest.xfail(reason)

    retained_blocks = get_result(obj, num=num, f=_retained_blocks)
    if retained_blocks is None:
        return
    first, expected, second = retained_blocks

    np.testing.assert_array_equal(first, expected)
    assert not np.shares_memory(first, second)


@parametrize('num', [1])
@parametrize_with_cases('obj', cases=Generators)
def test_result_does_not_modify_source_blocks(obj, num):
    """Processors must not modify data owned by their input source."""
    xfails = {
        ac.Mixer: 'Issue #595',
        ac.SourceMixer: 'Issue #595',
    }
    if reason := xfails.get(type(obj)):
        pytest.xfail(reason)

    source_data = [(source, np.array(source.data, copy=True)) for source in _time_samples_sources(obj)]
    get_result(obj, num=num)

    for source, expected in source_data:
        np.testing.assert_array_equal(source.data, expected)
