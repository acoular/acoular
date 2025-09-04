# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for result generators."""

import acoular as ac
import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

from tests.cases.test_generator_cases import Generators
from tests.utils import get_result_list

xfails = {
    ac.BeamformerTime: 'Issue #',
    ac.IntegratorSectorTime: 'Issue #',  # test case relies on BeamformerTime as source
    ac.PointSourceConvolve: 'Issue #',  # test case relies on TimeConvolve as generator
}


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
    for gen, reason in xfails.items():
        if isinstance(obj, gen):
            pytest.xfail(reason)

    blocks = get_result_list(obj, num=num)
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
