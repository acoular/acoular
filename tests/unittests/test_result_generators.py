# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Basic tests for result generators."""

import acoular as ac
import numpy as np
from pytest_cases import parametrize, parametrize_with_cases

from tests.cases.test_generator_cases import Generators
from tests.utils import get_result


# @given(num=st.integers(min_value=1, max_value=1000))
# suppress health check (fixtures are not reset!)
# @settings(max_examples=4, suppress_health_check=HealthCheck)
@parametrize('num', [1, 32], ids=['num=1', 'num=32'])
@parametrize_with_cases('obj', cases=Generators)
def test_result_generator(obj, num):  # don't use default value if @parametrize is used
    """Basic test for result generators if they return numpy arrays with correct shape.

    Parameters
    ----------
    obj : instance of acoular.base.Generator
        Generator instance to be tested (cases from Generators)
    num : int
        Number of samples to return by the generator
    """
    result = get_result(obj, num=num)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == min(obj.num_samples, num)
    obj_copy = obj
    while isinstance(obj_copy, ac.InOut):
        obj_copy = obj_copy.source
    if isinstance(obj_copy, ac.SamplesGenerator):
        assert result.shape[1] == obj.num_channels
    elif isinstance(obj_copy, ac.SpectraGenerator):
        assert result.shape[1] == obj.num_channels * obj.num_freqs
