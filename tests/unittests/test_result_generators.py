# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Basic tests for result generators."""

import acoular as ac
import numpy as np
from pytest_cases import parametrize, parametrize_with_cases

from tests.cases.test_generator_cases import Generators


# @given(num=st.integers(min_value=1, max_value=1000))
# @settings(max_examples=4, suppress_health_check=HealthCheck) # suppress health check (fixtures are not reset!)
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
    result = next(obj.result(num=num))
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == min(obj.numsamples, num)
    if isinstance(obj, ac.SamplesGenerator):
        assert result.shape[1] == obj.numchannels
    elif isinstance(obj, ac.SpectraGenerator):
        assert result.shape[1] == obj.numchannels * obj.numfreqs
