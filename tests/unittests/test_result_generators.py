import acoular as ac
import numpy as np
from pytest_cases import parametrize, parametrize_with_cases
from tests.regression.test_generator_cases import Generators


# @given(num=st.integers(min_value=1, max_value=1000))
# @settings(max_examples=4, suppress_health_check=HealthCheck) # suppress health check (fixtures are not reset!)
@parametrize('num', [1, 32], ids=['num=1', 'num=32'])
@parametrize_with_cases('obj', cases=Generators)
def test_result_generator(obj, num):  # don't use default value if @parametrize is used
    result = next(obj.result(num=num))
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == min(obj.numsamples, num)
    if isinstance(obj, ac.SamplesGenerator):
        assert result.shape[1] == obj.numchannels
    elif isinstance(obj, ac.SpectraGenerator):
        assert result.shape[1] == obj.numchannels * obj.numfreqs
