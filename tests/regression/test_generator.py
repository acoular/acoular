# get all classes in module acoular.tprocess


from pytest_cases import parametrize_with_cases
from test_generator_cases import Generators


@parametrize_with_cases("obj", cases=Generators)
def test_generators(snapshot, obj):
    gen = obj.result(num=1)
    result = next(gen)  # Trigger calculation
    snapshot.check(result, rtol=5e-5, atol=5e-8)  # uses numpy.testing.assert_allclose

